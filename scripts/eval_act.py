#!/usr/bin/env python
"""Unified offline evaluation tool for ACT policy models.

Usage:
    # Evaluate single model (full metrics)
    python scripts/eval_act.py --model outputs/train/MODEL/checkpoints/100000/pretrained_model

    # Evaluate with custom eval steps (simulate n_action_steps)
    python scripts/eval_act.py --model PATH --eval_steps 20

    # Compare multiple models
    python scripts/eval_act.py --model PATH1 PATH2 PATH3

    # Quick diagnose: single frame detail output
    python scripts/eval_act.py --model PATH --diagnose

    # Named models (shortcut)
    python scripts/eval_act.py --model v0.1 chunk16 chunk16_dec7_aug
"""

import argparse
from collections import defaultdict
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "lepao/so101_test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SHORTCUTS = {
    "v0.1": "outputs/train/act_so101_test_v0.1/checkpoints/100000/pretrained_model",
    "chunk16": "outputs/train/act_so101_chunk16_act16/checkpoints/100000/pretrained_model",
    "chunk16_dec7_aug": "outputs/train/act_so101_chunk16_dec7_aug/checkpoints/100000/pretrained_model",
    "exp2_kl5": "outputs/train/act_so101_v02_exp2_chunk100_act20_kl5/checkpoints/100000/pretrained_model",
    "exp3_dec7": "outputs/train/act_so101_v02_exp3_chunk100_act20_dec7/checkpoints/100000/pretrained_model",
    "exp4_aug": "outputs/train/act_so101_v02_exp4_chunk100_act20_aug/checkpoints/100000/pretrained_model",
}

DEFAULT_EPISODE_SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def resolve_path(name):
    return SHORTCUTS.get(name, name)


def summarize_metric(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def parse_episode_splits(split_spec):
    if not split_spec:
        return DEFAULT_EPISODE_SPLITS.copy()

    splits = {}
    for raw_part in split_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid split specification '{part}'. Expected format name:fraction")
        name, fraction = part.split(":", 1)
        splits[name.strip()] = float(fraction.strip())

    if not splits:
        raise ValueError("No valid episode splits provided")

    total = sum(splits.values())
    if total <= 0 or total > 1.0:
        raise ValueError(f"Episode split fractions must sum to > 0 and <= 1.0, got {total}")

    return splits


def split_episode_indices(total_episodes, splits):
    ordered = list(range(total_episodes))
    result = {}
    start_idx = 0
    split_names = list(splits.keys())
    for i, split_name in enumerate(split_names):
        fraction = splits[split_name]
        end_idx = start_idx + int(total_episodes * fraction)
        if i == len(split_names) - 1:
            end_idx = total_episodes
        result[split_name] = ordered[start_idx:end_idx]
        start_idx = end_idx
    return result


def resolve_episode_subset(dataset, split_name=None, split_spec=None):
    if split_name is None:
        return None, None

    split_map = split_episode_indices(dataset.meta.total_episodes, parse_episode_splits(split_spec))
    if split_name not in split_map:
        raise ValueError(f"Unknown split '{split_name}'. Available: {sorted(split_map)}")

    episodes = split_map[split_name]
    if not episodes:
        raise ValueError(f"Split '{split_name}' has no episodes")

    return episodes, split_map


def build_sample_indices_from_groups(index_groups, num_samples, sampling_mode="global_uniform"):
    flat_indices = [idx for group in index_groups for idx in group]
    if not flat_indices:
        return []

    if num_samples is None or num_samples <= 0 or num_samples >= len(flat_indices):
        return sorted(flat_indices)

    if sampling_mode == "global_uniform":
        step = max(1, len(flat_indices) // num_samples)
        return flat_indices[::step]

    if sampling_mode != "balanced_episodes":
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    num_episodes = len(index_groups)
    target = min(num_samples, len(flat_indices))
    base = max(1, target // max(1, num_episodes))
    remainder = max(0, target - base * num_episodes)
    sampled = []

    for episode_pos, group in enumerate(index_groups):
        length = len(group)
        if length == 0:
            continue

        quota = base + (1 if episode_pos < remainder else 0)
        quota = min(quota, length)
        if quota <= 0:
            continue
        offsets = np.linspace(0, length - 1, num=quota, dtype=int).tolist()
        sampled.extend(group[offset] for offset in offsets)

    return sorted(set(sampled))


def get_local_episode_groups(dataset):
    dataset._ensure_hf_dataset_loaded()
    episode_indices = dataset.hf_dataset["episode_index"]
    episode_groups = []
    current_episode = None
    current_group = []

    for local_idx, episode_idx in enumerate(episode_indices):
        episode_idx = int(episode_idx)
        if current_episode is None or episode_idx != current_episode:
            if current_group:
                episode_groups.append(current_group)
            current_episode = episode_idx
            current_group = []
        current_group.append(local_idx)

    if current_group:
        episode_groups.append(current_group)

    return episode_groups


def build_sample_indices(dataset, num_samples, sampling_mode="global_uniform"):
    episode_groups = get_local_episode_groups(dataset)
    return build_sample_indices_from_groups(episode_groups, num_samples, sampling_mode=sampling_mode)


def evaluate_window(pred, gt, n_steps):
    diff = pred[:n_steps] - gt[:n_steps]
    return diff.pow(2).mean().item(), diff.abs().mean().item(), diff


def load_policy(model_path):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy = ACTPolicy.from_pretrained(model_path)
    policy.to(DEVICE)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )
    return policy, preprocessor, postprocessor, policy_cfg


def load_dataset(chunk_size, episodes=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(DATASET_REPO)
    fps = ds_meta.fps
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(DATASET_REPO, episodes=episodes, delta_timestamps=delta_timestamps)


def evaluate(model_path, eval_steps=None, num_samples=500, split_name=None, split_spec=None,
             sampling_mode="balanced_episodes", mode="standard", rollout_horizon=None):
    model_name = Path(model_path).parts[2] if len(Path(model_path).parts) > 2 else Path(model_path).name

    policy, preprocessor, postprocessor, policy_cfg = load_policy(model_path)
    chunk_size = policy_cfg.chunk_size
    n_eval = eval_steps or chunk_size
    base_dataset = load_dataset(chunk_size)
    selected_episodes, split_map = resolve_episode_subset(base_dataset, split_name=split_name, split_spec=split_spec)
    dataset = load_dataset(chunk_size, episodes=selected_episodes) if selected_episodes is not None else base_dataset
    active_split = split_name or "all"
    replan_horizon = rollout_horizon or n_eval

    all_mse_full = []
    all_mae_full = []
    all_mse_eval = []
    all_mae_eval = []
    episode_mse_eval = defaultdict(list)
    episode_mae_eval = defaultdict(list)
    step_errors_sq = [[] for _ in range(n_eval)]
    step_errors_abs = [[] for _ in range(n_eval)]
    step_deltas_pred = [[] for _ in range(n_eval - 1)]
    step_deltas_gt = [[] for _ in range(n_eval - 1)]

    if mode == "rollout":
        anchor_groups = []
        for group in get_local_episode_groups(dataset):
            anchor_groups.append(group[::replan_horizon])
        sample_indices = build_sample_indices_from_groups(anchor_groups, num_samples, sampling_mode=sampling_mode)
    else:
        sample_indices = build_sample_indices(dataset, num_samples, sampling_mode=sampling_mode)

    logger.info(
        f"Model: {model_name} | split={active_split} | chunk={chunk_size} | eval_steps={n_eval} | "
        f"samples={len(sample_indices)} | sampling={sampling_mode} | mode={mode}"
    )
    if split_map is not None:
        split_summary = ", ".join(f"{name}={len(episodes)}ep" for name, episodes in split_map.items())
        logger.info(f"Episode split: {split_summary}")
    if mode == "rollout":
        logger.info(
            f"Rollout mode uses ground-truth observations every {replan_horizon} step(s); "
            "this approximates replanning but is not a simulator rollout."
        )

    t0 = time.time()

    for i, idx in enumerate(sample_indices):
        item = dataset[idx]
        n_valid = (~item["action_is_pad"]).sum().item()
        if n_valid == 0:
            continue

        gt = item["action"]
        with torch.no_grad():
            pred = postprocessor(policy.predict_action_chunk(preprocessor(item))).squeeze(0).cpu()

        n_full = min(n_valid, chunk_size)
        n_cut = min(n_valid, n_eval)

        diff_full = pred[:n_full] - gt[:n_full]
        window_horizon = min(n_cut, replan_horizon) if mode == "rollout" else n_cut
        mse_eval, mae_eval, diff_eval = evaluate_window(pred, gt, window_horizon)

        all_mse_full.append(diff_full.pow(2).mean().item())
        all_mae_full.append(diff_full.abs().mean().item())
        all_mse_eval.append(mse_eval)
        all_mae_eval.append(mae_eval)

        if "episode_index" in item:
            episode_idx = int(item["episode_index"].item())
            episode_mse_eval[episode_idx].append(all_mse_eval[-1])
            episode_mae_eval[episode_idx].append(all_mae_eval[-1])

        for s in range(window_horizon):
            step_errors_sq[s].append(diff_eval[s].pow(2).mean().item())
            step_errors_abs[s].append(diff_eval[s].abs().mean().item())

        for s in range(min(window_horizon - 1, n_eval - 1)):
            step_deltas_pred[s].append((pred[s + 1] - pred[s]).abs().mean().item())
            step_deltas_gt[s].append((gt[s + 1] - gt[s]).abs().mean().item())

        if (i + 1) % 100 == 0:
            logger.info(f"  [{i+1}/{len(sample_indices)}] "
                        f"MSE({n_eval}step)={np.mean(all_mse_eval):.4f} "
                        f"MSE({chunk_size}step)={np.mean(all_mse_full):.4f} ({time.time()-t0:.1f}s)")

    per_step_mse = [float(np.mean(x)) if x else 0.0 for x in step_errors_sq]
    per_step_mae = [float(np.mean(x)) if x else 0.0 for x in step_errors_abs]
    avg_deltas_pred = [float(np.mean(x)) if x else 0.0 for x in step_deltas_pred]
    avg_deltas_gt = [float(np.mean(x)) if x else 0.0 for x in step_deltas_gt]
    delta_ratio = float(np.mean(avg_deltas_pred) / np.mean(avg_deltas_gt)) if avg_deltas_gt else 0.0
    eval_mse_stats = summarize_metric(all_mse_eval)
    eval_mae_stats = summarize_metric(all_mae_eval)
    episode_mse_stats = summarize_metric([np.mean(v) for v in episode_mse_eval.values() if v])
    episode_mae_stats = summarize_metric([np.mean(v) for v in episode_mae_eval.values() if v])

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {model_name} (eval_steps={n_eval}, split={active_split}, mode={mode})")
    logger.info(f"{'='*60}")
    logger.info(f"  MSE ({n_eval:3d} steps): {eval_mse_stats['mean']:.4f}")
    logger.info(f"  MAE ({n_eval:3d} steps): {eval_mae_stats['mean']:.4f}")
    logger.info(f"  MSE ({chunk_size:3d} steps): {np.mean(all_mse_full):.4f}")
    logger.info(f"  MAE ({chunk_size:3d} steps): {np.mean(all_mae_full):.4f}")
    if n_eval < chunk_size:
        imp = (1 - np.mean(all_mse_eval) / np.mean(all_mse_full)) * 100
        logger.info(f"  Improvement: MSE -{imp:.1f}%")
    logger.info(f"  Delta ratio (diagnostic): {delta_ratio:.2%}")

    logger.info("\n  Frame-level distribution:")
    logger.info(
        f"  MSE std={eval_mse_stats['std']:.4f} p50={eval_mse_stats['p50']:.4f} p90={eval_mse_stats['p90']:.4f}"
    )
    logger.info(
        f"  MAE std={eval_mae_stats['std']:.4f} p50={eval_mae_stats['p50']:.4f} p90={eval_mae_stats['p90']:.4f}"
    )

    logger.info(f"\n  Episode-level distribution ({len(episode_mse_eval)} episodes sampled):")
    logger.info(
        f"  MSE mean={episode_mse_stats['mean']:.4f} std={episode_mse_stats['std']:.4f} "
        f"p50={episode_mse_stats['p50']:.4f} p90={episode_mse_stats['p90']:.4f}"
    )
    logger.info(
        f"  MAE mean={episode_mae_stats['mean']:.4f} std={episode_mae_stats['std']:.4f} "
        f"p50={episode_mae_stats['p50']:.4f} p90={episode_mae_stats['p90']:.4f}"
    )

    logger.info(f"\n  Per-Step MSE/MAE (first {n_eval} steps):")
    logger.info(f"  {'Step':<6} {'MSE':>10} {'MAE':>10} {'GT delta':>12} {'Pred delta':>12} {'Ratio':>8}")
    logger.info(f"  {'-'*58}")
    for s in range(n_eval):
        mse_s = per_step_mse[s]
        mae_s = per_step_mae[s]
        if s < n_eval - 1:
            gt_d = avg_deltas_gt[s]
            pr_d = avg_deltas_pred[s]
            ratio = pr_d / gt_d if gt_d > 1e-8 else 0
            logger.info(f"  {s:<6} {mse_s:>10.4f} {mae_s:>10.4f} {gt_d:>12.6f} {pr_d:>12.6f} {ratio:>7.2%}")
        else:
            logger.info(f"  {s:<6} {mse_s:>10.4f} {mae_s:>10.4f}")

    results = {
        "model": model_name,
        "model_path": model_path,
        "chunk_size": chunk_size,
        "eval_steps": n_eval,
        "split": active_split,
        "sampling_mode": sampling_mode,
        "evaluation_mode": mode,
        "rollout_horizon": replan_horizon if mode == "rollout" else None,
        "mse_eval": float(np.mean(all_mse_eval)),
        "mae_eval": float(np.mean(all_mae_eval)),
        "mse_full": float(np.mean(all_mse_full)),
        "mae_full": float(np.mean(all_mae_full)),
        "delta_ratio": delta_ratio,
        "delta_ratio_note": "Diagnostic only; compares action magnitude, not direction or task success.",
        "frame_stats": {
            "mse_eval": eval_mse_stats,
            "mae_eval": eval_mae_stats,
        },
        "episode_stats": {
            "num_episodes_sampled": len(episode_mse_eval),
            "mse_eval": episode_mse_stats,
            "mae_eval": episode_mae_stats,
        },
        "per_step_mse": per_step_mse,
        "per_step_mae": per_step_mae,
        "avg_gt_delta": float(np.mean(avg_deltas_gt)) if avg_deltas_gt else 0.0,
        "avg_pred_delta": float(np.mean(avg_deltas_pred)) if avg_deltas_pred else 0.0,
    }

    del policy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def diagnose(model_path):
    logger.info(f"\n{'='*70}")
    logger.info(f"DIAGNOSE: {model_path}")
    logger.info(f"{'='*70}")

    policy, preprocessor, postprocessor, policy_cfg = load_policy(model_path)
    chunk_size = policy_cfg.chunk_size
    dataset = load_dataset(chunk_size)

    ep0 = dataset.meta.episodes[0]
    mid_idx = ep0["dataset_from_index"] + 50
    item = dataset[mid_idx]
    gt_actions = item["action"]
    gt_state = item["observation.state"]
    n_valid = (~item["action_is_pad"]).sum().item()

    logger.info(f"  Frame {mid_idx} (episode 0, step 50), valid={n_valid}/{chunk_size}")
    logger.info(f"  GT state:    {gt_state.numpy()}")
    logger.info(f"  GT action[0]: {gt_actions[0].numpy()}")

    with torch.no_grad():
        pred = postprocessor(policy.predict_action_chunk(preprocessor(item))).squeeze(0).cpu()

    logger.info(f"  Pred action[0]: {pred[0].numpy()}")
    logger.info(f"  Error[0]:      {(pred[0] - gt_actions[0]).numpy()}")

    n = min(n_valid, chunk_size)
    gt_deltas = (gt_actions[1:n] - gt_actions[:n-1]).abs().mean(-1).numpy()
    pred_deltas = (pred[1:n] - pred[:n-1]).abs().mean(-1).numpy()
    logger.info(f"\n  Action delta comparison:")
    logger.info(f"  GT delta mean:   {gt_deltas.mean():.6f}")
    logger.info(f"  Pred delta mean: {pred_deltas.mean():.6f}")
    logger.info(f"  Delta ratio:     {pred_deltas.mean()/gt_deltas.mean():.2%}" if gt_deltas.mean() > 0 else "  Delta ratio: N/A")

    logger.info(f"\n  Per-dimension range (first {n} steps):")
    for d in range(6):
        gt_min, gt_max = gt_actions[:n, d].min().item(), gt_actions[:n, d].max().item()
        pr_min, pr_max = pred[:n, d].min().item(), pred[:n, d].max().item()
        logger.info(f"  Dim {d}: GT=[{gt_min:.2f}, {gt_max:.2f}] Pred=[{pr_min:.2f}, {pr_max:.2f}]")

    del policy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    parser = argparse.ArgumentParser(description="Unified ACT offline evaluation")
    parser.add_argument("--model", nargs="+", required=True, help="Model paths or shortcuts (e.g. v0.1 chunk16)")
    parser.add_argument("--eval_steps", type=int, default=None, help="Only evaluate first N steps (simulate n_action_steps)")
    parser.add_argument("--diagnose", action="store_true", help="Single-frame detailed diagnosis")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of frames to sample")
    parser.add_argument("--split", type=str, default=None, help="Episode split to evaluate: train/val/test")
    parser.add_argument(
        "--episode_splits",
        type=str,
        default=None,
        help="Comma-separated split fractions, e.g. train:0.8,val:0.1,test:0.1",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["global_uniform", "balanced_episodes"],
        default="balanced_episodes",
        help="Frame sampling strategy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "rollout"],
        default="standard",
        help="Evaluation mode: standard teacher-forced or rollout-style replanning windows",
    )
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=None,
        help="Steps executed before replanning in rollout mode; defaults to eval_steps",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    requested_models = [resolve_path(m) for m in args.model]
    models = []
    for p in requested_models:
        if not Path(p).exists():
            logger.warning(f"Model not found: {p}, skipping")
            continue
        models.append(p)

    if not models:
        logger.error("No valid models found")
        return

    if args.diagnose:
        for p in models:
            diagnose(p)
        return

    all_results = {}
    for p in models:
        r = evaluate(
            p,
            eval_steps=args.eval_steps,
            num_samples=args.num_samples,
            split_name=args.split,
            split_spec=args.episode_splits,
            sampling_mode=args.sampling,
            mode=args.mode,
            rollout_horizon=args.rollout_horizon,
        )
        all_results[r["model"]] = r

    if len(all_results) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON")
        logger.info(f"{'='*70}")
        logger.info(f"  {'Model':<30} {'MSE':>8} {'MAE':>8} {'MSEstd':>8} {'Delta%':>8} {'Chunk':>6} {'Eval':>6}")
        logger.info(f"  {'-'*75}")
        for name, r in all_results.items():
            logger.info(f"  {name:<30} {r['mse_eval']:>8.4f} {r['mae_eval']:>8.4f} "
                        f"{r['frame_stats']['mse_eval']['std']:>8.4f} {r['delta_ratio']:>7.1%} {r['chunk_size']:>6} {r['eval_steps']:>6}")

    output_path = args.output or "outputs/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
