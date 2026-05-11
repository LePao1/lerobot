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
    "exp3_kl5_obs3": "outputs/train/act_so101_v02_exp3_chunk100_act20_kl5_obs3/checkpoints/100000/pretrained_model",
    "exp4_dec7": "outputs/train/act_so101_v02_exp4_chunk100_act20_dec7/checkpoints/100000/pretrained_model",
    "exp5_aug": "outputs/train/act_so101_v02_exp5_chunk100_act20_aug/checkpoints/100000/pretrained_model",
}


def resolve_path(name):
    return SHORTCUTS.get(name, name)


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


def load_dataset(chunk_size):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(DATASET_REPO)
    fps = ds_meta.fps
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(DATASET_REPO, delta_timestamps=delta_timestamps)


def evaluate(model_path, eval_steps=None, num_samples=500):
    model_name = Path(model_path).parts[2] if len(Path(model_path).parts) > 2 else Path(model_path).name

    policy, preprocessor, postprocessor, policy_cfg = load_policy(model_path)
    chunk_size = policy_cfg.chunk_size
    n_eval = eval_steps or chunk_size
    dataset = load_dataset(chunk_size)

    sample_indices = list(range(0, len(dataset), max(1, len(dataset) // num_samples)))
    logger.info(f"Model: {model_name} | chunk={chunk_size} | eval_steps={n_eval} | samples={len(sample_indices)}")

    all_mse_full = []
    all_mae_full = []
    all_mse_eval = []
    all_mae_eval = []
    step_errors_sq = [[] for _ in range(n_eval)]
    step_errors_abs = [[] for _ in range(n_eval)]
    step_deltas_pred = [[] for _ in range(n_eval - 1)]
    step_deltas_gt = [[] for _ in range(n_eval - 1)]

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
        diff_eval = pred[:n_cut] - gt[:n_cut]

        all_mse_full.append(diff_full.pow(2).mean().item())
        all_mae_full.append(diff_full.abs().mean().item())
        all_mse_eval.append(diff_eval.pow(2).mean().item())
        all_mae_eval.append(diff_eval.abs().mean().item())

        for s in range(n_cut):
            step_errors_sq[s].append(diff_eval[s].pow(2).mean().item())
            step_errors_abs[s].append(diff_eval[s].abs().mean().item())

        for s in range(min(n_cut - 1, n_eval - 1)):
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

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS: {model_name} (eval_steps={n_eval})")
    logger.info(f"{'='*60}")
    logger.info(f"  MSE ({n_eval:3d} steps): {np.mean(all_mse_eval):.4f}")
    logger.info(f"  MAE ({n_eval:3d} steps): {np.mean(all_mae_eval):.4f}")
    logger.info(f"  MSE ({chunk_size:3d} steps): {np.mean(all_mse_full):.4f}")
    logger.info(f"  MAE ({chunk_size:3d} steps): {np.mean(all_mae_full):.4f}")
    if n_eval < chunk_size:
        imp = (1 - np.mean(all_mse_eval) / np.mean(all_mse_full)) * 100
        logger.info(f"  Improvement: MSE -{imp:.1f}%")
    logger.info(f"  Delta ratio: {delta_ratio:.2%}")

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
        "mse_eval": float(np.mean(all_mse_eval)),
        "mae_eval": float(np.mean(all_mae_eval)),
        "mse_full": float(np.mean(all_mse_full)),
        "mae_full": float(np.mean(all_mae_full)),
        "delta_ratio": delta_ratio,
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
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    models = [resolve_path(m) for m in args.model]

    for p in models:
        if not Path(p).exists():
            logger.warning(f"Model not found: {p}, skipping")
            models.remove(p)

    if not models:
        logger.error("No valid models found")
        return

    if args.diagnose:
        for p in models:
            diagnose(p)
        return

    all_results = {}
    for p in models:
        r = evaluate(p, eval_steps=args.eval_steps, num_samples=args.num_samples)
        all_results[r["model"]] = r

    if len(all_results) > 1:
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON")
        logger.info(f"{'='*70}")
        logger.info(f"  {'Model':<30} {'MSE':>8} {'MAE':>8} {'Delta%':>8} {'Chunk':>6} {'Eval':>6}")
        logger.info(f"  {'-'*66}")
        for name, r in all_results.items():
            logger.info(f"  {name:<30} {r['mse_eval']:>8.4f} {r['mae_eval']:>8.4f} "
                        f"{r['delta_ratio']:>7.1%} {r['chunk_size']:>6} {r['eval_steps']:>6}")

    output_path = args.output or "outputs/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
