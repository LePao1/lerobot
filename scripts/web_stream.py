import argparse
import platform
import time
from pathlib import Path
from urllib.parse import quote, unquote

import cv2
from flask import Flask, Response

app = Flask(__name__)
cameras = {}


class Camera:
    def __init__(self, device, width=640, height=480, fourcc="MJPG"):
        self.device = device
        self.cap = cv2.VideoCapture(device)
        if self.cap.isOpened():
            if fourcc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Opened {device}")
        else:
            print(f"Failed to open {device}")

    def get_frame(self):
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            return None

        return jpeg.tobytes()

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.close()


def parse_device(value):
    try:
        return int(value)
    except ValueError:
        return value


def get_default_devices(max_index):
    if platform.system() == "Linux":
        devices = sorted(Path("/dev").glob("video*"))
        return [str(device) for device in devices]

    # Windows and macOS use OpenCV numeric camera indices instead of /dev/video* paths.
    return list(range(max_index + 1))


def initialize_cameras(devices, width, height, fourcc):
    for device in devices:
        try:
            cam = Camera(device, width=width, height=height, fourcc=fourcc)
            if cam.cap.isOpened():
                cameras[str(device)] = cam
        except Exception as exc:
            print(f"Error initializing {device}: {exc}")


@app.route("/")
def index():
    html = """
    <html>
    <head>
        <title>Camera Stream</title>
        <style>
            .camera-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .camera-box {
                border: 1px solid #ccc;
                padding: 10px;
                text-align: center;
            }
            img {
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>Camera Streams</h1>
        <div class="camera-container">
    """

    if not cameras:
        html += "<p>No cameras found. Please check connections or pass --devices.</p>"

    for device in cameras.keys():
        html += f"""
            <div class="camera-box">
                <h3>{device}</h3>
                <img src="/video_feed/{quote(device, safe='')}" width="640" height="480">
            </div>
        """

    html += """
        </div>
    </body>
    </html>
    """
    return html


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            time.sleep(0.1)


@app.route("/video_feed/<device>")
def video_feed(device):
    device = unquote(device)
    if device in cameras:
        return Response(gen(cameras[device]), mimetype="multipart/x-mixed-replace; boundary=frame")

    return "Camera not found", 404


def main():
    parser = argparse.ArgumentParser(description="Stream OpenCV cameras through a local Flask web page.")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated camera devices. Examples: '/dev/video0,/dev/video2' on Linux or '0,1' on Windows.",
    )
    parser.add_argument("--max-index", type=int, default=5, help="Highest numeric camera index to scan.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.devices:
        devices = [parse_device(device.strip()) for device in args.devices.split(",") if device.strip()]
    else:
        devices = get_default_devices(args.max_index)

    print(f"Scanning camera devices: {devices}")
    initialize_cameras(devices, width=args.width, height=args.height, fourcc=args.fourcc)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
