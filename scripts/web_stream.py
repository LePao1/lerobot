import cv2
import glob
from flask import Flask, Response
import time

app = Flask(__name__)

class Camera:
    def __init__(self, device_path):
        self.device_path = device_path
        self.cap = cv2.VideoCapture(device_path)
        if self.cap.isOpened():
             # Force MJPG
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Opened {device_path}")
        else:
            print(f"Failed to open {device_path}")

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None
            
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

cameras = {}
# Initialize cameras
# We try video0 and video2 as they were detected earlier
for dev in ['/dev/video0', '/dev/video2']:
    try:
        cam = Camera(dev)
        if cam.cap.isOpened():
            cameras[dev] = cam
    except Exception as e:
        print(f"Error initializing {dev}: {e}")

@app.route('/')
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
        html += "<p>No cameras found. Please check connections.</p>"
    
    for dev_path in cameras.keys():
        # Encode path safely for url
        safe_path = dev_path.replace('/', '_')
        html += f"""
            <div class="camera-box">
                <h3>{dev_path}</h3>
                <img src="/video_feed/{safe_path}" width="640" height="480">
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
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed/<device_path_safe>')
def video_feed(device_path_safe):
    # Decode path
    device_path = device_path_safe.replace('_', '/')
    if device_path in cameras:
        return Response(gen(cameras[device_path]),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
