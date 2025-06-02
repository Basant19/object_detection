import os
import sys
import glob
import torch
import cv2
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64

app = Flask(__name__)
CORS(app)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv5 model once
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt', force_reload=True)
model.to(device)
model.conf = 0.5

camera = None  # Only initialized when needed

def gen_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, size=416)
            rendered_frame = results.render()[0]
            ret, buffer = cv2.imencode('.jpg', rendered_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

UPLOAD_FOLDER = os.path.join(os.getcwd(), "data")
INPUT_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "inputImage.jpg")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ClientApp:
    def __init__(self):
        self.filename = INPUT_IMAGE_PATH

clApp = ClientApp()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        return "Training Successful!!"
    except Exception as e:
        return Response(f"Training failed: {str(e)}", status=500)

@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        image_data = request.json.get('image')
        if not image_data:
            return Response("No image data provided", status=400)

        decodeImage(image_data, clApp.filename)
        os.system(f"cd yolov5 && python detect.py --weights best.pt --img 416 --conf 0.5 --source {clApp.filename}")
        exp_paths = sorted(glob.glob("yolov5/runs/detect/exp*"), key=os.path.getmtime, reverse=True)
        if not exp_paths:
            return Response("Detection failed, no output folder found", status=500)

        output_image_path = os.path.join(exp_paths[0], "inputImage.jpg")
        if not os.path.exists(output_image_path):
            return Response("Detected image not found", status=500)

        opencodedbase64 = encodeImageIntoBase64(output_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        os.system("rm -rf yolov5/runs")

        return jsonify(result)
    except Exception as e:
        return Response(f"An error occurred: {str(e)}", status=500)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080,debug=True)
