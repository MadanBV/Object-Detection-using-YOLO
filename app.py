from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import torch
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Create necessary directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "runs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano model (you can use 'yolov8s.pt' for better accuracy)


def detect_objects(image_path):
    # Perform YOLO object detection
    results = model(image_path)
    
    # Process the results
    result_image_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    for result in results:
        img = result.plot()  # Plot the detected objects
        cv2.imwrite(result_image_path, img)

    detected_objects = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]  # Get object name from YOLO class index
            detected_objects.append(label)

    return result_image_path, list(set(detected_objects))  # Remove duplicate detections


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Perform object detection
    output_image, detected_objects = detect_objects(file_path)

    return jsonify({"objects": detected_objects, "image_url": f"/output/{os.path.basename(output_image)}"}), 200


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
