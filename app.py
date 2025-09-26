# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ...
# We'll skip the license boilerplate here for brevity.

import signal
import sys
from types import FrameType
import os
from unittest import result

from flask import Flask, request, jsonify
from flask_cors import CORS

import torchxrayvision as xrv
import skimage, torch, torchvision
from matplotlib.colors import TABLEAU_COLORS
import copy

from fracture import Fracture
from cancer_predictor import CancerPredictor
from cxray_from_mknoon.util import classify
from cxray_from_mknoon.classifier import model as cxray_model
import base64
from io import BytesIO
from chest_xray import TBClassifier

# For reading DICOM
import pydicom
import numpy as np
from io import BytesIO
import cv2

app = Flask(__name__)
CORS(app)

# loading the models once at startup
# chest_model = xrv.models.DenseNet(weights="densenet121-res224-all")
chest_model = TBClassifier()
fracture_model = Fracture()
cancer_prediction = CancerPredictor()

def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    # Clean up logs or resources if needed
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

###################################
# 1) UTILITY: READ IMAGE OR DICOM
###################################
def read_image_or_dicom(file_storage):
    """
    Checks if the uploaded file is a .dcm (DICOM) or a standard image.
    Returns a 2D NumPy array (grayscale).
    """
    # Read raw bytes once
    data = file_storage.read()
    # Reset stream position in case someone else reads it later
    try:
        file_storage.seek(0)
    except Exception:
        pass

    filename = file_storage.filename.lower()

    if filename.endswith('.dcm'):
        # DICOM: parse from the in-memory bytes
        ds = pydicom.dcmread(BytesIO(data))
        img = ds.pixel_array
    else:
        # Standard image: load from bytes
        img = skimage.io.imread(BytesIO(data))

    # Ensure it’s a 2D grayscale image
    if img.ndim > 2:
        # convert RGB/BGR to gray
        # skimage returns RGB; cv2.cvtColor expects BGR, but gray result is same
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Force float32 for consistency downstream
    return img.astype(np.float32)

###################################
# 2) DETAILED CXR PREDICTION
###################################
def get_detailed_prediction(img):
    """
    This model returns the probability of having 14 different pathologies
    using TorchXRayVision's DenseNet.
    """
    # If the image has 3 channels (RGB), convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(axis=2)

    # Normalize to the range [-1024, 1024]
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]  # Add a channel dimension for single color channel

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

    img = transform(img)
    img = torch.from_numpy(img)

    outputs = chest_model(img[None, ...])
    # Convert outputs to standard Python types for JSON
    pathologies = chest_model.pathologies
    scores = outputs[0].detach().cpu().numpy()
    results = {p: float(s) for p, s in zip(pathologies, scores)}

    return results

###################################
# 3) CUSTOM CXR MODEL
###################################
def get_prediction_from_mknoon(file_storage):
    """
    Uses the pneumonia_classification_v3 from Mknoon,
    plus DenseNet for more details if abnormal.
    """
    cxray_model.load_weights("cxray_from_mknoon/pneumonia_classification_v3.h5")

    with open('cxray_from_mknoon/labels.txt', 'r') as f:
        class_names = [line.strip().split(' ')[1] for line in f]

    # read as image or DICOM
    img = read_image_or_dicom(file_storage)
    img_copy = copy.deepcopy(img)

    class_name, conf_score = classify(img, cxray_model, class_names)
    result = str(class_name).lower()

    if result == "normal":
        return {"status": "normal", "details": None}
    else:
        detailed_prediction = get_detailed_prediction(img_copy)
        return {"status": "abnormal", "details": detailed_prediction}

def get_prediction_for_tb(file_storage):
    img = read_image_or_dicom(file_storage)
    result = chest_model.predict(img)
    return {"status": result}

###################################
# 4) CANCER PREDICTION
###################################
def encode_image_to_base64(image):
    """ Convert an OpenCV/Numpy image to a base64-encoded string """
    if image is None:
        print("image is None")
        return None
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")

def get_cancer_prediction(file1, file2):
    """
    Cancer predictor expects two images: CC, MLO (or DICOM).
    Returns classification labels and bounding box images (if applicable) as base64 strings.
    """
    img1 = read_image_or_dicom(file1)
    img2 = read_image_or_dicom(file2)

    result = cancer_prediction.cancer_prediction(img1, img2)

    # Convert images to base64 for JSON serialization
    result["image1_with_boxes"] = encode_image_to_base64(result["image1_with_boxes"])
    result["image2_with_boxes"] = encode_image_to_base64(result["image2_with_boxes"])

    return result

###################################
# FLASK ROUTES
###################################
@app.route("/")
def hello() -> str:
    return "Hello, World!"

@app.route('/chestXray', methods=['POST'])
def classify_chestxray():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_storage = request.files['file']
    filename = file_storage.filename.lower()

    # We handle either standard images or .dcm
    if not (filename.endswith('.png') or filename.endswith('.jpg')
            or filename.endswith('.jpeg') or filename.endswith('.dcm')):
        return jsonify({"error": "File must be .png, .jpg, .jpeg, or .dcm"}), 400

    try:
        result = get_prediction_for_tb(file_storage)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': "Server could not handle your file", 'details': str(e)}), 500

    return jsonify({"prediction": result}), 200

@app.route('/fracture', methods=['POST'])
def classify_fracture():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_storage = request.files['file']
    filename = file_storage.filename.lower()

    if not (filename.endswith('.png') or filename.endswith('.jpg')
            or filename.endswith('.jpeg') or filename.endswith('.dcm')):
        return jsonify({"error": "File must be .png, .jpg, .jpeg, or .dcm"}), 400

    try:
        from io import BytesIO
        if filename.endswith('.dcm'):
            # read as DICOM
            img_data = read_image_or_dicom(file_storage)
            # Convert to BytesIO in PNG format if your Fracture model expects standard images
            # or adapt the "fracture_model" to accept arrays directly
            # For example:
            from PIL import Image
            import cv2
            import numpy as np

            # Make sure the array is 8-bit
            # If it's large range, scale it or cast
            if img_data.dtype != np.uint8:
                # normalize to 0-255
                normalized = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
                img_data = (normalized * 255).astype(np.uint8)
            
            # Convert to PIL
            pil_img = Image.fromarray(img_data)
            temp_buffer = BytesIO()
            pil_img.save(temp_buffer, format="PNG")
            temp_buffer.seek(0)
            result_label = fracture_model.predict(temp_buffer)
        else:
            # If it's a normal image
            img_stream = BytesIO(file_storage.read())
            result_label = fracture_model.predict(img_stream)

        result = {"status": result_label, "details": None}
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': "Server could not process your file", 'details': str(e)}), 500

    return jsonify({"prediction": result}), 200

@app.route('/cancer', methods=['POST'])
def classify_cancer():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Two image files are required"}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    # Check allowed extensions
    allowed_exts = ('.png', '.jpg', '.jpeg', '.dcm')
    if (not file1.filename.lower().endswith(allowed_exts)
        or not file2.filename.lower().endswith(allowed_exts)):
        return jsonify({"error": "Both files must be .png, .jpg, .jpeg, or .dcm"}), 400

    try:
        # 1) Read the bytes for each file
        data1 = file1.read()
        data2 = file2.read()

        # 2) If both byte sequences are identical, return an error
        if data1 == data2:
            return jsonify({
                "error": "The two uploaded images are identical. "
                         "Please upload two distinct images."
            }), 400

        # 3) Reset the file pointers so `get_cancer_prediction` can read them again
        file1.seek(0)
        file2.seek(0)

        # 4) Proceed with your existing logic
        result = get_cancer_prediction(file1, file2)
    except Exception as e:
        print("Error during cancer prediction:", e)
        return jsonify({
            'error': "Server could not process the images",
            'details': str(e)
        }), 500

    return jsonify({"prediction": result}), 200


if __name__ == "__main__":
    # Running application locally
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
