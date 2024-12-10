# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import sys
from types import FrameType

from flask import Flask


from flask import Flask, request, jsonify
from flask_cors import CORS
# import some_ml_library  # Replace with TensorFlow, PyTorch, etc., as needed
import torchxrayvision as xrv
import skimage, torch, torchvision
from fracture import Fracture
from matplotlib.colors import TABLEAU_COLORS
import copy
import os


from cancer_predictor import cancer_prediction
from cxray_from_mknoon.util import classify
from cxray_from_mknoon.classifier import model as cxray_model

def get_detailed_prediction(img):
    """
        This model will return the probability of having 14 different pathologies
    """
    # img = skimage.io.imread(files)
    
    # Check if the image has 3 channels; if so, convert it to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(axis=2)  # Average across RGB channels to convert to grayscale
    
    # Ensure image is in the required range
    img = xrv.datasets.normalize(img, 255)  # Convert 8-bit image to range [-1024, 1024]
    img = img[None, ...]  # Add a channel dimension for single color channel

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    img = transform(img)
    img = torch.from_numpy(img)

    # Load model and process image
    outputs = chest_model(img[None,...]) # or model.features(img[None,...]) 

    # Convert outputs to standard Python types for JSON serialization
    results = {pathology: float(score) for pathology, score in zip(chest_model.pathologies, outputs[0].detach().cpu().numpy())}
    
    return results

def get_prediction_from_mknoon(file):
    cxray_model.load_weights("cxray_from_mknoon/pneumonia_classification_v3.h5")

    # load class names
    with open('cxray_from_mknoon/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()
    img = skimage.io.imread(file)
    img_copy = copy.deepcopy(img)
    # # Check if the image has 3 channels; if so, convert it to grayscale
    # if img.ndim == 3 and img.shape[2] == 3:
    #     img = img.mean(axis=2)  # Average across RGB channels to convert to grayscale
    
    # # Ensure image is in the required range
    # img = xrv.datasets.normalize(img, 255)  # Convert 8-bit image to range [-1024, 1024]
    # img = img[None, ...]  # Add a channel dimension for single color channel
    print(img.shape)
    # cv2.imshow("Output Image", img)
    # cv2.waitKey(0)
    class_name, conf_score = classify(img, cxray_model, class_names)
    print("Conf: ",conf_score)
    result = str(class_name).lower()
    print("result", result)
    if result == "normal":
        return {"status": "normal", "details": None}
    else:
        detailed_prediction = get_detailed_prediction(img_copy)
        return {"status": "abnormal", "details": detailed_prediction}
    
def get_cancer_prediction(image1, image2):
    # Load the two images
    img1 = skimage.io.imread(image1)
    img2 = skimage.io.imread(image2)
    pred = cancer_prediction(img1, img2)
    print(pred)
    if pred > 0.5:
        return ["malignant"]
    else:
        return ["benign"]


app = Flask(__name__)
CORS(app)

#loading the models
chest_model = xrv.models.DenseNet(weights="densenet121-res224-all")
fracture_model = Fracture()

@app.route("/")
def hello() -> str:

    return "Hello, World!"

@app.route('/chestXray', methods=['POST'])
def classify_chestxray():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check MIME type
    if not file.mimetype.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400
    
    # Process the image if MIME type is valid
    try:
        # result = get_prediction(file, chest_model)
        result = get_prediction_from_mknoon(file)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': "Server could not deal with your file", 'details': str(e)}), 500

    return jsonify({"prediction": result}), 200

@app.route('/elbow', methods=['POST'])
def classify_elbow():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check MIME type
    if not file.mimetype.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400
    
    try:
        # Read the image in a way that the model can process it
        from io import BytesIO
        img_stream = BytesIO(file.read())  # Convert uploaded file to stream
        result = fracture_model.predict(img_stream, "Elbow")
        result = {"status": result, "details": None}
        print(result)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': "Server could not process your file", 'details': str(e)}), 500

    return jsonify({"prediction": result}), 200

@app.route('/hand', methods=['POST'])
def classify_hand():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check MIME type
    if not file.mimetype.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400
    
    try:
        # Read the image in a way that the model can process it
        from io import BytesIO
        img_stream = BytesIO(file.read())  # Convert uploaded file to stream
        result = fracture_model.predict(img_stream, "Hand")
        result = {"status": result, "details": None}
        print(result)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': "Server could not process your file", 'details': str(e)}), 500

    return jsonify({"prediction": result}), 200

@app.route('/shoulder', methods=['POST'])
def classify_shoulder():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check MIME type
    if not file.mimetype.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400
    
    try:
        # Read the image in a way that the model can process it
        from io import BytesIO
        img_stream = BytesIO(file.read())  # Convert uploaded file to stream
        result = fracture_model.predict(img_stream, "Shoulder")
        result = {"status": result, "details": None}
        print(result)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': "Server could not process your file", 'details': str(e)}), 500

    return jsonify({"prediction": result}), 200

@app.route('/cancer', methods=['POST'])
def classify_cancer():
    print(request.files)
    # Check if both images are provided
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Two image files are required"}), 400
    image1 = request.files['file1']
    image2 = request.files['file2']
    # Check MIME type for both images
    if not image1.mimetype.startswith('image/') or not image2.mimetype.startswith('image/'):
        return jsonify({"error": "Both uploaded files must be images"}), 400
    
    try:
        # Pass the images to the cancer prediction function
        result = get_cancer_prediction(image1, image2)
    except Exception as e:
        print("Error during cancer prediction:", e)
        return jsonify({'error': "Server could not process the images", 'details': str(e)}), 500

    # Return the prediction result
    return jsonify({"prediction": result}), 200


def shutdown_handler(signal_int: int, frame: FrameType) -> None:

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
