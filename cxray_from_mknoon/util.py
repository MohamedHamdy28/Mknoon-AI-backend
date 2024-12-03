import base64

import numpy as np
import cv2

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    img_size = 150

    # # Convert image to grayscale
    # image = ImageOps.grayscale(image)
    # # image = image.convert('L')

    # convert image to (224, 224)
    # image = ImageOps.fit(image, (img_size, img_size), Image.Resampling.LANCZOS)
    # image = image.resize((150, 150))
    image = cv2.resize(image, (img_size, img_size)) 

    # convert image to numpy array
    # image_array = np.asarray(image)
    image_array = np.array(image)

    # normalize image
    # normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    normalized_image_array = image_array / 255

    # set model input
    # data = np.ndarray(shape=(1, img_size, img_size, 3), dtype=np.float32)
    # data[0] = normalized_image_array
    data = normalized_image_array.reshape(-1, img_size, img_size, 1)


    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    # confidence_score = prediction÷[0][index]
    confidence_score = prediction[0][0]
    
    return class_name, confidence_score
