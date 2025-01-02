
import numpy as np
import tensorflow as tf
import keras.utils as image

class Fracture:
    def __init__(self) -> None:
        # load the models when import "predictions.py"
        self.model_elbow_frac = tf.keras.models.load_model("./model/fracture/weights/ResNet50_Elbow_frac.h5")
        self.model_hand_frac = tf.keras.models.load_model("./model/fracture/weights/ResNet50_Hand_frac.h5")
        self.model_shoulder_frac = tf.keras.models.load_model("./model/fracture/weights/ResNet50_Shoulder_frac.h5")
        self.model_parts = tf.keras.models.load_model("./model/fracture/weights/ResNet50_BodyParts.h5")
        #   0-Elbow     1-Hand      2-Shoulder
        self.categories_parts = ["Elbow", "Hand", "Shoulder"]

        #   0-fractured     1-normal
        self.categories_fracture = ['fractured', 'normal']

    def predict(self, img):
        size = 224

        # Step 1: Use the model_parts model to determine the body part
        temp_img = image.load_img(img, target_size=(size, size))
        x = image.img_to_array(temp_img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        body_part_prediction = np.argmax(self.model_parts.predict(images), axis=1)
        body_part = self.categories_parts[body_part_prediction.item()]  # Get the body part as a string

        # Step 2: Select the appropriate model based on the determined body part
        if body_part == 'Elbow':
            chosen_model = self.model_elbow_frac
        elif body_part == 'Hand':
            chosen_model = self.model_hand_frac
        elif body_part == 'Shoulder':
            chosen_model = self.model_shoulder_frac
        else:
            raise ValueError(f"Unknown body part detected: {body_part}")

        print("Predicted body part ", body_part)
        # Step 3: Predict fracture status using the chosen model
        fracture_prediction = np.argmax(chosen_model.predict(images), axis=1)
        prediction_str = self.categories_fracture[fracture_prediction.item()]  # Get the prediction as a string

        return prediction_str

    

if __name__ == '__main__':
    # Instantiate the Fracture class
    fracture = Fracture()

    # Path to the test image
    img_path = r"D:\Work\test data\fracture\elbow\normal\elbow3.jpg"

    # Call the predict function and print the result
    try:
        result = fracture.predict(img_path)
        print(f"Prediction for the image '{img_path}': {result}")
    except Exception as e:
        print(f"An error occurred: {e}")

    