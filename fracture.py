
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

    def predict(self, img, model="Parts"):
        size = 224
        if model == 'Parts':
            chosen_model = self.model_parts
        else:
            if model == 'Elbow':
                chosen_model = self.model_elbow_frac
            elif model == 'Hand':
                chosen_model = self.model_hand_frac
            elif model == 'Shoulder':
                chosen_model = self.model_shoulder_frac

        # load image with 224px224p (the training model image size, rgb)
        temp_img = image.load_img(img, target_size=(size, size))
        x = image.img_to_array(temp_img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        prediction = np.argmax(chosen_model.predict(images), axis=1)

        # chose the category and get the string prediction
        if model == 'Parts':
            prediction_str = self.categories_parts[prediction.item()]
        else:
            prediction_str = self.categories_fracture[prediction.item()]

        return prediction_str
    

if __name__ == '__main__':
    fracture = Fracture()
    img_path = "../test data/fracture/elbow/negative/image4.png"
    print(fracture.predict(img_path, "Elbow"))
    