import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import cv2
import numpy as np

num_classes = 3  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CancerPredictor:
    def __init__(self):
        classification_path = "model\cancer\cancer_classification_vgg16.pt"
        detection_path = "model\cancer\cancer_detection_YOLOv11.pt"

        # Load classification model
        self.classification_model = models.vgg16(pretrained=False)
        in_features = self.classification_model.classifier[6].in_features
        self.classification_model.classifier[6] = nn.Linear(in_features, num_classes)
        self.classification_model = self.classification_model.to(device)

        # Load the saved state dictionary
        checkpoint = torch.load(classification_path, map_location=device)
        self.classification_model.load_state_dict(checkpoint)
        self.classification_model.eval()

        # Load detection model
        self.detection_model = YOLO(detection_path)
        self.detection_model = self.detection_model.to(device)
        self.detection_model.eval()

        self.val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Use standard RGB mean/std
        ])
        self.class_names = ["Benign", "Malignant", "Normal"]  # Update this based on your dataset
        print("Cancer Predictor initialized")

    def extract_detection(self, image, detection_result):
        # Extract detections
        threshold = 0.25  # Confidence threshold
        boxes = detection_result[0].boxes.data.cpu().numpy()  # Extract bounding boxes
        for box in boxes:
            if len(box) < 6:  # Ensure the box has enough values
                continue

            x, y, w, h, conf, cls = box
            if conf < threshold:
                continue

            # Convert YOLO format to OpenCV format
            x1 = int(x)
            y1 = int(y)
            x2 = int(w)
            y2 = int(h)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Tumor: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
    def cancer_prediction(self, image1, image2):
        # check if the images are rgb or not and convert them to rgb
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
        image1_pil = Image.fromarray(image1)
        image2_pil = Image.fromarray(image2)
        img1_transformed = self.val_transform(image1_pil).unsqueeze(0).to(device)
        img2_transformed = self.val_transform(image2_pil).unsqueeze(0).to(device)
        print(image1.shape, image2.shape)
        image1_with_boxes = None
        image2_with_boxes = None
        with torch.no_grad():
            cls_output1 = self.classification_model(img1_transformed)
            _, classification_preds1 = torch.max(cls_output1, 1)
            cls_output2 = self.classification_model(img2_transformed)
            _, classification_preds2 = torch.max(cls_output2, 1)
            print(classification_preds1, classification_preds2) 
            classification_label1 = self.class_names[classification_preds1.item()]
            classification_label2 = self.class_names[classification_preds2.item()]

            if classification_label1 != "Normal":
                # Perform detection on the images
                detection_output1 = self.detection_model(image1_pil)
                image1_with_boxes = self.extract_detection(image1, detection_output1)
            
            if classification_label2 != "Normal":
                detection_output2 = self.detection_model(image2_pil)
                image2_with_boxes = self.extract_detection(image2, detection_output2)
        result = {
            "image1_label": classification_label1,
            "image2_label": classification_label2,
            "image1_with_boxes": image1_with_boxes,
            "image2_with_boxes": image2_with_boxes
        }
        return result
            
            
        


