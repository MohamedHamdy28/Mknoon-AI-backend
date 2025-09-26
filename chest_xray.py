from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from transformers import AutoModel

class TBClassifier:
    def __init__(self):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load classification model
        self.classifier = self._load_classifier()

        # Load segmentation model
        self.segmenter = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
        self.segmenter = self.segmenter.eval().to(self.device)

        # Define preprocessing transform (same as during training)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Map class indices to names
        self.class_names = ["Normal", "TB", "Others"]

    def _load_classifier(self):
        # Locate model file relative to this script
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / "model" / "chest xray" / "TB_with_others.pt"

        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: Normal, TB, Others

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def _extract_lungs_and_heart(self, img):
        """Given an image path, segment and return the masked image (as PIL.Image)"""
        # Load grayscale image
        if img is None:
            raise ValueError(f"Failed to load image: {img}")

        original_shape = img.shape

        # Preprocess for segmentation model
        x = self.segmenter.preprocess(img)  # returns numpy
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Run segmentation model
        with torch.inference_mode():
            out = self.segmenter(x)
        mask = out["mask"].argmax(1).squeeze().cpu().numpy()

        # Resize mask back to original image shape
        resized_mask = cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create binary mask for lungs and heart (classes 1, 2, 3)
        combined_mask = np.isin(resized_mask, [1, 2, 3]).astype(np.uint8)

        # Apply mask
        masked_img = cv2.bitwise_and(img, img, mask=combined_mask)

        # Convert to 3-channel RGB for classifier
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2RGB)
        masked_pil = Image.fromarray(masked_img_rgb)

        return masked_pil

    def predict(self, image_path):
        """Predict the class (Normal, TB, Others) for a given chest X-ray image."""
        # Step 1: Segment lungs and heart
        masked_pil_image = self._extract_lungs_and_heart(image_path)

        # Step 2: Preprocess the masked image
        input_tensor = self.transform(masked_pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        # Step 3: Predict
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            _, preds = torch.max(outputs, 1)
            predicted_class = preds.item()

        return self.class_names[predicted_class]
