# 2 views CLASSIFIER - test script
#
# Test inference for 2 views mammograms
#
# run: python3 2views_clf_test.py -c [cc image file] -m [mlo image file]
#
# DGPP 06/Sep/2021

import argparse
import numpy as np
import torch
from torch.autograd import Variable
import cv2

from two_views_net import SideMIDBreastModel
import torch
import torch.nn as nn
from torchvision import models, transforms

# Define the model class
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Output: 16x224x224
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Output: 16x112x112
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: 32x112x112
        self.fc1 = nn.Linear(32 * 56 * 56, 128)                            # Fully connected layer
        self.fc2 = nn.Linear(128, 2)                                       # Output layer for binary classification
        self.dropout = nn.Dropout(0.5)                                    # Dropout for regularization
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


TRAIN_DS_MEAN = 13369
NETWORK = 'EfficientNet-b0'
TOPOLOGY = 'side_mid_clf'
DEVICE = 'cpu'
gpu_number = 0

TOP_LAYER_N_BLOCKS = 2
TOP_LAYER_BLOCK_TYPE = 'mbconv'
USE_AVG_POOL = True
STRIDES=2


def get_2views_model(model, model_file, device):
    """ Load model weights from file  """
    print('Model 2views: ', model_file)
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model


def load_model(network, topology):
    """ load model structure and device """

    device = torch.device("cpu")
    if topology == 'side_mid_clf':
        model = SideMIDBreastModel(device, network, TOP_LAYER_N_BLOCKS,
                                   b_type=TOP_LAYER_BLOCK_TYPE, avg_pool=USE_AVG_POOL,
                                   strides=STRIDES)
    else:
        raise NotImplementedError(f"Net type error: {topology}")

    model = model.to(device)

    return model, device

def standard_normalize(image):
    """ Normalize accordingly for model """
    image = np.float32(image)
    image -= TRAIN_DS_MEAN
    image /= 65535    # float [-1,1]

    return image


def make_prediction(image_cc, image_mlo, model, device):
    """ 
    Execute deep learning inference
    inputs: [vector of] image
    output: full image mask
    """
    img_cc = standard_normalize(image_cc)
    img_mlo = standard_normalize(image_mlo)

    img_cc_t = torch.from_numpy(img_cc.transpose(2, 0, 1))
    img_mlo_t = torch.from_numpy(img_mlo.transpose(2, 0, 1))
    batch_t = torch.cat([img_cc_t, img_mlo_t], dim=0)
    batch_t = batch_t.unsqueeze(0)

    # prediction
    with torch.no_grad():
        model.eval()        # if not here, BN is enabled and mess everything
        input = Variable(batch_t.to(device))
        output_t = model(input)

    pred = output_t.squeeze()
    pred = torch.softmax(pred, dim=0)

    return pred, batch_t


def simple_prediction(image_cc, image_mlo, model, device):
    """ Execute simple inference """
    tta_predictions = np.array([])
    for i in range(1,2):
        aug_image_cc = image_cc
        aug_image_mlo = image_mlo
        prediction, _ = make_prediction(aug_image_cc, aug_image_mlo, model, device)
        tta_predictions = np.append(tta_predictions, prediction[1].cpu().detach().numpy())
    
    return tta_predictions


def translation_aug(image_cc, image_mlo, model, device, type=None):
    """ Execute inference with translation augmentation """
    tta_predictions = np.array([])
    rows, cols, _ = image_cc.shape
    # Translation
    for i in range(-1, +2):
        for j in range(-1, +2):
            M = np.float32([[1, 0, i*cols//40], [0, 1, j*rows//40]]) # de 0.8414=>0.8476
            aug_image_cc = cv2.warpAffine(image_cc, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
            aug_image_mlo = cv2.warpAffine(image_mlo, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
            prediction, _ = make_prediction(aug_image_cc, aug_image_mlo, model, device)
            tta_predictions = np.append(tta_predictions, prediction[1].cpu().detach().numpy())
    
    return tta_predictions

def arrange_images(image1, image2, view_model):
    """
    Arrange images based on their predicted views (CC or MLO).
    The function predicts the views of the two input images (image1 and image2)
    and ensures they are aligned as CC (Cranio-Caudal) and MLO (Medio-Lateral Oblique).

    Args:
        image1: First input image (numpy array or tensor).
        image2: Second input image (numpy array or tensor).
        view_model: Trained model to classify views (CC or MLO).

    Returns:
        image_cc: The image classified as CC.
        image_mlo: The image classified as MLO.
    """
    # Preprocess images to match the input format for the model
    def preprocess(image):
        # change from grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image = cv2.resize(image, (224, 224))  # Resize to model input size
        image = np.float32(image) / 255.0  # Normalize to [0, 1]
        image = image.transpose(2, 0, 1)  # Channel-first format
        image_tensor = torch.from_numpy(image).unsqueeze(0).to('cpu')  # Add batch dimension

        return image_tensor

    # Preprocess both images
    image1_tensor = preprocess(image1)
    image2_tensor = preprocess(image2)

    # Predict view type for both images
    with torch.no_grad():
        view_model.eval()
        pred1 = torch.softmax(view_model(image1_tensor), dim=1)
        pred2 = torch.softmax(view_model(image2_tensor), dim=1)

    # Get predicted labels (1 = MLO, 0 = CC)
    label1 = torch.argmax(pred1).item()
    label2 = torch.argmax(pred2).item()
    print(pred1, pred2)

    # Arrange images based on their predicted labels
    if label1 == 1 and label2 == 0:  # image1 is MLO, image2 is CC
        image_cc = image2
        image_mlo = image1
        print("image1 is MLO, image2 is CC")
    elif label1 == 0 and label2 == 1:  # image1 is CC, image2 is MLO
        image_cc = image1
        image_mlo = image2
        print("image1 is CC, image2 is MLO")
    else:
        print(pred1, pred2)
        image_cc = image1
        image_mlo = image2

    return image_cc, image_mlo



# <<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def cancer_prediction(image1, image2):
    
    ap = argparse.ArgumentParser(description='[Poli-USP] Two Views Breast Cancer inference')
    ap.add_argument("-d", "--model", help="two-views detector model")
    ap.add_argument("-a", "--aug", help="select to use translation augmentation: -a true")

    args = vars(ap.parse_args())

    model_file = 'models_side_mid_clf_efficientnet-b0/2021-08-03-03h54m_100ep_1074n_last_model_BEST.pt'

    use_aug = False

    print(f'\n--> {NETWORK} {TOPOLOGY} \n')

    view_model = SimpleCNN().to('cpu')
    checkpoint_path = r"model\cancer\breast_cancer_view_classifier (1).pth"
    view_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    view_model.eval()
    image_cc, image_mlo = arrange_images(image1, image2, view_model)
    model, device = load_model(NETWORK, TOPOLOGY)

    # now overwirte the original model with 2-views-pre-trained for test
    model = get_2views_model(model, model_file, device)

    image = cv2.resize(image_cc, (896, 1152))
    image_cc = np.zeros((*image_cc.shape[0:2], 3), dtype=np.uint16)
    image_cc[:, :, 0] = image
    image_cc[:, :, 1] = image
    image_cc[:, :, 2] = image

    image = cv2.resize(image_mlo, (896, 1152))
    image_mlo = np.zeros((*image_mlo.shape[0:2], 3), dtype=np.uint16)
    image_mlo[:, :, 0] = image
    image_mlo[:, :, 1] = image
    image_mlo[:, :, 2] = image

    if not use_aug:
        tta_predictions = simple_prediction(image_cc, image_mlo, model, device)
    pred = np.mean(tta_predictions)

    return pred

if __name__ == '__main__':
    cc_image_path = r"D:\Work\test data\cancer\Benign\Calc-Test_P_00127_RIGHT_MLO.png"
    mlo_image_path = r"D:\Work\test data\cancer\Benign\Calc-Test_P_00127_RIGHT_CC.png"

    cc_image = cv2.imread(cc_image_path, cv2.IMREAD_UNCHANGED)
    mlo_image = cv2.imread(mlo_image_path, cv2.IMREAD_UNCHANGED)
    print(cancer_prediction(cc_image, mlo_image))
