from PIL import Image
import numpy as np
import pydicom
from skimage import transform
import torch.utils.data
from torchvision import transforms
import torch.utils.data
import torch
import torch.utils.data

def get_slice(path):
    ##############################################
    # Reads .dcm file in path and outputs pydicom class
    ##############################################
    dataset = pydicom.dcmread(path)
    return dataset

def prepro_isolate_lungs(pydicom_dataset):
    ##############################################
    # reads pydicom dataset, selects lungs and outpts 2d array 512 x 512 of 0 and 1's
    ##############################################
    img = pydicom_dataset.pixel_array
    img = (img + pydicom_dataset.RescaleIntercept) / pydicom_dataset.RescaleSlope
    img = np.logical_and(img > -2000, img < -500)
    img = transform.resize(img, (512,512))
    img = img.astype(int)
    return img

def bw2d_to_RGB3d(img):
    ##############################################
    # Reads 2d numpy array (bw image) outputs RGB PIL image
    ##############################################
    img = img + np.abs(np.min(img))
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    img = np.array(np.hsplit(img, img.shape[1]))
    img = np.concatenate((img, img, img), axis=2)
    return Image.fromarray(img).convert('RGB')

def feat_extr(model, path):
    ##############################################
    # Returns the 1000 dim feature vector after pre-trained feature extractor of <model>
    ##############################################
    img = bw2d_to_RGB3d(prepro_isolate_lungs(get_slice(path)))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    # move to cuda if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    return model(input_batch)
