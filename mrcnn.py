import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob
from sklearn.model_selection import KFold
import pickle
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

# helper
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')

    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns, only_target=True): 
    image_fps = get_dicom_fps(dicom_dir)

    # filter out non-target
    target_images = []

    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')

        if only_target:
            if (row['Target'] == 1) and (fp in image_fps) and (fp not in target_images):
                target_images.append(fp)
        elif (fp in image_fps) and (fp not in target_images):
            target_images.append(fp)

    # create annotation list
    image_annotations = {fp: [] for fp in target_images}

    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')

        if fp in image_fps:
            if row['Target'] == 1:
                image_annotations[fp].append(row)

    return target_images, image_annotations 

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Mask RCNN Pipeline')
parser.add_argument('--lr', default=0.004, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--transfer', action='store_true', help='transfer learning from COCO pre-weights')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--lib', default='../', help='Mask RCNN library location')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root directory')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--epochs', default=200, type=int, help='stop epoch')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--steps', default=200, type=int, help='steps(batch) in each epoch')
parser.add_argument('--plot', action='store_true', help='plot images')
flags = parser.parse_args()

# load Mask RCNN library
print('Loading Mask RCNN library from {}'.format(flags.lib))

sys.path.append(flags.lib)
from config import Config
import utils
import model as modellib
import visualize
from model import log

device = torch.device(flags.device)

# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 
class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet101'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_PADDING = False

    RPN_NMS_THRESHOLD  = 0.9
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.25, 0.33, 0.5, 1, 2, 3, 4]
    TRAIN_ROIS_PER_IMAGE = 16

    USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (256, 256)  # (height, width) of the mini-mask
    
    MAX_GT_INSTANCES = 5
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.3

config = DetectorConfig()
config.display()

class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image(
                'pneumonia',
                image_id=i,
                path=fp, 
                annotations=annotations,
                orig_height=orig_height,
                orig_width=orig_width
            )
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array

        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
  
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)

        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)

            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])

                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1

        return mask.astype(np.bool), class_ids.astype(np.int32)

# training dataset
train_dicom_dir = os.path.join(flags.root, 'train')
val_dicom_dir = os.path.join(flags.root, 'val')
anns = pd.read_csv(os.path.join(flags.root, 'stage_1_train_labels.csv'))

print(anns.head())

train_image_fps, train_image_annotations = parse_dataset(train_dicom_dir, anns=anns, only_target=True)
val_image_fps, val_image_annotations = parse_dataset(val_dicom_dir, anns=anns, only_target=False)

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

image_fps_train = list(train_image_fps)
image_fps_val = list(val_image_fps)

image_annotations = {}
image_annotations.update(train_image_annotations)
image_annotations.update(val_image_annotations)

print('Samples in train set: {}, val set: {}'.format(len(image_fps_train), len(image_fps_val)))
# print(image_fps_val[:6])

# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

# prepare the training dataset
dataset_train = DetectorDataset(
    image_fps_train,
    image_annotations,
    ORIG_SIZE,
    ORIG_SIZE
)
dataset_train.prepare()

# prepare the validation dataset
dataset_val = DetectorDataset(
    image_fps_val,
    image_annotations,
    ORIG_SIZE,
    ORIG_SIZE
)
dataset_val.prepare()

model = modellib.MaskRCNN(
    config=config,
    model_dir=flags.lib,
    device=device
)
model.to(device)

if flags.resume:
    model.load_weights(flags.checkpoint)

    model.train_model(
        dataset_train,
        dataset_val,
        learning_rate=flags.lr,
        epochs=flags.epochs,
        BatchSize=flags.batch_size,
        steps=flags.steps,
        layers='all',
        augmentation=augmentation
    )
else:
    if flags.transfer:
        coco_weights_path = os.path.join(flags.lib, 'mask_rcnn_coco.pth')
        model.load_pre_weights(coco_weights_path)
    else:
        model.load_pre_weights('./checkpoint.pth') # a quick fix for log path problem

    # Train Mask-RCNN Model 
    ## train heads with higher lr to speedup the learning
    model.train_model(
        dataset_train,
        dataset_val,
        learning_rate=flags.lr * 2,
        epochs=2,
        BatchSize=flags.batch_size,
        steps=flags.steps,
        layers='heads',
        augmentation=None
    )  ## no need to augment yet

    model.train_model(
        dataset_train,
        dataset_val,
        learning_rate=flags.lr,
        epochs=6,
        BatchSize=flags.batch_size,
        steps=flags.steps,
        layers='all',
        augmentation=augmentation
    )

    model.train_model(
        dataset_train,
        dataset_val,
        learning_rate=flags.lr / 5,
        epochs=flags.epochs,
        BatchSize=flags.batch_size,
        steps=flags.steps,
        layers='all',
        augmentation=augmentation
    )
