import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
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

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Mask RCNN Pipeline')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--lib', default='../', help='Mask RCNN library location')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root directory')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
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
    GPU_COUNT = 2
    IMAGES_PER_GPU = 256
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_PADDING = False
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 4
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.97  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# training dataset
test_dicom_dir = os.path.join(flags.root, 'test')
test_image_fps = get_dicom_fps(test_dicom_dir)

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

model = modellib.MaskRCNN(
    config=config,
    model_dir=flags.lib,
    device=device
)
model.to(device)

model.load_weights(flags.checkpoint)

# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='submission.csv', min_conf=0.95):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    #resize_factor = ORIG_SIZE
    with open(filepath, 'w') as file:
        file.write("patientId,PredictionString\n")

        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            image, window, scale, padding = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM
            )

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])

            if results is None:
                r = {'rois': [], 'class_ids': [], 'scores': [], 'masks': []}
            else:
                r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","

            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )

            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                           width*resize_factor, height*resize_factor)
                        out_str += bboxes_str

            file.write(out_str+"\n")

submission_fp = os.path.join(flags.root, 'submission.csv')
predict(test_image_fps, filepath=submission_fp, min_conf=config.DETECTION_MIN_CONFIDENCE)

print(submission_fp)
