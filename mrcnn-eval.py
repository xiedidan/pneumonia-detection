import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
else:
    plt.switch_backend('tkagg')
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

def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)

    # allow all images
    target_images = []

    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')

        if (fp in image_fps) and (fp not in target_images):
            target_images.append(fp)

    # create annotation list
    image_annotations = {fp: [] for fp in target_images}

    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')

        if (fp in image_fps) and (row['Target'] == 1):
            image_annotations[fp].append(row)

    return target_images, image_annotations 

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

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
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_PADDING = False

    # RPN_ANCHOR_SCALES = (16, 32, 64, 128)

    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (256, 256)  # (height, width) of the mini-mask

    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 5
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.95 # 0.78  ## match target distribution
    DETECTION_NMS_THRESHOLD = 0.01

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

# eval dataset
val_dicom_dir = os.path.join(flags.root, 'eval')
anns = pd.read_csv(os.path.join(flags.root, 'stage_1_train_labels.csv'))
print(anns.head())

image_fps, image_annotations = parse_dataset(val_dicom_dir, anns=anns)

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

image_fps_list = list(image_fps)
image_fps_val = image_fps_list

print('Samples in eval set: {}'.format(len(image_fps_val)))
# print(image_fps_val[:6])

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
model.load_weights(flags.checkpoint)

# ok, eval
aps = []
ap_dist = {'hit': 0, 'fp': 0, 'fn': 0, 'miss': 0, 'neg': 0}
image_ids = dataset_val.image_ids

model.eval()

if flags.plot:
    for image_id in image_ids:
        # load original image w/o mini-mask
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset_val,
            config,
            image_id,
            use_mini_mask=False
        )

        plt.ion()
        fig = plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        visualize.display_instances(
            image,
            gt_bbox,
            gt_mask,
            gt_class_id,
            dataset_val.class_names,
            # colors=get_colors_for_class_ids(gt_class_id),
            ax=fig.axes[-1]
        )

        plt.subplot(1, 2, 2)
        results = model.detect([image]) #, verbose=1)
        if results is None:
            visualize.display_instances(
                image,
                np.array([]),
                [],
                [],
                dataset_val.class_names,
                # colors=get_colors_for_class_ids(gt_class_id),
                ax=fig.axes[-1]
            )
        else:
            r = results[0]

            visualize.display_instances(
                image,
                r['rois'],
                r['masks'],
                r['class_ids'],
                dataset_val.class_names,
                r['scores'],
                # colors=get_colors_for_class_ids(r['class_ids']),
                ax=fig.axes[-1]
            )
        
        plt.ioff()
        plt.show()

for image_id in tqdm(image_ids):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_val,
        config,
        image_id
    )

    results = model.detect([image])
    if results is None:
        r = {'rois': [], 'class_ids': [], 'scores': [], 'masks': []}
    else:
        r = results[0]

    if len(r['class_ids']) == 0:
        # no pred bbox
        if len(gt_class_id) == 1 and gt_class_id[0] == 0:
            # also no gt bbox, no count, goto next image
            ap_dist['neg'] += 1
            continue
        else:
            # got gt bbox, false-negative, count a 0
            ap_dist['fn'] += 1
            aps.append(0.)
            continue
    else:
        # got pred box
        if len(gt_class_id) == 1 and gt_class_id[0] == 0:
            # no gt bbox, false-positive, count a 0
            ap_dist['fp'] += 1
            aps.append(0.)
            continue

    ap = utils.compute_ap_range(
        gt_bbox,
        gt_class_id,
        gt_mask,
        r['rois'],
        r['class_ids'],
        r['scores'],
        r['masks'],
        iou_thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    )

    if ap > 0.:
        ap_dist['hit'] += 1
    else:
        ap_dist['miss'] += 1

    aps.append(ap)

print(len(aps))
print('mAP: {}'.format(np.mean(aps)))
print('hit:\t{}\nmiss:\t{}\nfp:\t{}\nfn:\t{}\nneg:\t{}'.format(
    ap_dist['hit'],
    ap_dist['miss'],
    ap_dist['fp'],
    ap_dist['fn'],
    ap_dist['neg']
))
