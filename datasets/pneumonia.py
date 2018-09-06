import sys
import os
import math

import pydicom
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from .transforms import *

# fix for 'RuntimeError: received 0 items of ancdata' problem
if sys.platform == 'linux':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def load_dicom_image(filename):
    ds = sitk.ReadImage(filename)

    image = sitk.GetArrayFromImage(ds)
    c, w, h = image.shape

    # we know these are uint8 gray pictures already
    image = Image.fromarray(image[0])

    return image, w, h

def get_groundtruth(df, patientId, w, h):
    lines = df[df['patientId']==patientId]

    locs = []
    confs = []

    for index, line in lines.iterrows():
        if line['Target'] == 0:
            # background label
            locs.append([0., 0., w, h])
            confs.append(0)
        else:
            locs.append([line['x'], line['y'], line['width'], line['height']])
            confs.append(1)

    return np.array(locs), np.array(confs)

def detectionCollate(batch):
    images = []
    locs = []
    confs = []
    ws = []
    hs = []
    ids = []

    for i, sample in enumerate(batch):
        image, gt, w, h, patientId = sample
        loc, conf = gt

        images.append(image)
        locs.append(loc.to(torch.float32))
        confs.append(conf.to(torch.uint8))
        ws.append(w)
        hs.append(h)
        ids.append(patientId)

    images = torch.stack(images, 0)

    return images, (locs, confs), ws, hs, ids

def classificationCollate(batch):
    images = []
    gts = []
    ws = []
    hs = []
    ids = []

    for i, sample in enumerate(batch):
        image, gt, w, h, patientId = sample

        images.append(image)
        gts.append(gt)
        ws.append(w)
        hs.append(h)
        ids.append(patientId)

    images = torch.stack(images, 0)
    gts = torch.stack(gts, 0)

    return images, gts, ws, hs, ids

def get_class_name(classMapping, value):
    for i, item in enumerate(classMapping):
        if i == value:
            return item

class PneumoniaClassificationDataset(Dataset):
    def __init__(self, root, classMapping, num_classes=3, phase='train', transform=None, target_transform=None):
        self.root = root
        self.classMapping = classMapping
        self.num_classes = num_classes
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform

        self.image_path = os.path.join(self.root, self.phase)
        self.gt_path = os.path.join(self.root, 'stage_1_detailed_class_info.csv')

        # load gt
        self.df = pd.read_csv(self.gt_path)

        self.image_files = os.listdir(self.image_path)
        self.df = self.df[df['class'].isin(self.image_files)]

        self.groups = self.df.groupby('class')
        print(self.groups.size())
        self.max_class_size = self.groups.size().max()

        self.total_len = self.num_classes * self.max_class_size

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        classIndex = index // self.max_class_size
        className = get_class_name(self.classMapping, classIndex)

        group = self.groups.get_group(className)
        class_size = group.shape[0]
        itemIndex = (index % self.max_class_size) % class_size

        row = group.iloc[itemIndex]
        patientId = row['patientId']

        filename = '{}.dcm'.format(patientId)
        image_file = os.path.join(self.image_path, filename)

        image, w, h = load_dicom_image(image_file)
        gt = classIndex

        if self.transform is not None:
            image = self.transform(image)

        return image, gt, w, h, patientId

class PneumoniaDetectionDataset(Dataset):
    def __init__(self, root, num_classes=2, phase='train', transform=None, target_transform=None):
        self.root = root
        self.num_classes = num_classes
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform

        self.image_path = os.path.join(self.root, self.phase)
        self.gt_path = os.path.join(self.root, 'stage_1_train_labels.csv')

        # list images
        self.image_files = os.listdir(self.image_path)
        self.total_len = len(self.image_files)

        # load gt
        self.df = pd.read_csv(self.gt_path)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        filename = self.image_files[index]

        patientId = filename.split('.')[0]
        image_file = os.path.join(self.image_path, filename)

        image, w, h = load_dicom_image(image_file)
        gt = get_groundtruth(self.df, patientId, w, h)

        if self.transform is not None:
            image, gt, w, h = self.transform(image, gt, w, h)

        if self.target_transform is not None:
            gt, w, h = self.target_transform(gt, w, h)

        return image, gt, w, h, patientId
