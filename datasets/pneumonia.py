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

'''
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
'''

def get_groundtruth(df, patientId, w, h):
    lines = df[df['patientId']==patientId]

    gts = []

    for index, line in lines.iterrows():
        # convert [xmin, ymin, width, height] to point-from [xmin, ymin, xmax, ymax]
        gts.append([line['x'], line['y'], line['x'] + line['width'], line['y'] + line['height'], 0])

    return gts

def detectionCollate(batch):
    images = []
    gts = []
    ws = []
    hs = []
    ids = []

    for i, sample in enumerate(batch):
        image, gt, w, h, patientId = sample

        images.append(torch.tensor(image))
        gts.append(torch.from_numpy(gt))
        ws.append(w)
        hs.append(h)
        ids.append(patientId)

    images = torch.stack(images, 0)
    images.transpose_(1, 3)
    images.transpose_(2, 3)

    return images, gts, ws, hs, ids

def classificationCollate(batch):
    images = []
    gts = []
    ws = []
    hs = []
    ids = []

    for i, sample in enumerate(batch):
        image, gt, w, h, patientId = sample

        images.append(image)

        gt = torch.tensor(gt, dtype=torch.uint8)
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

        if self.phase == 'train' or self.phase == 'val':
            self.gt_path = os.path.join(self.root, 'stage_1_detailed_class_info.csv')

            # load gt
            self.df = pd.read_csv(self.gt_path)

            self.image_files = os.listdir(self.image_path)

            ids = [filename.split('.')[0] for filename in self.image_files]
            self.df = self.df[self.df['patientId'].isin(ids)]

            if self.phase == 'train':
                self.groups = self.df.groupby('class')
                self.max_class_size = self.groups.size().max()

                self.total_len = self.num_classes * self.max_class_size
            else: # val
                self.total_len = len(self.image_files)
        else: # test
            self.image_files = os.listdir(self.image_path)
            self.total_len = len(self.image_files)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if self.phase == 'train':
            classIndex = index // self.max_class_size
            className = get_class_name(self.classMapping, classIndex)

            group = self.groups.get_group(className)
            class_size = group.shape[0]
            itemIndex = (index % self.max_class_size) % class_size

            row = group.iloc[itemIndex]
            patientId = row['patientId']

            filename = '{}.dcm'.format(patientId)

            gt = classIndex
        else: # val and test
            filename = self.image_files[index]
            patientId = filename.split('.')[0]

            gt = 0 # dummy gt for test phase

            if self.phase == 'val':
                # query gt from df with patientId
                rows = self.df[self.df['patientId'] == patientId]
                row = rows.iloc[0]
                gt = self.classMapping[row['class']]

        image_file = os.path.join(self.image_path, filename)
        image, w, h = load_dicom_image(image_file)

        if self.transform is not None:
            image = self.transform(image)

        return image, gt, w, h, patientId

class PneumoniaDetectionDataset(Dataset):
    def __init__(self, root, classMapping, num_classes=2, phase='train', transform=None, target_transform=None, classification_path=None):
        self.root = root
        self.classMapping = classMapping
        self.num_classes = num_classes
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform

        self.image_path = os.path.join(self.root, self.phase)

        if self.phase == 'train' or self.phase == 'val':
            self.gt_path = os.path.join(self.root, 'stage_1_train_labels.csv')

            # list images
            self.image_files = os.listdir(self.image_path)
            ids = [filename.split('.')[0] for filename in self.image_files]

            # load gt
            self.df = pd.read_csv(self.gt_path)
            self.df = self.df[self.df['Target'] == 1]
            self.df = self.df[self.df['patientId'].isin(ids)]

            self.total_len = len(self.df)
        else: # test
            self.classification_path = classification_path
            self.df = pd.read_csv(self.classification_path)

            self.df = self._pick_sample(self.df, self.classMapping)
            self.total_len = len(self.df)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        row = self.df.iloc[index]
        patientId = row['patientId']

        filename = '{}.dcm'.format(patientId)
        image_file = os.path.join(self.image_path, filename)

        image, w, h = load_dicom_image(image_file)
        image = np.array(image)
        image = image[:, :, np.newaxis]
        image = np.tile(image, (1, 1, 3))

        if self.phase == 'train' or self.phase == 'val':
            gt = get_groundtruth(self.df, patientId, w, h)
        else: # test
            gt = [[0, 0, w, h, 0]] # dummpy gt

        if self.transform is not None:
            gt = np.array(gt, dtype='float')
            image, boxes, labels = self.transform(image, gt[:, :4], gt[:, 4])

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return image, gt, w, h, patientId

    def _pick_sample(self, df, classMapping):
        # TODO : more complicated pick method?
        return df[df['classNo'] == classMapping['Lung Opacity']]
