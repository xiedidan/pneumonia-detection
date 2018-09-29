import sys
import os
import pickle
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

def verificationCollate(batch):
    images = []
    gts = []
    ws = []
    hs = []
    ids = []

    for i, sample in enumerate(batch):
        image, gt, w, h, patientId = sample

        images.append(image)

        gt = torch.tensor(gt, dtype=torch.int)
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
        return df[df['classNo'] != classMapping['No Lung Opacity / Not Normal']]

class PneumoniaVerificationDataset(Dataset):
    def __init__(self, root, classMapping, num_classes=3, crop_ratio=1.25, phase='train', transform=None, target_transform=None, detection_path='./detection.pth'):
        self.root = root
        self.classMapping = classMapping
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.detection_path = detection_path

        self.image_path = os.path.join(self.root, self.phase)

        if self.phase == 'train' or self.phase == 'val':
            self.class_path = os.path.join(self.root, 'stage_1_detailed_class_info.csv')
            self.gt_path = os.path.join(self.root, 'stage_1_train_labels.csv')

            # list image files
            self.image_files = os.listdir(self.image_path)
            ids = [filename.split('.')[0] for filename in self.image_files]

            # load class
            self.class_df = pd.read_csv(self.class_path)
            self.class_df = self.class_df[self.class_df['patientId'].isin(ids)]
            self.class_groups = self.class_df.groupby('class')

            # load gt
            self.df = pd.read_csv(self.gt_path)
            self.df = self.df[self.df['patientId'].isin(ids)]

            self.target_df = self.df[self.df['Target'] == 1]
            self.background_df = self.df[self.df['Target'] == 0]

            self.total_len = self.num_classes * len(self.target_df)
        else: # test
            self.image_files = os.listdir(self.image_path)

            with open(self.detection_path, 'rb') as dump:
                detections = pickle.load(dump)
            self.ids = detections['ids']
            self.bboxes = detections['bboxes']

            self.total_len = 0
            for bboxes in self.bboxes:
                self.total_len += len(bboxes)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if self.phase == 'train' or self.phase == 'val':
            target_offset = (self.num_classes - 1) * len(self.target_df)

            if index >= target_offset: # target
                # get sample from target
                row = self.target_df.iloc[index - target_offset]
                patientId = row['patientId']

                # read image
                filename = '{}.dcm'.format(patientId)
                image_file = os.path.join(self.image_path, filename)
                image, _, __ = load_dicom_image(image_file)

                x = row['x'] - row['width'] * (self.crop_ratio - 1.) / 2.
                y = row['y'] - row['height'] * (self.crop_ratio - 1.) / 2.
                w = row['width'] * self.crop_ratio
                h = row['height'] * self.crop_ratio

                # crop
                crop = transforms.functional.crop(image, y, x, h, w)

                gt = self.classMapping['Lung Opacity']
            else: # background
                # get class_index and group size
                target_len = len(self.target_df)

                class_index = index // target_len

                class_name = get_class_name(self.classMapping, class_index)
                group = self.class_groups.get_group(class_name)

                class_size = group.shape[0]

                if self.phase == 'train':
                    # randomly pick a target bbox and background image
                    target_index = random.randint(0, target_len - 1)
                    item_index = random.randint(0, class_size - 1)
                else: # val
                    # get bbox from target
                    target_index = index % target_len
                    item_index = (index % target_len) % class_size
                
                # get target bbox
                target_row = self.target_df.iloc[target_index]

                # get image file from background group
                row = group.iloc[item_index]
                patientId = row['patientId']

                # read image
                filename = '{}.dcm'.format(patientId)
                image_file = os.path.join(self.image_path, filename)
                image, _, __ = load_dicom_image(image_file)

                # crop background image with target bbox
                x = target_row['x'] - target_row['width'] * (self.crop_ratio - 1.) / 2.
                y = target_row['y'] - target_row['height'] * (self.crop_ratio - 1.) / 2.
                w = target_row['width'] * self.crop_ratio
                h = target_row['height'] * self.crop_ratio

                # crop
                crop = transforms.functional.crop(image, y, x, h, w)

                gt = class_index # equals to self.classMapping[row['class']]
        else: # test
            patient_index = 0
            bbox_count = 0
            bbox_index = 0

            # search for patient_index and bbox_index
            for i, bboxes in enumerate(self.bboxes):
                lower_bound = bbox_count
                upper_bound = bbox_count + len(bboxes)

                if lower_bound <= index < upper_bound:
                    patient_index = i
                    bbox_index = index - lower_bound
                    break
                else:
                    bbox_count += len(bboxes)

            bbox = self.bboxes[patient_index][bbox_index, :4]
            patientId = self.ids[patient_index]

            # read image
            filename = '{}.dcm'.format(patientId)
            image_file = os.path.join(self.image_path, filename)
            image, _, __ = load_dicom_image(image_file)

            # crop background image with target bbox
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            x = bbox[0] - w * (self.crop_ratio - 1.) / 2.
            y = bbox[1] - h * (self.crop_ratio - 1.) / 2.
            w = w * self.crop_ratio
            h = h * self.crop_ratio

            # crop
            crop = transforms.functional.crop(image, y, x, h, w)

            # use gt to pass bbox (not in point form)
            gt = self.bboxes[patient_index][bbox_index, :4]

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, gt, w, h, patientId
