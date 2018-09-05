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

# fix for 'RuntimeError: received 0 items of ancdata' problem
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

    for line in lines:
        if line['Target'] == 0:
            # background label
            locs.append([0., 0., w, h])
            confs.append(0)
        else:
            locs.append([line['x'], line['y'], line['width'], line['height']])
            confs.append(1)

    return np.array(locs), np.array(confs)

def collate(batch):
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

    return images, gts, ws, hs, ids

class PneumoniaDataset(Dataset):
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

# mod from torchvision
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, gt, w, h):
        for t in self.transforms:
            image, gt, w, h = t(image, gt, w, h)
        return image, gt, w, h

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ComposeTarget(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, gt, w, h):
        for t in self.transforms:
            gt, w, h = t(gt, w, h)
        return gt, w, h

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# ToTensor wrapper
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image, gt, w, h):
        image = transforms.functional.to_tensor(image)
        gt = (
            torch.as_tensor(gt[0], dtype=torch.float32),
            torch.as_tensor(gt[1], dtype=torch.uint8)
        )

        return image, gt, w, h

# resize transform
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt, w, h):
        image = transforms.functional.resize(image, self.size)

        w_ratio = float(self.size[0]) / w
        h_ratio = float(self.size[1]) / h
        ratio = np.array([w_ratio, h_ratio, w_ratio, h_ratio], dtype=np.float32)

        gt[0] = gt[0] * ratio
        
        return image, gt, w, h

# convert gt to percentage format
class Percentage(object):
    def __init__(self, size):
        self.size = size
        self.scale = np.array([self.size[0], self.size[1], self.size[0], self.size[1]], dtype=np.float32)

    def __call__(self, gt, w, h):
        # Tensor.numpy() shares memory with the tensor itself
        locs_numpy = gt[0].numpy()
        locs_numpy = locs_numpy / self.scale

        return gt, w, h

# convert gt from (xmin, ymin, w, h) to bbox (xmin, ymin, xmax, ymax)
class ToBbox(object):
    def __init__(self):
        self.trans_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]).transpose(1, 0)

    def __call__(self, gt, w, h):
        locs_numpy = gt[0].numpy()
        locs_numpy = locs_numpy @ self.trans_matrix

        return gt, w, h
