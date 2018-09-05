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

def collate(batch):
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
        confs.append(conf.to(torch.float32))
        ws.append(w)
        hs.append(h)
        ids.append(patientId)

    images = torch.stack(images, 0)

    return images, (locs, confs), ws, hs, ids

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
