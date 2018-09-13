# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
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

import caffe
from caffe import layers as L
from caffe import params as P

from datasets.pneumonia import *
from datasets.transforms import *
from utils.plot import *

# constants & configs
model_path = './caffe/caffemodel/'
prototxt_path = './caffe/prototxt/'

start_layer_name = 'conv1_1'
loss_layer_name = 'mbox_loss'

data_blob_name = 'data'
label_blob_name = 'label'
loss_blob_name = 'mbox_loss'

num_classes = 2
snapshot_interval = 1000
pretrained = False

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 2

size = 512
mean = [0.49043187350911405]
std = [0.22854086980778032]

# be ware - this is different from mapping in classification
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 0,
    'Lung Opacity': 1
}

# variables
start_epoch = 0
best_loss = float('inf')

# helper
def get_device(device):
    arr = device.split(':')

    if len(arr) == 1:
        device_type = 'cpu'
        device_id = 0
    else:
        device_type = 'cuda'
        device_id = int(arr[1])
    
    return device_type, device_id

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Classifier Training')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
flags = parser.parse_args()

# data
trainTransform = Compose([
    RandomResizedCrop(size=size, p=0.7, scale=(0.9, 1.), ratio=(0.9, 1/0.9)),
    Percentage(size=size),
    ToTensor()
])

trainSet = PneumoniaDetectionDataset(
    root=flags.root,
    phase='train',
    transform=trainTransform,
    classMapping=classMapping,
    num_classes=num_classes
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers,
    collate_fn=detectionCollate,
)

valTransform = Compose([
    Resize(size=size),
    Percentage(size=size),
    ToTensor(),
])

valSet = PneumoniaDetectionDataset(
    root=flags.root,
    phase='val',
    transform=valTransform,
    classMapping=classMapping,
    num_classes=num_classes
)
valLoader = torch.utils.data.DataLoader(
    valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=detectionCollate,
)

# model
device_type, device_id = get_device(flags.device)

if device_type == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    
solver = caffe.get_solver(os.path.join(prototxt_path, 'solver.prototxt'))

def train(epoch):
    print('\nTraining Epoch: {}'.format(epoch))

    train_loss = 0
    batch_count = len(trainLoader)

    for batch_index, (images, gts, ws, hs, ids) in enumerate(trainLoader):
        images = images.numpy()
        gts = gts.numpy()

        # set data to data layer
        solver.net.blobs[data_blob_name].data[...] = images
        solver.net.blobs[label_blob_name].data[...] = gts

        # train - do NOT use solver.step(1), or data layer will be overwritten
        solver.net.forward(start=start_layer_name, end=loss_layer_name)
        solver.net.backward()

        # get loss from net - should be a scalar
        loss = solver.net.blobs[loss_blob_name].data[0]

        train_loss += loss

        print('e:{}/{}, b:{}/{}, b_l:{:.2f}, e_l:{:.2f}'.format(
            epoch,
            flags.end_epoch - 1,
            batch_index,
            batch_count - 1,
            loss,
            train_loss / (batch_index + 1)
        ))

def val(epoch):
    print('\nVal')

    val_loss = 0
    batch_count = len(valLoader)

    # just perfrom forward on training net
    for batch_index, (images, gts, ws, hs, ids) in enumerate(valLoader):
        images = images.numpy()
        gts = gts.numpy()

        # set data to data layer
        solver.net.blobs[data_blob_name].data[...] = images
        solver.net.blobs[label_blob_name].data[...] = gts

        # train - do NOT use solver.step(1), or data layer will be overwritten
        solver.net.forward(start=start_layer_name, end=loss_layer_name)
        solver.net.backward()

        # get loss from net - should be a scalar
        loss = solver.net.blobs[loss_blob_name].data[0]

        val_loss += loss

        print('e:{}/{}, b:{}/{}, b_l:{:.2f}, e_l:{:.2f}'.format(
            epoch,
            flags.end_epoch - 1,
            batch_index,
            batch_count - 1,
            loss,
            val_loss / (batch_index + 1)
        ))

    global best_loss
    val_loss /= batch_count

    # save checkpoint
    if val_loss < best_loss:
        print('Saving checkpoint, best loss: {}'.format(val_loss))

        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        solver.net.save(os.path.join(
            model_path,
            'epoch_{:0>5}_loss_{:.5f}.caffemodel'.format(epoch, val_loss)
        ))

        best_loss = val_loss

# ok, main loop
if __name__ == '__main__':
    for epoch in range(start_epoch, flags.end_epoch):
        train(epoch)
        val(epoch)
