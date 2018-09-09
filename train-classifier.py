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

from resnet import *
from datasets.pneumonia import *
from utils.plot import *

# constants & configs
num_classes = 3
snapshot_interval = 1000
pretrained = False

# spawned workers on windows take too much gmem
num_workers = 8
if sys.platform == 'win32':
    number_workers = 2

size = [512, 512]
mean = [0.49043187350911405]
std = [0.22854086980778032]
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# variables
start_epoch = 0
best_loss = float('inf')

# helper functions
def xavier(param):
    init.xavier_uniform_(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def snapshot(epoch, batch, model, loss):
    print('Taking snapshot, loss: {}'.format(loss))

    state = {
        'net': model.state_dict(),
        'loss': loss,
        'epoch': epoch
    }
    
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')

    torch.save(state, './snapshot/e_{:0>6}_b_{:0>6}_loss_{:.5f}.pth'.format(
        epoch,
        batch,
        loss
    ))

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Classifier Training')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--transfer', action='store_true', help='fintune pretrained model')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--lock_feature', action='store_true', help='lock feature layers for baseline')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
flags = parser.parse_args()

device = torch.device(flags.device)

# data loader
'''
trainTransform = Compose([
    RandomResizedCrop(size=size, p=0.7, scale=(0.1, 1.), ratio=(0.5, 2)),
    RandomSaltAndPepper(size=size),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    RandomHorizontalFlip(size=size, p=0.5),
    Percentage(size=size),
    ToTensor()
])
'''
# TODO : data arguument
trainTransform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

valTransform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainSet = PneumoniaClassificationDataset(
    root=flags.root,
    classMapping=classMapping,
    phase='train',
    transform=trainTransform,
    num_classes=num_classes
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers,
    collate_fn=classificationCollate,
)

valSet = PneumoniaClassificationDataset(
    root=flags.root,
    classMapping=classMapping,
    phase='val',
    transform=valTransform,
    num_classes=num_classes
)
valLoader = torch.utils.data.DataLoader(
    valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=classificationCollate,
)

# model
resnet = resnet152(pretrained, num_classes=num_classes)
resnet.to(device)

if flags.resume:
    checkpoint = torch.load(flags.checkpoint)
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    resnet.load_state_dict(checkpoint['net'])
    updating_parameters = resnet.parameters()
elif flags.transfer:
    checkpoint = torch.load(flags.checkpoint)
    updating_parameters = resnet.transfer(checkpoint, flags.lock_feature)
else:
    # train from scratch - init weights
    # resnet.apply(init_weight)
    updating_parameters = resnet.parameters()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    updating_parameters,
    lr=flags.lr,
    momentum=0.9,
    weight_decay=1e-4
)

def train(epoch):
    print('\nTraining Epoch: {}'.format(epoch))

    resnet.train()
    train_loss = 0
    batch_count = len(trainLoader)

    for batch_index, (samples, gts, ws, hs, ids) in enumerate(trainLoader):
        samples = samples.to(device)
        gts = gts.to(device=device, dtype=torch.long)

        optimizer.zero_grad()

        if torch.cuda.device_count() > 1:
            outputs = nn.parallel.data_parallel(resnet, samples)
        else:
            outputs = resnet(samples)
        
        if torch.cuda.device_count() > 1:
            loss = nn.parallel.data_parallel(criterion, (outputs, gts))
        else:
            loss = criterion(outputs, gts)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('e:{}/{}, b:{}/{}, b_l:{:.2f}, e_l:{:.2f}'.format(
            epoch + 1,
            flags.end_epoch,
            batch_index + 1,
            batch_count,
            loss.item(),
            train_loss / (batch_index + 1)
        ))

        if (batch_index + 1) % snapshot_interval == 0:
            snapshot(epoch, batch_index, resnet, train_loss / (batch_index + 1))

def val(epoch):
    print('\nVal')

    with torch.no_grad():
        resnet.eval()
        val_loss = 0
        batch_count = len(valLoader)

        # perfrom forward
        for batch_index, (samples, gts, ws, hs, ids) in enumerate(valLoader):
            samples = samples.to(device)
            gts = gts.to(device=device, dtype=torch.long)

            if torch.cuda.device_count() > 1:
                outputs = nn.parallel.data_parallel(resnet, samples)
            else:
                outputs = resnet(samples)

            loss = criterion(outputs, gts)

            val_loss += loss.item()

            print('e:{}/{}, b:{}/{}, b_l:{:.2f}, e_l:{:.2f}'.format(
                epoch + 1,
                flags.end_epoch,
                batch_index + 1,
                batch_count,
                loss.item(),
                val_loss / (batch_index + 1)
            ))

        # save checkpoint
        global best_loss
        val_loss /= len(valLoader)

        if val_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(val_loss))

            state = {
                'net': resnet.state_dict(),
                'loss': val_loss,
                'epoch': epoch,
            }
            
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/epoch_{:0>5}_loss_{:.5f}.pth'.format(
                epoch,
                val_loss
            ))

            best_loss = val_loss

# ok, main loop
if __name__ == '__main__':
    for epoch in range(start_epoch, flags.end_epoch):
        train(epoch)
        val(epoch)
