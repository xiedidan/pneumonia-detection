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

from faf import build_faf
from datasets.vid import *
from utils.plot import *
from layers import MultiFrameBoxLoss

# constants & configs
packs = [
    'ILSVRC2015_VID_train_0000',
    'ILSVRC2015_VID_train_0001',
    'ILSVRC2015_VID_train_0002',
    'ILSVRC2015_VID_train_0003',
    'ILSVRC2017_VID_train_0000'
]
num_frames = 5
num_classes = 31
class_filename = 'class.mapping'
number_workers = 8
snapshot_interval = 1000

# variables
start_epoch = 0
best_loss = float('inf')

# helper functions
def collate(batch):
    # batch = [(image, gt), (image, gt), ...]
    images = []
    gts = []
    for i, sample in enumerate(batch):
        image, gt, w, h = sample
        
        images.append(image)
        gts.append(gt)
    
    # N, D, C, H, W
    images = torch.stack(images, 0)
            
    return images, gts
    
def xavier(param):
    init.xavier_uniform_(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def snapshot(epoch, batch, loss):
    print('Taking snapshot, loss: {}'.format(loss))

    state = {
        'net': faf.state_dict(),
        'loss': loss,
        'epoch': epoch,
    }
    
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')

    torch.save(state, './snapshot/e_{:0>6}_b_{:0>6}_loss_{:.5f}.pth'.format(
        epoch,
        batch,
        loss
    ))

# argparser
parser = argparse.ArgumentParser(description='FaF Training')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/ILSVRC/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--loss_device', default='cuda:0', help='device to calculate loss')
flags = parser.parse_args()

print('Got flags: {}'.format(flags))

if not os.path.exists(os.path.join(flags.root, 'dump/')):
    os.mkdir(os.path.join(flags.root, 'dump/'))

device = torch.device(flags.device)
loss_device = torch.device(flags.loss_device)

# data loader
size = [300, 300]

trainTransform = Compose([
    RandomResizedCrop(size=size, p=0.7, scale=(0.1, 1.), ratio=(0.5, 2)),
    RandomSaltAndPepper(size=size),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    RandomHorizontalFlip(size=size, p=0.5),
    Percentage(size=size),
    ToTensor()
])

valTransform = Compose([
    Resize(size=size),
    Percentage(size=size),
    ToTensor(),
])

# load or create class mapping
# remember to clear mapping before switching data set
class_path = os.path.join(flags.root, 'dump/', class_filename)

if file_exists(class_path) == True:
    with open(class_path, 'rb') as file:
        data = pickle.load(file)
        num_classes, classMapping = data['num_classes'], data['classMapping']
else:
    num_classes, classMapping = create_class_mapping(os.path.join(flags.root, 'Annotations/VID/val/'))

print('num_classes: {}\nclassMapping: {}'.format(num_classes, classMapping))

trainSet = VidDataset(
    root=flags.root,
    packs=packs,
    phase='train',
    transform=trainTransform,
    classDict=classMapping,
    num_classes=num_classes
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers,
    collate_fn=collate,
)

valSet = VidDataset(
    root=flags.root,
    packs=packs,
    phase='val',
    transform=valTransform,
    classDict=classMapping,
    num_classes=num_classes
)
valLoader = torch.utils.data.DataLoader(
    valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=collate,
)

# model
# cfg - for prior box and (maybe) detection
cfg = {
    'min_dim': 300,
    'aspect_ratios': [
        [2],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2., 3.],
        [2.]
    ],
    'variance': [1., 1.],
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_sizes': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    'max_sizes': [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    'clip': True,
    'name': 'VID2017',
}

faf = build_faf('train', cfg=cfg, num_classes=num_classes)
faf.to(device)

if (flags.resume):
    checkpoint = torch.load(flags.checkpoint)
    faf.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # train from scratch - init weights
    faf.vgg.apply(init_weight)
    faf.extras.apply(init_weight)
    faf.loc.apply(init_weight)
    faf.conf.apply(init_weight)

criterion = MultiFrameBoxLoss(
    3,
    0.5,
    cfg['variance'],
    num_frames,
    num_classes,
    loss_device
)
optimizer = optim.SGD(
    faf.parameters(),
    lr=flags.lr,
    momentum=0.9,
    weight_decay=1e-4
)

def train(epoch):
    print('\nTraining Epoch: {}'.format(epoch))

    faf.train()
    train_loss = 0
    anchor = faf.anchors.to(loss_device)
    batch_count = len(trainLoader)

    for batch_index, (samples, gts) in enumerate(trainLoader):
        samples = samples.to(device)
        for index, gt in enumerate(gts):
            gts[index] = [frame.to(loss_device) for frame in gt]

        optimizer.zero_grad()

        if torch.cuda.device_count() > 1:
            loc, conf = nn.parallel.data_parallel(faf, samples)
        else:
            loc, conf = faf(samples)

        # offload loss to loss_device
        loss_l, loss_c = criterion((loc.to(loss_device), conf.to(loss_device), anchor), gts)
        loss = loss_l + loss_c
        loss = loss.to(device)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('e:{}/{}, b:{}/{}, b_l:{:.2f} = l{:.2f} + c{:.2f}, e_l:{:.2f}'.format(
            epoch + 1,
            flags.end_epoch,
            batch_index + 1,
            batch_count,
            loss.item(),
            loss_l.item(),
            loss_c.item(),
            train_loss / (batch_index + 1)
        ))

        if (batch_index + 1) % snapshot_interval == 0:
            snapshot(epoch, batch_index, train_loss / (batch_index + 1))

def val(epoch):
    print('\nVal')

    with torch.no_grad():
        faf.eval()
        val_loss = 0
        anchor = faf.anchors.to(loss_device)
        batch_count = len(valLoader)

        # perfrom forward
        for batch_index, (samples, gts) in enumerate(valLoader):
            samples = samples.to(device)
            for index, gt in enumerate(gts):
                gts[index] = [frame.to(loss_device) for frame in gt]

            if torch.cuda.device_count() > 1:
                loc, conf = nn.parallel.data_parallel(faf, samples)
            else:
                loc, conf = faf(samples)

            loss_l, loss_c = criterion((loc.to(loss_device), conf.to(loss_device), anchor), gts)
            loss = loss_l + loss_c

            val_loss += loss.item()

            print('e:{}/{}, b:{}/{}, b_l:{:.2f} = l{:.2f} + c{:.2f}, e_l:{:.2f}'.format(
                epoch + 1,
                flags.end_epoch,
                batch_index + 1,
                batch_count,
                loss.item(),
                loss_l.item(),
                loss_c.item(),
                val_loss / (batch_index + 1)
            ))

        # save checkpoint
        global best_loss
        val_loss /= len(valLoader)
        if val_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(best_loss))
            state = {
                'net': faf.state_dict(),
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
