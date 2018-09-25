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

from datasets.pneumonia import *
from datasets.transforms import *
from datasets.config import *
from utils.plot import *
from utils.augmentations import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd

# constants & configs
snapshot_interval = 50
pretrained = False

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 1

mean = [125]
# mean = [0.49043187350911405]
# std = [0.22854086980778032]

classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# variables
start_epoch = 0
curr_batch = 0
best_loss = float('inf')

# helper
def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def snapshot(epoch, batch, model, loss):
    print('Taking snapshot, loss: {}'.format(loss))

    state = {
        'net': model.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'batch': batch
    }
    
    if not os.path.isdir('snapshot'):
        os.mkdir('snapshot')

    torch.save(state, './snapshot/e_{:0>6}_b_{:0>6}_loss_{:.5f}.pth'.format(
        epoch,
        batch,
        loss
    ))

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Detector Training')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=float, help='epcoh to stop training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot result')
flags = parser.parse_args()

device = torch.device(flags.device)
cfg = rsna

# data
'''
trainTransform = Compose([
    SSDAugmentation(cfg['min_dim'], [mean, mean, mean])
    # Resize(size=cfg['min_dim']),
    # RandomResizedCrop(size=cfg['min_dim'], p=0.7, scale=(0.9, 1.), ratio=(0.9, 1/0.9)),
    # Percentage(size=size),
    # ToTensor()
])
'''

trainSet = PneumoniaDetectionDataset(
    root=flags.root,
    phase='train',
    transform=SSDAugmentation(cfg['min_dim'], (mean, mean, mean)),
    classMapping=classMapping,
    num_classes=cfg['num_classes']
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=flags.batch_size,
    shuffle=True,
    num_workers=number_workers,
    collate_fn=detectionCollate,
)

'''
valTransform = Compose([
    Resize(size=cfg['min_dim']),
    Percentage(size=size),
    ToTensor(),
])
'''

valSet = PneumoniaDetectionDataset(
    root=flags.root,
    phase='val',
    transform=SSDTransformation(cfg['min_dim'], (mean, mean, mean)),
    classMapping=classMapping,
    num_classes=cfg['num_classes']
)
valLoader = torch.utils.data.DataLoader(
    valSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=detectionCollate,
)

# model
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], device)
net = ssd_net

if flags.resume:
    print('Resuming training, loading {}...'.format(flags.checkpoint))
    checkpoint = torch.load(flags.checkpoint)

    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    curr_batch = checkpoint['batch']

    ssd_net.load_state_dict(checkpoint['net'])
else:
    print('Loading base network...')
    vgg_weights = torch.load(flags.checkpoint)
    ssd_net.vgg.load_state_dict(vgg_weights)

net.to(device)

if not flags.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(
    net.parameters(),
    lr=flags.lr,
    momentum=0.9,
    weight_decay=1e-4
)
criterion = MultiBoxLoss(
    cfg['num_classes'],
    0.5,
    True,
    0,
    True,
    3,
    0.5,
    False,
    device)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.1,
    patience=10,
    verbose=True
)

def train(epoch):
    print('\nTraining Epoch: {}'.format(epoch))

    net.train()

    train_loss = 0
    loc_loss = 0
    conf_loss = 0
    batch_count = len(trainLoader)

    for batch_index, (images, gts, ws, hs, ids) in enumerate(trainLoader):
        if flags.plot:
            plot_detection_batch(images, gts, 2)

        images = images.to(device)

        gts = [gt.to(device=device, dtype=torch.float32) for gt in gts]

        # forward
        '''
        if torch.cuda.device_count() > 1  and flags.device != 'cpu':
            out = nn.parallel.data_parallel(net, images)
        else:
            out = net(images)
        '''
        out = net(images)

        # backward
        optimizer.zero_grad()

        '''
        if torch.cuda.device_count() > 1 and flags.device != 'cpu':
            loss_l, loss_c  = nn.parallel.data_parallel(criterion, (out, gts))
            loss_l = loss_l.squeeze()
            loss_c = loss_c.squeeze()
        else:
            loss_l, loss_c = criterion(out, gts)
        '''
        loss_l, loss_c = criterion(out, gts)

        loss = loss_l + loss_c
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        loc_loss = loss_l.item()
        conf_loss = loss_c.item()

        print('e:{}/{}, b:{}/{}, b_l:{:.2f} = l_l:{:.2f} + c_l:{:.2f}, e_l:{:.2f}'.format(
            epoch,
            flags.end_epoch - 1,
            batch_index,
            batch_count - 1,
            loss.item(),
            loc_loss,
            conf_loss,
            train_loss / (batch_index + 1)
        ))

        # take snapshot
        iteration = batch_count * epoch + batch_index + 1
        if iteration % snapshot_interval == 0:
            snapshot(epoch, batch_index, net, best_loss)

        if epoch == start_epoch:
            if curr_batch + batch_index + 1 == batch_count:
                break

def val(epoch):
    print('\nVal')

    with torch.no_grad():
        net.eval()

        val_loss = 0
        loc_loss = 0
        conf_loss = 0
        batch_count = len(valLoader)

        # just perfrom forward on training net
        for batch_index, (images, gts, ws, hs, ids) in enumerate(valLoader):
            images = images.to(device)

            gts = [gt.to(device=device, dtype=torch.float32) for gt in gts]
            
            # forward
            '''
            if torch.cuda.device_count() > 1 and flags.device != 'cpu':
                out = nn.parallel.data_parallel(net, images)
            else:
                out = net(images)
            '''
            out = net(images)

            '''
            if torch.cuda.device_count() > 1 and flags.device != 'cpu':
                loss_l, loss_c  = nn.parallel.data_parallel(criterion, (out, gts))
                loss_l = loss_l.squeeze()
                loss_c = loss_c.squeeze()
            else:
                loss_l, loss_c = criterion(out, gts)
            '''
            loss_l, loss_c = criterion(out, gts)

            loss = loss_l + loss_c

            val_loss += loss.item()
            loc_loss = loss_l.item()
            conf_loss = loss_c.item()

            print('e:{}/{}, b:{}/{}, b_l:{:.2f} = l_l:{:.2f} + c_l:{:.2f}, e_l:{:.2f}'.format(
                epoch,
                flags.end_epoch - 1,
                batch_index,
                batch_count - 1,
                loss.item(),
                loc_loss,
                conf_loss,
                val_loss / (batch_index + 1)
            ))

        global best_loss
        val_loss /= batch_count

        # update lr
        scheduler.step(val_loss)

        # save checkpoint
        if val_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(val_loss))

            state = {
                'net': net.state_dict(),
                'loss': val_loss,
                'epoch': epoch,
                'batch': 0
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
