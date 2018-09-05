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

num_frames = 5
num_classes = 31
class_filename = 'class.mapping'
number_workers = 8

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

# argparser
parser = argparse.ArgumentParser(description='FaF Testing')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/ILSVRC/', help='dataset root path')
parser.add_argument('--save_folder', default='test/', type=str, help='Dir to save results')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot result bbox')
flags = parser.parse_args()

print('Got flags: {}'.format(flags))

if not os.path.exists(os.path.join(flags.root, 'dump/')):
    os.mkdir(os.path.join(flags.root, 'dump/'))

device = torch.device(flags.device)

# data loader
size = [300, 300]
transform = Compose([
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

testSet = VidDataset(
    root=flags.root,
    phase='test',
    transform=transform,
    classDict=classMapping,
    num_classes=num_classes
)
testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=collate
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

faf = build_faf('test', cfg=cfg, num_classes=num_classes)
faf.to(device)

# load from checkpoint
checkpoint = torch.load(flags.checkpoint)
faf.load_state_dict(checkpoint['net'])

def test():
    print('\Test')

    with torch.no_grad():
        faf.eval()

        anchor = faf.anchors.to(device)
        batch_count = len(testLoader)

        # perfrom forward
        for batch_index, (samples, gts) in enumerate(testLoader):
            samples = samples.to(device)

            if torch.cuda.device_count() > 1:
                loc, conf = nn.parallel.data_parallel(faf, samples)
            else:
                loc, conf = faf(samples)

            # TODO : get bboxes

            # TODO : get labels

            if flags.plot:
                plot_result_batch((samples.numpy(), (bboxes, labels)))
            
            # TODO : save results

if __name__ == '__main__':
    test()
