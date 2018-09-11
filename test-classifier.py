# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import pickle
import argparse
import itertools
from tqdm import tqdm

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
from utils.export import *

# constants & configs
num_classes = 3
pretrained = False

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 2

size = 512
mean = [0.49043187350911405]
std = [0.22854086980778032]
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Classifier Testing')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--plot', action='store_true', help='plot result')
parser.add_argument('--save_file', default='./classification.csv', type=str, help='Filename to save results')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
flags = parser.parse_args()

device = torch.device(flags.device)

testTransform = transforms.Compose([
    transforms.Resize(size=size, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

testSet = PneumoniaClassificationDataset(
    root=flags.root,
    classMapping=classMapping,
    phase='test',
    transform=testTransform,
    num_classes=num_classes
)

testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=classificationCollate,
)

# model
resnet = resnet152(pretrained, num_classes=num_classes)
resnet.to(device)

checkpoint = torch.load(flags.checkpoint)
resnet.load_state_dict(checkpoint['net'])

def test():
    print('\nTest')

    with torch.no_grad():
        resnet.eval()

        # write csv header
        if not flags.plot:
            with open(flags.save_file, 'w') as csv:
                csv.write('patientId,class,classNo,confidence\n')

        # perform forward
        for (samples, gts, ws, hs, ids) in tqdm(testLoader):
            samples = samples.to(device)

            if torch.cuda.device_count() > 1:
                outputs = nn.parallel.data_parallel(resnet, samples)
            else:
                outputs = resnet(samples)

            confs = nn.functional.softmax(outputs)
            max_confs, results = torch.max(confs, dim=outputs.dim() - 1)

            labels = [get_class_name(classMapping, result.item()) for result in results]

            if flags.plot:
                plot_classification(samples.cpu(), labels, 2)
            else:
                # export to csv - patientId, class, classNo, confidence
                with open(flags.save_file, 'a') as csv:
                    export_classification_csv(csv, ids, labels, results, max_confs)

if __name__ == '__main__':
    test()
