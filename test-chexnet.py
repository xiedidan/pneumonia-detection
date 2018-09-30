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

from densenet import *
from datasets.pneumonia import *
from utils.plot import *
from utils.export import *

# constants & configs

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 2

# data configs
num_classes = 3
size = 224
mean = [0.49043187350911405]
std = [0.22854086980778032]
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# network configs
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

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

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
imagenet_normalize = transforms.Normalize(imagenet_mean, imagenet_std)

testTransform = transforms.Compose([
    transforms.Resize(size=256, interpolation=Image.BILINEAR),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([imagenet_normalize(crop) for crop in crops]))
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
    pin_memory=True,
    collate_fn=classificationCollate
)

# model
model = DenseNet121(N_CLASSES)
model.to(device)

checkpoint = torch.load(flags.checkpoint)
model.transfer(checkpoint['state_dict'])

def test():
    print('\nTest')

    with torch.no_grad():
        model.eval()

        # perform forward - use gts for original pictures
        for (samples, gts, ws, hs, ids) in tqdm(testLoader):
            batch_size, n_crops, c, h, w = samples.size()
            samples = samples.view(-1, c, h, w)

            samples = samples.to(device)

            if torch.cuda.device_count() > 1:
                outputs = nn.parallel.data_parallel(model, samples)
            else:
                outputs = model(samples)

            output_means = outputs.detach().view(batch_size, n_crops, -1).mean(dim=-2)
            max_confs, results = torch.max(output_means, dim=-1)
            pneumonia_confs = output_means[:, 6]

            # get CAM
            gts = gts.to(device)
            cams = cam(gts, model)

            labels = [CLASS_NAMES[result.item()] for result in results]
            labels = ['{}\n{}: {:.2f} / {:.2f}'.format(ids[i], label, max_confs[i], pneumonia_confs[i]) for i, label in enumerate(labels)]

            if flags.plot:
                plot_classification(torch.from_numpy(cams.cpu().numpy()[:, np.newaxis, :, :]), labels, 2)

def cam(images, model):
    feature_layers = model.densenet121.features

    if torch.cuda.device_count() > 1:
        features = nn.parallel.data_parallel(feature_layers, images)
    else:
        features = feature_layers(images)

    features = torch.transpose(features, 1, 3)
    features = torch.transpose(features, 1, 2)

    class_weights = model.densenet121.classifier[0].weight[6, :]
    feature_maps = torch.matmul(features, class_weights)
    
    return feature_maps

if __name__ == '__main__':
    test()
