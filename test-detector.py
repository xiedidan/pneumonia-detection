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
from utils.export import export_detection_csv
from layers.modules import MultiBoxLoss
from ssd import build_ssd

# constants & configs
pretrained = False

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 1

mean = [0.49043187350911405]
std = [0.22854086980778032]

# be ware - this is different from mapping in classification
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 0,
    'Lung Opacity': 1
}

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Detector Testing')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot result')
parser.add_argument('--classfication_file', default='./classification.csv', type=str, help='Classifier results')
parser.add_argument('--save_file', default='./detection.csv', type=str, help='Filename to save results')
flags = parser.parse_args()

device = torch.device(flags.device)
cfg = rsna

# data
# TODO : transform without gt
testSet = PneumoniaDetectionDataset(
    root=flags.root,
    phase='test',
    transform=SSDTransformation(cfg['min_dim'], (mean, mean, mean)),
    classMapping=classMapping,
    num_classes=cfg['num_classes']
)
testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=detectionCollate,
)

# model
# TODO : update parameters
ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'], device)
net = ssd_net

print('Loading {}...'.format(flags.checkpoint))
checkpoint = torch.load(flags.checkpoint)
ssd_net.load_state_dict(checkpoint['net'])

net.to(device)

def test():
    print('\nTest')

    with torch.no_grad():
        net.eval()

        # write csv header
        if not flags.plot:
            with open(flags.save_file, 'w') as csv:
                csv.write('patientId,PredictionString\n')

        sample_count = len(testSet)
        batch_count = len(testLoader)

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(sample_count)]
                    for _ in range(cfg['num_classes'])]
        all_ids = []

        i = 0

        for batch_index, (images, gts, ws, hs, ids) in enumerate(testLoader):
            images = images.to(device)

            # forward
            if torch.cuda.device_count() > 1:
                out = nn.parallel.data_parallel(net, images)
            else:
                out = net(images)

            # detections.shape = [batch_size, num_classes, num_detections, 5]
            # detection = [score, xmin, ymin, xmax, ymax]
            detections = out.detach()

            # convert bboxes to absolute coords
            for sample_index in range(detections.size(0)):
                w = ws[sample_index]
                h = hs[sample_index]
                all_ids.append(ids[sample_index])

                for class_index in range(1, detections.size(1)):
                    dets = detections[sample_index, class_index, :]

                    # only keep score > 0.
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.dim() == 0:
                        continue # jump to next class
                    
                    boxes = dets[: 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()

                    cls_dets = np.hstack((
                        boxes.cpu().numpy(),
                        scores[:, np.newaxis]
                    )).astype(np.float32, copy=False)

                    all_boxes[class_index][i] = cls_dets

            # export bboxes
            with open(flags.save_file, 'a') as csv:
                export_detection_csv(csv, all_ids, all_boxes[1, :])

            # TODO : plot

# ok, main loop
if __name__ == '__main__':
    test()
