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
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

pretrained = False

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 4

score_threshold = 0.6
size = 512
mean = [0.49043187350911405]
std = [0.22854086980778032]

# helper functions
def xavier(param):
    init.xavier_uniform_(param)

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Verifier Testing')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--detection_file', default='./detection.pth', help='detection dump file path')
parser.add_argument('--save_file', default='./verification.csv', type=str, help='Filename to save results')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot images')
flags = parser.parse_args()

device = torch.device(flags.device)

# data loader
testTransform = transforms.Compose([
    transforms.Resize(size=(size, size), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

testSet = PneumoniaVerificationDataset(
    root=flags.root,
    classMapping=classMapping,
    phase='test',
    transform=testTransform,
    num_classes=num_classes,
    detection_path=flags.detection_file
)
testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=verificationCollate,
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
                csv.write('patientId,PredictionString\n')

        all_target_ids = []
        targets = {}

        # perfrom forward
        for (samples, gts, ws, hs, ids) in tqdm(testLoader):
            samples = samples.to(device)
            gts = gts.to(device=device, dtype=torch.long)

            if torch.cuda.device_count() > 1:
                outputs = nn.parallel.data_parallel(resnet, samples)
            else:
                outputs = resnet(samples)

            # deal with results
            confs = nn.functional.softmax(outputs.detach(), dim=-1)
            max_confs, results = torch.max(confs, dim=-1)

            # plot
            if flags.plot:
                labels = ['{}\nr: {}, s: {:.2f}'.format(ids[i], result, max_confs[i]) for i, result in enumerate(results)]
                plot_classification(samples.cpu(), labels, 2)

            mask = results.eq(2)
            scores = torch.masked_select(max_confs, mask)
            # target_ids = torch.masked_select(ids, mask)
            target_ids = list(itertools.compress(ids, mask))

            mask = mask.expand(4, confs.size(0)).t()
            bboxes = torch.masked_select(gts, mask).view(-1, 4)

            if bboxes.dim() == 0:
                continue # go to next batch

            rows = np.hstack((
                bboxes.cpu().numpy(),
                scores[:, np.newaxis]
            )).astype(np.float32, copy=False)

            # collect results
            all_target_ids = list(set(all_target_ids + target_ids)) # unique target ids

            for i, row in enumerate(rows):
                if target_ids[i] in targets.keys():
                    targets[target_ids[i]].append(row)
                else:
                    targets[target_ids[i]] = [ row ]

        # export
        with open(flags.save_file, 'a') as csv:
            # collect non-targets
            all_ids = [filename.split('.')[0] for filename in testSet.image_files]

            for patientId in all_ids:
                if patientId not in all_target_ids:
                    targets[patientId] = []

            export_verification_csv(csv, targets, score_threshold)

# ok, main loop
if __name__ == '__main__':
    test()
