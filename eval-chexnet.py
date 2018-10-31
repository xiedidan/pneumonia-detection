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
from utils.cam import *
from utils.metric import map_iou

# constants & configs
phase = 'val' # train or val, they have gt for eval

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
PNEUMONIA_POSITION = 6

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

evalTransform = transforms.Compose([
    transforms.Resize(size=256, interpolation=Image.BILINEAR),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([imagenet_normalize(crop) for crop in crops]))
])

evalSet = PneumoniaClassificationDataset(
    root=flags.root,
    classMapping=classMapping,
    phase=phase,
    transform=evalTransform,
    num_classes=num_classes
)

evalLoader = torch.utils.data.DataLoader(
    evalSet,
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

# keys = model.state_dict().keys()
# for key in keys:
#    print(key)

def eval():
    print('\nEval')

    mean_aps = []
    # hit, false-positive, miss, false-negative, negative
    ap_counts = [0, 0, 0, 0, 0]

    with torch.no_grad():
        model.eval()

        # perform forward - use gts for eval
        for (samples, gts, ws, hs, ids, origins) in tqdm(evalLoader):
            batch_size, n_crops, c, h, w = samples.size()
            samples = samples.view(-1, c, h, w)

            samples = samples.to(device)

            if torch.cuda.device_count() > 1:
                outputs = nn.parallel.data_parallel(model, samples)
            else:
                outputs = model(samples)

            output_means = outputs.detach().view(batch_size, n_crops, -1).mean(dim=-2)
            max_confs, results = torch.max(output_means, dim=-1)
            pneumonia_confs = output_means[:, PNEUMONIA_POSITION]
            pneumonia_preds = torch.gt(pneumonia_confs, 0.5).to(dtype=torch.long)

            pneumonia_truths = torch.tensor([1 if len(item) > 0 else 0 for item in gts]).to(dtype=torch.long)
            p_results = torch.eq(pneumonia_truths.cpu(), pneumonia_preds.cpu())

            # print(pneumonia_preds)
            # print(pneumonia_truths)
            # print(pneumonia_preds.cpu() - pneumonia_truths.cpu())

            # get CAM - samples are ten-cropped, use original images instead
            origins = origins.to(device)
            cams = chexnet_cam(origins, model, PNEUMONIA_POSITION)

            # get bboxes from cams
            results, resized_cams = export_bboxes(cams, ws, hs)

            # measure metric
            truths = [gt.cpu().numpy() for gt in gts]

            for truth, result, pred in zip(truths, results, pneumonia_preds.cpu().numpy()):
                bboxes, scores = result
                # print(bboxes, truth)
                if pred == 1:
                    mean_ap = map_iou(
                        truth,
                        bboxes,
                        scores
                    )
                else:
                    mean_ap = map_iou(
                        truth,
                        [],
                        []
                    )

                if mean_ap is not None and mean_ap > 0.: # hit
                    ap_counts[0] += 1
                    print('mAP: {}'.format(mean_ap))
                elif mean_ap is not None:
                    if len(truth) == 0: # false-positive
                        ap_counts[1] += 1
                    elif len(bboxes) == 0 or pred == 0: # false-negative
                        ap_counts[3] += 1
                    else: # miss
                        ap_counts[2] += 1
                else: # negative
                    ap_counts[4] += 1

                if mean_ap is not None:
                    mean_aps.append(mean_ap)

            # labels = [CLASS_NAMES[result.item()] for result in results]
            # labels = ['{}\n{}: {:.2f} / {:.2f}'.format(ids[i], label, max_confs[i], pneumonia_confs[i]) for i, label in enumerate(labels)]

            if flags.plot:
                # conver gts to bbox form
                for i, gt in enumerate(gts):
                    bboxes = torch.tensor([to_bbox(p) for p in gt])
                    gts[i] = bboxes
                '''
                for i, result in enumerate(results):
                    bboxes, _ = result
                    bboxes = torch.from_numpy(np.array([to_bbox(p) for p in bboxes]))
                    gts[i] = bboxes
                '''
                plot_detection_batch(resized_cams.cpu(), gts, 2)

        # final score
        mean_aps = np.array(mean_aps)
        mean_ap = mean_aps.mean()
        print('mAP:\t{}\nhit:\t{}\nfp:\t{}\nmiss:\t{}\nfn:\t{}\nneg:\t{}'.format(
            mean_ap,
            ap_counts[0],
            ap_counts[1],
            ap_counts[2],
            ap_counts[3],
            ap_counts[4]
        ))

if __name__ == '__main__':
    eval()