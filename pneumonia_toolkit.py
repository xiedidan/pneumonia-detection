import sys
import os
import argparse
import pickle

import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from datasets.pneumonia import *
from utils.plot import *
from utils.statistics import *

size = [512, 512]
mean = [0.49043187350911405]
std = [0.22854086980778032]
classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# argparser
parser = argparse.ArgumentParser(description='Pneumonia detection dataset toolkit')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--stage', default='classification', help='stage in [classification / detection / review]')
parser.add_argument('--plot', action='store_true', help='plot batch')
parser.add_argument('--statistics', action='store_true', help='dataset statistics')
flags = parser.parse_args()

if __name__ == '__main__':
    if flags.stage == 'detection':
        transform = Compose([
            Resize(size=size),
            ToTensor(),
            Normalize(mean, std)
        ])

        target_transform = ComposeTarget([
            ToBbox(),
            Percentage(size=size)
        ])

        # data
        trainSet = PneumoniaDetectionDataset(
            root=flags.root,
            phase='train',
            transform=transform,
            target_transform=target_transform,
            num_classes=2
        )

        trainLoader = torch.utils.data.DataLoader(
            trainSet,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            collate_fn=detectionCollate,
        )

        clsCounter = ClassCounter([0, 1])
        boxCounter = BoxCounter()
        detCounter = DetectionCounter()
        counter = MeanAndStd()

        for samples in tqdm(trainLoader):
            images, gts, ws, hs, ids = samples
            locs, confs = gts

            if flags.statistics:
                for i, conf in enumerate(confs):
                    if conf[0] == 0:
                        clsCounter.inc(0)
                        boxCounter.inc(0)
                    else:
                        clsCounter.inc(1)
                        boxCounter.inc(conf.shape[0])
                    
                    detCounter.inc(locs[i])
            
            counter.inc(images)

            if flags.plot:
                plot_batch(images, gts, 4)

        if flags.statistics:
            print(clsCounter.get_result())
            print(boxCounter.get_result())
            print(counter.get_result())

            xs, ys, ratios, levels = detCounter.get_result()
            plot_scatter(xs, ys)
            plot_scatter(ratios, levels)
    elif flags.stage == 'classification':
        transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # data
        trainSet = PneumoniaClassificationDataset(
            root=flags.root,
            classMapping=classMapping,
            phase='train',
            transform=transform,
            num_classes=3
        )

        trainLoader = torch.utils.data.DataLoader(
            trainSet,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            collate_fn=classificationCollate,
        )

        for i, batch in enumerate(trainLoader):
            pass

    else: # stage = 'review'
        pass
