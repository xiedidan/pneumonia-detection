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

num_classes = 2
size = [512, 512]

# argparser
parser = argparse.ArgumentParser(description='Pneumonia detection dataset toolkit')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--plot', action='store_true', help='plot batch')
flags = parser.parse_args()

if __name__ == '__main__':
    transform = Compose([
        Resize(size=size),
        ToTensor()
    ])

    target_transform = ComposeTarget([
        ToBbox(),
        Percentage(size=size)
    ])

    # data
    trainSet = PneumoniaDataset(
        root=flags.root,
        phase='train',
        transform=transform,
        target_transform=target_transform,
        num_classes=num_classes
    )

    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=collate,
    )

    for samples in tqdm(trainLoader):
        images, gts, ws, hs, ids = samples
        if flags.plot:
            plot_batch(images, gts, 2)
