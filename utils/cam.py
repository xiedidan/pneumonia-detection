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

def s(images, model, position):
    feature_layers = model.densenet121.features

    if torch.cuda.device_count() > 1:
        features = nn.parallel.data_parallel(feature_layers, images)
    else:
        features = feature_layers(images)

    features = torch.transpose(features, 1, 3)
    features = torch.transpose(features, 1, 2)

    class_weights = model.densenet121.classifier[0].weight[position, :]
    feature_maps = torch.matmul(features, class_weights)
    
    return feature_maps
