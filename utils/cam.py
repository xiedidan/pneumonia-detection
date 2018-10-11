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
from skimage import measure
from skimage.transform import resize

from densenet import *
from datasets.pneumonia import *
from utils.plot import *
from utils.export import *

import matplotlib.pyplot as plt

FEATURE_THRESHOLD = 0.5

def chexnet_cam(images, model, position):
    feature_layers = model.densenet121.features

    # if torch.cuda.device_count() > 1:
    #    features = nn.parallel.data_parallel(feature_layers, images)
    # else:

    # features.shape = [batch_size, 1024, h, w]
    features = feature_layers(images)
    
    # convert to [batch_size, h, w, 1024]
    features = torch.transpose(features, 1, 3)
    features = torch.transpose(features, 1, 2)

    # class_weights.shape = [1024]
    class_weights = model.densenet121.classifier[0].weight[position, :]
    # feature_maps.shape = [batch_size, h, w, 1024] x [1024, 1] = [batch_size, h, w]
    feature_maps = torch.matmul(features, class_weights)

    feature_maps = torch.sigmoid(feature_maps)
    
    return feature_maps
    
def export_bboxes(cams, ws, hs):
    results = []
    resized_cams = []

    for i, cam in enumerate(cams):
        bboxes = []
        scores = []

        # resize cam to original size
        w = ws[i]
        h = hs[i]
        
        pil_pos = transforms.functional.to_pil_image(torch.unsqueeze(cam, 0).cpu())
        resized_pos = transforms.functional.resize(pil_pos, (h, w))
        resized_pos = transforms.functional.to_tensor(resized_pos).squeeze().cpu().numpy()

        binary_pos = resized_pos[:, :] > FEATURE_THRESHOLD
        # print(resized_pos.max())

        ##plt.imshow(resized_pos)
        # plt.show()

        components = measure.label(binary_pos)

        for region in measure.regionprops(components):
            ymin, xmin, ymax, xmax = region.bbox

            # return bbox in (xmin, ymin, width, height)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            bboxes.append(bbox)

            # conf
            conf = np.mean(resized_pos[ymin:ymax, xmin:xmax])
            scores.append(conf)
        
        # print(bboxes)
        results.append((np.array(bboxes), np.array(scores)))
        resized_cams.append(resized_pos)

    return results, torch.tensor(resized_cams)
