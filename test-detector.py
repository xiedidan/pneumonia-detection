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

from datasets.pneumonia import *
from datasets.transforms import *
from datasets.config import *
from utils.plot import *
from utils.augmentations import *
from utils.export import export_detection_csv
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from densenet import *
from utils.filter import *

# constants & configs
pretrained = False

# spawned workers on windows take too much gmem
number_workers = 8
if sys.platform == 'win32':
    number_workers = 1

mean = [125]
# mean = [0.49043187350911405]
# std = [0.22854086980778032]

classMapping = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}

# chexnet configs
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
PNEUMONIA_POSITION = 6
# TODO : check this
CHEX_CP = './pretrained/???'
CHEX_THRESHOLD = 0.5

# argparser
parser = argparse.ArgumentParser(description='Pneumonia Detector Testing')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot result')
parser.add_argument('--classification_file', default='./classification.csv', type=str, help='Classifier results')
parser.add_argument('--save_file', default='./detection.csv', type=str, help='Filename to save results')
parser.add_argument('--dump_file', default='./detection.pth', type=str, help='Filename to dump results, for next stage')
flags = parser.parse_args()

device = torch.device(flags.device)
cfg = rsna

# data
testSet = PneumoniaDetectionDataset(
    root=flags.root,
    phase='test',
    transform=SSDTransformation(cfg['min_dim'], (mean, mean, mean)),
    classMapping=classMapping,
    num_classes=cfg['num_classes'],
    classification_path=flags.classification_file
)
testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=number_workers,
    collate_fn=detectionCollate,
)

# chexnet model
chexnet = DenseNet121(CHEX_N_CLASSES)
chexnet.to(device)

chex_cp = torch.load(CHEX_CP)
chexnet.transfer(chex_cp['state_dict'])

# model
ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'], device)
net = ssd_net

print('Loading {}...'.format(flags.checkpoint))
checkpoint = torch.load(flags.checkpoint)
ssd_net.load_state_dict(checkpoint['net'])

net.to(device)

def test():
    print('\nTest')

    with torch.no_grad():
        chexnet.eval()
        net.eval()

        ssd_transformations = Compose([
            Resize(cfg['min_dim']),
            SubtractMeans((mean, mean, mean)
        ])

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

        for (images, gts, ws, hs, ids) in tqdm(testLoader):
            images = images.to(device)

            # 3 stage forward pass

            # chexnet for global check
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]

            chex_pils = transforms.functional.to_pil(images)
            chex_pils = transforms.functional.resize(chex_pils, (256, 256))
            chex_pils = transforms.functional.ten_crop(224)
            chex_images = torch.stack([transforms.functional.to_tensor(pil) for pil in chex_piles])
            chex_images = transforms.functional.normalize(chex_pils, imagenet_mean, imagenet_std)
            
            batch_size, n_crops, cc, ch, cw = chex_images.size()
            chex_images = chex_images.view(-1, cc, ch, cw)

            chex_outputs = chexnet(chex_images)
            chex_means = chex_outputs.detach().view(batch_size, n_crops, -1).mean(dim=-2)
            chex_confs = chex_means[:, PNEUMONIA_POSITION]
            chex_preds = torch.gt(chex_confs, CHEX_THRESHOLD).to(dtype=torch.long)

            # ssd detector
            '''
            if torch.cuda.device_count() > 1:
                out = nn.parallel.data_parallel(net, images)
            else:
                out = net(images)
            '''

            ssd_images = ssd_transformations(images)
            out = net(ssd_images)

            # detections.shape = [batch_size, num_classes, num_detections, 5]
            # detection = [score, xmin, ymin, xmax, ymax]
            detections = out.detach()

            if flags.plot:
                plot_detection(ssd_images.cpu().to(dtype=torch.uint8), detections, ws, hs, ids, 2)

            # convert bboxes to absolute coords
            for sample_index in range(detections.size(0)):
                w = ws[sample_index]
                h = hs[sample_index]
                all_ids.append(ids[sample_index])

                if chex_preds[sample_index] == 0:
                    continue # jump to next sample

                for class_index in range(1, detections.size(1)):
                    dets = detections[sample_index, class_index, :]

                    # only keep score > 0.
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)

                    # filter
                    dets = check_bbox(dets)

                    if dets.dim() == 0:
                        continue # jump to next class
                    
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()

                    # TODO : got bboxes and scores, now verify

                    cls_dets = np.hstack((
                        boxes.cpu().numpy(),
                        scores[:, np.newaxis]
                    )).astype(np.float32, copy=False)

                    all_boxes[class_index][i] = cls_dets
                    i += 1

        # no need for this anymore
        # serialize
        '''
        results = {
            'bboxes': all_boxes[1],
            'ids': all_ids
        }
        with open(flags.dump_file, 'wb') as dump:
            pickle.dump(results, dump)
        '''
        
        # export bboxes
        with open(flags.save_file, 'a') as csv:
            export_detection_csv(csv, all_ids, all_boxes[1])

            df = pd.read_csv(flags.classification_file)
            other_ids = []
            dummy_dets = []

            for index, line in df.iterrows():
                if line['patientId'] not in all_ids:
                    other_ids.append(line['patientId'])
                    dummy_dets.append([])

            export_detection_csv(csv, other_ids, dummy_dets)

# ok, main loop
if __name__ == '__main__':
    test()
