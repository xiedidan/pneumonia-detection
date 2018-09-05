import sys
import os
import argparse
import pickle

from datasets.vid import *
from utils.plot import *

import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

# config
class_filename = 'class.mapping'
sample_prefix = 'sample.{}.dump'
gt_prefix = 'gt.{}.dump'

packs=[
    'ILSVRC2015_VID_train_0000',
    'ILSVRC2015_VID_train_0001',
    'ILSVRC2015_VID_train_0002',
    'ILSVRC2015_VID_train_0003',
    'ILSVRC2017_VID_train_0000'
]

size = [300, 300]
transform = Compose([
    # Resize(size=size),
    RandomResizedCrop(size=size, p=1., scale=(0.5, 1.), ratio=(0.1, 10)),
    RandomSaltAndPepper(size=size),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    RandomHorizontalFlip(size=size, p=0.99),
    Percentage(size=size),
    ToTensor()
])

# helper
def collate(batch):
    # batch = [(image, gt), (image, gt), ...]
    images = []
    gts = []
    for i, sample in enumerate(batch):
        image, gt, w, h = sample
        
        images.append(image)
        gts.append(gt)
        
    images = torch.stack(images, 0)
        
    return images, gts

def store_class_mapping(root, class_path):
    num_classes, classMapping = create_class_mapping(os.path.join(root, 'Annotations/VID/val/'))

    data = {'num_classes': num_classes, 'classMapping': classMapping}
    with open(class_path, 'wb') as file:
        pickle.dump(data, file)

    return num_classes, classMapping

def load_class_mapping(class_path):
    with open(class_path, 'rb') as file:
        data = pickle.load(file)
        num_classes, classMapping = data['num_classes'], data['classMapping']

    return num_classes, classMapping

def serialize(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def deserialize(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data

# argparser
parser = argparse.ArgumentParser(description='FaF ImageNet VID dataset toolkit')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/ILSVRC/', help='dataset root path')
parser.add_argument('--create-mapping', action='store_true', help='force to create new mapping')
parser.add_argument('--snapshot', action='store_true', help='create dataset snapshot')
parser.add_argument('--plot', action='store_true', help='plot batch')
flags = parser.parse_args()

if __name__ == '__main__':
    print('\nProcessing data...')

    if not os.path.exists(os.path.join(flags.root, 'dump/')):
        os.mkdir(os.path.join(flags.root, 'dump/'))
    
    # class mapping
    class_path = os.path.join(flags.root, 'dump/', class_filename)

    if (file_exists(class_path) == True) and (not flags.create_mapping):
        num_classes, classMapping = load_class_mapping(class_path)
    else:
        num_classes, classMapping = store_class_mapping(flags.root, class_path)

    print('num_classes: {}\nclassMapping: {}'.format(num_classes, classMapping))

    if flags.snapshot:
        # clear exited snapshots
        sample_train_filename = os.path.join(flags.root, 'dump/', sample_prefix).format('train')
        if os.path.exists(sample_train_filename):
            os.remove(sample_train_filename)

        sample_val_filename = os.path.join(flags.root, 'dump/', sample_prefix).format('val')
        if os.path.exists(sample_val_filename):
            os.remove(sample_val_filename)

    # data
    trainSet = VidDataset(
        root=flags.root,
        packs=packs,
        phase='train',
        transform=transform,
        classDict=classMapping,
        num_classes=num_classes
    )

    valSet = VidDataset(
        root=flags.root,
        packs=packs,
        phase='val',
        transform=transform,
        classDict=classMapping,
        num_classes=num_classes
    )

    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
    )

    # snapshot
    if flags.snapshot:
        print('Dumping snapshots...')
        train_samples = trainSet.get_sample()
        serialize(os.path.join(flags.root, 'dump/', sample_prefix).format('train'), train_samples)
        val_samples = valSet.get_sample()
        serialize(os.path.join(flags.root, 'dump/', sample_prefix).format('val'), val_samples)

    # plot
    if flags.plot:
        print('Plotting batches...')
        for samples, gts in tqdm(trainLoader):
            plot_batch((samples, gts))
