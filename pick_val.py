import sys
import os
import shutil
import argparse
import pickle
import random

from tqdm import tqdm

# argparser
parser = argparse.ArgumentParser(description='Pneumonia detection dataset split tool')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--ratio', default=0.1, help='val:train sample ratio')
flags = parser.parse_args()

def pick_val(root, ratio):
    val_path = os.path.join(root, 'val/')
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    train_path = os.path.join(root, 'train/')
    train_list = os.listdir(train_path)

    val_list = random.sample(train_list, int(len(train_list) * ratio))

    for filename in val_list:
        src_path = os.path.join(train_path, filename)
        dest_path = os.path.join(root, 'val/', filename)
        
        shutil.move(src_path, dest_path)

if __name__ == '__main__':
    pick_val(flags.root, flags.ratio)
