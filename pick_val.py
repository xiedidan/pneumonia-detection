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
parser.add_argument('--val', type=int, default=1000, help='samples picked for val')
parser.add_argument('--eval', type=int, default=0, help='samples picked for evaluation')
parser.add_argument('--random', default=False, help='randomly pick')
flags = parser.parse_args()

def pick_randomly(root, src_path, dest_path, file_list, count):
    # mk output dir
    sample_path = os.path.join(root, dest_path)
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    # pick randomly
    sample_list = random.sample(file_list, count)

    # move
    for filename in sample_list:
        src_file = os.path.join(root, src_path, filename)
        dest_file = os.path.join(root, dest_path, filename)
        shutil.move(src_file, dest_file)

    return sample_list

def pick_last(root, src_path, dest_path, file_list, count):
    # mk output dir
    sample_path = os.path.join(root, dest_path)
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    # pick last
    sample_list = file_list[:count]

    # move
    for filename in sample_list:
        src_file = os.path.join(root, src_path, filename)
        dest_file = os.path.join(root, dest_path, filename)
        shutil.move(src_file, dest_file)

    return sample_list

if __name__ == '__main__':
    # pick val first
    train_path = os.path.join(flags.root, 'train/')
    train_list = os.listdir(train_path)

    if flags.random:
        val_list = pick_randomly(
            flags.root,
            'train/',
            'val/',
            train_list,
            flags.val
        )
    else:
        val_list = pick_last(
            flags.root,
            'train/',
            'val/',
            train_list,
            flags.val
        )

    # pick eval list
    train_list = os.listdir(train_path)

    if flags.random:
        eval_list = pick_randomly(
            flags.root,
            'train/',
            'eval/',
            train_list,
            flags.eval
        )
    else:
        eval_list = pick_last(
            flags.root,
            'train/',
            'eval/',
            train_list,
            flags.eval
        )
