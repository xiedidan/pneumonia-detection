import torch
from torch.utils.data import *
import torchvision.transforms as transforms

from PIL import Image
import sys
import os
import pickle
import random
import math
import xml.etree.ElementTree as ET
import multiprocessing as mp
from multiprocessing.dummy import Pool
import numpy as np

from tqdm import tqdm

# fix for 'RuntimeError: received 0 items of ancdata' problem
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

dict_file = './words.txt'

# helper
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def overlap_numpy(box_a, box_b):
    # compute A ∩ B / A
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))
    
    return inter / area_a

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def file_exists(path):
    try:
        with open(path) as f:
            return True
    except IOError:
        return False

def load_wordnet_dict(path):
    with open(path) as file:
        wordnet_dict = {}

        for line in file.readlines():
            line.strip("\n")
            arr = line.split("\t")
            wordnet_dict[arr[0]] = arr[1]

        return wordnet_dict

# class mapping should be saved since it maybe unstable
# we have a background class by default
def create_class_mapping(val_annotation_path='./ILSVRC/Annotations/VID/val/'):
    print('\ncreating class mapping from: {}'.format(val_annotation_path))
    seqs = os.listdir(val_annotation_path)
    
    num_classes = 1
    classDict = {'background': 0}

    for seq_name in tqdm(seqs):
        seq_path = os.path.join(val_annotation_path, seq_name)
        frames = os.listdir(seq_path)
        frame_paths = [os.path.join(seq_path, frame) for frame in frames]

        for xml_path in frame_paths:
            tree = ET.ElementTree(file=xml_path)
            root = tree.getroot()

            for obj in root.iterfind('object'):
                # create wnid to classNo mapping
                className = obj.find('name').text
                if classDict.get(className) is None:
                    classDict[className] = num_classes
                    num_classes += 1
        
    return num_classes, classDict

# data augmentation
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, images, gts, w, h):
        brightness_factor = random.uniform(max(0., 1. - self.brightness), 1. + self.brightness)
        contrast_factor = random.uniform(max(0., 1. - self.contrast), 1. + self.contrast)
        saturation_factor = random.uniform(max(0., 1. - self.saturation), 1. + self.saturation)
        hue_factor = random.uniform(-1. * self.hue, self.hue)

        images = [transforms.functional.adjust_brightness(image, brightness_factor) for image in images]
        images = [transforms.functional.adjust_contrast(image, contrast_factor) for image in images]
        images = [transforms.functional.adjust_saturation(image, saturation_factor) for image in images]
        images = [transforms.functional.adjust_hue(image, hue_factor) for image in images]

        return images, gts, w, h

class RandomHorizontalFlip(object):
    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, images, gts, w, h):
        if random.uniform(0., 1.) < self.p:
            # filp both images and gts
            images = [transforms.functional.hflip(image) for image in images]

            for i, frame in enumerate(gts):
                new_frame = []

                for j, bbox in enumerate(frame):
                    new_bbox = [
                        self.size[0] - bbox[2],
                        bbox[1],
                        self.size[0] - bbox[0],
                        bbox[3],
                        bbox[4]
                    ]

                    new_frame.append(new_bbox)

                new_frame = np.array(new_frame, dtype=np.float32)
                gts[i] = new_frame

        return images, gts, w, h

class RandomResizedCrop(object):
    def __init__(self, size, p=0.5, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2, threshold=0.5):
        self.size = size
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.threshold = threshold

    @staticmethod
    def get_params(w, h, scale, ratio):
        for attempt in range(10):
            scale_factor = random.uniform(*scale)
            ratio_factor = random.uniform(*ratio)
            
            target_area = w * h * scale_factor
            new_w = int(round(math.sqrt(target_area * ratio_factor)))
            new_h = int(round(math.sqrt(target_area / ratio_factor)))

            if random.random() < 0.5:
                new_w, new_h = new_h, new_w

            if new_w < w and new_h < h:
                i = random.randint(0, h - new_h)
                j = random.randint(0, w - new_w)

                return i, j, h, w

        # fallback
        new_w = min(w, h)
        i = (h - new_w) // 2
        j = (w - new_w) // 2

        return i, j, new_w, new_w

    def __call__(self, images, gts, w, h):
        if random.uniform(0., 1.) < self.p:
            i, j, new_h, new_w = self.get_params(w, h, self.scale, self.ratio)
            images = [transforms.functional.resized_crop(image, i, j, new_h, new_w, self.size, self.interpolation) for image in images]

            # gt transform
            offset = np.array([j, i, j, i, 0])
            scale = np.array([self.size[0] / new_w, self.size[1] / new_h, self.size[0] / new_w, self.size[1] / new_h, 1])

            new_fullframe = np.array([0, 0, self.size[0], self.size[1]])

            for i, frame in enumerate(gts):
                new_frame = (frame - offset) * scale

                # remove bbox out of the frame
                inside_mask = overlap_numpy(new_frame[:, :4], new_fullframe) - self.threshold
                inside_mask = inside_mask > 0.

                cleared_frame = np.compress(inside_mask, new_frame, axis=0)

                if len(cleared_frame) == 0:
                    cleared_frame = np.array([[0., 0., self.size[0], self.size[1], 0.]], dtype=float)

                # clip into the frame
                cleared_frame = cleared_frame.reshape(5, -1)
                np.clip(cleared_frame[0], 0., self.size[0], out=cleared_frame[0])
                np.clip(cleared_frame[2], 0., self.size[0], out=cleared_frame[2])
                np.clip(cleared_frame[1], 0., self.size[1], out=cleared_frame[1])
                np.clip(cleared_frame[3], 0., self.size[1], out=cleared_frame[3])
                cleared_frame = cleared_frame.reshape(-1, 5)

                gts[i] = cleared_frame
        else:
            # fall back to simple resize
            images = transforms.Lambda(lambda frames: [transforms.Resize(self.size)(frame) for frame in frames])(images)

            w_ratio = float(self.size[0]) / w
            h_ratio = float(self.size[1]) / h
            ratio = np.array([w_ratio, h_ratio, w_ratio, h_ratio, 1.], dtype=np.float32)

            for i, frame in enumerate(gts):
                new_frame = frame * ratio

                gts[i] = new_frame

        return images, gts, w, h

class RandomSaltAndPepper(object):
    def __init__(self, size, p=0.5, ratio=0.05):
        self.size = size
        self.p = p
        self.ratio = ratio

    def __call__(self, images, gts, w, h):
        if random.uniform(0., 1.) < self.p:
            ratio_factor = random.uniform(0., self.ratio)
            images = [self.salt_and_pepper(image, self.size[0], self.size[1], ratio_factor) for image in images]

        return images, gts, w, h

    def salt_and_pepper(self, image, w, h, ratio_factor):
        noise_count = math.floor(w * h * ratio_factor)
        np_image = np.array(image)

        for i in range(noise_count):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            noise = 0 if random.uniform(0., 1.) < 0.5 else 255
            np_image[y, x, :] = noise

        return transforms.functional.to_pil_image(np_image)

# mod from torchvision
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, gts, w, h):
        for t in self.transforms:
            images, gts, w, h = t(images, gts, w, h)
        return images, gts, w, h

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# resize transform
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, gts, w, h):
        images = transforms.Lambda(lambda frames: [transforms.Resize(self.size)(frame) for frame in frames])(images)

        w_ratio = float(self.size[0]) / w
        h_ratio = float(self.size[1]) / h
        ratio = np.array([w_ratio, h_ratio, w_ratio, h_ratio, 1.], dtype=np.float32)

        for i, frame in enumerate(gts):
            new_frame = frame * ratio

            gts[i] = new_frame
        
        return images, gts, w, h

# convert gt to percentage format
class Percentage(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, gts, w, h):
        scale = np.array([self.size[0], self.size[1], self.size[0], self.size[1], 1], dtype=np.float32)

        for frame_index, frame in enumerate(gts):
            scaled_frame = frame / scale

            gts[frame_index] = scaled_frame

        return images, gts, w, h

# ToTensor wrapper
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, images, gts, w, h):
        # TODO : image channel swapping?
        images = transforms.Lambda(lambda frames: torch.stack([transforms.ToTensor()(frame) for frame in frames]))(images)
        gts = [torch.as_tensor(frame, dtype=torch.float32) for frame in gts]

        return images, gts, w, h

class VidDataset(Dataset):
    def __init__(self, root, packs, classDict, num_classes=31, phase='train', transform=None, target_transform=None, num_frames=5):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.total_len = 0

        self.dict = classDict
        self.num_classes = num_classes

        self.image_root = os.path.join(self.root, 'Data/VID/', self.phase)
        self.groundtruth_root = os.path.join(self.root, 'Annotations/VID/', self.phase)

        self.samples = []

        # load dump sample file if it exists
        sample_dump_file = os.path.join(self.root, 'dump/sample.{}.dump').format(self.phase)

        if file_exists(sample_dump_file):
            print('\nloading {} sample dump'.format(self.phase))
            with open(sample_dump_file, 'rb') as file:
                self.samples = pickle.load(file)
                self.total_len = len(self.samples)
        elif self.phase == 'train':
            for pack in packs:
                print('\nlisting pack: {}'.format(pack))
                # get samples and gts
                image_pack_path = os.path.join(self.image_root, pack)
                label_pack_path = os.path.join(self.groundtruth_root, pack)
                seq = os.listdir(image_pack_path)

                for seq_name in tqdm(seq):
                    # image paths
                    image_seq_path = os.path.join(image_pack_path, seq_name)
                    image_frames = os.listdir(image_seq_path)
                    image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                    image_frame_paths.sort()

                    # read w, h from first frame of this seq
                    with Image.open(image_frame_paths[0]) as image:
                        w, h = image.size

                    # label paths
                    label_seq_path = os.path.join(label_pack_path, seq_name)
                    label_frames = os.listdir(label_seq_path)
                    label_frame_paths = [os.path.join(label_seq_path, frame) for frame in label_frames]
                    label_frame_paths.sort()

                    # parse xml lable files
                    labels = [self.parse_groundtruth(label_frame_path, w, h) for label_frame_path in label_frame_paths]

                    # split seq into 3D samples
                    for i in range(len(image_frames) - (self.num_frames - 1)):
                        images = image_frame_paths[i:i + self.num_frames]

                        frame_labels = labels[i:i + self.num_frames]
                        gts = [np.array(frame, dtype=np.float32) for frame in frame_labels]

                        self.samples.append((images, gts, w, h))

                        self.total_len += 1
        elif self.phase == 'val':
            print('\nlisting pack: val')
            # get samples and gts
            seq = os.listdir(self.image_root)

            for seq_name in tqdm(seq):
                # image paths
                image_seq_path = os.path.join(self.image_root, seq_name)
                image_frames = os.listdir(image_seq_path)
                image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                image_frame_paths.sort()

                # read w, h from first frame of this seq
                with Image.open(image_frame_paths[0]) as image:
                    w, h = image.size

                # label paths
                label_seq_path = os.path.join(self.groundtruth_root, seq_name)
                label_frames = os.listdir(label_seq_path)
                label_frame_paths = [os.path.join(label_seq_path, frame) for frame in label_frames]
                label_frame_paths.sort()

                # parse xml lable files
                labels = [self.parse_groundtruth(label_frame_path, w, h) for label_frame_path in label_frame_paths]

                # split seq into 3D samples
                for i in range(len(image_frames) - (self.num_frames - 1)):
                    images = image_frame_paths[i:i + self.num_frames]

                    frame_labels = labels[i:i + self.num_frames]
                    gts = [np.array(frame, dtype=np.float32) for frame in frame_labels]

                    self.samples.append((images, gts, w, h))

                    self.total_len += 1
        else:
            # test has no groundtruth
            print('\nlisting pack: test')
            seq = os.listdir(self.image_root)

            for seq_name in tqdm(seq):
                # image paths
                image_seq_path = os.path.join(self.image_root, seq_name)
                image_frames = os.listdir(image_seq_path)
                image_frame_paths = [os.path.join(image_seq_path, frame) for frame in image_frames]
                image_frame_paths.sort()

                # read w, h from first frame of this seq
                with Image.open(image_frame_paths[0]) as image:
                    w, h = image.size

                # split seq into 3D samples
                for i in range(len(image_frames) - (self.num_frames - 1)):
                    images = image_frame_paths[i:i + self.num_frames]
                    gts = []

                    # create background gts just for transformation
                    for j in range(self.num_frames):
                        gts.append([np.array([0, 0, w, h, 0], dtype=np.float32)])
                    
                    self.samples.append((images, gts, w, h))
                    self.total_len += 1

    def __getitem__(self, index):
        images, gts, w, h = self.samples[index]
        images = [Image.open(img_path) for img_path in images]

        if self.transform is not None:
            images, gts, w, h = self.transform(images, gts, w, h)

        return images, gts, w, h

    def __len__(self):
        return self.total_len

    def parse_groundtruth(self, xml_path, width, height):
        label = []
        tree = ET.ElementTree(file=xml_path)
        root = tree.getroot()

        for obj in root.iterfind('object'):
            # load labels
            xmax = int(obj.find('bndbox/xmax').text)
            xmin = int(obj.find('bndbox/xmin').text)
            ymax = int(obj.find('bndbox/ymax').text)
            ymin = int(obj.find('bndbox/ymin').text)

            # load class number
            className = obj.find('name').text
            classNo = self.dict[className]

            # lower-left to upper-right
            bbox = [xmin, ymin, xmax, ymax, classNo]
            label.append(bbox)

        # if no obj exists in this frame,
        # then we have a background label
        if len(label) == 0:
            label.append([0, 0, width, height, 0])

        return label

    def get_sample(self):
        return self.samples
