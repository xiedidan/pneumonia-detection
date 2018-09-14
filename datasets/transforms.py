import random
import math

import numpy as np

import torch
from torch.utils.data import *
import torchvision.transforms as transforms

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

# mod from torchvision
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, gt, w, h):
        for t in self.transforms:
            image, gt, w, h = t(image, gt, w, h)
        return image, gt, w, h

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# ToTensor wrapper
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image, gt, w, h):
        image = transforms.functional.to_tensor(image)
        gt = torch.as_tensor(gt, dtype=torch.long)

        return image, gt, w, h

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, gt, w, h):
        image = transforms.functional.normalize(image, self.mean, self.std)
        return image, gt, w, h

# resize transform
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt, w, h):
        image = transforms.functional.resize(image, self.size)

        w_ratio = float(self.size) / w
        h_ratio = float(self.size) / h
        ratio = np.array([w_ratio, h_ratio, w_ratio, h_ratio, 1], dtype=np.float32)

        gt = gt * ratio
        
        return image, gt, w, h

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
    
# target transforms
class ComposeTarget(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, gt, w, h):
        for t in self.transforms:
            gt, w, h = t(gt, w, h)
        return gt, w, h

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# convert gt to percentage format
class Percentage(object):
    def __init__(self, size):
        self.size = size
        self.scale = np.array([self.size[0], self.size[1], self.size[0], self.size[1]], dtype=np.float32)

    def __call__(self, gt, w, h):
        # Tensor.numpy() shares memory with the tensor itself
        locs_numpy = gt[0].numpy()
        locs_numpy = locs_numpy / self.scale

        return (torch.from_numpy(locs_numpy), gt[1]), w, h

# convert gt from (xmin, ymin, w, h) to bbox (xmin, ymin, xmax, ymax)
class ToBbox(object):
    def __init__(self):
        self.trans_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]).transpose(1, 0)

    def __call__(self, gt, w, h):
        locs_numpy = gt[0].numpy()
        locs_numpy = locs_numpy @ self.trans_matrix

        return (torch.from_numpy(locs_numpy), gt[1]), w, h
    