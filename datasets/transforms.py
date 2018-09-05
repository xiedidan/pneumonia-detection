import numpy as np

import torch
from torch.utils.data import *
import torchvision.transforms as transforms

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
        gt = (
            torch.as_tensor(gt[0], dtype=torch.float32),
            torch.as_tensor(gt[1], dtype=torch.uint8)
        )

        return image, gt, w, h

# resize transform
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt, w, h):
        image = transforms.functional.resize(image, self.size)

        w_ratio = float(self.size[0]) / w
        h_ratio = float(self.size[1]) / h
        ratio = np.array([w_ratio, h_ratio, w_ratio, h_ratio], dtype=np.float32)

        gt = (gt[0] * ratio, gt[1])
        
        return image, gt, w, h
    
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
    