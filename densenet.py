import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

import re

def convert_model(state_dict):
    # Code modified from torchvision densenet source for loading from pre .4 densenet weights.
    remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)

    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        match = pattern.match(key)
        new_key = match.group(1) + match.group(2) if match else key
        new_key = new_key[7:] if remove_data_parallel else new_key
        state_dict[new_key] = state_dict[key]

        # Delete old key only if modified.
        if match or remove_data_parallel: 
            del state_dict[key]
    
    return state_dict

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def transfer(self, state_dict):
        # load pretrained chexnet parameters (from old pytorch version)
        my_state_dict = convert_model(state_dict)
        self.load_state_dict(my_state_dict)
