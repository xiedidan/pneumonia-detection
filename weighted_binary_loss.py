import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init

def weighted_binary_cross_entropy(input, target, N_POS, N_NEG, N_TOTAL):
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    weights = torch.div(torch.neg(torch.tensor([N_NEG, N_POS])), N_TOTAL) # [2]
    selector = torch.stack([target, torch.add(1.0, torch.neg(target))]) # [N, 2] 
    p = torch.log(torch.sigmoid(input)) # [N, 2]

    loss = torch.tensor([torch.dot(weights, line) for line in torch.mul(selector, p)])

    return loss

class WeightedBinaryCrossEntropyLoss(nn._Loss):
    def __init__(
        self,
        N_POS,
        N_NEG,
        size_average=None,
        reduce=None,
        reduction='elementwise_mean'
    ):
        super(WeightedBinaryCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.N_POS = N_POS
        self.N_NEG = N_NEG
        self.N_TOTAL = N_POS + N_NEG

    def forward(self, input, target):
        return weighted_binary_cross_entropy(
            input,
            target,
            self.N_POS,
            self.N_NEG,
            self.N_TOTAL
        )
        