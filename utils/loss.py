import torch
from torch.nn.modules.loss import _Loss


import torch
from torchvision import models
from torchvision.transforms import Normalize
from collections import namedtuple
import sys
from pathlib import Path

import sys
from pathlib import Path




class MaskedCriterion(torch.nn.Module):
    """
    calculates average loss on x of 2d image while enabling the user to specify a float mask ranging from 0 ... 1
    specifying the weights of the different regions to the loss.
    Can be used to exclude the background for loss calculation.
    """

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, x, y, mask=None):
        """
        :param x: N x C x H x W
        :param y: N x C x H x W
        :param mask: N x 1 x H x W range from 0 ... 1, if mask not given, mask = torch.ones_like(x)
                -> falls back to standard mean reduction
        :return:
        """
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        assert mask.dtype == torch.float
        assert 0 <= mask[0, 0, 0, 0] <= 1.  # exemplary check first tensor entry for range
        assert x.shape == y.shape

        mask = mask.expand_as(x)  # adding channels to mask if necessary

        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = self.criterion(x, y)
            if len(loss.shape) == 3:
                loss.unsqueeze_(1)
            loss = (loss * mask).sum() / mask_sum
        else:
            loss = 0

        return loss



