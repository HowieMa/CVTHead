import torch
import torch.nn.functional as F
from torch import nn

from typing import Union


class SegmentationLoss(nn.Module):
    def __init__(self, loss_type = 'bce_with_logits'):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, 
                pred_seg_logits: Union[torch.Tensor, list], 
                target_segs: Union[torch.Tensor, list]) -> torch.Tensor:
        if isinstance(pred_seg_logits, list):
            # Concat alongside the batch axis
            pred_seg_logits = torch.cat(pred_seg_logits)
            target_segs = torch.cat(target_segs)

        if target_segs.shape[2] != pred_seg_logits.shape[2]:
            target_segs = F.interpolate(target_segs, size=pred_seg_logits.shape[2:], mode='bilinear')

        if self.loss_type == 'bce_with_logits':
            loss = self.criterion(pred_seg_logits, target_segs)
        
        elif self.loss_type == 'dice':
            # pred_segs = torch.sigmoid(pred_seg_logits)
            pred_segs = pred_seg_logits
            intersection = (pred_segs * target_segs).view(pred_segs.shape[0], -1)
            cardinality = (pred_segs**2 + target_segs**2).view(pred_segs.shape[0], -1)
            loss = 1 - ((2. * intersection.mean(1)) / (cardinality.mean(1) + 1e-7)).mean(0)

        return loss