import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        if logits.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W] logits, got {tuple(logits.shape)}")

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)[:, 1:2]

        target = target.float()
        probs = probs.reshape(-1)
        target = target.reshape(-1)

        intersection = (probs * target).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        loss = 0.0
        if self.ce_weight:
            loss = loss + self.ce_weight * self.ce(logits, target)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss
