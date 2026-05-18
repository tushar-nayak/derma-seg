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


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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

        true_pos = (probs * target).sum()
        false_pos = (probs * (1.0 - target)).sum()
        false_neg = ((1.0 - probs) * target).sum()
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits, target):
        loss = self.tversky(logits, target)
        return loss.pow(self.gamma)


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        ce_weight=0.5,
        overlap_weight=2.0,
        class_weights=(0.1, 0.9),
        overlap_mode="dice",
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_tversky_gamma=1.33,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.overlap_weight = overlap_weight
        self.overlap_mode = overlap_mode
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        if overlap_mode == "dice":
            self.overlap = DiceLoss()
        elif overlap_mode == "tversky":
            self.overlap = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif overlap_mode == "focal_tversky":
            self.overlap = FocalTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=focal_tversky_gamma,
            )
        else:
            raise ValueError(f"Unknown overlap mode: {overlap_mode}")

    def forward(self, logits, target):
        loss = 0.0
        if self.ce_weight:
            ce = F.cross_entropy(logits, target, weight=self.class_weights.to(logits.device))
            loss = loss + self.ce_weight * ce
        if self.overlap_weight:
            loss = loss + self.overlap_weight * self.overlap(logits, target)
        return loss
