import torch
import torch.nn as nn
import torch.nn.functional as F


def _target_mask(target):
    if target.ndim == 3:
        return target.unsqueeze(1).float()
    if target.ndim == 4:
        return target.float()
    raise ValueError(f"Expected [N, H, W] or [N, 1, H, W] target, got shape {tuple(target.shape)}")


def _foreground_probabilities(logits):
    if logits.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] logits, got {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=1)[:, 1:2]


def boundary_target(target, kernel_size=3):
    target = _target_mask(target)
    padding = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp_(0.0, 1.0)


def probability_gradient_map(probabilities):
    grad_x = torch.abs(probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1])
    grad_y = torch.abs(probabilities[:, :, 1:, :] - probabilities[:, :, :-1, :])
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    return (grad_x + grad_y).clamp_(0.0, 1.0)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = _foreground_probabilities(logits)
        target = _target_mask(target)
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
        probs = _foreground_probabilities(logits)
        target = _target_mask(target)
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
        coarse_weight=0.5,
        boundary_weight=0.3,
        uncertainty_weight=0.2,
        consistency_weight=0.2,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.overlap_weight = overlap_weight
        self.overlap_mode = overlap_mode
        self.coarse_weight = coarse_weight
        self.boundary_weight = boundary_weight
        self.uncertainty_weight = uncertainty_weight
        self.consistency_weight = consistency_weight
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

    def _segmentation_loss(self, logits, target):
        loss = 0.0
        if self.ce_weight:
            ce = F.cross_entropy(logits, target, weight=self.class_weights.to(logits.device))
            loss = loss + self.ce_weight * ce
        if self.overlap_weight:
            loss = loss + self.overlap_weight * self.overlap(logits, target)
        return loss

    def forward(self, logits, target):
        if not isinstance(logits, dict):
            return self._segmentation_loss(logits, target)

        refined_logits = logits["logits"]
        loss = self._segmentation_loss(refined_logits, target)

        coarse_logits = logits.get("coarse_logits")
        if coarse_logits is not None and self.coarse_weight:
            loss = loss + self.coarse_weight * self._segmentation_loss(coarse_logits, target)

        boundary_logits = logits.get("boundary_logits")
        if boundary_logits is not None and self.boundary_weight:
            boundary = boundary_target(target)
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary)
            loss = loss + self.boundary_weight * boundary_loss

        uncertainty_logits = logits.get("uncertainty_logits")
        if uncertainty_logits is not None and self.uncertainty_weight:
            refined_probs = _foreground_probabilities(refined_logits).detach()
            uncertainty_target = torch.abs(refined_probs - _target_mask(target)).clamp_(0.0, 1.0)
            uncertainty_loss = F.binary_cross_entropy_with_logits(uncertainty_logits, uncertainty_target)
            loss = loss + self.uncertainty_weight * uncertainty_loss

        if boundary_logits is not None and self.consistency_weight:
            contour_map = probability_gradient_map(_foreground_probabilities(refined_logits))
            contour_loss = F.l1_loss(torch.sigmoid(boundary_logits), contour_map)
            loss = loss + self.consistency_weight * contour_loss

        return loss
