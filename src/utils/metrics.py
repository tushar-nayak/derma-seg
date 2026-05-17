import torch


def _foreground_prediction(logits):
    if logits.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] logits, got shape {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        return (torch.sigmoid(logits) > 0.5).long()

    return logits.argmax(dim=1, keepdim=True).long()


def _target_mask(target):
    if target.ndim == 3:
        return target.unsqueeze(1).long()
    if target.ndim == 4:
        return target.long()
    raise ValueError(f"Expected [N, H, W] or [N, 1, H, W] target, got shape {tuple(target.shape)}")


def dice_coeff(logits, target):
    """Hard Dice score for binary foreground segmentation."""
    smooth = 1e-6
    pred = _foreground_prediction(logits).float()
    target = _target_mask(target).float()

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(logits, target):
    """Hard IoU score for binary foreground segmentation."""
    smooth = 1e-6
    pred = _foreground_prediction(logits).float()
    target = _target_mask(target).float()

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)
