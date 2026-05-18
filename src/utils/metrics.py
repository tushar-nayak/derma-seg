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


def _per_sample_iou(pred, target):
    smooth = 1e-6
    pred_flat = pred.reshape(pred.shape[0], -1).float()
    target_flat = target.reshape(target.shape[0], -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_coeff(logits, target):
    """Hard Dice score for binary foreground segmentation."""
    pred = _foreground_prediction(logits).float()
    target = _target_mask(target).float()
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    smooth = 1e-6
    intersection = (pred_flat * target_flat).sum(dim=1)
    score = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return score.mean()


def iou_score(logits, target):
    """Hard IoU score for binary foreground segmentation."""
    pred = _foreground_prediction(logits).float()
    target = _target_mask(target).float()
    return _per_sample_iou(pred, target).mean()


def threshold_jaccard(logits, target, threshold=0.65):
    """
    ISIC-style thresholded Jaccard score.

    Returns 0 if the sample-level IoU is below the threshold.
    """
    jaccard = _per_sample_iou(_foreground_prediction(logits).float(), _target_mask(target).float())
    return torch.where(jaccard < threshold, torch.zeros_like(jaccard), jaccard).mean()
