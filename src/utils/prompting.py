import numpy as np


def mask_to_box(mask, pad=4):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)

    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + pad)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def mask_to_point(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return np.array([[w / 2.0, h / 2.0]], dtype=np.float32), np.array([0], dtype=np.int32)

    x = float(xs.mean())
    y = float(ys.mean())
    return np.array([[x, y]], dtype=np.float32), np.array([1], dtype=np.int32)


def grayscale_to_rgb(image):
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
