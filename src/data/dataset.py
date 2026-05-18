from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    """
    Paired image-mask dataset for flat PNG folders.

    This loader is intentionally forgiving about mask naming:
    it first looks for the same filename in the masks folder and
    then falls back to common image->mask prefix substitutions.
    """

    def __init__(self, data_dir, image_size=256, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.filenames = sorted(p.name for p in self.images_dir.glob("*.png"))

    def __len__(self):
        return len(self.filenames)

    def _mask_path_for(self, image_name):
        same_name = self.masks_dir / image_name
        if same_name.exists():
            return same_name

        stem = Path(image_name).stem
        suffix = Path(image_name).suffix
        candidates = [
            self.masks_dir / image_name.replace("img_", "mask_"),
            self.masks_dir / image_name.replace("image_", "mask_"),
            self.masks_dir / f"{stem.replace('img', 'mask')}{suffix}",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not find a mask for {image_name}")

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = self.images_dir / img_name
        mask_path = self._mask_path_for(img_name)

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        if image.shape != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if not torch.is_tensor(image):
                image = torch.from_numpy(np.asarray(image))
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(np.asarray(mask))
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).long()

        if image.ndim == 2:
            image = image.unsqueeze(0)
        return image.float(), mask.long()
