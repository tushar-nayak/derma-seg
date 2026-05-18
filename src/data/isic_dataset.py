from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def _resolve_dir(base, candidates):
    for rel in candidates:
        path = base / rel
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {', '.join(candidates)} under {base}")


def _mask_path_for(image_path, masks_dir):
    stem = image_path.stem
    candidates = [
        masks_dir / f"{stem}_segmentation.png",
        masks_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a mask for {image_path.name}")


def _augment(image, mask):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.rand() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 4)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    return image.copy(), mask.copy()


class ISICSegmentationDataset(Dataset):
    """
    2D dermoscopic segmentation dataset for the ISIC 2018 lesion boundary task.

    Expected folder layouts supported:
    - <root>/images + <root>/masks
    - <root>/ISIC2018_Task1-2_Training_Input + <root>/ISIC2018_Task1_Training_GroundTruth
    - <root>/train_images + <root>/train_masks
    """

    def __init__(self, data_dir, image_size=224, augment=False):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        self.images_dir = _resolve_dir(
            self.data_dir,
            ["images", "train_images", "ISIC2018_Task1-2_Training_Input", "training_images"],
        )
        self.masks_dir = _resolve_dir(
            self.data_dir,
            ["masks", "train_masks", "ISIC2018_Task1_Training_GroundTruth", "training_masks"],
        )

        self.image_paths = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name.startswith("ISIC_")
            ]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No ISIC images found under {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = _mask_path_for(image_path, self.masks_dir)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.int64)

        if self.augment:
            image, mask = _augment(image, mask)

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


def build_isic_loaders(data_dir, image_size=224, batch_size=16, seed=42, num_workers=0):
    base_dataset = ISICSegmentationDataset(data_dir, image_size=image_size, augment=False)
    n_samples = len(base_dataset)
    train_size = int(round(0.8 * n_samples))
    val_size = max(1, int(round(0.1 * n_samples)))
    test_size = n_samples - train_size - val_size
    if test_size <= 0:
        test_size = 1
        train_size = n_samples - val_size - test_size

    indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_dataset = Subset(ISICSegmentationDataset(data_dir, image_size=image_size, augment=True), train_idx)
    val_dataset = Subset(ISICSegmentationDataset(data_dir, image_size=image_size, augment=False), val_idx)
    test_dataset = Subset(ISICSegmentationDataset(data_dir, image_size=image_size, augment=False), test_idx)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
