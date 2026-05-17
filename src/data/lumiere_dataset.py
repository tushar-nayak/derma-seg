from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LumiereSample:
    patient_id: str
    week_id: str
    flair_path: str
    t1_path: str
    t2_path: str
    ct1_path: str
    mask_path: str
    slice_idx: int


def _normalise_volume(volume):
    volume = volume.astype(np.float32)
    min_val = float(volume.min())
    max_val = float(volume.max())
    if max_val <= min_val:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - min_val) / (max_val - min_val)


@lru_cache(maxsize=64)
def _load_volume(path):
    return nib.load(path).get_fdata(dtype=np.float32)


def discover_lumiere_samples(
    data_root,
    positive_only=True,
    min_mask_pixels=10,
):
    """
    Index all available LUMIERE FLAIR/segmentation slice pairs.

    The dataset is grouped by patient so that train/val/test splits can
    be done without leaking slices from the same patient across splits.
    """

    imaging_root = Path(data_root) / "Imaging" / "Imaging"
    samples = []
    patients = []

    for patient_dir in sorted(imaging_root.glob("Patient-*")):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        patients.append(patient_id)

        for week_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            flair_path = week_dir / "FLAIR.nii.gz"
            t1_path = week_dir / "T1.nii.gz"
            t2_path = week_dir / "T2.nii.gz"
            ct1_path = week_dir / "CT1.nii.gz"
            seg_path = week_dir / "HD-GLIO-AUTO-segmentation" / "registered" / "segmentation.nii.gz"
            if not flair_path.exists() or not t1_path.exists() or not t2_path.exists() or not ct1_path.exists() or not seg_path.exists():
                continue

            seg = _load_volume(str(seg_path))
            depth = seg.shape[2]

            for slice_idx in range(depth):
                mask_slice = (seg[:, :, slice_idx] > 0).astype(np.uint8)
                if positive_only and int(mask_slice.sum()) < min_mask_pixels:
                    continue
                samples.append(
                    LumiereSample(
                        patient_id=patient_id,
                        week_id=week_dir.name,
                        flair_path=str(flair_path),
                        t1_path=str(t1_path),
                        t2_path=str(t2_path),
                        ct1_path=str(ct1_path),
                        mask_path=str(seg_path),
                        slice_idx=slice_idx,
                    )
                )

    return samples, sorted(set(patients))


def split_patients(patients, seed=42, val_fraction=0.15, test_fraction=0.15):
    patients = np.array(sorted(set(patients)))
    if len(patients) < 3:
        raise ValueError("Need at least 3 patients for train/val/test splitting.")

    trainval_patients, test_patients = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=seed,
    ).split(patients, groups=patients).__next__()

    remaining = patients[trainval_patients]
    train_patients, val_patients = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction / (1.0 - test_fraction),
        random_state=seed,
    ).split(remaining, groups=remaining).__next__()

    return set(remaining[train_patients]), set(remaining[val_patients]), set(patients[test_patients])


class LumiereSliceDataset(Dataset):
    def __init__(
        self,
        samples,
        image_size=256,
        augment=False,
        cache_volumes=False,
        modalities=("flair",),
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment
        self.cache_volumes = cache_volumes
        self.modalities = tuple(modalities)
        self._volume_cache = {} if cache_volumes else None

    def __len__(self):
        return len(self.samples)

    def _get_volume(self, path):
        if self._volume_cache is None:
            return _load_volume(path)
        if path not in self._volume_cache:
            self._volume_cache[path] = _load_volume(path)
        return self._volume_cache[path]

    @staticmethod
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

    def __getitem__(self, idx):
        sample = self.samples[idx]
        volumes = {
            "flair": self._get_volume(sample.flair_path),
            "t1": self._get_volume(sample.t1_path),
            "t2": self._get_volume(sample.t2_path),
            "ct1": self._get_volume(sample.ct1_path),
        }
        mask_vol = self._get_volume(sample.mask_path)

        selected_volumes = [volumes[name] for name in self.modalities]
        depth = min([vol.shape[2] for vol in selected_volumes] + [mask_vol.shape[2]])
        slice_idx = min(sample.slice_idx, depth - 1)

        image = [_normalise_volume(vol[:, :, slice_idx]) for vol in selected_volumes]
        mask = (mask_vol[:, :, slice_idx] > 0).astype(np.uint8)

        resized = []
        for channel in image:
            resized.append(cv2.resize(channel, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR))
        image = np.stack(resized, axis=0)
        if mask.shape != (self.image_size, self.image_size):
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image_hw, mask = self._augment(np.transpose(image, (1, 2, 0)), mask)
            image = np.transpose(image_hw, (2, 0, 1))

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask.astype(np.int64)).long()

        return image, mask


def build_lumiere_splits(data_root, seed=42, val_fraction=0.15, test_fraction=0.15, positive_only=True):
    samples, patients = discover_lumiere_samples(data_root, positive_only=positive_only)
    train_patients, val_patients, test_patients = split_patients(
        patients, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction
    )

    train_samples = [s for s in samples if s.patient_id in train_patients]
    val_samples = [s for s in samples if s.patient_id in val_patients]
    test_samples = [s for s in samples if s.patient_id in test_patients]

    return train_samples, val_samples, test_samples


def load_lumiere_slice(sample, image_size=256):
    flair_vol = _load_volume(sample.flair_path)
    mask_vol = _load_volume(sample.mask_path)

    depth = min(flair_vol.shape[2], mask_vol.shape[2])
    slice_idx = min(sample.slice_idx, depth - 1)

    image = [_normalise_volume(flair_vol[:, :, slice_idx])]
    mask = (mask_vol[:, :, slice_idx] > 0).astype(np.uint8)

    resized = []
    for channel in image:
        resized.append(cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_LINEAR))
    image = np.stack(resized, axis=0)
    if mask.shape != (image_size, image_size):
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    return image.astype(np.float32), mask.astype(np.uint8)
