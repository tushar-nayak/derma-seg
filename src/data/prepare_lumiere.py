import os
import csv
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

def extract_slices(data_root, output_root, num_patients=None):
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'masks'), exist_ok=True)
    
    imaging_dir = os.path.join(data_root, 'Imaging', 'Imaging')
    patients = sorted(os.listdir(imaging_dir))
    if num_patients is not None:
        patients = patients[:num_patients]
    
    manifest_rows = []
    slice_counter = 0
    for patient in tqdm(patients, desc="Processing patients"):
        patient_path = os.path.join(imaging_dir, patient)
        weeks = sorted(os.listdir(patient_path))
        for week in weeks:
            week_path = os.path.join(patient_path, week)
            flair_path = os.path.join(week_path, 'FLAIR.nii.gz')
            seg_path = os.path.join(week_path, 'HD-GLIO-AUTO-segmentation', 'registered', 'segmentation.nii.gz')

            if not os.path.exists(flair_path) or not os.path.exists(seg_path):
                continue

            flair_img = nib.load(flair_path).get_fdata()
            seg_img = nib.load(seg_path).get_fdata()

            depth = min(flair_img.shape[2], seg_img.shape[2])
            flair_img = (flair_img - flair_img.min()) / (flair_img.max() - flair_img.min() + 1e-8)

            for s in range(depth):
                image_slice = flair_img[:, :, s]
                mask_slice = (seg_img[:, :, s] > 0).astype(np.uint8)

                if np.sum(mask_slice) < 10:
                    continue

                image_slice = cv2.resize(image_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
                mask_slice = cv2.resize(mask_slice, (256, 256), interpolation=cv2.INTER_NEAREST)

                img_filename = f"slice_{slice_counter:05d}.png"
                cv2.imwrite(os.path.join(output_root, 'images', img_filename), (image_slice * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(output_root, 'masks', img_filename), mask_slice.astype(np.uint8))

                manifest_rows.append([img_filename, patient, week, s])
                slice_counter += 1

    manifest_path = os.path.join(output_root, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "patient_id", "week_id", "slice_idx"])
        writer.writerows(manifest_rows)
    print(f"Saved {slice_counter} slices to {output_root}")

if __name__ == "__main__":
    DATA_ROOT = "/home/sofa/host_dir/hub/glioblastoma-evolution/data/lumiere"
    OUTPUT_ROOT = "data/lumiere_slices"
    extract_slices(DATA_ROOT, OUTPUT_ROOT, num_patients=50)
