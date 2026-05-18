# Medical Image Segmentation Benchmark

This repository is a 2D medical segmentation benchmark built around the ISIC 2018 lesion segmentation dataset. It compares classic CNN segmentation baselines with a transformer-based encoder/decoder model on patient-level splits.

## Models

- U-Net
- Attention U-Net
- SegNet
- U-Net++
- DeepLabV3+
- SwinUNetLite, a Swin Transformer encoder with a U-Net-style decoder
- SegFormerLite, a SegFormer-style transformer baseline with a hierarchical encoder

## Promptable Models

- SAM
- MedSAM

These are evaluated separately from the supervised benchmark with ground-truth-derived prompts so the comparison stays explicit and reproducible.

## Dataset

The main benchmark now uses ISIC 2018, which provides dermoscopic lesion images and binary mask PNGs through the official challenge data page.

Download the training images and masks from the official ISIC data page:

- [ISIC data page](https://challenge.isic-archive.com/data/)
- [ISIC 2018 lesion segmentation task](https://challenge.isic-archive.com/landing/2018/45/)

Place the files in one of these layouts:

- `data/isic2018/images` and `data/isic2018/masks`
- `data/isic2018/ISIC2018_Task1-2_Training_Input` and `data/isic2018/ISIC2018_Task1_Training_GroundTruth`
- `data/isic2018/ISIC2018_Task1-2_Training_Input`, `data/isic2018/ISIC2018_Task1_Training_GroundTruth`, `data/isic2018/ISIC2018_Task1-2_Validation_Input`, `data/isic2018/ISIC2018_Task1_Validation_GroundTruth`, `data/isic2018/ISIC2018_Task1-2_Test_Input`, and `data/isic2018/ISIC2018_Task1_Test_GroundTruth`

Masks are expected to follow the `ISIC_<image_id>_segmentation.png` naming convention used by the challenge.
If the official validation and test folders are present, the training pipeline uses that official split automatically.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train a model on ISIC 2018:

```bash
python scripts/train.py --dataset isic --model unet
```

3. Try the transformer-based model:

```bash
python scripts/train.py --dataset isic --model swin_unet --pretrained
```

4. Run a strong RGB baseline with pretrained weights:

```bash
python scripts/train.py --dataset isic --model deeplabv3 --pretrained
```

5. Run the PNG fallback dataset if needed:

```bash
python scripts/train.py --dataset png --data_dir data/lumiere_slices --model deeplabv3
```

## What Changed

- Patient-level splitting for the medical datasets used in this repo.
- Correct Dice, IoU, and ISIC threshold Jaccard computation for 2-class segmentation.
- Combined cross-entropy + Dice training loss with foreground weighting.
- Official ISIC split support with validation-based checkpoint selection on thresholded Jaccard.
- A Swin-based segmentation model for transformer comparison.
- A SegFormer-style non-U-Net transformer baseline.
- Optional SAM/MedSAM promptable evaluation with oracle box prompts.
- Reproducible metric saving under `runs/<dataset>/<model>/metrics.json`.

## Results

Populate the table below only after running the benchmark on your machine.

Legacy benchmark summaries are preserved in [results/final_benchmark.md](./results/final_benchmark.md) and [results/smoke_benchmark.md](./results/smoke_benchmark.md). They do not replace running the current ISIC benchmark locally.

| Model | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: |
| U-Net | TBD | TBD | TBD |
| Attention U-Net | TBD | TBD | TBD |
| SegNet | TBD | TBD | TBD |
| U-Net++ | TBD | TBD | TBD |
| DeepLabV3+ | TBD | TBD | TBD |
| SwinUNetLite | TBD | TBD | TBD |
| SegFormerLite | TBD | TBD | TBD |

Promptable SAM/MedSAM results are saved separately under `runs/promptable/`.

## Notes

- LUMIERE and the synthetic toy data are still present, but they are not the main benchmark.
- The repository does not invent metrics. If you want publishable numbers, run the benchmark and commit the saved `metrics.json` files or summarize them in a results table.
- SAM and MedSAM need the optional `segment-anything` package and their respective checkpoints. Install the package separately before running `scripts/evaluate_promptable.py`.
