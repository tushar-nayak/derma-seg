# DermaSeg

DermaSeg is a medical image segmentation project focused on skin lesion segmentation with deep learning. The project is built around the ISIC 2018 lesion segmentation task and is structured to show practical experimentation across classic CNNs, attention-based models, transformer-based models, and promptable foundation-model baselines.

## Project Goal

The point of this repository is not just to train a single model. It is to build a serious segmentation project that shows:

- supervised medical segmentation on a real dataset with ground-truth masks
- comparison across multiple model families
- reproducible training, validation, and test workflows
- room for extensions such as SAM, MedSAM, and newer hybrid approaches

## Methods Implemented

### Supervised segmentation models

- U-Net
- Attention U-Net
- SegNet
- U-Net++
- DeepLabV3
- SwinUNetLite
- SegFormerLite

### Promptable foundation-model track

- SAM
- MedSAM

The promptable models are kept separate from the supervised training track so the comparisons stay honest. They are useful here as project extensions, not as direct replacements for mask-supervised training.

## Dataset

The main project dataset is ISIC 2018 Task 1 for lesion boundary segmentation.

Official sources:

- [ISIC data page](https://challenge.isic-archive.com/data/)
- [ISIC 2018 lesion segmentation task](https://challenge.isic-archive.com/landing/2018/45/)

Supported layouts:

- `data/isic2018/images` and `data/isic2018/masks`
- `data/isic2018/ISIC2018_Task1-2_Training_Input` and `data/isic2018/ISIC2018_Task1_Training_GroundTruth`
- the full official split with train, validation, and test folders

If the official validation and test folders are present, the training pipeline uses that split automatically.

## Project Structure

- `scripts/train.py`: main training, validation, and test entrypoint
- `scripts/evaluate_promptable.py`: SAM and MedSAM evaluation path
- `scripts/summarize_metrics.py`: converts saved metrics into a Markdown summary
- `src/data/`: dataset loaders and preparation utilities
- `src/models/`: segmentation architectures
- `src/utils/`: losses, metrics, and prompting helpers
- `runs/`: checkpoints and metrics written by experiments
- `results/`: summarized experiment outputs for the repo

## Training Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a baseline U-Net:

```bash
python scripts/train.py --dataset isic --model unet
```

Train a stronger pretrained RGB baseline:

```bash
python scripts/train.py --dataset isic --model deeplabv3 --pretrained
```

Train a transformer-based model:

```bash
python scripts/train.py --dataset isic --model swin_unet --pretrained
```

Promptable evaluation:

```bash
python scripts/evaluate_promptable.py --model sam --checkpoint /path/to/checkpoint.pth
```

## Experiment Philosophy

This repository is organized like a real project rather than a one-off notebook dump. The focus is:

- reproducible runs with saved checkpoints and metrics
- comparing segmentation architectures on the same medical task
- keeping classical supervised models and foundation-model experiments clearly separated
- building a repo that can evolve into stronger experiments instead of freezing at a single result

## Outputs

Each supervised run writes artifacts under:

```text
runs/<dataset>/<model>/
```

That directory contains:

- `best.pt`
- `last.pt`
- `metrics.json`

You can turn completed runs into a project summary with:

```bash
python scripts/summarize_metrics.py
```

## Notes

- ISIC is the main project dataset now.
- LUMIERE and synthetic data are still in the repo as side datasets and utilities, but they are not the primary project path.
- Metrics include Dice, IoU, and thresholded Jaccard for ISIC-style evaluation.
- Pretrained RGB models use ImageNet-normalized inputs during ISIC training.
