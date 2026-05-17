# Medical Image Segmentation Benchmark

This repository is a 2D medical segmentation benchmark built around the LUMIERE glioblastoma dataset. It compares classic CNN segmentation baselines with a transformer-based encoder/decoder model on patient-level splits.

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

The main benchmark uses the local LUMIERE source data at:

`/home/sofa/host_dir/hub/glioblastoma-evolution/data/lumiere`

The training pipeline indexes LUMIERE volumes and segmentation masks directly, then builds slice-level samples with patient-level train/val/test splits to avoid leakage.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train a model on LUMIERE:

```bash
python scripts/train.py --dataset lumiere --modalities flair --model unet
```

3. Try the transformer-based model:

```bash
python scripts/train.py --dataset lumiere --modalities flair --model swin_unet --pretrained
```

4. Run the PNG fallback dataset if needed:

```bash
python scripts/train.py --dataset png --data_dir data/lumiere_slices --model deeplabv3
```

5. Run a promptable SAM-style evaluation:

```bash
python scripts/evaluate_promptable.py --model sam --checkpoint /path/to/sam_checkpoint.pth
```

## What Changed

- Patient-level splitting for the real dataset.
- Correct Dice and IoU computation for 2-class segmentation.
- Combined cross-entropy + Dice training loss.
- A Swin-based segmentation model for transformer comparison.
- A SegFormer-style non-U-Net transformer baseline.
- Optional SAM/MedSAM promptable evaluation with oracle box prompts.
- Reproducible metric saving under `runs/<dataset>/<model>/metrics.json`.
- Final benchmark runs currently use FLAIR-only inputs for tractable full-dataset training; multimodal support remains available via `--modalities flair,t1,t2,ct1`.

## Results

Populate the table below only after running the benchmark on your machine.

A reproducible final benchmark from the full PNG cache run is saved in [results/final_benchmark.md](./results/final_benchmark.md). A smaller smoke benchmark is also preserved in [results/smoke_benchmark.md](./results/smoke_benchmark.md).

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

- The previous synthetic toy data is still present, but it is not the main benchmark.
- The repository does not invent metrics. If you want publishable numbers, run the benchmark and commit the saved `metrics.json` files or summarize them in a results table.
- SAM and MedSAM need the optional `segment-anything` package and their respective checkpoints. Install the package separately before running `scripts/evaluate_promptable.py`.
