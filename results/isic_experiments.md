# ISIC Experiment Summary

Current completed run:

- dataset: `ISIC 2018 Task 1`
- model: `DeepLabV3`
- setup: pretrained RGB backbone, official train/validation/test split
- best validation threshold Jaccard: `0.7598`
- final test threshold Jaccard: `0.7320`

| Model | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deeplabv3 | 0.8900 | 0.8134 | 0.7598 | 0.8782 | 0.7991 | 0.7320 |

## Qualitative Results

Each panel shows:

- left: input dermoscopic image
- middle: ground-truth lesion mask overlay
- right: DeepLabV3 prediction overlay

![DeepLabV3 sample 1](figures/isic_deeplabv3_sample_001.png)
![DeepLabV3 sample 2](figures/isic_deeplabv3_sample_002.png)
![DeepLabV3 sample 3](figures/isic_deeplabv3_sample_003.png)
![DeepLabV3 sample 4](figures/isic_deeplabv3_sample_004.png)
