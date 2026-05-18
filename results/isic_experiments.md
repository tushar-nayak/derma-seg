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
