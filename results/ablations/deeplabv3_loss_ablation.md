# DeepLabV3 Loss Ablation

| Run | Model | Loss | LR | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deeplabv3 | deeplabv3 | ce_dice | 0.0003 | 0.8842 | 0.8054 | 0.7624 | 0.8749 | 0.7946 | 0.7375 |
| deeplabv3 | deeplabv3 | ce_focal_tversky | 0.0003 | 0.8650 | 0.7732 | 0.7003 | 0.8536 | 0.7616 | 0.6794 |
| deeplabv3 | deeplabv3 | ce_tversky | 0.0003 | 0.8606 | 0.7682 | 0.6854 | 0.8498 | 0.7549 | 0.6602 |
