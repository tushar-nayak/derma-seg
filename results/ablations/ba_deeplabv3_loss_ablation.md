# BA-DeepLabV3 Loss Ablation

| Run | Model | Loss | LR | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ba_deeplabv3 | ba_deeplabv3 | ce_dice | 0.0003 | 0.8747 | 0.7871 | 0.7160 | 0.8557 | 0.7665 | 0.6870 |
| ba_deeplabv3 | ba_deeplabv3 | ce_focal_tversky | 0.0003 | 0.8559 | 0.7587 | 0.6734 | 0.8448 | 0.7467 | 0.6574 |
| ba_deeplabv3 | ba_deeplabv3 | ce_tversky | 0.0003 | 0.8560 | 0.7592 | 0.6803 | 0.8485 | 0.7520 | 0.6717 |
