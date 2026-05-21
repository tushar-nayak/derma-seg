# ISIC Combined Ablation Summary

| Run | Model | Loss | LR | Best Val Dice | Best Val IoU | Best Val TJ | Test Dice | Test IoU | Test TJ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ba_deeplabv3 | ba_deeplabv3 | ce_dice | 0.0003 | 0.8747 | 0.7871 | 0.7160 | 0.8557 | 0.7665 | 0.6870 |
| ba_deeplabv3 | ba_deeplabv3 | ce_focal_tversky | 0.0003 | 0.8559 | 0.7587 | 0.6734 | 0.8448 | 0.7467 | 0.6574 |
| ba_deeplabv3 | ba_deeplabv3 | ce_tversky | 0.0001 | 0.8617 | 0.7671 | 0.7041 | 0.8393 | 0.7392 | 0.6344 |
| ba_deeplabv3 | ba_deeplabv3 | ce_tversky | 0.0003 | 0.8560 | 0.7592 | 0.6803 | 0.8485 | 0.7520 | 0.6717 |
| ba_deeplabv3 | ba_deeplabv3 | ce_tversky | 0.0010 | 0.8620 | 0.7711 | 0.7035 | 0.8603 | 0.7707 | 0.6978 |
| deeplabv3 | deeplabv3 | ce_dice | 0.0003 | 0.8842 | 0.8054 | 0.7624 | 0.8749 | 0.7946 | 0.7375 |
| deeplabv3 | deeplabv3 | ce_focal_tversky | 0.0003 | 0.8650 | 0.7732 | 0.7003 | 0.8536 | 0.7616 | 0.6794 |
| deeplabv3 | deeplabv3 | ce_tversky | 0.0001 | 0.8758 | 0.7898 | 0.7483 | 0.8563 | 0.7664 | 0.6829 |
| deeplabv3 | deeplabv3 | ce_tversky | 0.0003 | 0.8606 | 0.7682 | 0.6854 | 0.8498 | 0.7549 | 0.6602 |
| deeplabv3 | deeplabv3 | ce_tversky | 0.0010 | 0.8787 | 0.7937 | 0.7344 | 0.8645 | 0.7772 | 0.7138 |
