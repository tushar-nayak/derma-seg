# Smoke Benchmark

These are bounded train/val/test runs on the LUMIERE dataset using:

- `--max_train_samples 256`
- `--max_val_samples 64`
- `--max_test_samples 64`
- `--epochs 1`
- `--batch_size 16`

They are useful as a reproducible pipeline check, not as final paper-grade numbers.

## Results

| Model | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: |
| U-Net | 0.5000 | 0.5000 | 0.5000 |
| Attention U-Net | 0.0005 | 0.0016 | 0.0008 |
| SegFormerLite | 0.0005 | 0.0016 | 0.0008 |
| SwinUNetLite | 0.0009 | 0.0026 | 0.0013 |

## Notes

- The runs used the real LUMIERE source data, not synthetic images.
- SAM and MedSAM remain in the repo as a separate promptable evaluation track and were not included in this smoke benchmark.
