# Final Benchmark

This benchmark was run on a full PNG cache extracted from the original LUMIERE source data. The training protocol used:

- patient-level train/val/test splits
- foreground-weighted loss
- `--epochs 3` for U-Net and Attention U-Net
- `--epochs 5` for SegFormer and SwinUNetLite
- `--batch_size 32` for U-Net and Attention U-Net
- `--batch_size 16` for SegFormer and SwinUNetLite

## Results

| Model | Val Dice | Test Dice | Test IoU |
| --- | ---: | ---: | ---: |
| U-Net | 0.2296 | 0.2296 | 0.1304 |
| Attention U-Net | 0.1761 | 0.1761 | 0.0970 |
| SegFormerLite | 0.4116 | 0.4116 | 0.2613 |
| SwinUNetLite | 0.4421 | 0.4421 | 0.2861 |

## Notes

- These are the actual saved metrics from `runs_final_png/png/<model>/metrics.json`.
- The benchmark uses FLAIR-only PNG slices for tractable training, while the repository still supports the original LUMIERE NIfTI pipeline and multimodal inputs.
- SAM and MedSAM remain separate promptable baselines and were not part of this supervised table.
