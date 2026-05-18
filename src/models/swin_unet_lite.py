import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.fuse(x)


class SwinUNetLite(nn.Module):
    """
    A lightweight Swin-based encoder with a U-Net style decoder.

    This is not a full research-grade Swin-Unet clone, but it gives
    the repository a credible transformer-based segmentation model
    that is practical to train on 2D medical slices.
    """

    def __init__(self, n_channels=1, n_classes=2, pretrained=False, backbone="swin_tiny_patch4_window7_224", img_size=256):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=n_channels,
            img_size=img_size,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4 = channels

        self.center = ConvBlock(c4, 512)
        self.dec3 = DecoderBlock(512, c3, 256)
        self.dec2 = DecoderBlock(256, c2, 128)
        self.dec1 = DecoderBlock(128, c1, 64)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        x1, x2, x3, x4 = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        x = self.center(x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.head(x)
