import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _to_chw(feature):
    if feature.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got {tuple(feature.shape)}")
    if feature.shape[-1] > feature.shape[1] and feature.shape[-1] > feature.shape[2]:
        return feature.permute(0, 3, 1, 2).contiguous()
    return feature


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegFormerLite(nn.Module):
    """
    A compact SegFormer-style baseline using a hierarchical transformer encoder.

    timm does not ship the original MiT backbone, so this implementation uses
    a Pyramid Vision Transformer encoder with a SegFormer-inspired decode head.
    """

    def __init__(self, n_channels=1, n_classes=2, pretrained=False, backbone="pvt_v2_b0", decoder_dim=64):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=n_channels,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.encoder.feature_info.channels()
        self.projections = nn.ModuleList([ConvBNReLU(c, decoder_dim, kernel_size=1) for c in channels])
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_dim * 4, 256),
            ConvBNReLU(256, 128),
            ConvBNReLU(128, 128),
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )

    def forward(self, x):
        features = [_to_chw(feature) for feature in self.encoder(x)]
        target_size = features[0].shape[-2:]

        pyramids = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            pyramids.append(feature)

        x = torch.cat(pyramids, dim=1)
        x = self.fuse(x)
        return self.head(x)
