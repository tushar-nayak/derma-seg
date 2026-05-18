import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _foreground_probability(logits):
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=1)[:, 1:2]


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        padding = 0 if kernel_size == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pooled = self.proj(self.pool(x))
        return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)


class DynamicASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(in_channels, out_channels, kernel_size=1),
                ConvNormAct(in_channels, out_channels, kernel_size=3, dilation=rates[0]),
                ConvNormAct(in_channels, out_channels, kernel_size=3, dilation=rates[1]),
                ConvNormAct(in_channels, out_channels, kernel_size=3, dilation=rates[2]),
                ASPPPooling(in_channels, out_channels),
            ]
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, len(self.branches)),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        branch_features = [branch(x) for branch in self.branches]
        branch_weights = torch.softmax(self.gate(x), dim=1)
        weighted = [
            feature * branch_weights[:, idx].view(-1, 1, 1, 1)
            for idx, feature in enumerate(branch_features)
        ]
        fused = self.project(torch.cat(weighted, dim=1))
        return fused, branch_weights


class BoundaryRefinementDecoder(nn.Module):
    def __init__(self, context_channels=256, low_level_channels=256, low_level_proj=48, n_classes=2):
        super().__init__()
        self.low_level = ConvNormAct(low_level_channels, low_level_proj, kernel_size=1)
        self.refine = nn.Sequential(
            ConvNormAct(context_channels + low_level_proj + 4, 256, kernel_size=3),
            ConvNormAct(256, 128, kernel_size=3),
            nn.Conv2d(128, n_classes, kernel_size=1),
        )

    def forward(self, context, low_level, coarse_logits, boundary_logits, uncertainty_logits):
        low_level = self.low_level(low_level)
        context = F.interpolate(context, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        coarse_logits = F.interpolate(
            coarse_logits, size=low_level.shape[-2:], mode="bilinear", align_corners=False
        )
        boundary_logits = F.interpolate(
            boundary_logits, size=low_level.shape[-2:], mode="bilinear", align_corners=False
        )
        uncertainty_logits = F.interpolate(
            uncertainty_logits, size=low_level.shape[-2:], mode="bilinear", align_corners=False
        )
        coarse_prob = _foreground_probability(coarse_logits)
        boundary_prob = torch.sigmoid(boundary_logits)
        uncertainty_prob = torch.sigmoid(uncertainty_logits)
        uncertainty_boundary = boundary_prob * (1.0 + uncertainty_prob)
        refinement_input = torch.cat(
            [context, low_level, coarse_prob, boundary_prob, uncertainty_prob, uncertainty_boundary],
            dim=1,
        )
        return self.refine(refinement_input)


class BoundaryAwareDeepLabV3(nn.Module):
    """
    BA-DeepLabV3: ResNet-50 encoder + dynamic ASPP + lesion/boundary/uncertainty heads.

    The final segmentation logits come from a refinement decoder that fuses:
    - coarse lesion logits
    - boundary logits
    - uncertainty logits
    - low-level encoder features
    """

    def __init__(self, n_channels=3, n_classes=2, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        self._adapt_input_channels(n_channels, pretrained)

        self.context = DynamicASPP(in_channels=2048, out_channels=256)
        self.coarse_head = nn.Sequential(
            ConvNormAct(256, 256, kernel_size=3),
            nn.Conv2d(256, n_classes, kernel_size=1),
        )
        self.boundary_head = nn.Sequential(
            ConvNormAct(256, 128, kernel_size=3),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.uncertainty_head = nn.Sequential(
            ConvNormAct(256, 128, kernel_size=3),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.refinement = BoundaryRefinementDecoder(
            context_channels=256,
            low_level_channels=256,
            low_level_proj=48,
            n_classes=n_classes,
        )

    def _adapt_input_channels(self, n_channels, pretrained):
        if n_channels == 3:
            return

        original = self.encoder.conv1
        updated = nn.Conv2d(
            n_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                mean_weight = original.weight.mean(dim=1, keepdim=True)
                updated.weight.copy_(mean_weight.repeat(1, n_channels, 1, 1))
            else:
                nn.init.kaiming_normal_(updated.weight, mode="fan_out", nonlinearity="relu")
        self.encoder.conv1 = updated

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        low_level = self.encoder.layer1(x)
        x = self.encoder.layer2(low_level)
        x = self.encoder.layer3(x)
        high_level = self.encoder.layer4(x)

        context, branch_weights = self.context(high_level)
        coarse_logits = self.coarse_head(context)
        boundary_logits = self.boundary_head(context)
        uncertainty_logits = self.uncertainty_head(context)
        refined_logits = self.refinement(
            context,
            low_level,
            coarse_logits,
            boundary_logits,
            uncertainty_logits,
        )

        refined_logits = F.interpolate(refined_logits, size=input_size, mode="bilinear", align_corners=False)
        coarse_logits = F.interpolate(coarse_logits, size=input_size, mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(boundary_logits, size=input_size, mode="bilinear", align_corners=False)
        uncertainty_logits = F.interpolate(
            uncertainty_logits, size=input_size, mode="bilinear", align_corners=False
        )

        return {
            "logits": refined_logits,
            "coarse_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "aspp_branch_weights": branch_weights,
        }
