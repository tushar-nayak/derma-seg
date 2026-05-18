import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

class DeepLabV3Wrapper(nn.Module):
    """
    Wrapper for torchvision's DeepLabV3+ to adapt it to 
    1-channel input and configurable number of classes.
    """
    def __init__(self, n_channels=1, n_classes=2, pretrained=False):
        super().__init__()

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained and n_channels == 3 else None
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained and n_channels != 3 else None

        # Use pretrained weights when possible for RGB ISIC experiments.
        self.model = segmentation.deeplabv3_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=n_classes
        )

        # Modify the first convolutional layer to accept `n_channels` instead of 3
        if n_channels != 3:
            original_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = nn.Conv2d(
                n_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=False
            )
            
    def forward(self, x):
        # Torchvision segmentation models return an OrderedDict
        # 'out' contains the main output, 'aux' contains auxiliary classifier output
        return self.model(x)['out']
