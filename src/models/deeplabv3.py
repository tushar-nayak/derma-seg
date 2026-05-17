import torch.nn as nn
import torchvision.models.segmentation as segmentation

class DeepLabV3Wrapper(nn.Module):
    """
    Wrapper for torchvision's DeepLabV3+ to adapt it to 
    1-channel input and configurable number of classes.
    """
    def __init__(self, n_channels=1, n_classes=2):
        super().__init__()
        
        # Load the base model without pretrained weights
        self.model = segmentation.deeplabv3_resnet50(
            weights=None, 
            weights_backbone=None,
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
