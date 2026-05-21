from .unet import UNet
from .attention_unet import AttentionUNet
from .segnet import SegNet
from .unetplusplus import UNetPlusPlus
from .deeplabv3 import DeepLabV3Wrapper
from .ba_deeplabv3 import BoundaryAwareDeepLabV3
from .swin_unet_lite import SwinUNetLite
from .segformer_lite import SegFormerLite

def get_model(model_name, n_channels=1, n_classes=2, img_size=256, pretrained=False, **kwargs):
    model_name = model_name.lower()
    if model_name == 'unet':
        return UNet(n_channels, n_classes)
    elif model_name == 'attention_unet':
        return AttentionUNet(n_channels, n_classes)
    elif model_name == 'segnet':
        return SegNet(n_channels, n_classes)
    elif model_name == 'unet++' or model_name == 'unetplusplus':
        return UNetPlusPlus(n_channels, n_classes)
    elif model_name == 'deeplabv3':
        return DeepLabV3Wrapper(n_channels, n_classes, pretrained=pretrained)
    elif model_name in {'ba_deeplabv3', 'boundary_aware_deeplabv3', 'novel_deeplabv3'}:
        return BoundaryAwareDeepLabV3(
            n_channels,
            n_classes,
            pretrained=pretrained,
            use_boundary_head=kwargs.get("use_boundary_head", True),
            use_uncertainty_head=kwargs.get("use_uncertainty_head", True),
        )
    elif model_name in {'swin_unet', 'swinunet', 'swin_unet_lite'}:
        return SwinUNetLite(n_channels, n_classes, pretrained=pretrained, img_size=img_size)
    elif model_name in {'segformer', 'segformer_lite'}:
        return SegFormerLite(n_channels, n_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
