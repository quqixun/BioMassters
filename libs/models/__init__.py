from .vt2unet import VT2UNet
from .swin_unetr import SwinUNETR


def define_model(configs):
    
    if configs.name == 'swin_unetr':
        model = SwinUNETR(**configs.params)
    elif configs.name == 'vt2unet':
        model = VT2UNet(**configs.params)
    else:
        raise NotImplementedError(f'unknown model name {configs.name}')

    return model
