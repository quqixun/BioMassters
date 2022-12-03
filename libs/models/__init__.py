from .vt2unet import VT2UNet3D
from .swin_unetr import SwinUNETR


def define_model(configs):
    
    if configs.name == 'vt2unet':
        model = VT2UNet3D(**configs.params)
    elif configs.name == 'swin_unetr':
        model = SwinUNETR(**configs.params)
    else:
        raise NotImplementedError(f'unknown model name {configs.name}')

    return model
