from .vt2unet import VT2UNet3D


def define_model(configs):
    
    if configs.name == 'vt2unet':
        model = VT2UNet3D(**configs.params)
    else:
        raise NotImplementedError(f'unknown model name {configs.name}')

    return model
