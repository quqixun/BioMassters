import torch
import numpy as np
import torch.nn as nn

from .swin import SwinTransformer
from typing import Sequence, Tuple, Union

from monai.utils import ensure_tuple_rep
from monai.networks.blocks import UnetOutBlock


__all__ = ['VTUnet']


class VTUnet(nn.Module):
    '''
    '''

    def __init__(
        self,
        image_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        window_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = 'instance',
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_version: str = 'v1',
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        apply_cross_attn: bool = False
    ) -> None:
        '''
        Args:
            image_size: dimension of input image.
            patch_size: dimension of patch.
            window_size: dimension of window.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: drop path rate.
            attn_version: version of attention in swin transformer.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            apply_cross_attn: if apply cross attention.
        '''

        super().__init__()

        image_size = ensure_tuple_rep(image_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError('spatial dimension should be 2 or 3.')

        for m, p in zip(image_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError('input image size (image_size) should be divisible by stage-wise image resolution.')

        if not (0 <= drop_rate <= 1):
            raise ValueError('dropout rate should be between 0 and 1.')

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError('attention dropout rate should be between 0 and 1.')

        if not (0 <= drop_path_rate <= 1):
            raise ValueError('drop path rate should be between 0 and 1.')

        if feature_size % 12 != 0:
            raise ValueError('feature_size should be divisible by 12.')

        if attn_version not in ['v1', 'v2']:
            raise ValueError('attn_version should be v1 or v2.')

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            image_size=image_size,
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            attn_version=attn_version,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )
        resamples = self.swinViT.resamples

        self.out = nn.Sequential(
            UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        out = self.decoder1(dec0, enc0)
        out = torch.mean(out, dim=2)
        logits = self.out(out)
        return logits


if __name__ == '__main__':

    B = 1
    C = 15
    image_size = (12, 256, 256)
    patch_size = (1, 2, 2)
    window_size = (3, 7, 7)
    x = torch.rand(B, C, 12, 256, 256).cuda()

    model = SwinUNETR(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_channels=C,
        out_channels=1,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name='batch',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version='v2',
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3
    )
    model.cuda()

    output = model(x)
    print(output.size())
