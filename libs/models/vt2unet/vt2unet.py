import torch
import torch.nn as nn

from .swin import *
from copy import deepcopy
from einops import rearrange
from timm.models.layers import trunc_normal_
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock


# ------------------------------------------------------------------------------
# VT2UNet


class VT2UNet(nn.Module):

    def __init__(self,
        image_size,
        patch_size,
        window_size,
        input_dims       = 3,
        output_dims      = 1,
        embed_dims       = 96,
        depths           = [2, 2, 2, 2],
        num_heads        = [3, 6, 12, 24],
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        drop_rate        = 0.0,
        attn_drop_rate   = 0.0,
        drop_path_rate   = 0.1,
        norm_layer       = nn.LayerNorm,
        patch_norm       = True,
        skip_connect     = True,
        apply_cross_attn = True
    ):
        super(VT2UNet, self).__init__()
        assert len(depths) == len(num_heads)
        assert len(image_size) == len(patch_size) == len(window_size) == 3
        assert all([embed_dims % h == 0 for h in num_heads])
        assert all([p % 2 == 0 or p == 1 for p in patch_size])
        assert all([i % p == 0 for i, p in zip(image_size, patch_size)])

        self.mlp_ratio = mlp_ratio
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.image_size = image_size
        self.input_dims = input_dims
        self.num_features = int(embed_dims * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(
            image_size = image_size,
            patch_size = patch_size,
            input_dims = input_dims,
            embed_dims = embed_dims,
            norm_layer = norm_layer if patch_norm else None
        )
        self.num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        drop_path_list = [dpr[sum(depths[:i]):sum(depths[:i + 1])]
                          for i in range(self.num_layers)]

        # build layers
        self.encoder     = nn.ModuleList()
        self.decoder     = nn.ModuleList()
        self.upsampler   = nn.ModuleList()
        self.downsampler = nn.ModuleList()

        dim_scale  = 2
        input_resolution = self.patches_resolution
        for ith_layer in range(self.num_layers - 1):
            input_dims = int(embed_dims * 2 ** ith_layer)
            drop_path  = drop_path_list[ith_layer]

            encoder_layer = BasicLayer(
                input_dims       = input_dims,
                input_resolution = input_resolution,
                depth            = depths[ith_layer],
                num_heads        = num_heads[ith_layer],
                window_size      = window_size,
                mlp_ratio        = mlp_ratio,
                qkv_bias         = qkv_bias,
                drop             = drop_rate,
                attn_drop        = attn_drop_rate,
                drop_path        = drop_path,
                norm_layer       = norm_layer
            )
            self.encoder.append(encoder_layer)

            downsample_layer = PatchMerging(
                input_dims       = input_dims,
                input_resolution = input_resolution,
                dim_scale        = dim_scale,
                resample_scale   = 2,
                norm_layer       = norm_layer
            )
            self.downsampler.append(downsample_layer)
            input_resolution = deepcopy(downsample_layer.output_resolution)

        self.middle_layer = BasicLayer(
            input_dims       = downsample_layer.output_dims,
            input_resolution = input_resolution,
            depth            = depths[-1],
            num_heads        = num_heads[-1],
            window_size      = window_size,
            mlp_ratio        = mlp_ratio,
            qkv_bias         = qkv_bias,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = drop_path_list[-1],
            norm_layer       = norm_layer
        )

        input_dims = self.middle_layer.output_dims
        input_resolution = self.middle_layer.output_resolution

        for ith_layer in range(self.num_layers - 1):
            cur_layer = self.num_layers - ith_layer - 2
            window_size = self.encoder[cur_layer].window_size
            resample_scale = self.downsampler[cur_layer].resample_scale

            upsample_layer = PatchExpanding(
                input_dims       = input_dims,
                input_resolution = input_resolution,
                dim_scale        = dim_scale,
                resample_scale   = resample_scale,
                norm_layer       = norm_layer
            )
            self.upsampler.append(upsample_layer)

            decoder_layer = BasicLayer(
                input_dims       = upsample_layer.output_dims,
                input_resolution = upsample_layer.output_resolution,
                depth            = depths[cur_layer],
                num_heads        = num_heads[cur_layer],
                window_size      = window_size,
                mlp_ratio        = mlp_ratio,
                qkv_bias         = qkv_bias,
                drop             = drop_rate,
                attn_drop        = attn_drop_rate,
                drop_path        = drop_path,
                norm_layer       = norm_layer,
                skip_connect     = skip_connect,
                apply_cross_attn = apply_cross_attn
            )
            self.decoder.append(decoder_layer)

            input_dims = decoder_layer.output_dims
            input_resolution = decoder_layer.output_resolution

        self.encoder0 = UnetrBasicBlock(
            spatial_dims = 3,
            in_channels  = self.input_dims,
            out_channels = input_dims,
            kernel_size  = 3,
            stride       = 1,
            norm_name    = 'batch',
            res_block    = True
        )

        self.decoder0 = UnetrUpBlock(
            spatial_dims         = 3,
            in_channels          = input_dims,
            out_channels         = input_dims,
            kernel_size          = 3,
            upsample_kernel_size = patch_size,
            norm_name            = 'batch',
            res_block            = True
        )

        self.out = nn.Sequential(
            UnetOutBlock(
                spatial_dims = 2,
                in_channels  = input_dims,
                out_channels = output_dims
            ),
            nn.Sigmoid()
        )

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'cpb_mlp', 'logits_scale', 'relative_position_bias_table'}

    def forward(self, x):

        out = self.patch_embed(x)
        out = self.pos_drop(out)
        out = rearrange(out, 'b c d h w -> b d h w c')

        # encoder
        skip_list, enc_qkv_list = [], []
        for enc_layer, down_layer in zip(self.encoder, self.downsampler):
            out, qkv_list = enc_layer(out)
            skip_list.insert(0, out)
            enc_qkv_list.insert(0, qkv_list)
            out = down_layer(out)

        # middle
        out, _ = self.middle_layer(out)

        # decoder
        for i, (up_layer, dec_layer) in enumerate(zip(self.upsampler, self.decoder)):
            out = up_layer(out)
            skip = skip_list[i]
            prev_qkv_list = enc_qkv_list[i]
            out, qkv_list = dec_layer(out, skip=skip, prev_qkv_list=prev_qkv_list)

        out = rearrange(out, 'b d h w c -> b c d h w')
        out = self.decoder0(out, self.encoder0(x))
        out = torch.mean(out, dim=2)
        logits = self.out(out)
        return logits


if __name__ == '__main__':

    B = 1
    C = 15
    image_size = [12, 256, 256]
    patch_size = [1, 2, 2]
    window_size = [3, 7, 7]
    x = torch.rand(B, C, 12, 256, 256).cuda()

    model = VT2UNet(
        image_size       = image_size,
        patch_size       = patch_size,
        window_size      = window_size,
        input_dims       = C,
        output_dims      = 1,
        embed_dims       = 24,
        depths           = [2, 2, 2, 2],
        num_heads        = [3, 6, 12, 24],
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        drop_rate        = 0.0,
        attn_drop_rate   = 0.0,
        drop_path_rate   = 0.1,
        norm_layer       = nn.LayerNorm,
        patch_norm       = True,
        skip_connect     = True,
        apply_cross_attn = True
    )
    model.cuda()

    output = model(x)
    print(output.size())
