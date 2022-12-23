# adapted from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
# adapted from https://github.com/himashi92/VT-UNet/blob/main/version_1/vtunet/vt_unet.py


import torch
import operator
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from einops import rearrange
from functools import reduce
from timm.models.layers import DropPath


MIN_FEATURE_MAP_SIZE = 4


# ------------------------------------------------------------------------------
# Patch Operation


class PatchMerging(nn.Module):

    def __init__(self,
        input_resolution,
        input_dims,
        dim_scale      = 2,
        resample_scale = 2,
        norm_layer     = nn.LayerNorm
    ):
        super(PatchMerging, self).__init__()

        self.input_dims = input_dims
        self.input_resolution = input_resolution
        D, H, W = self.input_resolution

        assert resample_scale == 2, 'only support resample_scale is 2'
        self.merge_D = (D % 2 == 0) and (D > MIN_FEATURE_MAP_SIZE)
        self.merge_H = (H % 2 == 0) and (H > MIN_FEATURE_MAP_SIZE)
        self.merge_W = (W % 2 == 0) and (W > MIN_FEATURE_MAP_SIZE)

        input_dims_tmp = input_dims
        self.resample_scale = [1, 1, 1]
        if self.merge_D:
            input_dims_tmp *= 2
            self.resample_scale[0] = 2
        if self.merge_H:
            input_dims_tmp *= 2
            self.resample_scale[1] = 2
        if self.merge_W:
            input_dims_tmp *= 2
            self.resample_scale[2] = 2

        self.output_resolution = [
            i // s for i, s in
            zip(input_resolution, self.resample_scale)
        ]

        self.norm = norm_layer(input_dims_tmp)
        self.output_dims = dim_scale * input_dims
        self.reduction = nn.Linear(input_dims_tmp, self.output_dims, bias=False)

    def forward(self, x):
        # x: (B, D, H, W, C)

        if self.merge_D and self.merge_H and self.merge_W:
            out = torch.cat([
                x[:, 0::2, 0::2, 0::2, :],
                x[:, 1::2, 0::2, 0::2, :],
                x[:, 0::2, 1::2, 0::2, :],
                x[:, 0::2, 0::2, 1::2, :],
                x[:, 1::2, 1::2, 0::2, :],
                x[:, 1::2, 0::2, 1::2, :],
                x[:, 0::2, 1::2, 1::2, :],
                x[:, 1::2, 1::2, 1::2, :],
            ], dim=-1)  # x: (B, D//2, H//2, W//2, 8C)
        elif (not self.merge_D) and self.merge_H and self.merge_W:
            out = torch.cat([
                x[:, :, 0::2, 0::2, :],
                x[:, :, 1::2, 0::2, :],
                x[:, :, 0::2, 1::2, :],
                x[:, :, 1::2, 1::2, :],
            ], dim=-1)  # x: (B, D, H//2, W//2, 4C)
        elif self.merge_D and (not self.merge_H) and self.merge_W:
            out = torch.cat([
                x[:, 0::2, :, 0::2, :],
                x[:, 1::2, :, 0::2, :],
                x[:, 0::2, :, 1::2, :],
                x[:, 1::2, :, 1::2, :],
            ], dim=-1)  # x: (B, D//2, H, W//2, 4C)
        elif self.merge_D and self.merge_H and (not self.merge_W):
            out = torch.cat([
                x[:, 0::2, 0::2, :, :],
                x[:, 1::2, 0::2, :, :],
                x[:, 0::2, 1::2, :, :],
                x[:, 1::2, 1::2, :, :],
            ], dim=-1)  # x: (B, D//2, H//2, W, 4C)
        elif (not self.merge_D) and (not self.merge_H) and self.merge_W:
            out = torch.cat([
                x[:, :, :, 0::2, :],
                x[:, :, :, 1::2, :],
            ], dim=-1)  # x: (B, D, H, W//2, 2C)
        elif (not self.merge_D) and self.merge_H and (not self.merge_W):
            out = torch.cat([
                x[:, :, 0::2, :, :],
                x[:, :, 1::2, :, :],
            ], dim=-1)  # x: (B, D, H//2, W, 2C)
        elif self.merge_D and (not self.merge_H) and (not self.merge_W):
            out = torch.cat([
                x[:, 0::2, :, :, :],
                x[:, 1::2, :, :, :],
            ], dim=-1)  # x: (B, D//2, H, W, 2C)
        else:
            pass

        out = self.norm(out)
        out = self.reduction(out)
        # out: (B, D*, H*, W*, 2C)

        return out


class PatchExpanding(nn.Module):

    def __init__(self,
        input_dims,
        input_resolution,
        dim_scale      = 2,
        resample_scale = 2,
        norm_layer     = nn.LayerNorm
    ):
        super(PatchExpanding, self).__init__()

        self.dim_scale = dim_scale
        self.input_resolution = input_resolution

        if isinstance(resample_scale, list):
            assert all(r in [1, 2] for r in resample_scale)
            self.resample_scale = resample_scale
        elif isinstance(resample_scale, int):
            assert resample_scale in [1, 2]
            self.resample_scale = [resample_scale] * 3
        self.rD, self.rH, self.rW = self.resample_scale

        self.output_resolution = [
            int(i * s) for i, s in
            zip(input_resolution, [self.rD, self.rH, self.rW])
        ]

        shape_scale = reduce(operator.mul, self.resample_scale, 1)
        self.new_dims = shape_scale * input_dims // dim_scale
        self.expand = nn.Linear(input_dims, self.new_dims, bias=False)
        self.output_dims = input_dims // dim_scale
        self.norm = norm_layer(self.output_dims)

    def forward(self, x):
        # x: (B, D, H, W, C)

        out = self.expand(x)
        # out: [B, D, H, W, self.new_dims]
        out = rearrange(
            out,
            'b d h w (rd rh rw c) -> b (d rd) (h rh) (w rw) c',
            rd=self.rD, rh=self.rH, rw=self.rW, c=self.output_dims
        )
        out = self.norm(out)
        # out: [B, D*rD, H*rH, W*rW, self.output_dim]

        return out


class PatchEmbedding(nn.Module):

    def __init__(self,
        image_size,
        patch_size,
        input_dims,
        embed_dims,
        norm_layer = None
    ):
        super(PatchEmbedding, self).__init__()

        self.patches_resolution = [i // p for i, p in zip(image_size, patch_size)]
        self.num_patches = reduce(operator.mul, self.patches_resolution, 1)

        self.proj = nn.Conv3d(
            input_dims, embed_dims,
            kernel_size=patch_size, stride=patch_size
        )

        self.norm = None
        if norm_layer is not None:
            self.norm = norm_layer(embed_dims)

    def forward(self, x):
        # x: (B, C, D, H, W)

        out = self.proj(x)
        # out: (B, C, D//pD, H//pH, W//pW)
        if self.norm is not None:
            B, E, nD, nH, nW = out.size()
            out = out.flatten(2).transpose(1, 2)
            out = self.norm(out)
            # out: (B, nD*nH*nW, E)
            out = out.transpose(1, 2).view(B, E, nD, nH, nW)
            # out: (B, E, nD, nH, nW)

        return out


class PatchReversing(nn.Module):

    def __init__(self,
        patch_size,
        input_dims,
        output_dims,
        norm_layer = nn.LayerNorm,
        act_layer  = nn.GELU
    ):
        super(PatchReversing, self).__init__()

        self.reverse_proj = nn.ConvTranspose3d(
            input_dims, input_dims,
            kernel_size=patch_size, stride=patch_size
        )

        self.norm = None
        if norm_layer is not None:
            self.norm = norm_layer(input_dims)
        
        self.act = None
        if act_layer is not None:
            self.act = act_layer()

        self.out_conv = nn.Conv3d(
            input_dims, output_dims,
            kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        # x: (B, C, D, H, W)

        out = self.reverse_proj(x)
        # out: (B, C, D*pD, H*pH, w*pW)

        if self.norm is not None:
            B, C, D, H, W = out.size()
            out = out.flatten(2).transpose(1, 2)
            out = self.norm(out)
            # out: (B, D*H*W, C)
            out = out.transpose(1, 2).view(B, C, D, H, W)
            # out: (B, C, D, H, W)

        if self.act is not None:
            out = self.act(out)

        out = self.out_conv(out)

        return out


# ------------------------------------------------------------------------------
# Window Operation


def window_partition(x, window_size):
    '''

    Args:
        x: (B, D, H, W, C)
        window_size (tuple): (wD, wH, wW)
    Returns:
        windows: (B*(D//wD)*(H//wH)*(W//wW), wD, wH, wW, C)

    '''

    B, D, H, W, C = x.size()
    wD, wH, wW = window_size

    x = x.view(
        B,
        D // wD, wD,
        H // wH, wH,
        W // wW, wW,
        C
    )

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wD, wH, wW, C)

    return windows


def window_reverse(windows, window_size, D, H, W):
    '''

    Args:
        windows: (B*(D//wD)*(H//wH)*(W//wW), wD, wH, wW, C)
        window_size (tuple): (wD, wH, wW)
        D (int): depths of image
        H (int): height of image
        W (int): width of image
    Returns:
        out: (B, D, H, W, C)

    '''

    wD, wH, wW = window_size
    nD, nH, nW = D // wD, H // wH, W // wW
    B = windows.size(0) // (nD * nH * nW)
    out = windows.view(B, nD, nH, nW, wD, wH, wW, -1)
    out = out.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    out = out.view(B, D, H, W, -1)

    return out


# ------------------------------------------------------------------------------
# Window Attenton Block


class WindowAttentionV2(nn.Module):

    def __init__(self,
        input_dims,
        num_heads,
        window_size,
        qkv_bias         = True,
        attn_drop        = 0.0,
        proj_drop        = 0.0,
        apply_cross_attn = False
    ):
        super(WindowAttentionV2, self).__init__()

        self.num_heads = num_heads
        self.wD, self.wH, self.wW = window_size

        scale_params = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(scale_params, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_table = torch.stack(
            torch.meshgrid([
                torch.arange(-(self.wD - 1), self.wD, dtype=torch.float32),
                torch.arange(-(self.wH - 1), self.wH, dtype=torch.float32),
                torch.arange(-(self.wW - 1), self.wW, dtype=torch.float32),
            ], indexing='ij')
        ).permute(1, 2, 3, 0).contiguous().unsqueeze(0)
        # relative_coords_table: (1, 2*wD-1, 2*wH-1, 2*wW-1, 3)
        relative_coords_table[:, :, :, :, 0] /= 1 if self.wD == 1 else self.wD - 1
        relative_coords_table[:, :, :, :, 1] /= 1 if self.wH == 1 else self.wH - 1
        relative_coords_table[:, :, :, :, 2] /= 1 if self.wW == 1 else self.wW - 1
        relative_coords_table *= 8  # normalize to -8 to 8
        relative_coords_table = \
            torch.sign(relative_coords_table) * \
            torch.log2(torch.abs(relative_coords_table) + 1.0) / \
            np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords = torch.stack(
            torch.meshgrid([
                torch.arange(self.wD),
                torch.arange(self.wH),
                torch.arange(self.wW),
            ], indexing='ij')
        )  # coords: (3, wD, Wh, wW)
        coords = torch.flatten(coords, 1)
        # coords: (3, wD*wH*wW)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        # (3, wD*wH*wW, wD*wH*wW)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # (wD*wH*wW, wD*wH*wW, 3)
        relative_coords[:, :, 0] += self.wD - 1
        relative_coords[:, :, 1] += self.wH - 1
        relative_coords[:, :, 2] += self.wW - 1
        relative_coords[:, :, 0] *= (2 * self.wH - 1) * (2 * self.wW - 1)
        relative_coords[:, :, 1] *= 2 * self.wW - 1
        relative_position_index = relative_coords.sum(-1)
        # relative_position_index: (wD*wH*wW, wD*wH*wW)
        relative_position_index = relative_position_index.view(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.apply_cross_attn = apply_cross_attn
        if not apply_cross_attn:
            self.qkv = nn.Linear(input_dims, input_dims * 3, bias=qkv_bias)
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(input_dims))
                self.v_bias = nn.Parameter(torch.zeros(input_dims))
            else:
                self.q_bias = None
                self.v_bias = None

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(input_dims, input_dims)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, cross_qkv=None):
        '''

        Args:
            x: (nD*nH*nW*B, wD*wH*wW, C)
            mask: (nD*nH*nW, wD*wH*wW, wD*wH*wW)

        '''

        # B_ == nD*nH*nW*B, N == wD*wH*wW
        B_, N, C = x.size()
        device = x.device

        if not self.apply_cross_attn:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat([
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias
                ])
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B_, N, 3, self.num_heads, -1)
            # qkv: (B_, wD*wH*wW, 3, num_heads, C)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            # qkv: (3, B_, num_heads, wD*wH*wW, C)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # q, k, v: (B_, num_heads, wD*wH*wW, C)
        else:
            assert cross_qkv is not None, 'cross_qkv was not found'
            q, k, v = cross_qkv[0], cross_qkv[1], cross_qkv[2]
            # q, k, v: (B_, num_heads, wD*wH*wW, C)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        max_scale = torch.log(torch.tensor(1.0 / 0.01, device=device))
        logit_scale = torch.clamp(self.logit_scale, max=max_scale).exp()
        attn = attn * logit_scale
        # attn: (B_, num_heads, wD*wH*wW, wD*wH*wW)

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index].view(
            self.wD * self.wH * self.wW, self.wD * self.wH * self.wW, -1
        )  # relative_position_bias: (wD*wH*wW, wD*wH*wW, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        # relative_position_bias: (num_heads, wD*wH*wW, wD*wH*wW)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mN = mask.shape[0]
            attn = attn.view(B_ // mN, mN, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # attn: (B_, num_heads, wD*wH*wW, wD*wH*wW)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        # out: (B_, wD*wH*wW, C)

        return out, qkv


# ------------------------------------------------------------------------------
# Basic Layer Block


class Mlp(nn.Module):

    def __init__(self,
        input_dims,
        hidden_dims = None,
        output_dims = None,
        act_layer   = nn.GELU,
        drop        = 0.0
    ):
        super(Mlp, self).__init__()

        hidden_dims = hidden_dims or input_dims
        output_dims = output_dims or input_dims

        self.act  = act_layer()
        self.drop = nn.Dropout(p=drop)
        self.fc1  = nn.Linear(input_dims, hidden_dims)
        self.fc2  = nn.Linear(hidden_dims, output_dims)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, channels):
        super(PositionalEncoding, self).__init__()

        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        if len(tensor.shape) != 5:
            raise RuntimeError('The input tensor has to be 5d!')

        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        sin_inp_y = torch.einsum('i,j->ij', pos_y, self.inv_freq)
        sin_inp_z = torch.einsum('i,j->ij', pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class SwinTransformerBlock(nn.Module):

    def __init__(self,
        input_dims,
        input_resolution,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        drop             = 0.0,
        attn_drop        = 0.0,
        drop_path        = 0.0,
        act_layer        = nn.GELU,
        norm_layer       = nn.LayerNorm,
        apply_cross_attn = False,
        apply_fpe        = False
    ):
        super(SwinTransformerBlock, self).__init__()

        # check if window_size and shift_size are valid
        new_window_size = deepcopy(window_size)
        new_shift_size = deepcopy(shift_size)
        for i in range(len(window_size)):
            assert 0 <= shift_size[i] < window_size[i]
            if input_resolution[i] <= window_size[i]:
                # window_size[i] is larger than or equal to input_resolution[i]
                if new_shift_size[i] != 0:
                    # disable shift
                    new_shift_size[i] = 0
                # dimention i only has one window
                new_window_size[i] = input_resolution[i]
        window_size = new_window_size
        shift_size = new_shift_size

        D, H, W = input_resolution
        wD, wH, wW = window_size
        sD, sH, sW = shift_size

        # check if window_size is valid
        wD = wD if D % wD == 0 else 1
        wH = wH if H % wH == 0 else 1
        wW = wW if W % wW == 0 else 1
        window_size = [wD, wH, wW]

        self.input_resolutiuon = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn = WindowAttentionV2(
            input_dims,
            num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # cross attention
        self.apply_fpe = False
        self.apply_cross_attn = apply_cross_attn
        if self.apply_cross_attn:
            self.apply_fpe = apply_fpe
            self.cross_attn = WindowAttentionV2(
                input_dims,
                num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                apply_cross_attn=apply_cross_attn
            )

        mlp_hidden_dims = int(input_dims * mlp_ratio)
        self.mlp = Mlp(
            input_dims=input_dims,
            hidden_dims=mlp_hidden_dims,
            output_dims=input_dims,
            act_layer=act_layer,
            drop=drop
        )

        self.norm1 = norm_layer(input_dims)
        self.norm2 = norm_layer(input_dims)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if any([s > 0 for s in shift_size]):
            self.apply_shift = True

            image_mask = torch.zeros((1, D, H, W, 1))
            d_slices = (slice(0, -wD), slice(-wD, -sD), slice(-sD, None))
            h_slices = (slice(0, -wH), slice(-wH, -sH), slice(-sH, None))
            w_slices = (slice(0, -wW), slice(-wW, -sW), slice(-sW, None))

            count = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        image_mask[:, d, h, w, :] = count
                        count += 1

            mask_windows = window_partition(image_mask, window_size)
            # mask_windows: (B*(D//wD)*(H//wH)*(W//wW), wD, wH, wW, 1)
            mask_windows = mask_windows.view(-1, wD * wH * wW)
            # mask_windows: (B*(D//wD)*(H//wH)*(W//wW), wD*wH*wW)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
            # attn_mask: (B*(D//wD)*(H//wH)*(W//wW), wD*wH*wW, wD*wH*wW)
        else:
            self.apply_shift = False
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x, prev_qkv=None):

        D, H, W = self.input_resolutiuon
        wD, wH, wW = self.window_size
        sD, sH, sW = self.shift_size

        B, D, H, W, C = x.shape
        shortcut = x

        # cyclic shift
        if self.apply_shift:
            shifted = torch.roll(x, shifts=(-sD, -sH, -sW), dims=(1, 2, 3))
        else:
            shifted = x

        # partition windows
        windows = window_partition(shifted, self.window_size)
        # windows: (B*(D//wD)*(H//wH)*(W//wW), wD, wH, wW, C)
        windows = windows.view(-1, wD * wH * wW, C)
        # windows: (B*(D//wD)*(H//wH)*(W//wW), wD*wH*wW, C)

        # W-MSA / SW-MSA
        attn_windows, qkv = self.attn(windows, mask=self.attn_mask)
        # attn_windows: (B*(D//wD)*(H//wH)*(W//wW), wD*wH*wW, C)

        # merge windows
        attn_windows = attn_windows.view(-1, wD, wH, wW, C)
        # attn_windows: (B*(D//wD)*(H//wH)*(W//wW), wD, wH, wW, C)
        shifted = window_reverse(attn_windows, self.window_size, D, H, W)
        # shifted: (B, D, H, W, C)

        # reverse cyclic shift
        out = torch.roll(shifted, shifts=(sD, sH, sW), dims=(1, 2, 3)) if self.apply_shift else shifted

        out = shortcut + self.drop_path(self.norm1(out))
        out = out + self.drop_path(self.norm2(self.mlp(out)))
        # out: (B, D, H, W, C)

        # cross attention
        if self.apply_cross_attn:
            assert prev_qkv is not None, 'prev_qkv was not found'
            cross_qkv = torch.stack([qkv[0], prev_qkv[1], prev_qkv[2]])
            cross_attn_windows, _ = self.attn(windows, mask=self.attn_mask, cross_qkv=cross_qkv)
            # cross_attn_windows: (B*(D//wD)*(H//wH)*(W//wW), wD*wH*wW, C)
            cross_attn_windows = cross_attn_windows.view(-1, wD, wH, wW, C)
            cross_shifted = window_reverse(cross_attn_windows, self.window_size, D, H, W)
            cross_out = torch.roll(cross_shifted, shifts=(sD, sH, sW), dims=(1, 2, 3)) if self.apply_shift else shifted

            cross_out = shortcut + self.drop_path(self.norm1(cross_out))
            cross_out = cross_out + self.drop_path(self.norm2(self.mlp(cross_out)))
            out = 0.5 * out + 0.5 * cross_out

            if self.apply_fpe:
                # fourier feature positional encoding
                FPE = PositionalEncoding(out.shape[4])
                pos_embed = self.norm2(self.mlp(FPE(out)))
                out = out + pos_embed

        return out, qkv


class BasicLayer(nn.Module):

    def __init__(self,
        input_dims,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        drop             = 0.0,
        attn_drop        = 0.0,
        drop_path        = 0.0,
        norm_layer       = nn.LayerNorm,
        skip_connect     = False,
        apply_cross_attn = False,
        apply_fpe        = False
    ):
        super(BasicLayer, self).__init__()

        self.skip_connect = skip_connect
        self.apply_cross_attn = apply_cross_attn

        # original input
        self.resample = None
        self.resample_first = False
        self.output_dims = input_dims
        self.resample_scale = [1, 1, 1]
        self.output_resolution = deepcopy(input_resolution)

        if self.skip_connect:
            # channel reduction for cat(input, skip)
            self.cat_mlp = nn.Linear(input_dims * 2, input_dims, bias=False)

        # build blocks
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            shift_size = [0, 0, 0] if i % 2 == 0 else [w // 2 for w in window_size]
            cur_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path

            block = SwinTransformerBlock(
                input_dims       = input_dims,
                input_resolution = input_resolution,
                num_heads        = num_heads,
                window_size      = window_size,
                shift_size       = shift_size,
                mlp_ratio        = mlp_ratio,
                qkv_bias         = qkv_bias,
                drop             = drop,
                attn_drop        = attn_drop,
                drop_path        = cur_drop_path,
                norm_layer       = norm_layer,
                apply_cross_attn = apply_cross_attn,
                apply_fpe        = apply_fpe
            )

            self.blocks.append(block)

        # update window size and shift size after checking in block initialization
        self.window_size = block.window_size
        self.shift_size = block.shift_size

    def forward(self, x, skip=None, prev_qkv_list=None):
        if self.apply_cross_attn:
            assert prev_qkv_list is not None, 'prev_qkv_list was not found'
            assert len(prev_qkv_list) == len(self.blocks), \
                'lenght of prev_qkv_list is not same as blocks'

        out = x
        if self.skip_connect:
            assert skip is not None, 'skip input was not found'
            # skip = rearrange(skip, 'b c d h w -> b d h w c')
            out = torch.cat([out, skip], dim=-1)
            out = self.cat_mlp(out)

        qkv_list = []
        for i, block in enumerate(self.blocks):
            prev_qkv = prev_qkv_list[i] if self.apply_cross_attn else None
            out, qkv = block(out, prev_qkv=prev_qkv)
            qkv_list.append(qkv)

        return out, qkv_list
