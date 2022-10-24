# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------

import math

import numpy as np
from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore import ops, nn, context, Tensor

from src.models.initializer import trunc_normal_, zeros_, ones_, kaiming_uniform_
from src.models.initializer import uniform_, _calculate_fan_in_and_fan_out
from src.models.layers.drop_path import DropPath1D
from src.models.layers.identity import Identity


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Cell):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        # 加速, has_bias=True的时候会很慢
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, group=dim, has_bias=False)
        self.get_v_bias = Parameter(Tensor(np.zeros(dim), mstype.float32))
        # self.get_v = Identity()
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.batch_matmul = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)
        self.unstack = ops.Unstack(axis=0)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = self.resolution
        x = x.transpose(0, 2, 1).ravel().reshape(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        return x

    def get_lepe(self, x):
        B, N, C = x.shape
        H = W = self.resolution
        x = x.transpose(0, 2, 1).ravel().reshape(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.reshape(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.transpose(0, 2, 4, 1, 3, 5).ravel().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
        lepe = self.get_v(x)  ### B', C, H', W'

        # print(lepe.shape)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).transpose(0, 1, 3, 2)
        get_v_bias = self.get_v_bias.reshape(1, self.num_heads, 1, -1)
        lepe = lepe + get_v_bias
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).transpose(0, 1, 3, 2)
        return x, lepe

    def construct(self, qkv):
        """
        x: B L C
        """
        q, k, v = self.unstack(qkv)

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)
        attn = self.batch_matmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = ops.BatchMatMul()(attn, v) + lepe
        x = x.transpose(0, 2, 1, 3).ravel().reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).reshape(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Cell):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.norm1 = norm_layer((dim,))

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - drop)

        if last_stage:
            self.attns0 = LePEAttention(
                dim, resolution=self.patches_resolution, idx=-1,
                split_size=split_size, num_heads=num_heads, dim_out=dim,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        else:
            self.attns0 = LePEAttention(
                dim // 2, resolution=self.patches_resolution, idx=0,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.attns1 = LePEAttention(
                dim // 2, resolution=self.patches_resolution, idx=1,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer((dim,))
        self.unstack = ops.Unstack(axis=-2)

    def construct(self, x):
        """
        x: B, H*W, C
        """

        B, L, C = x.shape
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).transpose(2, 0, 1, 3)

        if self.branch_num == 2:
            B, C, H, W = qkv.shape
            qkv = qkv.reshape(B, C, H, 2, -1)
            x1, x2 = self.unstack(qkv)
            x1 = self.attns0(x1)
            x2 = self.attns1(x2)
            # x1 = self.attns[0](qkv[:, :, :, :C // 2])
            # x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = ops.Concat(2)((x1, x2))
        else:
            attened_x = self.attns0(qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.reshape(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.transpose(0, 2, 4, 3, 5, 1).ravel().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = img_splits_hw.shape[0] // (H * W // H_sp // W_sp)

    img = img_splits_hw.reshape(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.transpose(0, 1, 3, 2, 4, 5).ravel().reshape(B, H, W, -1)
    return img


class Merge_Block(nn.Cell):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm, resolutio=224):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, has_bias=True)
        self.norm = norm_layer((dim_out,))
        self.resolutio = resolutio

    def construct(self, x):
        B, new_HW, C = x.shape
        x = x.transpose(0, 2, 1).ravel().reshape(B, C, self.resolutio, self.resolutio)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.reshape(B, C, -1).transpose(0, 2, 1)
        x = self.norm(x)

        return x


class CSWinTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
                 split_size=[3, 5, 7], num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads
        self.stage1_conv_embed = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=7, stride=4,
                                           pad_mode='same', has_bias=True)
        self.stage1_conv_norm = nn.LayerNorm((embed_dim,), epsilon=1e-05)

        curr_dim = embed_dim
        dpr = [x for x in np.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.SequentialCell([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2, resolutio=img_size // 4)

        curr_dim = curr_dim * 2
        self.stage2 = nn.SequentialCell(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2, resolutio=img_size // 8)
        curr_dim = curr_dim * 2
        self.stage3 = nn.SequentialCell([CSWinBlock(
            dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2, resolutio=img_size // 16)
        curr_dim = curr_dim * 2
        self.stage4 = nn.SequentialCell(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.norm = norm_layer((curr_dim,))
        # Classifier head
        self.head = nn.Dense(curr_dim, num_classes) if num_classes > 0 else Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            ones_(m.gamma)
            zeros_(m.beta)
        elif isinstance(m, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            kaiming_uniform_(m.weight)
            if m.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    uniform_(m.bias, bound)

    def apply(self, func):
        for _, cell in self.cells_and_names():
            func(cell)

    def construct_features(self, x):
        x = self.stage1_conv_embed(x)
        B, C = x.shape[:2]
        x = ops.Reshape()(x, (B, C, -1))
        x = ops.Transpose()(x, (0, 2, 1))
        x = self.stage1_conv_norm(x)
        x = self.stage1(x)
        x = self.merge1(x)
        x = self.stage2(x)

        x = self.merge2(x)
        x = self.stage3(x)

        x = self.merge3(x)
        x = self.stage4(x)

        x = self.norm(x)
        return ops.ReduceMean()(x, 1)

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


### 224 models

def CSWin_64_12211_tiny_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    return model


def CSWin_64_24322_small_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    return model


def CSWin_96_24322_base_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    return model


def CSWin_144_24322_large_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    return model


### 384 models

def CSWin_96_24322_base_384(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    return model


def CSWin_144_24322_large_384(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    return model


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([1, 3, 224, 224]), dtype=mstype.float32)
    model = CSWin_96_24322_base_224()  # 77382184
    out = model(data)
    print(out.shape)
    params = 0.
    num = 0.
    for name, param in model.parameters_and_names():
        params += np.prod(param.shape)
        print(name, param.shape)
        num += 1
    print(params, num)
    # assert params == 22320552
    # model.stage1_conv_embed.weight
