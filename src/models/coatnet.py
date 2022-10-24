import collections.abc
import math
import os
from itertools import repeat

import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, context, Tensor, ops, Parameter, load_checkpoint, load_param_into_net

from src.models.initializer import trunc_normal_, zeros_, ones_, kaiming_uniform_
from src.models.initializer import uniform_, _calculate_fan_in_and_fan_out
from src.models.layers.drop_path import DropPath2D
from src.models.layers.identity import Identity

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Swish(nn.Cell):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


class SE(nn.Cell):
    def __init__(self, output_filters, se_filters):
        super().__init__()
        self.fc = nn.SequentialCell([
            nn.Conv2d(in_channels=output_filters, out_channels=se_filters, kernel_size=1, has_bias=True),
            Swish(),
            nn.Conv2d(in_channels=se_filters, out_channels=output_filters, kernel_size=1, has_bias=True),
            nn.Sigmoid()
        ])

    def construct(self, x):
        y = ops.ReduceMean(True)(x, (2, 3))
        y = self.fc(y)
        return x * y


class FFN(nn.Cell):
    def __init__(self, input_size, hidden_size, expansion_rate=4, dropout=0.):
        super().__init__()
        self.net = nn.SequentialCell([
            nn.Conv2d(in_channels=input_size, kernel_size=1, pad_mode='pad', has_bias=True,
                      out_channels=int(hidden_size * expansion_rate)),
            nn.GELU(),
            nn.Dropout(keep_prob=1.0 - dropout),
            nn.Conv2d(out_channels=hidden_size, kernel_size=1, pad_mode='pad', has_bias=True,
                      in_channels=int(hidden_size * expansion_rate)),
        ])

    def construct(self, x):
        return self.net(x)


class Attention(nn.Cell):
    def __init__(self, image_size, input_size, hidden_size, head_size, num_heads=None,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads or hidden_size // head_size
        self.head_size = head_size
        self.height, self.width = image_size
        self.scale = head_size ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        # define a parameter table of relative position bias
        self.relative_position_bias_table = Parameter(
            Tensor(np.zeros([(2 * self.height - 1) * (2 * self.width - 1), self.num_heads]),
                   mstype.float32))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.height)
        coords_w = np.arange(self.width)
        coords = np.stack(np.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.height - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.width - 1
        relative_coords[:, :, 0] *= 2 * self.width - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.relative_position_index = Parameter(Tensor(relative_position_index.reshape(-1), mstype.int32),
                                                 requires_grad=False)

        # self.one_hot = nn.OneHot(axis=-1, depth=(2 * self.height - 1) * (2 * self.width - 1),
        #                          dtype=mstype.float32)

        self.attend = nn.Softmax(axis=-1)
        self.qkv = nn.Dense(in_channels=input_size, has_bias=qkv_bias,
                            out_channels=int(self.num_heads * self.head_size * 3))

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(out_channels=hidden_size, in_channels=int(self.num_heads * self.head_size))
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()
        self.unstack = ops.Unstack(axis=0)

    def construct(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, C, H, W = x.shape
        x = ops.Transpose()(ops.Reshape()(x, (B_, C, -1,)), (0, 2, 1))
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_size).transpose(2, 0, 3, 1, 4)
        q, k, v = self.unstack(qkv)

        q = q * self.scale
        attn = ops.BatchMatMul()(q, k.transpose(0, 1, 3, 2))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
        #     self.height * self.width, self.height * self.width, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = ops.Gather()(self.relative_position_bias_table,
                                              self.relative_position_index.reshape(-1), 0)
        relative_position_bias = relative_position_bias.reshape(self.height * self.width, self.height * self.width, -1)
        relative_position_bias = ops.Transpose()(relative_position_bias, (2, 0, 1,))  # nH, Wh*Ww, Wh*Ww

        # one_hot_index = ops.Cast()(self.one_hot(self.relative_position_index), mstype.float16)
        # relative_position_bias_table = ops.Cast()(self.relative_position_bias_table, mstype.float16)
        # relative_position_bias = ops.MatMul()(one_hot_index, relative_position_bias_table)
        # relative_position_bias = relative_position_bias.reshape(
        #     self.height * self.width, self.height * self.width, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = ops.Transpose()(relative_position_bias, (2, 0, 1,))  # nH, Wh*Ww, Wh*Ww

        attn = attn + ops.ExpandDims()(relative_position_bias, 0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = self.batch_matmul(attn, v).transpose(0, 2, 1, 3).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = ops.Reshape()(ops.Transpose()(x, (0, 2, 1,)), (B_, -1, H, W,))
        return x


class TransformerBlock(nn.Cell):
    def __init__(self, image_size, input_size, hidden_size, head_size, num_heads=None, dropout=0., expansion_rate=4,
                 stride=1, drop_path_rate=0.):
        super(TransformerBlock, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.stride = stride
        self.dropout = dropout
        # for shortcut
        if self.stride != 1:
            self._downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=stride, pad_mode='valid')
        else:
            self._downsample = Identity()

        if input_size != hidden_size:
            self._shortcut_proj = nn.Dense(in_channels=input_size, out_channels=hidden_size)
        else:
            self._shortcut_proj = Identity()

        self._attn_layer_norm = LayerNorm(normalized_shape=input_size, epsilon=1e-05)
        self._attention = Attention(self.image_size, input_size, hidden_size, head_size, num_heads=num_heads)
        self._ffn_layer_norm = BatchNorm2d(num_features=hidden_size, momentum=0.9)
        self._ffn = FFN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, expansion_rate=expansion_rate)
        self._drop_path = DropPath2D(drop_prob=drop_path_rate)
        self._drop_out = nn.Dropout(keep_prob=1 - self.dropout)

    def shortcut_branch(self, shortcut):
        shortcut = self._downsample(shortcut)
        B, C, H, W = shortcut.shape
        shortcut = ops.Transpose()(ops.Reshape()(shortcut, (B, C, -1,)), (0, 2, 1,))
        shortcut = self._shortcut_proj(shortcut)
        shortcut = ops.Reshape()(ops.Transpose()(shortcut, (0, 2, 1,)), (B, -1, H, W,))
        return shortcut

    def attn_branch(self, inputs):
        B, C, H, W = inputs.shape
        inputs = ops.Transpose()(ops.Reshape()(inputs, (B, C, -1,)), (0, 2, 1,))
        output = self._attn_layer_norm(inputs)
        output = ops.Reshape()(ops.Transpose()(output, (0, 2, 1,)), (B, C, H, W,))
        output = self._downsample(output)
        output = self._attention(output)
        return output

    def ffn_branch(self, inputs):
        output = self._ffn_layer_norm(inputs)
        output = self._ffn(output)
        return output

    def construct(self, x):

        shortcut = self.shortcut_branch(x)
        output = self.attn_branch(x)
        if self.dropout > 0:
            output = self._drop_out(output)
        output = shortcut + self._drop_path(output)

        shortcut = output
        output = self.ffn_branch(output)
        if self.dropout > 0:
            output = self._drop_out(output)
        output = shortcut + self._drop_path(output)
        return output


class MBConvBlock(nn.Cell):
    def __init__(self, input_size, hidden_size, stride=1, expansion_rate=4, se_ratio=0.25, drop_path_rate=0.):
        super(MBConvBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        inner_size = int(hidden_size * expansion_rate)
        self.stride = stride
        self._activation_fn = nn.GELU()

        if self.stride != 1:
            self._downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=stride, pad_mode='valid')
        else:
            self._downsample = Identity()

        # shortcut
        if input_size != hidden_size:
            self._shortcut_conv = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=1,
                                            pad_mode='pad', has_bias=True)
        else:
            self._shortcut_conv = Identity()
        # Pre-Activation norm
        self._pre_norm = LayerNorm(input_size, dims=2)
        if self.expansion_rate:
            self._expand_conv = nn.Conv2d(in_channels=input_size, out_channels=inner_size, kernel_size=1, stride=stride,
                                          pad_mode='pad', has_bias=False)
            self._expand_norm = LayerNorm(inner_size, dims=2)

        # Depth-wise convolution phase.
        self._depthwise_conv = nn.Conv2d(in_channels=inner_size, out_channels=inner_size, kernel_size=3, stride=1,
                                         pad_mode='pad', padding=1, group=inner_size, has_bias=False)
        self._depthwise_norm = LayerNorm(inner_size, dims=2)

        se_filters = max(1, int(hidden_size * se_ratio))
        self._se = SE(se_filters=se_filters, output_filters=inner_size)

        # Output phase.
        self._shrink_conv = nn.Conv2d(in_channels=inner_size, out_channels=hidden_size, kernel_size=1, stride=1,
                                      pad_mode='pad', has_bias=True)
        self._drop_path = DropPath2D(drop_prob=drop_path_rate)

    def shortcut_branch(self, shortcut):
        shortcut = self._downsample(shortcut)
        shortcut = self._shortcut_conv(shortcut)
        return shortcut

    def residual_branch(self, inputs):
        output = self._pre_norm(inputs)
        output = self._expand_conv(output)
        output = self._expand_norm(output)
        output = self._activation_fn(output)

        output = self._depthwise_conv(output)
        output = self._depthwise_norm(output)
        output = self._activation_fn(output)

        output = self._se(output)
        output = self._shrink_conv(output)

        return output

    def construct(self, x):
        residual = self.residual_branch(x)
        shortcut = self.shortcut_branch(x)
        output = shortcut + self._drop_path(residual)
        return output


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, dims=1, **kwargs):
        super(LayerNorm, self).__init__((normalized_shape,), **kwargs)
        self.dims = dims

    def construct(self, input_x):
        if self.dims == 1:
            x = super(LayerNorm, self).construct(input_x)
        else:
            B, C, H, W = input_x.shape
            x = input_x.reshape(B, C, -1).transpose(0, 2, 1)
            x = super(LayerNorm, self).construct(x)
            x = x.transpose(0, 2, 1).reshape(B, C, H, W)
        return x


class CoAtNet(nn.Cell):
    def __init__(self, image_size=(224, 224), num_blocks=[2, 2, 3, 5, 2], hidden_size=[64, 96, 192, 384, 768],
                 num_classes=1000, block_types=['C', 'C', 'T', 'T'], in_channels=3, drop_path_rate=0., attn_drop=0.,
                 proj_drop=0., expansion_rate=4, se_ratio=0.25, head_size=32):
        super().__init__()
        self.image_size = image_size
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.stem = self._make_stem(in_channels=in_channels)
        dprs = [x for x in
                np.linspace(0, drop_path_rate, np.sum(num_blocks[1:]))]  # stochastic depth decay rule
        blocks = []
        start = 0
        for index, num_block in enumerate(num_blocks[1:]):
            blocks.append(
                self._make_block(block_types[index],
                                 image_size=(image_size[0] >> (2 + index), image_size[1] >> (2 + index)),
                                 input_size=hidden_size[index], hidden_size=hidden_size[index + 1],
                                 num_block=num_blocks[index + 1], expansion_rate=expansion_rate, se_ratio=se_ratio,
                                 drop_path_rates=dprs[start:start + num_block], head_size=head_size))
            start += num_block
        self.blocks = nn.SequentialCell(blocks)

        # final layer normalization
        self._final_layer_norm = LayerNorm(normalized_shape=hidden_size[4], epsilon=1e-05)
        self.classfier = nn.SequentialCell([
            nn.Dense(in_channels=hidden_size[4], out_channels=hidden_size[4]),
            nn.Tanh(),
            nn.Dense(in_channels=hidden_size[4], out_channels=num_classes)
        ])
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

    def _make_stem(self, in_channels):
        stem_layers = []
        for i in range(self.num_blocks[0]):
            stem_layers.append(
                nn.Conv2d(in_channels=in_channels if i == 0 else self.hidden_size[0], out_channels=self.hidden_size[0],
                          kernel_size=3, stride=2 if i == 0 else 1, pad_mode='pad', padding=1, has_bias=True))
            if i < self.num_blocks[0] - 1:
                stem_layers.append(BatchNorm2d(num_features=self.hidden_size[0], momentum=0.9))
                stem_layers.append(nn.GELU())
        return nn.SequentialCell(stem_layers)

    def _make_block(self, block_type, image_size, input_size, hidden_size, num_block, expansion_rate, se_ratio,
                    drop_path_rates, head_size):
        block = []
        assert len(drop_path_rates) == num_block
        if block_type == "C":
            for i in range(num_block):
                block.append(MBConvBlock(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    stride=2 if i == 0 else 1,
                    expansion_rate=expansion_rate,
                    se_ratio=se_ratio,
                    drop_path_rate=drop_path_rates[i]
                ))
        elif block_type == 'T':
            for i in range(num_block):
                block.append(TransformerBlock(
                    image_size=image_size,
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size, head_size=head_size, dropout=0., expansion_rate=4,
                    stride=2 if i == 0 else 1, drop_path_rate=drop_path_rates[i]
                ))
        else:
            raise NotImplementedError
        return nn.SequentialCell([*block])

    def construct(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = ops.ReduceMean(False)(x, (2, 3))
        x = self._final_layer_norm(x)
        x = self.classfier(x)
        return x


def CoAtNet2(drop_path_rate, img_size=224, num_classes=1000):
    image_size = to_2tuple(img_size)
    num_blocks = [2, 2, 6, 14, 2]
    hidden_size = [128, 128, 256, 512, 1024]
    return CoAtNet(num_classes=num_classes, hidden_size=hidden_size,
                   num_blocks=num_blocks, image_size=image_size,
                   drop_path_rate=drop_path_rate)


def CoAtNet1(drop_path_rate, img_size=224, num_classes=1000):
    image_size = to_2tuple(img_size)
    num_blocks = [2, 2, 6, 14, 2]
    hidden_size = [64, 96, 192, 384, 768]
    return CoAtNet(num_classes=num_classes, hidden_size=hidden_size,
                   num_blocks=num_blocks, image_size=image_size,
                   drop_path_rate=drop_path_rate)


def CoAtNet0(drop_path_rate, img_size=224, num_classes=1000):
    image_size = to_2tuple(img_size)
    num_blocks = [2, 2, 3, 5, 2]
    hidden_size = [64, 96, 192, 384, 768]
    return CoAtNet(num_classes=num_classes, hidden_size=hidden_size,
                   num_blocks=num_blocks, image_size=image_size,
                   drop_path_rate=drop_path_rate)


if __name__ == "__main__":
    img_size = 384
    pretrained = "../../_384.ckpt"
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    data = Tensor(np.ones([1, 3, img_size, img_size]), dtype=mstype.float32)
    model = CoAtNet2(drop_path_rate=0.2, img_size=img_size)
    param_dict = load_checkpoint(pretrained)
    for key in param_dict.copy().keys():
        print(f'=> {key}')
        if key.endswith("relative_position_index") or "adam" in key:
            param_dict.pop(key)

    load_param_into_net(model, param_dict, strict_load=True)
    out = model(data)
    print(out.shape)
    params = 0.
    num = 0.
    for name, param in model.parameters_and_names():
        params += np.prod(param.shape)
        print(name, param.shape)
        num += 1
    print(params, num)
