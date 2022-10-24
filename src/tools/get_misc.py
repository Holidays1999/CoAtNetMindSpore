# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""misc functions for program"""
import collections.abc
import math
import os
from itertools import repeat

import cv2
from mindspore import nn, context
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import data, models
from src.data.data_utils.moxing_adapter import sync_data
from src.trainer.train_one_step_with_ema import TrainOneStepWithEMA


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    else:
        raise ValueError("Unsupported platform.")

    return rank


def get_dataset(args, training=True):
    """"Get model according to args.set"""
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args, training)

    return dataset


def get_model(args):
    """"Get model according to args.arch"""
    print("==> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](drop_path_rate=args.drop_path_rate, num_classes=args.num_classes,
                                       img_size=args.image_size)

    return model


def pretrained(args, model):
    """"Load pretrained weights if args.pretrained is given"""
    if args.run_modelarts:
        print('Syncing data.')
        local_data_path = '/cache/weight/model.ckpt'
        sync_data(args.pretrained, local_data_path, threads=128)
        args.pretrained = local_data_path
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    elif os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def resize_pos_embed(pos_embed_table, new_pos_embed_table_shape):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    # (729, 16) -> math.sqrt(729) -> 27
    gs_old = int(math.sqrt(pos_embed_table.shape[0]))
    print('Resized position embedding: %s to %s', (gs_old, gs_old, pos_embed_table.shape[1]), new_pos_embed_table_shape)
    posemb_grid = pos_embed_table.reshape(gs_old, gs_old, -1)
    new_h, new_w = new_pos_embed_table_shape[:2]
    posemb_grid = cv2.resize(posemb_grid, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)

    return posemb_grid


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        print(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 10, scale_factor=2,
                                                                    scale_window=2000)
    else:
        print(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)
    net_with_loss = TrainOneStepWithEMA(net_with_loss, optimizer, global_norm=args.clip_global_norm_value,
                                        scale_sense=scale_sense, with_ema=args.with_ema, ema_decay=args.ema_decay)

    return net_with_loss
