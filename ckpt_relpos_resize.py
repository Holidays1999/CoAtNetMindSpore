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
"""for relative position embedding resize for images with different size"""
import argparse
import math
import os

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import load_checkpoint, save_checkpoint

from src.tools.get_misc import resize_pos_embed


def parse_arguments():
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="MindSpore Checkpoint Position Embedding Resize")

    parser.add_argument("--ori_img_size", default=224, type=int, help="Origin Image Size(default: 224).")
    parser.add_argument("--cur_img_size", default=384, type=int, help="Origin Image Size(default: 384).")
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model",
                        required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
    else:
        raise ValueError(f"{args.pretrained} is not a file or not exsit.")
    weights = []
    for key in param_dict.keys():
        collects = {}
        print(key)
        if key.endswith("relative_position_bias_table"):
            heads_num = param_dict[key].shape[1]
            H = W = (int(math.sqrt(param_dict[key].shape[0])) + 1) / 2
            print(f'H W: {H} {W}')
            new_H = new_W = args.cur_img_size / (args.ori_img_size / H)
            new_pos_embed_table_shape = (int(2 * new_H - 1), int(2 * new_W - 1), heads_num)
            pos_embed_table = resize_pos_embed(pos_embed_table=param_dict[key].asnumpy(),
                                               new_pos_embed_table_shape=new_pos_embed_table_shape)
            collects['name'] = key
            collects['data'] = Tensor(pos_embed_table.reshape(-1, heads_num), dtype=mstype.float32)
        elif key.endswith("relative_position_index") or key.startswith("adam"):
            continue
        else:
            collects['name'] = key
            collects['data'] = Tensor(param_dict[key].data.asnumpy(), dtype=mstype.float32)
        weights.append(collects)
    save_path = f"{os.path.basename(args.pretrained).split('.')[0]}_{args.cur_img_size}.ckpt"
    print(f'save path: {save_path}')

    save_checkpoint(weights, save_path)


if __name__ == '__main__':
    main()
