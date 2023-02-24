import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, ngf=32, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            # ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("/", nn.Conv2d(input_ch, ngf, 3, stride=1, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(ngf, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(ngf, [1, 3, 1], [ngf, ngf, ngf*4], 3, stride=1)
        self.d1 = ResidualBlock(ngf*4, [1, 3, 1], [ngf*2, ngf*2, ngf*8], 4, stride=2)
        self.d2 = ResidualBlock(ngf*8, [1, 3, 1], [ngf*4, ngf*4, ngf*16], 6, stride=2)
        self.d3 = ResidualBlock(ngf*16, [1, 3, 1], [ngf*8, ngf*8, ngf*32], 3, stride=2)

        self.conv_bot = nn.Conv2d(ngf*32, ngf*16, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(ngf*16, ngf*4, ksize, stride=1, padding=2, bias=False)),
                ("dense", DenseBlock(ngf*4, [1, ksize], [ngf*2, ngf//2], 8, split=4)),
                ("convf", nn.Conv2d(ngf*8, ngf*8, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(ngf*8, ngf*2, ksize, stride=1, padding=2, bias=False)),
                ("dense", DenseBlock(ngf*2, [1, ksize], [ngf*2, ngf//2], 4, split=4)),
                ("convf", nn.Conv2d(ngf*4, ngf*4, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(ngf*4, ngf, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(ngf, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(ngf, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):
        # import pdb
        # pdb.set_trace()
        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        # if self.mode == 'original':
        #     d[0] = crop_op(d[0], [184, 184])
        #     d[1] = crop_op(d[1], [72, 72])
        # else:
        #     d[0] = crop_op(d[0], [92, 92])
        #     d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            # import pdb
            # pdb.set_trace()
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)

