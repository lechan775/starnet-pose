# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/ma-xu/Rewrite-the-Stars
"""
StarNet backbone for MMPose.

Paper: Rewrite the Stars (CVPR 2024)
Original implementation: https://github.com/ma-xu/Rewrite-the-Stars
"""
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from mmcv.cnn.bricks import DropPath
    from mmengine.model.weight_init import trunc_normal_


class ConvBN(nn.Sequential):
    """Conv + BatchNorm block."""

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module(
            'conv',
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class StarBlock(BaseModule):
    """Star Block with element-wise multiplication.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim. Default: 3.
        drop_path (float): Stochastic depth rate. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, dim, mlp_ratio=3, drop_path=0., init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # Star operation
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


@MODELS.register_module()
class StarNet(BaseBackbone):
    """StarNet backbone.

    A PyTorch implementation of: `Rewrite the Stars` (CVPR 2024)
    https://arxiv.org/abs/2403.19967

    Args:
        arch (str): Architecture of StarNet. Choose from 's1', 's2', 's3', 's4',
            's050', 's100', 's150'. Default: 's3'.
        base_dim (int): Base channel dimension. Default: 32.
        depths (list[int]): Depths of each stage. Default: [2, 2, 8, 4].
        mlp_ratio (int): Ratio of mlp hidden dim. Default: 4.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (3,).
        frozen_stages (int): Stages to be frozen. Default: -1.
        norm_eval (bool): Whether to set norm layers to eval mode.
            Default: False.
        init_cfg (dict, optional): Initialization config dict.
    """

    arch_settings = {
        's050': {'base_dim': 16, 'depths': [1, 1, 3, 1], 'mlp_ratio': 3},
        's100': {'base_dim': 20, 'depths': [1, 2, 4, 1], 'mlp_ratio': 4},
        's150': {'base_dim': 24, 'depths': [1, 2, 4, 2], 'mlp_ratio': 3},
        's1': {'base_dim': 24, 'depths': [2, 2, 8, 3], 'mlp_ratio': 4},
        's2': {'base_dim': 32, 'depths': [1, 2, 6, 2], 'mlp_ratio': 4},
        's3': {'base_dim': 32, 'depths': [2, 2, 8, 4], 'mlp_ratio': 4},
        's4': {'base_dim': 32, 'depths': [3, 3, 12, 5], 'mlp_ratio': 4},
    }

    def __init__(self,
                 arch='s3',
                 base_dim=None,
                 depths=None,
                 mlp_ratio=None,
                 drop_path_rate=0.0,
                 out_indices=(3,),
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1, layer=['_BatchNorm'])
                 ]):
        super().__init__(init_cfg=init_cfg)

        if arch in self.arch_settings:
            arch_cfg = self.arch_settings[arch]
            base_dim = base_dim or arch_cfg['base_dim']
            depths = depths or arch_cfg['depths']
            mlp_ratio = mlp_ratio or arch_cfg['mlp_ratio']
        else:
            assert base_dim is not None and depths is not None
            mlp_ratio = mlp_ratio or 4

        self.arch = arch
        self.base_dim = base_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # Validate out_indices
        for index in out_indices:
            if index not in range(len(depths)):
                raise ValueError(f'out_indices must be in range(0, {len(depths)}), '
                                 f'but got {index}')

        self.in_channel = 32
        # Stem layer
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6()
        )

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [
                StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        # Output channels for each stage
        self.out_channels = [base_dim * 2 ** i for i in range(len(depths))]

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
