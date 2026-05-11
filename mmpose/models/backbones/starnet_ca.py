# Copyright (c) OpenMMLab. All rights reserved.
"""StarNetCA backbone - StarNet with Coordinate Attention for MMPose."""
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .utils import CoordinateAttention
from .starnet import ConvBN, StarBlock

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from mmcv.cnn.bricks import DropPath
    from mmengine.model.weight_init import trunc_normal_


class CAStarBlock(BaseModule):
    """Star Block with optional Coordinate Attention."""

    def __init__(self, dim, mlp_ratio=3, drop_path=0., use_ca=True,
                 ca_reduction=32, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.use_ca = use_ca

        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if use_ca:
            self.ca = CoordinateAttention(dim, reduction=ca_reduction)
        else:
            self.ca = None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        if self.use_ca and self.ca is not None:
            x = self.ca(x)
        x = self.dwconv2(x)
        x = input + self.drop_path(x)
        return x


@MODELS.register_module()
class StarNetCA(BaseBackbone):
    """StarNet backbone with Coordinate Attention."""

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
                 use_ca=True,
                 ca_stages=[2, 3],
                 ca_reduction=32,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained

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
        self.use_ca = use_ca
        self.ca_stages = ca_stages if use_ca else []
        self.ca_reduction = ca_reduction

        for index in out_indices:
            if index not in range(len(depths)):
                raise ValueError(f'out_indices must be in range(0, {len(depths)})')

        for stage_idx in self.ca_stages:
            if stage_idx not in range(len(depths)):
                raise ValueError(f'ca_stages must be in range(0, {len(depths)})')

        self.in_channel = 32
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6()
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim

            use_ca_in_stage = i_layer in self.ca_stages
            if use_ca_in_stage:
                blocks = [
                    CAStarBlock(self.in_channel, mlp_ratio, dpr[cur + i],
                                use_ca=True, ca_reduction=ca_reduction)
                    for i in range(depths[i_layer])
                ]
            else:
                blocks = [
                    StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
                    for i in range(depths[i_layer])
                ]

            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.out_channels = [base_dim * 2 ** i for i in range(len(depths))]

    def init_weights(self):
        """Initialize weights with pretrained StarNet weights."""
        super().init_weights()
        
        if self.pretrained:
            from mmengine.logging import print_log
            
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print_log(f'Missing keys (CA layers): {len(missing_keys)} keys', logger='current', level=30)
            if unexpected_keys:
                print_log(f'Unexpected keys: {len(unexpected_keys)} keys', logger='current', level=30)
            
            print_log(f'Loaded pretrained StarNet weights from {self.pretrained}', logger='current')

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
