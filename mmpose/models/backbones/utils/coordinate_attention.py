# Copyright (c) OpenMMLab. All rights reserved.
"""Coordinate Attention Module.

Paper: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
https://arxiv.org/abs/2103.02907

This module encodes positional information into channel attention,
enabling the network to capture long-range dependencies while
preserving precise spatial information.
"""
import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention Module.

    Decomposes channel attention into two 1D feature encoding processes
    (horizontal and vertical) to preserve precise positional information.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Channel reduction ratio. The intermediate channel
            dimension will be max(8, in_channels // reduction) to prevent
            information bottleneck. Default: 32.

    Input:
        x: Tensor of shape (B, C, H, W)

    Output:
        out: Tensor of shape (B, C, H, W) - same shape as input
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Protect intermediate channel dimension from being too small
        mid_channels = max(8, in_channels // reduction)

        # Adaptive pooling for X and Y directions
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)

        # Shared 1x1 Conv for dimension reduction
        self.conv_reduce = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # Separate 1x1 Conv for attention weight generation
        self.conv_h = nn.Conv2d(
            mid_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.conv_w = nn.Conv2d(
            mid_channels, in_channels, kernel_size=1, stride=1, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with zero initialization strategy.

        The conv_h and conv_w layers are initialized to zero so that
        sigmoid output is ~0.5 at initialization, making the CA module
        act as a pass-through initially. This preserves pretrained features.
        """
        # Normal initialization for conv_reduce
        nn.init.kaiming_normal_(
            self.conv_reduce.weight, mode='fan_out', nonlinearity='relu')

        # Zero initialization for conv_h and conv_w (critical for pretrained weights)
        nn.init.constant_(self.conv_h.weight, 0)
        nn.init.constant_(self.conv_h.bias, 0)
        nn.init.constant_(self.conv_w.weight, 0)
        nn.init.constant_(self.conv_w.bias, 0)

        # BatchNorm initialization
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Coordinate Attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()

        # Step 1: X-direction pooling (B, C, H, W) -> (B, C, H, 1)
        x_h = self.pool_h(x)

        # Step 2: Y-direction pooling (B, C, H, W) -> (B, C, 1, W)
        x_w = self.pool_w(x)

        # Step 3: Transpose x_w for concatenation: (B, C, 1, W) -> (B, C, W, 1)
        x_w_transposed = x_w.permute(0, 1, 3, 2)

        # Step 4: Concatenate along spatial dimension: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w_transposed], dim=2)

        # Step 5: Shared 1x1 Conv + BN + ReLU for dimension reduction
        y = self.conv_reduce(y)
        y = self.bn(y)
        y = self.act(y)

        # Step 6: Split back into H and W components
        x_h_out, x_w_out = torch.split(y, [height, width], dim=2)

        # Step 7: Transpose x_w back: (B, mid_channels, W, 1) -> (B, mid_channels, 1, W)
        x_w_out = x_w_out.permute(0, 1, 3, 2)

        # Step 8: Generate attention weights through separate 1x1 Conv + Sigmoid
        attn_h = torch.sigmoid(self.conv_h(x_h_out))  # (B, C, H, 1)
        attn_w = torch.sigmoid(self.conv_w(x_w_out))  # (B, C, 1, W)

        # Step 9: Apply attention: out = x * attn_h * attn_w
        out = x * attn_h * attn_w

        return out
