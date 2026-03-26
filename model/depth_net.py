"""
depth_net.py
------------
Predicts a discrete depth distribution over D bins for every image pixel.

Takes the backbone feature map (B, C_in, fH, fW) and outputs:
    depth_dist: (B, D, fH, fW)  — softmax probability over depth bins
    context:    (B, C_ctx, fH, fW) — rich context features to attach to each point

These are the two outputs needed by the Lift step in lift_splat.py.
"""

import torch
import torch.nn as nn
import numpy as np


class DepthNet(nn.Module):
    """
    Lightweight depth + context head on top of backbone features.

    Architecture:
        shared_conv → split into two heads:
            depth_head  : predicts D-bin softmax depth distribution
            context_head: predicts C_ctx context features per pixel
    """

    def __init__(
        self,
        in_channels:  int,   # backbone output channels (2048 for ResNet-50)
        depth_bins:   int,   # D — number of discrete depth values
        ctx_channels: int,   # C_ctx — context feature dimension
    ):
        super().__init__()

        self.depth_bins   = depth_bins
        self.ctx_channels = ctx_channels

        # Reduce backbone channels first (2048 → 512)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Depth distribution head
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, depth_bins, kernel_size=1),
        )

        # Context feature head
        self.context_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ctx_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (B, C_in, fH, fW)  backbone feature map

        Returns:
            depth_dist: (B, D, fH, fW)    softmax depth probabilities
            context:    (B, C_ctx, fH, fW) context features
        """
        x          = self.reduce(features)           # (B, 512, fH, fW)
        depth_dist = self.depth_head(x).softmax(1)   # (B, D, fH, fW)
        context    = self.context_head(x)             # (B, C_ctx, fH, fW)
        return depth_dist, context


def build_depth_bins(d_min: float, d_max: float, num_bins: int) -> torch.Tensor:
    """
    Returns a (D,) tensor of evenly-spaced depth values.
    Used during the Lift step to convert bin index → metric depth.
    """
    return torch.linspace(d_min, d_max, num_bins)    # (D,)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DepthNet(in_channels=2048, depth_bins=112, ctx_channels=64)
    feat  = torch.randn(2, 2048, 14, 25)
    d, c  = model(feat)
    print(f"depth_dist : {d.shape}")   # (2, 112, 14, 25)
    print(f"context    : {c.shape}")   # (2,  64, 14, 25)
    print(f"depth sums : {d[0,:,0,0].sum():.4f}")  # should be 1.0