"""
bev_decoder.py
--------------
Lightweight CNN decoder that takes the (B, C, H_bev, W_bev) BEV feature map
from LiftSplat and produces a (B, 1, H_bev, W_bev) occupancy logit map.

No sigmoid here — raw logits are returned so we can use BCEWithLogitsLoss
which is numerically more stable.
"""

import torch
import torch.nn as nn


class BEVDecoder(nn.Module):
    """
    3-block conv decoder:
        BEV features → refined features → per-cell occupancy logit
    """

    def __init__(self, in_channels: int, dropout: float = 0.3):
        """
        Args:
            in_channels: C_ctx from LiftSplat output (e.g. 64)
        """
        super().__init__()

        self.decoder = nn.Sequential(
            # Block 1: keep spatial size, increase capacity
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout), 

            # Block 2
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout), 

            # Block 3
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Output head: 1 logit per cell
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_feat: (B, C, H_bev, W_bev)

        Returns:
            logits: (B, 1, H_bev, W_bev)  raw (pre-sigmoid) occupancy scores
        """
        return self.decoder(bev_feat)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model  = BEVDecoder(in_channels=64)
    dummy  = torch.randn(2, 64, 100, 100)
    logits = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {logits.shape}")    # (2, 1, 100, 100)