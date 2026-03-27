"""
metrics.py
----------
Evaluation metrics for BEV occupancy prediction.

    - Occupancy IoU       : standard IoU between predicted and GT binary grids
    - Distance-weighted Error : penalises mistakes closer to the ego vehicle more
"""

import torch
import numpy as np


def occupancy_iou(
    logits: torch.Tensor,   # (B, 1, H, W) or (B, H, W)
    gt:     torch.Tensor,   # (B, H, W)  binary float
    threshold: float = 0.5,
) -> float:
    """
    Computes mean Intersection-over-Union over the batch.

    Args:
        logits:    raw model output (pre-sigmoid)
        gt:        binary ground truth (0 or 1)
        threshold: probability threshold after sigmoid

    Returns:
        iou: scalar float
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)   # (B, H, W)

    pred = (torch.sigmoid(logits) > threshold).float()   # (B, H, W)

    intersection = (pred * gt).sum(dim=(1, 2))            # (B,)
    union        = ((pred + gt) > 0).float().sum(dim=(1, 2))  # (B,)

    # Avoid division by zero for empty scenes
    iou = torch.where(union > 0, intersection / union, torch.ones_like(intersection))
    return iou.mean().item()


def distance_weighted_error(
    logits:  torch.Tensor,   # (B, 1, H, W) or (B, H, W)
    gt:      torch.Tensor,   # (B, H, W)  binary float
    H_bev:   int,
    fwd_min:   float = -25,          # forward range min (metres)
    fwd_max:   float = 25,          # forward range max (metres)
    threshold: float = 0.5,
) -> float:
    """
    Computes a distance-weighted binary error.
    Errors at cells closer to the ego vehicle get higher weight.

    Weight per row = (fwd_max - x) / (fwd_max - fwd_min)
    where fwd_max = farthest row (weight→0), x = ego row (weight→1).

    Args:
        logits:  raw model output
        gt:      binary ground truth
        H_bev:   number of BEV rows
        fwd_min:   minimum forward distance in metres
        fwd_max:   maximum forward distance in metres

    Returns:
        weighted_error: scalar float
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)

    pred = (torch.sigmoid(logits) > threshold).float()
    error = (pred - gt).abs()                            # (B, H, W)

    # Build weight map: row 0 = far, row H-1 = near ego
    # near rows should have higher weight
    row_weights = torch.linspace(0.0, 1.0, H_bev,
                                  device=logits.device)  # (H,) near=1, far=0
    # Actually row 0 is far, row H-1 is near, so:
    row_weights = row_weights.flip(0)                    # row 0=1.0 (near), H-1=0.0
    # Wait: our grid: row 0 = far (fwd_max), row H-1 = near (fwd_min)
    # We want near to be high weight → row H-1 gets weight 1.0
    row_weights = torch.linspace(0.0, 1.0, H_bev,
                                  device=logits.device)  # (H,) 0=far, 1=near

    # Expand to (1, H, 1) for broadcasting
    row_weights = row_weights.view(1, H_bev, 1)

    weighted_error = (error * row_weights).sum(dim=(1, 2)) / row_weights.sum()
    return weighted_error.mean().item()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fwd_min, fwd_max = -25, 25
    B, H, W = 4, 100, 100
    logits = torch.randn(B, 1, H, W)
    gt     = (torch.rand(B, H, W) > 0.9).float()   # sparse GT

    iou    = occupancy_iou(logits, gt)
    dwe    = distance_weighted_error(logits, gt, H, fwd_min, fwd_max)

    print(f"Occupancy IoU            : {iou:.4f}")
    print(f"Distance-weighted Error  : {dwe:.4f}")