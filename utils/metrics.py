"""
metrics.py
----------
Evaluation metrics for BEV occupancy prediction.

    - occupancy_iou          : standard IoU between predicted and GT binary grids
    - distance_weighted_error: penalises mistakes closer to the ego vehicle more
"""

import torch


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

    intersection = (pred * gt).sum(dim=(1, 2))                     # (B,)
    union        = ((pred + gt) > 0).float().sum(dim=(1, 2))       # (B,)

    # Avoid division by zero: if both pred and GT are empty, IoU = 1
    iou = torch.where(union > 0, intersection / union, torch.ones_like(intersection))
    return iou.mean().item()


def distance_weighted_error(
    logits:    torch.Tensor,   # (B, 1, H, W) or (B, H, W)
    gt:        torch.Tensor,   # (B, H, W)  binary float
    H_bev:     int,
    fwd_min:   float = -25.0,  # forward range min (metres)
    fwd_max:   float =  25.0,  # forward range max (metres)
    threshold: float = 0.5,
) -> float:
    """
    Computes a distance-weighted binary error.
    Errors at cells closer to the ego vehicle get higher weight.

    BEV grid convention:
        row 0     = far front  (fwd_max)
        row H//2  = ego centre (forward ≈ 0)
        row H-1   = far back   (fwd_min)

    Weight per row peaks at the ego row (H//2) and decays toward 0 at
    both extremes, using a simple 1/(1+|row - center|) kernel — matching
    the same scheme used by the training loss in train.py.

    BUG FIX #2: previous code had three problems:
        1. A .flip(0) was applied then immediately overwritten (dead code).
        2. linspace(0, 1, H) weights row H-1 highest, but row H-1 is the
           *far back* edge, not the ego position.
        3. The ego row is H//2, not H-1 or 0.

    Args:
        logits:  raw model output
        gt:      binary ground truth
        H_bev:   number of BEV rows
        fwd_min: minimum forward distance in metres
        fwd_max: maximum forward distance in metres

    Returns:
        weighted_error: scalar float
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)

    pred  = (torch.sigmoid(logits) > threshold).float()
    error = (pred - gt).abs()                             # (B, H, W)

    device = logits.device
    center = H_bev // 2

    # Distance of each row from the ego centre row
    row_idx = torch.arange(H_bev, device=device).float()  # (H,)
    dist    = (row_idx - center).abs()                     # (H,)

    # Weight = 1 / (1 + dist), normalised so ego row = 1.0
    row_weights = 1.0 / (1.0 + dist)                      # (H,)
    row_weights = row_weights / row_weights.max()          # peak = 1.0

    # Expand to (1, H, 1) for broadcasting over (B, H, W)
    row_weights = row_weights.view(1, H_bev, 1)

    weighted_error = (error * row_weights).sum(dim=(1, 2)) / row_weights.sum()
    return weighted_error.mean().item()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fwd_min, fwd_max = -25, 25
    B, H, W = 4, 200, 200
    logits = torch.randn(B, 1, H, W)
    gt     = (torch.rand(B, H, W) > 0.9).float()   # sparse GT

    iou = occupancy_iou(logits, gt)
    dwe = distance_weighted_error(logits, gt, H, fwd_min, fwd_max)

    print(f"Occupancy IoU           : {iou:.4f}")
    print(f"Distance-weighted Error : {dwe:.4f}")

    # Verify weights peak at centre row
    import torch
    center = H // 2
    row_idx = torch.arange(H).float()
    dist    = (row_idx - center).abs()
    weights = 1.0 / (1.0 + dist)
    weights = weights / weights.max()
    print(f"Weight at row 0 (far)   : {weights[0]:.4f}")
    print(f"Weight at row {center} (ego): {weights[center]:.4f}")
    print(f"Weight at row {H-1} (back) : {weights[-1]:.4f}")
