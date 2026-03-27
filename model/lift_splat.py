"""
lift_splat.py
-------------
Implements the Lift → Splat pipeline (from LSS: "Lift, Splat, Shoot").

LIFT:
    For each image pixel (u, v):
    - Predict depth distribution over D bins → d_1 ... d_D
    - Unproject pixel to D 3D points in camera space using intrinsics K
    - Weight each 3D point by its depth probability
    - Attach context feature C to each point
    → Produces a frustum of (D × fH × fW) weighted 3D feature points

SPLAT:
    - Transform frustum points from camera frame → ego frame using extrinsics E
    - Discard points outside the BEV grid bounds
    - Sum (pool) features of all points that fall in the same BEV cell
    → Produces a (B, C_ctx, H_bev, W_bev) BEV feature map
"""

import torch
import torch.nn as nn
import numpy as np


class LiftSplat(nn.Module):
    """
    Differentiable Lift-Splat module.

    Takes per-image depth distributions + context features + calibration
    and returns a BEV feature map.
    """

    def __init__(self, cfg: dict, ctx_channels: int):
        """
        Args:
            cfg:          full config dict
            ctx_channels: C_ctx (must match DepthNet ctx_channels)
        """
        super().__init__()

        bev   = cfg["bev"]
        depth = cfg["depth"]

        self.lat_min, self.lat_max = bev["lateral_range"]   # forward  (ego X)
        self.fwd_min, self.fwd_max = bev["forward_range"]   # lateral  (ego Y)
        self.res = bev["resolution"]

        self.H_bev = int((self.lat_max - self.lat_min) / self.res)
        self.W_bev = int((self.fwd_max - self.fwd_min) / self.res)

        self.ctx_channels = ctx_channels

        # Fixed depth bin values (not learned)
        d_bins = torch.linspace(depth["min"], depth["max"], depth["bins"])
        self.register_buffer("d_bins", d_bins)   # (D,)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _make_frustum(
        self,
        fH: int, fW: int,
        K: torch.Tensor,       # (B, 3, 3)
    ) -> torch.Tensor:
        """
        Build a frustum of 3D points in camera space.

        For each (depth_bin, pixel_row, pixel_col) we compute the
        corresponding 3D point in camera coordinates.

        Returns:
            frustum_cam: (B, D, fH, fW, 3)  XYZ in camera frame
        """
        B = K.shape[0]
        D = self.d_bins.shape[0]
        device = K.device

        # Pixel grid in feature-map space
        # We need to know the original image size to scale K.
        # Instead, we work directly in feature pixel coords and
        # assume K has already been scaled to feature map resolution
        # by the caller (see BEVOccupancyNet.forward).

        # u, v pixel coords on feature map
        v_coords = torch.arange(fH, device=device).float()   # (fH,)
        u_coords = torch.arange(fW, device=device).float()   # (fW,)

        # Shape: (fH, fW)
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")

        # Homogeneous pixel coords: (3, fH*fW)
        ones  = torch.ones_like(uu)
        uvh   = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)  # (3, fH*fW)

        # Unproject: direction in camera space = K_inv @ [u, v, 1]^T
        K_inv = torch.linalg.inv(K)                   # (B, 3, 3)
        dirs  = torch.bmm(K_inv, uvh.unsqueeze(0)
                          .expand(B, -1, -1))          # (B, 3, fH*fW)

        # Scale each direction by each depth bin
        # d_bins: (D,) → (1, D, 1)
        d = self.d_bins.view(1, D, 1)                 # (1, D, 1)

        # dirs: (B, 3, fH*fW) → (B, 1, 3, fH*fW)
        dirs = dirs.unsqueeze(1)                        # (B, 1, 3, fH*fW)

        # points_cam: (B, D, 3, fH*fW)
        points_cam = dirs * d.unsqueeze(2)

        # Reshape to (B, D, fH, fW, 3)
        points_cam = points_cam.permute(0, 1, 3, 2)   # (B, D, fH*fW, 3)
        points_cam = points_cam.reshape(B, D, fH, fW, 3)

        return points_cam                              # (B, D, fH, fW, 3)

    def _cam_to_ego(
        self,
        points_cam: torch.Tensor,   # (B, D, fH, fW, 3)
        E: torch.Tensor,            # (B, 4, 4)  cam→ego
    ) -> torch.Tensor:
        """
        Apply rigid transform E to convert camera-frame points to ego frame.

        Returns:
            points_ego: (B, D, fH, fW, 3)
        """
        B, D, fH, fW, _ = points_cam.shape

        R = E[:, :3, :3]   # (B, 3, 3)
        t = E[:, :3,  3]   # (B, 3)

        pts = points_cam.reshape(B, -1, 3)             # (B, D*fH*fW, 3)
        pts = torch.bmm(pts, R.transpose(1, 2)) + t.unsqueeze(1)  # (B, N, 3)
        pts = pts.reshape(B, D, fH, fW, 3)

        return pts

    # ── Splat ─────────────────────────────────────────────────────────────────

    def _splat(
        self,
        points_ego:  torch.Tensor,   # (B, D, fH, fW, 3)
        depth_dist:  torch.Tensor,   # (B, D, fH, fW)
        context:     torch.Tensor,   # (B, C, fH, fW)
    ) -> torch.Tensor:
        """
        Pool weighted context features into the BEV grid.

        For each 3D point:
            weighted_feat = depth_prob * context_feat
        Then sum into the BEV cell the point lands in.

        Returns:
            bev_feat: (B, C, H_bev, W_bev)
        """
        B, D, fH, fW, _ = points_ego.shape
        C = context.shape[1]
        device = context.device

        # Ego-frame X (forward), Y (lateral)
        px = points_ego[..., 0]   # (B, D, fH, fW)
        py = points_ego[..., 1]   # (B, D, fH, fW)

        # Convert to BEV grid indices
        row = ((px - self.lat_min) / self.res)   # (B, D, fH, fW)  forward → row
        col = ((py - self.fwd_min) / self.res)   # (B, D, fH, fW)  lateral → col

        # Flip row so near = bottom (row 0 = far)
        row = (self.H_bev - 1) - row

        # Valid mask: points inside BEV bounds
        valid = (
            (row >= 0) & (row < self.H_bev) &
            (col >= 0) & (col < self.W_bev)
        )                                      # (B, D, fH, fW) bool

        row = row.long().clamp(0, self.H_bev - 1)
        col = col.long().clamp(0, self.W_bev - 1)

        # Weighted context: depth_dist × context
        # depth_dist: (B, D, fH, fW) → expand for C
        # context:    (B, C, fH, fW) → expand for D
        w   = depth_dist.unsqueeze(2)          # (B, D, 1, fH, fW)
        ctx = context.unsqueeze(1)             # (B, 1, C, fH, fW)
        weighted = (w * ctx)                   # (B, D, C, fH, fW)

        # Flatten everything to scatter into BEV grid
        # BEV cell index: row * W_bev + col
        cell_idx = (row * self.W_bev + col)    # (B, D, fH, fW)
        cell_idx = cell_idx.unsqueeze(2).expand(-1, -1, C, -1, -1)
        # → (B, D, C, fH, fW)

        valid_exp = valid.unsqueeze(2).expand_as(weighted)

        # Zero out invalid points
        weighted  = weighted  * valid_exp.float()

        # Reshape for scatter: (B, C, D*fH*fW)
        weighted  = weighted.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        cell_idx  = cell_idx.permute(0, 2, 1, 3, 4).reshape(B, C, -1)

        # Scatter sum into (B, C, H_bev*W_bev)
        bev_flat  = torch.zeros(B, C, self.H_bev * self.W_bev,
                                device=device, dtype=weighted.dtype)
        bev_flat.scatter_add_(2, cell_idx, weighted)

        bev_feat  = bev_flat.reshape(B, C, self.H_bev, self.W_bev)
        return bev_feat

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        depth_dist: torch.Tensor,   # (B, D, fH, fW)
        context:    torch.Tensor,   # (B, C, fH, fW)
        K_feat:     torch.Tensor,   # (B, 3, 3) intrinsics scaled to feat map
        E:          torch.Tensor,   # (B, 4, 4) cam→ego extrinsic
    ) -> torch.Tensor:
        """
        Returns:
            bev_feat: (B, C, H_bev, W_bev)
        """
        fH = depth_dist.shape[2]
        fW = depth_dist.shape[3]

        frustum_cam = self._make_frustum(fH, fW, K_feat)       # (B, D, fH, fW, 3)
        frustum_ego = self._cam_to_ego(frustum_cam, E)         # (B, D, fH, fW, 3)
        bev_feat    = self._splat(frustum_ego, depth_dist, context)

        return bev_feat   # (B, C, H_bev, W_bev)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    B, D, fH, fW, C = 2, 112, 14, 25, 64

    ls          = LiftSplat(cfg, ctx_channels=C)
    depth_dist  = torch.rand(B, D, fH, fW).softmax(1)
    context     = torch.randn(B, C, fH, fW)

    # Dummy intrinsics (scaled to feature map)
    K_feat = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K_feat[:, 0, 0] = 25.0;  K_feat[:, 1, 1] = 14.0
    K_feat[:, 0, 2] = 12.5;  K_feat[:, 1, 2] =  7.0

    E = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()

    bev = ls(depth_dist, context, K_feat, E)
    print(f"BEV feature map: {bev.shape}")   # (2, 64, 100, 100)