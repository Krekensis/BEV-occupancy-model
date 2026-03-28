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

BEV grid convention (matches bev_gt_generator.py):
    H_bev rows  = forward/back axis  (cfg forward_range  → ego X)
    W_bev cols  = lateral axis       (cfg lateral_range  → ego Y)
    row 0       = far front  (fwd_max)
    row H-1     = far back   (fwd_min)
    col 0       = far left   (lat_max, nuScenes Y+ = left)
    col W-1     = far right  (lat_min)
"""

import torch
import torch.nn as nn


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
            ctx_channels: C_ctx (must match encoder ctx_channels)
        """
        super().__init__()

        bev   = cfg["bev"]
        depth = cfg["depth"]

        self.lat_min, self.lat_max = bev["lateral_range"]   # ego Y  (left/right)
        self.fwd_min, self.fwd_max = bev["forward_range"]   # ego X  (front/back)
        self.res = bev["resolution"]

        self.H_bev = int((self.fwd_max - self.fwd_min) / self.res)  # forward → rows
        self.W_bev = int((self.lat_max - self.lat_min) / self.res)  # lateral → cols

        self.ctx_channels = ctx_channels

        d_bins = torch.linspace(depth["min"], depth["max"], depth["bins"])
        self.register_buffer("d_bins", d_bins)   # (D,)

        # cache-ing the pixel-grid part of the frustum.
        # The meshgrid over (fH, fW) is constant for a fixed input resolution.
        # im stupid and lazily built it on first forward call and re-use it afterwards, avoiding repeated allocation on every forward pass.
        self._cached_uvh: torch.Tensor | None = None
        self._cached_fH: int = -1
        self._cached_fW: int = -1

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _get_pixel_grid(self, fH: int, fW: int, device: torch.device) -> torch.Tensor:
        """
        Return homogeneous pixel coords (3, fH*fW), cached for speed.
        Invalidated automatically if fH/fW changes (e.g. different input res).
        """
        if (fH != self._cached_fH or fW != self._cached_fW
                or self._cached_uvh is None
                or self._cached_uvh.device != device):
            v_coords = torch.arange(fH, device=device).float()
            u_coords = torch.arange(fW, device=device).float()
            vv, uu   = torch.meshgrid(v_coords, u_coords, indexing="ij")
            ones     = torch.ones_like(uu)
            self._cached_uvh = torch.stack([uu, vv, ones], dim=0).reshape(3, -1)
            self._cached_fH  = fH
            self._cached_fW  = fW
        return self._cached_uvh   # (3, fH*fW)

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

        uvh  = self._get_pixel_grid(fH, fW, device)   # (3, fH*fW)

        # Unproject: direction in camera space = K_inv @ [u, v, 1]^T
        K_inv = torch.linalg.inv(K)                    # (B, 3, 3)
        dirs  = torch.bmm(K_inv, uvh.unsqueeze(0)
                          .expand(B, -1, -1))           # (B, 3, fH*fW)

        # Scale each direction by each depth bin
        d    = self.d_bins.view(1, D, 1)               # (1, D, 1)
        dirs = dirs.unsqueeze(1)                        # (B, 1, 3, fH*fW)

        # points_cam: (B, D, 3, fH*fW)
        points_cam = dirs * d.unsqueeze(2)

        # Reshape to (B, D, fH, fW, 3)
        points_cam = points_cam.permute(0, 1, 3, 2)    # (B, D, fH*fW, 3)
        points_cam = points_cam.reshape(B, D, fH, fW, 3)

        return points_cam                               # (B, D, fH, fW, 3)

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

        pts = points_cam.reshape(B, -1, 3)
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

        px = points_ego[..., 0]   # ego X = forward  → row index  (B, D, fH, fW)
        py = points_ego[..., 1]   # ego Y = lateral  → col index  (B, D, fH, fW)

        # BEV row: row 0 = fwd_max (far front), row H-1 = fwd_min (far back)
        row = (self.fwd_max - px) / self.res              # (B, D, fH, fW)
        # BEV col: col 0 = lat_max (far left), col W-1 = lat_min (far right)
        col = (self.lat_max - py) / self.res              # (B, D, fH, fW)

        # Valid mask: points strictly inside BEV bounds (before clamping)
        valid = (
            (row >= 0) & (row < self.H_bev) &
            (col >= 0) & (col < self.W_bev)
        )                                                  # (B, D, fH, fW) bool

        # Clamp only to keep scatter indices in range (invalid points will be
        # zeroed out via the valid mask ... the clamp just prevents an index error)
        row = row.long().clamp(0, self.H_bev - 1)
        col = col.long().clamp(0, self.W_bev - 1)

        # Weighted context: depth_dist × context
        w       = depth_dist.unsqueeze(2)                 # (B, D, 1, fH, fW)
        ctx     = context.unsqueeze(1)                    # (B, 1, C, fH, fW)
        weighted = w * ctx                                # (B, D, C, fH, fW)

        # Zero out invalid points before scatter
        valid_exp = valid.unsqueeze(2).expand_as(weighted)
        weighted  = weighted * valid_exp.float()

        # BEV flat index
        cell_idx = (row * self.W_bev + col)               # (B, D, fH, fW)
        cell_idx = cell_idx.unsqueeze(2).expand(-1, -1, C, -1, -1)
        # → (B, D, C, fH, fW)

        # Reshape for scatter: (B, C, D*fH*fW)
        weighted = weighted.permute(0, 2, 1, 3, 4).reshape(B, C, -1)
        cell_idx = cell_idx.permute(0, 2, 1, 3, 4).reshape(B, C, -1)

        # Scatter sum into (B, C, H_bev * W_bev)
        bev_flat = torch.zeros(B, C, self.H_bev * self.W_bev,
                               device=device, dtype=weighted.dtype)
        bev_flat.scatter_add_(2, cell_idx, weighted)

        bev_feat = bev_flat.reshape(B, C, self.H_bev, self.W_bev)
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

        frustum_cam = self._make_frustum(fH, fW, K_feat)        # (B, D, fH, fW, 3)
        frustum_ego = self._cam_to_ego(frustum_cam, E)           # (B, D, fH, fW, 3)
        bev_feat    = self._splat(frustum_ego, depth_dist, context)

        return bev_feat   # (B, C, H_bev, W_bev)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    B, D, fH, fW, C = 2, 112, 32, 57, 64 

    ls         = LiftSplat(cfg, ctx_channels=C)
    depth_dist = torch.rand(B, D, fH, fW).softmax(1)
    context    = torch.randn(B, C, fH, fW)

    K_feat = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K_feat[:, 0, 0] = 57.0;  K_feat[:, 1, 1] = 32.0
    K_feat[:, 0, 2] = 28.5;  K_feat[:, 1, 2] = 16.0

    E = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()

    bev = ls(depth_dist, context, K_feat, E)
    print(f"BEV feature map: {bev.shape}")   # (2, 64, 200, 200) for 0.25 res, ±25 m
