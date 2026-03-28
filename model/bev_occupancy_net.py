"""
bev_occupancy_net.py
---------------------
Assembles the full pipeline:

    RGB image (B, 3, H, W)
        ↓  DepthAnythingEncoder  (Depth Anything V2 backbone)
    depth_dist (B, D, fH, fW)  +  context (B, C_ctx, fH, fW)
        ↓  LiftSplat  [uses K, E from nuScenes calibration]
    bev_feat (B, C_ctx, H_bev, W_bev)
        ↓  BEVDecoder
    logits (B, 1, H_bev, W_bev)

Key change vs. original:
    ResNetBackbone + DepthNet  →  DepthAnythingEncoder
    The DA2 encoder jointly predicts metric depth (converted to a soft
    bin distribution) and rich context features from ViT patch embeddings.
"""

import torch
import torch.nn as nn

from model.depth_anything import DepthAnythingEncoder
from model.lift_splat     import LiftSplat
from model.bev_decoder    import BEVDecoder


class BEVOccupancyNet(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()

        ctx_ch   = cfg["model"]["bev_channels"]    # 64
        img_h, img_w = cfg["data"]["image_size"]   # [448, 800]

        self.img_h = img_h
        self.img_w = img_w

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.encoder     = DepthAnythingEncoder(cfg, ctx_channels=ctx_ch)
        self.lift_splat  = LiftSplat(cfg, ctx_channels=ctx_ch)
        self.bev_decoder = BEVDecoder(
            in_channels=ctx_ch,
            dropout=cfg["model"].get("dropout", 0.3),
        )

    # ── Intrinsic scaling helper ──────────────────────────────────────────────

    def _scale_intrinsics(
        self,
        K: torch.Tensor,    # (B, 3, 3) original image intrinsics
        fH: int, fW: int,   # feature map spatial size
    ) -> torch.Tensor:
        """
        Scale K from original image resolution to feature map resolution.
        DA2 ViT patch size = 14, so fH = img_H // 14, fW = img_W // 14.
        """
        K_feat = K.clone()
        K_feat[:, 0, :] *= (fW / self.img_w)   # scale x (columns)
        K_feat[:, 1, :] *= (fH / self.img_h)   # scale y (rows)
        return K_feat                            # (B, 3, 3)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,   # (B, 3, H, W)
        K:     torch.Tensor,   # (B, 3, 3)
        E:     torch.Tensor,   # (B, 4, 4)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, 1, H_bev, W_bev)  raw occupancy logits
        """
        # 1. Depth distribution + context features (DA2 encoder)
        depth_dist, context = self.encoder(image)        # (B,D,fH,fW), (B,C,fH,fW)
        fH, fW = context.shape[2], context.shape[3]

        # 2. Scale intrinsics to match the ViT patch-feature resolution
        K_feat = self._scale_intrinsics(K, fH, fW)       # (B, 3, 3)

        # 3. Lift-Splat: image frustum → BEV feature map
        bev_feat = self.lift_splat(depth_dist, context, K_feat, E)

        # 4. Decode BEV features → occupancy logits
        logits = self.bev_decoder(bev_feat)              # (B, 1, H_bev, W_bev)

        return logits


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model  = BEVOccupancyNet(cfg).to(device)

    B = 2
    image = torch.randn(B, 3, 448, 800).to(device)
    K     = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone().to(device)
    K[:, 0, 0] = 800; K[:, 1, 1] = 448
    K[:, 0, 2] = 400; K[:, 1, 2] = 224
    E     = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone().to(device)

    logits = model(image, K, E)
    print(f"Input image  : {image.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Device       : {logits.device}")
