"""
bev_occupancy_net.py
---------------------
Assembles the full pipeline:

    RGB image (B, 3, H, W)
        ↓  ResNetBackbone
    features (B, 2048, fH, fW)
        ↓  DepthNet
    depth_dist (B, D, fH, fW)  +  context (B, C, fH, fW)
        ↓  LiftSplat  [uses K, E from nuScenes calibration]
    bev_feat (B, C, H_bev, W_bev)
        ↓  BEVDecoder
    logits (B, 1, H_bev, W_bev)
"""

import torch
import torch.nn as nn

from model.backbone    import ResNetBackbone
from model.depth_net   import DepthNet
from model.lift_splat  import LiftSplat
from model.bev_decoder import BEVDecoder


class BEVOccupancyNet(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()

        ctx_ch     = cfg["model"]["bev_channels"]   # 64
        depth_bins = cfg["depth"]["bins"]            # 112
        img_h, img_w = cfg["data"]["image_size"]    # [448, 800]

        self.img_h = img_h
        self.img_w = img_w

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.backbone = ResNetBackbone(pretrained=cfg["model"]["pretrained"])
        self.depth_net = DepthNet(
            in_channels=self.backbone.out_channels,   # 2048
            depth_bins=depth_bins,
            ctx_channels=ctx_ch,
        )
        self.lift_splat = LiftSplat(cfg, ctx_channels=ctx_ch)
        self.bev_decoder = BEVDecoder(
            in_channels=ctx_ch,
            dropout=cfg["model"].get("dropout", 0.3)
        )

    # ── Intrinsic scaling helper ──────────────────────────────────────────────

    def _scale_intrinsics(
        self,
        K: torch.Tensor,    # (B, 3, 3) original image intrinsics
        fH: int, fW: int,   # feature map spatial size
    ) -> torch.Tensor:
        """
        Scale K from original image resolution to feature map resolution.
        ResNet-50 uses stride 32, so fH = img_H/32, fW = img_W/32.
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
        # 1. Extract image features
        features = self.backbone(image)                        # (B, 2048, fH, fW)
        fH, fW   = features.shape[2], features.shape[3]

        # 2. Predict depth distribution + context
        depth_dist, context = self.depth_net(features)        # (B, D, fH, fW), (B, C, fH, fW)

        # 3. Scale intrinsics to feature map resolution
        K_feat = self._scale_intrinsics(K, fH, fW)            # (B, 3, 3)

        # 4. Lift-Splat: image frustum → BEV feature map
        bev_feat = self.lift_splat(depth_dist, context, K_feat, E)  # (B, C, H_bev, W_bev)

        # 5. Decode BEV features → occupancy logits
        logits = self.bev_decoder(bev_feat)                    # (B, 1, H_bev, W_bev)

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
    print(f"Input image : {image.shape}")
    print(f"Output logits: {logits.shape}")   # (2, 1, 100, 100)
    print(f"Device: {logits.device}")