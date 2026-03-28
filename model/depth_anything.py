"""
depth_anything.py
-----------------
Wraps Depth Anything V2 (ViT-based) as a drop-in replacement for the
ResNetBackbone + DepthNet pair.

Instead of:
    backbone  → (B, 2048, fH, fW)
    depth_net → depth_dist (B, D, fH, fW)  +  context (B, C, fH, fW)

We now have a single module that produces the same two outputs by:
    1. Running the Depth Anything V2 encoder to get rich ViT patch features.
    2. Decoding a *metric* depth map (B, 1, fH, fW) using the DA2 DPT head,
       then converting it to a soft distribution over D discrete depth bins
       via a temperature-scaled softmax (differentiable — gradients flow back).
    3. Learning a lightweight context head on top of the ViT features.

Depth Anything V2 repo: https://github.com/DepthAnything/Depth-Anything-V2
Install:
    pip install depth-anything-v2
  or clone the repo and add it to PYTHONPATH.

Supported encoder sizes (set cfg["model"]["da2_encoder"]):
    "vits"  — ViT-Small  (fastest,  ~25 M params)
    "vitb"  — ViT-Base   (balanced, ~97 M params)   ← default
    "vitl"  — ViT-Large  (best,    ~335 M params)

The DA2 model weights are loaded from a local .pth file whose path is set
in cfg["model"]["da2_weights"].  Download from the DA2 release page.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/kaggle/working/Depth-Anything-V2")


# ---------------------------------------------------------------------------
# Encoder channel sizes per DA2 variant
# ---------------------------------------------------------------------------
_DA2_FEAT_CHANNELS = {
    "vits": 384,
    "vitb": 768,
    "vitl": 1024,
}


class DepthAnythingEncoder(nn.Module):
    """
    Wraps the Depth Anything V2 model and exposes:
        depth_dist : (B, D, fH, fW)   soft depth distribution over D bins
        context    : (B, C_ctx, fH, fW)  rich context features

    The DA2 metric depth output is used to build the distribution;
    the internal ViT patch features drive the context head.

    Args:
        cfg          : full config dict
        ctx_channels : number of context feature channels (C_ctx)
    """

    def __init__(self, cfg: dict, ctx_channels: int):
        super().__init__()

        model_cfg   = cfg["model"]
        depth_cfg   = cfg["depth"]
        encoder_key = model_cfg.get("da2_encoder", "vitb")   # "vits"|"vitb"|"vitl"
        weights_path = model_cfg.get("da2_weights", None)

        self.depth_min  = depth_cfg["min"]    # e.g. 4.0
        self.depth_max  = depth_cfg["max"]    # e.g. 50.0
        self.depth_bins = depth_cfg["bins"]   # D

        # ── Load Depth Anything V2 ───────────────────────────────────────────
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(
                "Depth Anything V2 is not installed.\n"
                "Clone https://github.com/DepthAnything/Depth-Anything-V2 and "
                "add it to your PYTHONPATH, or install via pip."
            ) from e

        # DA2 model configs
        model_configs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        assert encoder_key in model_configs, \
            f"da2_encoder must be one of {list(model_configs)}; got '{encoder_key}'"

        self.da2 = DepthAnythingV2(**model_configs[encoder_key])

        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            self.da2.load_state_dict(state)
            print(f"[DepthAnythingEncoder] Loaded DA2 weights from {weights_path}")
        else:
            print("[DepthAnythingEncoder] WARNING: no da2_weights path set — "
                  "using random initialisation.")

        # ── Freeze DA2 by default; set cfg model.freeze_da2=false to fine-tune ─
        if model_cfg.get("freeze_da2", True):
            for p in self.da2.parameters():
                p.requires_grad_(False)
            print("[DepthAnythingEncoder] DA2 weights frozen.")

        # ── Context head (on top of ViT patch features) ──────────────────────
        vit_ch = _DA2_FEAT_CHANNELS[encoder_key]   # raw patch embedding dim

        self.context_head = nn.Sequential(
            nn.Conv2d(vit_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ctx_channels, kernel_size=1),
        )

        # Learnable temperature for depth-bin softmax (log-scale for stability)
        self.log_temperature = nn.Parameter(torch.zeros(1))   # init T=1

        # Register bin centres as a buffer (no gradient)
        bins = torch.linspace(self.depth_min, self.depth_max, self.depth_bins)
        self.register_buffer("d_bins", bins)   # (D,)

    # -----------------------------------------------------------------------

    def _depth_to_distribution(
        self, depth_metric: torch.Tensor   # (B, 1, fH, fW)  metric depth
    ) -> torch.Tensor:
        """
        Convert a metric depth map to a soft distribution over D fixed bins.

        For each pixel with predicted depth d, we assign probability using
        a temperature-scaled Laplace-like kernel:

            logit_k = -|d - bin_k| / T

        then softmax over k.  This is differentiable and lets the network
        learn the sharpness of the distribution.

        Returns:
            dist: (B, D, fH, fW)
        """
        B, _, fH, fW = depth_metric.shape
        D = self.depth_bins
        T = self.log_temperature.exp().clamp(min=1e-3)   # (1,)

        # Clamp depth to valid range before computing distances
        depth_clamped = depth_metric.clamp(self.depth_min, self.depth_max)

        # (B, 1, fH, fW) vs (1, D, 1, 1)
        bins = self.d_bins.view(1, D, 1, 1)
        dist_to_bins = (depth_clamped - bins).abs()       # (B, D, fH, fW)
        logits = -dist_to_bins / T                        # (B, D, fH, fW)

        return logits.softmax(dim=1)                      # (B, D, fH, fW)

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: (B, 3, H, W)  ImageNet-normalised RGB

        Returns:
            depth_dist: (B, D, fH, fW)     soft depth distribution
            context:    (B, C_ctx, fH, fW) context features
        """
        B, _, H, W = image.shape
        patch_size = 14
        pH = H // patch_size
        pW = W // patch_size

        # Run ViT once, get both depth and patch features
        with torch.set_grad_enabled(not self._da2_frozen()):
            layer_feats = self.da2.pretrained.get_intermediate_layers(
                image, n=1, return_class_token=False
            )   # list of (B, pH*pW, embed_dim)
            depth_metric = self.da2(image)   # (B, H, W)

        # Patch features → spatial grid
        feats = layer_feats[0]                      # (B, pH*pW, embed_dim)
        embed_dim = feats.shape[-1]
        patch_feats = feats.permute(0, 2, 1).reshape(B, embed_dim, pH, pW)

        # Resize depth to patch resolution
        depth_resized = F.interpolate(
            depth_metric.unsqueeze(1),
            size=(pH, pW),
            mode="bilinear",
            align_corners=False,
        )

        depth_dist = self._depth_to_distribution(depth_resized)
        context    = self.context_head(patch_feats)

        return depth_dist, context

    def _extract_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract last-block ViT patch features from DA2's pretrain backbone.

        DA2 uses DINOv2 as pretrain; its get_intermediate_layers() returns
        a list of (B, num_patches, embed_dim) tensors.

        Returns:
            feats: (B, embed_dim, pH, pW)
        """
        B, _, H, W = image.shape
        patch_size = 14   # DA2 / DINOv2 patch size

        # Ensure image dimensions are divisible by patch_size
        pH = H // patch_size
        pW = W // patch_size

        # get_intermediate_layers returns list; take last layer
        with torch.set_grad_enabled(not self._da2_frozen()):
            layer_feats = self.da2.pretrained.get_intermediate_layers(
                image, n=1, return_class_token=False
            )   # list of (B, pH*pW, embed_dim)

        feats = layer_feats[0]                     # (B, pH*pW, embed_dim)
        embed_dim = feats.shape[-1]
        feats = feats.permute(0, 2, 1)             # (B, embed_dim, pH*pW)
        feats = feats.reshape(B, embed_dim, pH, pW)

        return feats   # (B, embed_dim, pH, pW)

    def _da2_frozen(self) -> bool:
        """Check if DA2 params are frozen (used for grad context)."""
        return not next(self.da2.parameters()).requires_grad
