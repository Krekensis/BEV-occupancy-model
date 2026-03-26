"""
backbone.py
-----------
ResNet-50 image feature extractor.
Strips the classification head and returns multi-scale feature maps.

Output (for input (B, 3, 448, 800)):
    features: (B, 2048, 14, 25)   — stride-32 final feature map
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    Pretrained ResNet-50 up to layer4 (stride 32).
    Returns the spatial feature map before average-pooling.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base    = resnet50(weights=weights)

        # Keep everything except avgpool and fc
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # stride  4  → 256 ch
        self.layer2 = base.layer2   # stride  8  → 512 ch
        self.layer3 = base.layer3   # stride 16  → 1024 ch
        self.layer4 = base.layer4   # stride 32  → 2048 ch

        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  normalised RGB image

        Returns:
            feat: (B, 2048, H/32, W/32)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ResNetBackbone(pretrained=False)
    dummy = torch.randn(2, 3, 448, 800)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expect (2, 2048, 14, 25)