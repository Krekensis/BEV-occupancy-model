"""
infer.py
--------
Run inference on a single front-camera image and visualise the BEV prediction.

Usage:
    python infer.py --checkpoint checkpoints/best.pth --sample_idx 0
"""

import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes

from model.bev_occupancy_net import BEVOccupancyNet
from data.nuscenes_loader    import NuScenesDataset
from utils.visualization     import denormalise_image, plot_bev_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Index into the val split")
    p.add_argument("--save",       default="infer_output.png")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    model = BEVOccupancyNet(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"best IoU={ckpt.get('best_iou', '?'):.4f})")

    # ── Load one val sample ───────────────────────────────────────────────────
    val_ds = NuScenesDataset(cfg, split="val")
    item   = val_ds[args.sample_idx]

    image  = item["image"].unsqueeze(0).to(device)   # (1, 3, H, W)
    K      = item["K"].unsqueeze(0).to(device)        # (1, 3, 3)
    E      = item["E"].unsqueeze(0).to(device)        # (1, 4, 4)
    gt     = item["bev_gt"]                           # (H_bev, W_bev)

    # ── Inference ─────────────────────────────────────────────────────────────
    logits = model(image, K, E)                       # (1, 1, H_bev, W_bev)
    prob   = torch.sigmoid(logits[0, 0]).cpu()        # (H_bev, W_bev)

    # ── Plot ──────────────────────────────────────────────────────────────────
    img_np = denormalise_image(item["image"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"BEV Occupancy Inference | token: {item['token'][:12]}...",
                 fontsize=11)

    axes[0].imshow(img_np)
    axes[0].set_title(f"Camera: {item['camera']}") 
    axes[0].axis("off")

    axes[1].imshow(gt.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT Occupancy (from LiDAR)")
    axes[1].set_xlabel("Lateral →"); axes[1].set_ylabel("Far ↑  Near ↓")

    im = axes[2].imshow(prob.numpy(), cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted BEV Occupancy")
    axes[2].set_xlabel("Lateral →")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.save}")
    plt.show()


if __name__ == "__main__":
    main()