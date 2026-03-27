"""
visualize_predictions.py
------------------------
Run this script to visualise model predictions vs GT.
Usage:
    python visualize_predictions.py
    python visualize_predictions.py --samples 10
    python visualize_predictions.py --split train
"""

import os
import sys
import argparse
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, "/kaggle/working/bev_occupancy")

from model.bev_occupancy_net import BEVOccupancyNet
from data.nuscenes_loader    import NuScenesDataset
from utils.visualization     import denormalise_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--checkpoint", default="/kaggle/working/checkpoints/best.pth")
    p.add_argument("--split",      default="val")
    p.add_argument("--samples",    type=int, default=5)
    p.add_argument("--out_dir",    default="/kaggle/working/predictions")
    return p.parse_args()


@torch.no_grad()
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load model
    model = BEVOccupancyNet(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint — best IoU: {ckpt.get('best_iou', '?'):.4f}")

    # Load dataset
    ds = NuScenesDataset(cfg, split=args.split)
    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(min(args.samples, len(ds))):
        item   = ds[i]
        image  = item["image"].unsqueeze(0).to(device)
        K      = item["K"].unsqueeze(0).to(device)
        E      = item["E"].unsqueeze(0).to(device)
        gt     = item["bev_gt"]
        camera = item.get("camera", "CAM_FRONT")

        logits = model(image, K, E)
        prob   = torch.sigmoid(logits[0, 0]).cpu()
        img_np = denormalise_image(item["image"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Sample {i} | {camera}", fontsize=12)

        axes[0].imshow(img_np)
        axes[0].set_title("Camera Input")
        axes[0].axis("off")

        axes[1].imshow(gt.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT Occupancy (LiDAR)")
        axes[1].axis("off")

        im = axes[2].imshow(prob.numpy(), cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("Predicted Occupancy")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = f"{args.out_dir}/sample_{i:03d}_{camera}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")

    # Display all saved images inline
    from IPython.display import Image, display
    for i in range(min(args.samples, len(ds))):
        camera = ds[i].get("camera", "CAM_FRONT")
        path   = f"{args.out_dir}/sample_{i:03d}_{camera}.png"
        if os.path.exists(path):
            display(Image(path))


if __name__ == "__main__":
    main()