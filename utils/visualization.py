"""
visualization.py
----------------
Utilities for visualising BEV predictions during training and inference.

    plot_bev_comparison  : side-by-side GT vs prediction heatmap
    overlay_bev_on_image : project BEV grid back onto the front camera image
    save_batch_viz       : save a grid of visualisations for a whole batch
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalise_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised (3, H, W) tensor back to uint8 HWC numpy image.
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)      # HWC
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def plot_bev_comparison(
    gt:     torch.Tensor,    # (H, W) binary float
    logits: torch.Tensor,    # (1, H, W) or (H, W) raw logits
    title:  str = "",
    save_path: str = None,
) -> None:
    """
    Plot GT occupancy alongside predicted occupancy.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)

    pred_prob = torch.sigmoid(logits).cpu().numpy()
    gt_np     = gt.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(gt_np, cmap="gray", vmin=0, vmax=1, origin="upper")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Lateral →"); axes[0].set_ylabel("← Far | Near →")

    axes[1].imshow(pred_prob, cmap="hot", vmin=0, vmax=1, origin="upper")
    axes[1].set_title("Predicted Probability")
    axes[1].set_xlabel("Lateral →")

    # Difference map
    pred_bin = (pred_prob > 0.5).astype(float)
    diff     = np.zeros((*gt_np.shape, 3))
    diff[..., 0] = np.clip(pred_bin - gt_np, 0, 1)   # False Positive → red
    diff[..., 2] = np.clip(gt_np - pred_bin, 0, 1)   # False Negative → blue
    axes[2].imshow(diff, origin="upper")
    axes[2].set_title("Error (red=FP, blue=FN)")
    axes[2].set_xlabel("Lateral →")

    fp_patch = mpatches.Patch(color="red",  label="False Positive")
    fn_patch = mpatches.Patch(color="blue", label="False Negative")
    axes[2].legend(handles=[fp_patch, fn_patch], loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_batch_viz(
    batch:  dict,
    logits: torch.Tensor,   # (B, 1, H_bev, W_bev)
    epoch:  int,
    step:   int,
    out_dir: str = "logs/viz",
) -> None:
    """
    Save visualisations for the first 4 items in a batch.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    B = min(4, logits.shape[0])

    for i in range(B):
        img_np = denormalise_image(batch["image"][i])
        gt     = batch["bev_gt"][i]                    # (H, W)
        lg     = logits[i].detach().cpu()              # (1, H, W)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Epoch {epoch} | Step {step} | Sample {i}", fontsize=11)

        axes[0].imshow(img_np)
        axes[0].set_title("Front Camera")
        axes[0].axis("off")

        axes[1].imshow(gt.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("BEV GT")
        axes[1].axis("off")

        axes[2].imshow(torch.sigmoid(lg.squeeze()).numpy(),
                       cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("BEV Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = f"{out_dir}/epoch{epoch:03d}_step{step:05d}_sample{i}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()