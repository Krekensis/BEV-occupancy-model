"""
train.py
--------
Full training loop for BEVOccupancyNet.

Usage:
    python train.py
    python train.py --config configs/default.yaml
    python train.py --resume checkpoints/last.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from data.nuscenes_loader    import build_dataloaders
from model.bev_occupancy_net import BEVOccupancyNet
from utils.metrics           import occupancy_iou, distance_weighted_error
from utils.visualization     import save_batch_viz


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--resume", default=None)
    return p.parse_args()


# ── Distance-weighted BCE loss ────────────────────────────────────────────────

def distance_weighted_bce(
    logits: torch.Tensor,   # (B, 1, H, W)
    gt:     torch.Tensor,   # (B, H, W)
    cfg:    dict,
) -> torch.Tensor:
    """
    BCE loss where cells closer to the ego vehicle are penalised more heavily.

    Weight map (per row):
        row H//2  (ego center) → weight = 1.0   ← highest
        row 0     (far front)  → weight ≈ 0     ← lowest
        row H-1   (far back)   → weight ≈ 0     ← lowest

    This makes the model prioritise accuracy near the car, which matters
    most for real driving decisions.
    """
    device = logits.device
    H      = logits.shape[2]
    center = H // 2

    row_idx = torch.arange(H, device=device).float()   # (H,)
    dist    = (row_idx - center).abs()                 # (H,)

    weights = 1.0 / (1.0 + dist)
    weights = weights / weights.max()

    weights = weights.view(1, 1, H, 1)

    pos_weight = torch.tensor(
        [cfg["train"]["pos_weight"]], device=device
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        gt.unsqueeze(1).float(),
        pos_weight=pos_weight,
        reduction="none", # (B, 1, H, W)
    )

    loss = (bce * weights).mean()
    return loss


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: str, model, optimizer, scheduler, device):

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_iou    = ckpt.get("best_iou", 0.0)
    print(f"  [ckpt] Resumed from epoch {ckpt['epoch']}  (best IoU={best_iou:.4f})")
    return start_epoch, best_iou


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, epoch, writer, cfg):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for step, batch in enumerate(pbar):
        image = batch["image"].to(device)
        K     = batch["K"].to(device)
        E     = batch["E"].to(device)
        gt    = batch["bev_gt"].to(device)

        optimizer.zero_grad()

        logits = model(image, K, E) # (B, 1, H, W)
        loss   = distance_weighted_bce(logits, gt, cfg)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        iou = occupancy_iou(logits.detach(), gt)
        total_loss += loss.item()
        total_iou  += iou

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

        global_step = epoch * len(loader) + step
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        writer.add_scalar("train/iou_step",  iou,         global_step)

        if epoch == 0 and step == 0:
            save_batch_viz(batch, logits.detach().cpu(), epoch, step,
                           out_dir=cfg["paths"]["logs"] + "/viz")

    n = len(loader)
    return total_loss / n, total_iou / n


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device, epoch, writer, cfg):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    total_dwe  = 0.0

    bev = cfg["bev"]
    H_bev = int(
        (bev["forward_range"][1] - bev["forward_range"][0]) / bev["resolution"]
    )

    pbar = tqdm(loader, desc=f"Val   E{epoch}", leave=False)
    for step, batch in enumerate(pbar):
        image = batch["image"].to(device)
        K     = batch["K"].to(device)
        E     = batch["E"].to(device)
        gt    = batch["bev_gt"].to(device)

        logits = model(image, K, E)
        loss   = distance_weighted_bce(logits, gt, cfg)

        iou = occupancy_iou(logits, gt)
        dwe = distance_weighted_error(
            logits, gt, H_bev,
            bev["forward_range"][0],
            bev["forward_range"][1],
        )

        total_loss += loss.item()
        total_iou  += iou
        total_dwe  += dwe

        if step == 0:
            save_batch_viz(batch, logits.cpu(), epoch, step,
                           out_dir=cfg["paths"]["logs"] + "/viz_val")

    n        = len(loader)
    avg_loss = total_loss / n
    avg_iou  = total_iou  / n
    avg_dwe  = total_dwe  / n

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/iou",  avg_iou,  epoch)
    writer.add_scalar("val/dwe",  avg_dwe,  epoch)

    return avg_loss, avg_iou, avg_dwe


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path(cfg["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model     = BEVOccupancyNet(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    writer = SummaryWriter(log_dir=cfg["paths"]["logs"])

    start_epoch = 0
    best_iou    = 0.0

    if args.resume:
        start_epoch, best_iou = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    patience   = cfg["train"].get("early_stopping_patience", 10)
    no_improve = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_loss, train_iou = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, cfg
        )
        val_loss, val_iou, val_dwe = val_epoch(
            model, val_loader, device, epoch, writer, cfg
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} iou={train_iou:.4f} | "
            f"val loss={val_loss:.4f} iou={val_iou:.4f} dwe={val_dwe:.4f}"
        )

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("train/iou_epoch",  train_iou,  epoch)

        is_best = val_iou > best_iou
        if is_best:
            best_iou   = val_iou
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(
            {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_iou":  best_iou,
            },
            path=f"{cfg['paths']['checkpoints']}/last.pth",
        )

        if is_best:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(), "best_iou": best_iou},
                path=f"{cfg['paths']['checkpoints']}/best.pth",
            )
            print(f"  ★ New best IoU: {best_iou:.4f}")
        else:
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    writer.close()
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()