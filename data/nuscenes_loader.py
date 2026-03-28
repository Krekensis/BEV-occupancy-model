"""
nuscenes_loader.py
------------------
PyTorch Dataset that returns:
    image  : (3, H, W)  float32 tensor  — normalised RGB
    K      : (3, 3)     float32 tensor  — camera intrinsic matrix
    E      : (4, 4)     float32 tensor  — cam-to-ego extrinsic (R|t)
    bev_gt : (H_bev, W_bev) float32 tensor — binary occupancy ground truth

Key fixes vs. original:
    BUG FIX #67 — Augmentation order: horizontal flip is applied *before*
                  ColorJitter/RandomErasing are baked in via Compose, so
                  geometric and photometric augmentations are fully consistent.
    BUG FIX #69 — BEV GT is pre-computed and cached in memory after the first
                  epoch so LiDAR point clouds are not re-parsed on every access.
    BUG FIX #42 — Train/val scene split is sorted + seeded for reproducibility.
    BUG FIX #00 — pin_memory is conditioned on CUDA availability.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from pyquaternion import Quaternion
import yaml

from nuscenes.nuscenes import NuScenes
from data.bev_gt_generator import generate_bev_gt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ALL_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg     = cfg
        self.split   = split
        self.cameras = cfg["data"].get("cameras", ALL_CAMERAS)

        img_h, img_w = cfg["data"]["image_size"]

        # Separate the photometric transforms from the geometric
        # flip so we can apply flip first (before baking in augmentation).
        # _photo_transform is applied to a PIL image; _tensor_transform converts and normalises.  
        # For val, there is no photometric augmentation.
        if split == "train":
            self._photo_transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            self._photo_transform = transforms.Resize((img_h, img_w))

        self._tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._erasing = (
            transforms.RandomErasing(p=0.2) if split == "train" else None
        )

        self.nusc = NuScenes(
            version=cfg["data"]["version"],
            dataroot=cfg["data"]["nuscenes_root"],
            verbose=False,
        )

        all_scenes   = sorted(self.nusc.scene, key=lambda s: s["token"])
        n_train      = int(len(all_scenes) * 0.8)
        train_scenes = {s["token"] for s in all_scenes[:n_train]}
        val_scenes   = {s["token"] for s in all_scenes[n_train:]}
        scene_set    = train_scenes if split == "train" else val_scenes

        self.items = []
        for s in self.nusc.sample:
            if s["scene_token"] not in scene_set:
                continue
            for cam in self.cameras:
                if cam in s["data"]:
                    self.items.append((s, cam))

        n_samples = len({id(s) for s, _ in self.items})
        print(f"[NuScenesDataset] {split}: {len(self.items)} items "
              f"({n_samples} samples × {len(self.cameras)} cameras)")

        self._gt_cache: dict[str, np.ndarray] = {}

    # ── Augmentation helpers ───────────────────────────────────────────────────

    def _apply_flip(
        self,
        image: Image.Image,
        bev_gt: np.ndarray,
        K: np.ndarray,
    ):
        """
        Random horizontal flip applied to the *PIL* image (before ToTensor)
        and consistently to the BEV GT and intrinsic matrix.

        FIX: Flip is now applied at PIL stage, before photometric
        augmentation is locked in, so erasing patches are applied to the
        correct (post-flip) image.
        """
        if random.random() > 0.5:
            image  = TF.hflip(image)
            bev_gt = np.fliplr(bev_gt).copy()   # lateral axis mirror
            K      = K.copy()
            img_w  = self.cfg["data"]["image_size"][1]
            K[0, 2] = img_w - K[0, 2]           # shift principal point
        return image, bev_gt, K

    # ── Calibration helpers ────────────────────────────────────────────────────

    def _get_intrinsics(self, sample: dict, camera: str) -> np.ndarray:
        sd_token = sample["data"][camera]
        sd       = self.nusc.get("sample_data", sd_token)
        calib    = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return np.array(calib["camera_intrinsic"], dtype=np.float32)

    def _get_extrinsics(self, sample: dict, camera: str) -> np.ndarray:
        sd_token = sample["data"][camera]
        sd       = self.nusc.get("sample_data", sd_token)
        calib    = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        R = Quaternion(calib["rotation"]).rotation_matrix
        t = np.array(calib["translation"], dtype=np.float32)
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R
        E[:3,  3] = t
        return E

    def _load_image(self, sample: dict, camera: str) -> Image.Image:
        sd_token = sample["data"][camera]
        sd       = self.nusc.get("sample_data", sd_token)
        img_path = os.path.join(self.nusc.dataroot, sd["filename"])
        return Image.open(img_path).convert("RGB")

    # ── GT cache ──────────────────────────────────────────────────────────────

    def _get_bev_gt(self, sample_token: str) -> np.ndarray:
        """Return cached BEV GT, computing it on first access."""
        if sample_token not in self._gt_cache:
            self._gt_cache[sample_token] = generate_bev_gt(
                self.nusc, sample_token, self.cfg
            )
        return self._gt_cache[sample_token]

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sample, camera = self.items[idx]

        image  = self._load_image(sample, camera)   # PIL
        K      = self._get_intrinsics(sample, camera)
        bev_gt = self._get_bev_gt(sample["token"])  # (H, W) uint8, possibly cached

        if self.split == "train":
            image, bev_gt, K = self._apply_flip(image, bev_gt, K)

        # Photometric augmentation + resize (PIL → PIL)
        image = self._photo_transform(image)

        # PIL → normalised tensor
        image = self._tensor_transform(image)        # (3, H, W)

        # RandomErasing after tensor conversion, after flip (fix #3)
        if self._erasing is not None:
            image = self._erasing(image)

        E = torch.from_numpy(self._get_extrinsics(sample, camera))
        K = torch.from_numpy(K)
        bev_gt_tensor = torch.from_numpy(bev_gt.copy()).float()

        return {
            "image":  image,
            "K":      K,
            "E":      E,
            "bev_gt": bev_gt_tensor,
            "token":  sample["token"],
            "camera": camera,
        }


def build_dataloaders(cfg: dict):
    train_ds = NuScenesDataset(cfg, split="train")
    val_ds   = NuScenesDataset(cfg, split="val")

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=use_pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=use_pin,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader = build_dataloaders(cfg)
    batch = next(iter(train_loader))

    print("image  :", batch["image"].shape)
    print("K      :", batch["K"].shape)
    print("E      :", batch["E"].shape)
    print("bev_gt :", batch["bev_gt"].shape)
    print("camera :", batch["camera"])
    print("token  :", batch["token"][0])
