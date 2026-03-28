"""
bev_gt_generator.py
--------------------
Converts nuScenes LiDAR point clouds into 2D BEV occupancy grids.

For each sample:
  1. Load LiDAR points in sensor frame
  2. Transform to ego-vehicle frame using calibration
  3. Filter by height (remove ground / overhead returns)
  4. Keep points within the configured BEV XY range
  5. Mark grid cells as occupied (1) if any point falls in them

nuScenes ego frame convention:
    points_ego[:, 0] = forward  (X in nuScenes) → rows in BEV
    points_ego[:, 1] = lateral  (Y in nuScenes, positive = LEFT) → cols in BEV
    points_ego[:, 2] = up       (Z in nuScenes) → used for height filter

Config convention:
    lateral_range : [min, max]  left/right  e.g. [-25, 25]  → W (cols)
    forward_range : [min, max]  front/back  e.g. [-25, 25]  → H (rows)

Grid convention:
    row 0       = far front  (+fwd_max)
    row H-1     = far back   (+fwd_min)
    row H//2    = ego        (forward=0)
    col 0       = far LEFT   (+lat_max)  ← nuScenes Y+ = left
    col W-1     = far RIGHT  (+lat_min)
    col W//2    = ego        (lateral=0)

Output: binary numpy array of shape (H, W) where
        H = (forward_range[1]  - forward_range[0])  / resolution
        W = (lateral_range[1]  - lateral_range[0])  / resolution
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


def get_bev_grid_shape(cfg: dict) -> tuple:
    """Return (H, W) of the BEV grid from config."""
    bev     = cfg["bev"]
    lat_min, lat_max = bev["lateral_range"]
    fwd_min, fwd_max = bev["forward_range"]
    res              = bev["resolution"]
    H = int((fwd_max - fwd_min) / res)   # forward → rows
    W = int((lat_max - lat_min) / res)   # lateral → cols
    return H, W


def lidar_to_ego(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Load LiDAR points and transform from LiDAR sensor frame to ego frame.

    Returns:
        points_ego: (N, 3)  columns = [forward, lateral, up]
    """
    sample      = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data  = nusc.get("sample_data", lidar_token)

    pc = LidarPointCloud.from_file(
        nusc.dataroot + "/" + lidar_data["filename"]
    )

    calib       = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    rotation    = Quaternion(calib["rotation"]).rotation_matrix   # (3, 3)
    translation = np.array(calib["translation"])                  # (3,)

    points_xyz  = pc.points[:3, :].T                              # (N, 3)
    points_ego  = (rotation @ points_xyz.T).T + translation       # (N, 3)

    return points_ego   # [:, 0]=forward  [:, 1]=lateral(+left)  [:, 2]=up


def points_to_bev_grid(points_ego: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Project ego-frame points onto a 2D BEV occupancy grid.

    BUG FIX #6: Added height (Z) filtering to exclude ground plane returns,
    overhead objects, and the ego vehicle's own roof — all of which would
    pollute the GT if left in.

    Args:
        points_ego: (N, 3) — forward (X), lateral (Y), up (Z) in ego frame
        cfg:        config dict

    Returns:
        grid: (H, W) binary uint8 — 1=occupied, 0=empty
    """
    bev = cfg["bev"]

    lat_min, lat_max = bev["lateral_range"]   # left/right
    fwd_min, fwd_max = bev["forward_range"]   # front/back
    res              = bev["resolution"]

    # Height filter bounds (metres in ego frame, configurable)
    z_min = bev.get("z_min", -1.5)   # below = ground / undercarriage
    z_max = bev.get("z_max",  3.0)   # above = overhead wires, bridges

    H = int((fwd_max - fwd_min) / res)   # forward → rows
    W = int((lat_max - lat_min) / res)   # lateral → cols

    forward = points_ego[:, 0]   # nuScenes X — front of car is positive
    lateral = points_ego[:, 1]   # nuScenes Y — LEFT is positive
    up      = points_ego[:, 2]   # nuScenes Z — up is positive

    # Keep only points inside the BEV region AND within height bounds
    mask = (
        (forward >= fwd_min) & (forward <  fwd_max) &
        (lateral >= lat_min) & (lateral <  lat_max) &
        (up      >= z_min)   & (up      <  z_max)
    )
    forward = forward[mask]
    lateral = lateral[mask]

    # row 0 = far front (fwd_max), row H-1 = far back (fwd_min)
    row = ((fwd_max - forward) / res).astype(int)

    # col 0 = far left (lat_max), col W-1 = far right (lat_min)
    col = ((lat_max - lateral) / res).astype(int)

    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    grid = np.zeros((H, W), dtype=np.uint8)
    grid[row, col] = 1
    return grid


def generate_bev_gt(nusc: NuScenes, sample_token: str, cfg: dict) -> np.ndarray:
    """
    Full pipeline: sample_token → binary BEV occupancy grid.
    """
    points_ego = lidar_to_ego(nusc, sample_token)
    grid       = points_to_bev_grid(points_ego, cfg)
    return grid


# ── Quick visual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    import yaml
    import matplotlib.pyplot as plt

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    nusc = NuScenes(
        version=cfg["data"]["version"],
        dataroot=cfg["data"]["nuscenes_root"],
        verbose=True
    )

    sample = random.choice(nusc.sample)
    grid   = generate_bev_gt(nusc, sample["token"], cfg)
    H, W   = grid.shape

    bev = cfg["bev"]
    lat_min, lat_max = bev["lateral_range"]
    fwd_min, fwd_max = bev["forward_range"]

    print(f"Grid shape     : {grid.shape}  (H={H} rows=forward, W={W} cols=lateral)")
    print(f"Occupied cells : {grid.sum()} / {grid.size}")
    print(f"Occupancy ratio: {grid.mean():.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="gray", origin="upper")

    ax.set_xticks(np.linspace(0, W, 5))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(lat_max, lat_min, 5)])
    ax.set_yticks(np.linspace(0, H, 5))
    ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(fwd_max, fwd_min, 5)])

    ax.set_xlabel("Lateral — left (+) | right (-) [m]")
    ax.set_ylabel("Forward (+) | Back (-) [m]")
    ax.set_title("BEV Occupancy GT")

    ax.plot(W // 2, H // 2, "r+", markersize=15, markeredgewidth=2, label="ego")
    ax.axhline(H // 2, color="red", linewidth=0.5, alpha=0.3)
    ax.axvline(W // 2, color="red", linewidth=0.5, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("bev_gt_test.png", dpi=150)
    print("Saved bev_gt_test.png")
    plt.show()