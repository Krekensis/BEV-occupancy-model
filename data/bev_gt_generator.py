"""
bev_gt_generator.py
--------------------
Converts nuScenes LiDAR point clouds into 2D BEV occupancy grids.

For each sample:
  1. Load LiDAR points in sensor frame
  2. Transform to ego-vehicle frame using calibration
  3. Keep points within the configured BEV range
  4. Mark grid cells as occupied (1) if any point falls in them

Output: binary numpy array of shape (H_bev, W_bev) where
        H_bev = (y_range[1] - y_range[0]) / resolution
        W_bev = (x_range[1] - x_range[0]) / resolution
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


def get_bev_grid_shape(cfg: dict) -> tuple:
    """Return (H, W) of the BEV grid from config."""
    bev = cfg["bev"]
    H = int((bev["y_range"][1] - bev["y_range"][0]) / bev["resolution"])
    W = int((bev["x_range"][1] - bev["x_range"][0]) / bev["resolution"])
    return H, W


def lidar_to_ego(nusc: NuScenes, sample_token: str) -> np.ndarray:
    """
    Load LiDAR points and transform them from LiDAR sensor frame
    to ego-vehicle frame.

    Returns:
        points: (N, 3) array of XYZ in ego frame
    """
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    # Load raw point cloud (x, y, z, intensity, ring)
    pc = LidarPointCloud.from_file(
        nusc.dataroot + "/" + lidar_data["filename"]
    )

    # ── Transform: LiDAR sensor frame → ego frame ────────────────────────
    calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

    rotation    = Quaternion(calib["rotation"]).rotation_matrix      # (3, 3)
    translation = np.array(calib["translation"])                     # (3,)

    points_xyz = pc.points[:3, :].T                                  # (N, 3)
    points_ego = (rotation @ points_xyz.T).T + translation           # (N, 3)

    return points_ego                                                 # (N, 3)


def points_to_bev_grid(points_ego: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Project ego-frame XYZ points onto a 2D BEV occupancy grid.

    nuScenes ego frame convention:
        X → forward (front of car)
        Y → left
        Z → up

    BEV grid convention (matches typical image layout):
        row  0   = far  (max X / y_range[1])
        row  H-1 = near (min X / y_range[0])
        col  0   = left (min Y / x_range[0])  — note Y is lateral axis
        col  W-1 = right

    Args:
        points_ego: (N, 3) XYZ in ego frame
        cfg: config dict

    Returns:
        grid: (H, W) binary uint8 array
    """
    bev     = cfg["bev"]
    x_min, x_max = bev["y_range"]   # forward range  (ego X axis)
    y_min, y_max = bev["x_range"]   # lateral range  (ego Y axis)
    res          = bev["resolution"]

    H = int((x_max - x_min) / res)
    W = int((y_max - y_min) / res)

    x = points_ego[:, 0]   # forward
    y = points_ego[:, 1]   # lateral

    # Filter to BEV region
    mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    x, y = x[mask], y[mask]

    # Convert to grid indices
    row = H - 1 - ((x - x_min) / res).astype(int)   # flip so near = bottom
    col = ((y - y_min) / res).astype(int)

    # Clip just in case of floating point edge cases
    row = np.clip(row, 0, H - 1)
    col = np.clip(col, 0, W - 1)

    grid = np.zeros((H, W), dtype=np.uint8)
    grid[row, col] = 1

    return grid


def generate_bev_gt(nusc: NuScenes, sample_token: str, cfg: dict) -> np.ndarray:
    """
    Full pipeline: sample_token → binary BEV occupancy grid.

    Args:
        nusc:         NuScenes instance
        sample_token: token string for the sample
        cfg:          config dict (from default.yaml)

    Returns:
        grid: (H, W) binary uint8 array
    """
    points_ego = lidar_to_ego(nusc, sample_token)
    grid       = points_to_bev_grid(points_ego, cfg)
    return grid


# ── Quick visual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    import matplotlib.pyplot as plt

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    nusc = NuScenes(
        version=cfg["data"]["version"],
        dataroot=cfg["data"]["nuscenes_root"],
        verbose=True
    )

    # Test on first sample
    sample_token = nusc.sample[0]["token"]
    grid = generate_bev_gt(nusc, sample_token, cfg)

    print(f"Grid shape     : {grid.shape}")
    print(f"Occupied cells : {grid.sum()} / {grid.size}")
    print(f"Occupancy ratio: {grid.mean():.4f}")

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray", origin="upper")
    plt.title("BEV Occupancy GT (white = occupied)")
    plt.xlabel("Lateral (Y)")
    plt.ylabel("Forward (X) — ego at bottom")
    plt.tight_layout()
    plt.savefig("bev_gt_test.png", dpi=150)
    print("Saved bev_gt_test.png")
    plt.show()