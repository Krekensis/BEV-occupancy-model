# BEV Occupancy Network

## Executive Summary

This project implements a **Bird's-Eye-View (BEV) Occupancy Prediction Network** that predicts which grid cells in a 2D top-down view of the road are occupied (contain objects/obstacles) using only a single front-facing camera image. The model is trained and evaluated on the **nuScenes dataset** and is designed for autonomous driving applications where understanding 3D scene geometry from 2D camera images is crucial for safe navigation.

---

## Table of Contents

1. [Overview](#overview)
   - What the project does
   - Why BEV occupancy matters
   - Key features
2. [Technical Architecture](#technical-architecture)
   - Pipeline overview
   - Core components
3. [Methods & Algorithms](#methods--algorithms)
   - Depth estimation via Depth Anything V2
   - Lift-Splat transformation
   - BEV occupancy ground truth generation
   - Training methodology
4. [Dataset & Data Processing](#dataset--data-processing)
5. [Model Components](#model-components)
6. [Training & Evaluation](#training--evaluation)
7. [Usage](#usage)
8. [File Structure](#file-structure)

---

## Overview

### What We're Doing

**Problem**: Autonomous vehicles need to understand the spatial layout of their environment in real-time. While LiDAR sensors provide direct 3D information, they're expensive and have limited range. Can we infer occupancy from cameras alone?

**Solution**: We built a **deep learning model** that:
- Takes a single front-facing camera image as input
- Predicts a 2D BEV occupancy grid showing which regions contain obstacles
- Uses camera intrinsics and extrinsics to transform 2D image features into 3D space
- Leverages modern monocular depth estimation (Depth Anything V2) for geometric understanding

### Why BEV Occupancy Matters

1. **Unified Representation**: BEV provides a canonical view independent of which camera is used
2. **Planning-Friendly**: Motion planning algorithms work naturally in BEV space (grid-based collision detection, sampling-based planning)
3. **Multi-View Fusion Ready**: Multiple camera views can be fused into a single BEV representation
4. **Scalability**: Unlike 3D object detection (which detects discrete objects), occupancy grids capture the full scene geometry including partial objects, curbs, vegetation, etc.

### Key Features

- **Depth Anything V2 Encoder**: State-of-the-art monocular depth estimation using Vision Transformers (ViT)
- **Lift-Splat Geometry**: Differentiable transformation from image frustum → BEV feature maps (from LSS: Lift, Splat, Shoot)
- **Real-Time Capable**: Efficient design suitable for deployment
- **Lightweight BEV Decoder**: Simple CNN-based occupancy decoder for interpretability
- **Distance-Weighted Loss**: Prioritizes accuracy near the ego vehicle where it matters most


---

## Technical Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Front Camera Image (3, H, W)                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ DepthAnythingEncoder (DA2)  │
        │  ViT-based monocular depth  │
        └──────────────┬──────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼─────┐           ┌────────▼──────┐
    │ Depth    │           │ Context       │
    │ Distrib. │           │ Features      │
    │(B,D,fH,fW)          │(B,C_ctx,fH,fW)│
    └────┬─────┘           └────────┬──────┘
         │                          │
        ┌└──────────────┬───────────┘
        │               │
        │  INPUT: Camera Calibration
        │  - K (intrinsics)
        │  - E (extrinsics)
        │
        │     ┌────────────────────────────┐
        ├────►│  LiftSplat Transformation  │
        │     │  Image → 3D → BEV          │
        │     └────────────────┬───────────┘
        │                      │
        │     ┌────────────────▼───────────┐
        └────►│ BEV Feature Map            │
              │ (B, C_ctx, H_bev, W_bev)   │
              └────────────┬───────────────┘
                           │
                   ┌───────▼────────┐
                   │  BEV Decoder   │
                   │  CNN layers    │
                   └───────┬────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │ OUTPUT: Occupancy Logits            │
        │ (B, 1, H_bev, W_bev)                │
        │ [Raw logits → apply sigmoid for    │
        │  probability in [0, 1]]             │
        └──────────────────────────────────────┘
```

### BEV Grid Convention

```
The BEV grid represents the ego-vehicle's surrounding space:

        ← Far Front  (forward_max)
     ┌────────────────────┐
     │    row 0           │  (fwd_max ≈ +25m ahead)
     │                    │
     │   row H//2         │  (ego center ≈ vehicle position)
     │   (ego center)     │
     │                    │
     │   row H-1          │  (fwd_min ≈ -25m behind)
     └────────────────────┘
     col 0            col W-1
     lat_max          lat_min
     (left)           (right)

Coordinate system (nuScenes ego frame):
  X (forward):  positive = ahead of vehicle
  Y (lateral):  positive = left (nuScenes convention)
  Z (vertical): positive = up

Each cell in the grid corresponds to a fixed spatial region (default 0.25m × 0.25m)
Cell value: 1 = occupied (obstacle present), 0 = free space
```

---

## Methods & Algorithms

### 1. Depth Estimation: Depth Anything V2

**What**: We use Depth Anything V2 (DA2), a foundation model trained on billions of images to predict metric depth from a single image.

**Architecture**:
- **Encoder**: Vision Transformer (ViT) variant (ViT-Small/Base/Large)
- **Decoder**: DPT-style decoder that gradually upsamples features to full resolution
- **Output**: Metric depth map in meters (per-pixel absolute depth)

**How It Works**:
1. Image is fed through the ViT encoder (patch size = 14 pixels)
   - Feature map size: (H/14) × (W/14) × channel_dim
2. ViT patches → rich feature embeddings (preserved in context head)
3. DPT decoder converts features to per-pixel depth map (B, 1, fH, fW)
4. **Critical conversion**: Metric depth → soft depth distribution
   - Depth map is converted to a **probability distribution over D discrete bins** (e.g., D=112 bins)
   - Uses temperature-scaled softmax: `P(d_i) = exp(log_prob_i / temp) / sum(exp(...))`
   - This is **differentiable**, allowing gradients to flow back through the depth estimation

**Why DA2?**
- Modern, well-trained on diverse imagery
- Robust to different lighting conditions, weather, textures
- Provides both depth and rich feature embeddings
- Can be frozen during training (transfer learning) or fine-tuned

**Code Location**: [model/depth_anything.py](model/depth_anything.py)

---

### 2. Lift-Splat: Camera → BEV Transformation

**Concept** (from Li et al. "Lift, Splat, Shoot"):

The core insight is to transform per-image depth distributions and features into a unified BEV coordinate frame through geometric projection.

#### **LIFT Phase**: Construct 3D Frustum

For each image pixel (u, v):

1. **Unproject to 3D Camera Space**:
   ```
   For each depth bin d in [d_min ... d_max]:
     - Pixel coord: [u, v, 1] (homogeneous)
     - Invert intrinsics K: P_cam = K^-1 @ [u, v, 1]ᵀ
     - Scale by depth: P_cam_3d = d × P_cam
     - Result: 3D point in camera frame
   ```

2. **Attach Depth Probability & Context**:
   - Weight each 3D point by its depth probability: `w_d = P(d_i)`
   - Associate context features C with each point
   - Create a "frustum": (D × fH × fW) weighted 3D points, each with:
     - 3D position in camera frame
     - Depth weight
     - Context feature vector

#### **SPLAT Phase**: Project to BEV Grid

1. **Transform to Ego Frame**:
   ```
   For each 3D point P_cam:
     P_ego = E @ [P_cam; 1]  (extrinsic matrix E is cam → ego)
   ```

2. **Map to BEV Grid**:
   ```
   For each point P_ego = [x, y, z]:
     Grid cell (row, col) based on x, y within BEV ranges:
       row = (x - fwd_min) / resolution
       col = (y - lat_min) / resolution
     (Check if cell is within grid bounds)
   ```

3. **Pool Features**:
   ```
   For each grid cell, sum all weighted features:
     BEV_feat[row, col] += sum(context_feature × depth_weight)
   ```

**Result**: (B, C_ctx, H_bev, W_bev) BEV feature map ready for decoding

**Code Location**: [model/lift_splat.py](model/lift_splat.py)

---

### 3. BEV Occupancy Ground Truth Generation

**Challenge**: How do we create training labels for occupancy?

**Solution**: Use LiDAR point clouds from nuScenes (available in the dataset).

#### **Process**:

1. **Load LiDAR Points**:
   - 32 LiDAR beams sampled at high frequency
   - Points recorded in sensor frame

2. **Transform to Ego Frame**:
   ```
   P_ego = E_lidar @ P_sensor
   Extract x, y coordinates
   ```

3. **Filter and Discretize into Grid**:
   ```
   For each 3D point P = [x, y, z]:
     if x in [fwd_min, fwd_max] and y in [lat_min, lat_max]:
       grid_row = int((x - fwd_min) / resolution)
       grid_col = int((y - lat_min) / resolution)
       BEV_GT[grid_row, grid_col] = 1
   ```

4. **Result**: Binary occupancy grid (H_bev, W_bev) where 1=occupied, 0=free

**Code Location**: [data/bev_gt_generator.py](data/bev_gt_generator.py)

---

### 4. Training Methodology

#### **Loss Function: Distance-Weighted BCE**

Standard Binary Cross-Entropy is used, but with a **spatial weighting scheme** that prioritizes accuracy near the vehicle:

```python
# Weight per row (decreases with distance from ego):
center = H_bev // 2
for row i:
    dist = |i - center|
    weight[i] = 1.0 / (1.0 + dist)
    
# Normalize weights to [0, 1]
weight = weight / weight.max()

# Apply to binary cross-entropy loss:
loss = BCE(logits, gt, weighted=weight) + pos_weight × positive_class_weight
```

**Intuition**:
- Row H//2 (ego center): weight ≈ 1.0 (highest)
- Rows 0 and H-1 (far edges): weight ≈ 0.1 (lowest)
- **Effect**: Mistakes near the vehicle hurt more → better safety margins

**Positive Weight**: Occupancy cells are typically sparser than free cells (class imbalance). We upweight positive examples via `pos_weight=5.0` in BCEWithLogitsLoss.

#### **Training Pipeline**:

```
1. Load batch of data:
   - image (B, 3, H_img, W_img)
   - K (B, 3, 3) intrinsics
   - E (B, 4, 4) extrinsics
   - bev_gt (B, H_bev, W_bev)

2. Forward pass:
   - image → Depth Anything Encoder → depth_dist, context
   - (depth_dist, context, K, E) → Lift-Splat → bev_feat
   - bev_feat → BEV Decoder → logits

3. Compute loss:
   loss = distance_weighted_bce(logits, bev_gt)

4. Backward & optimize:
   loss.backward()
   clip_grad_norm_(max_norm=5.0)  # Prevent exploding gradients
   optimizer.step()

5. Compute metrics:
   iou = occupancy_iou(logits, bev_gt)
```

#### **Key Hyperparameters**:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `batch_size` | 4 | Trade-off speed vs. memory |
| `lr` | 2.0e-4 | Learning rate for Adam optimizer |
| `weight_decay` | 1.0e-4 | L2 regularization |
| `pos_weight` | 5.0 | Upweight occupied (positive) cells |
| `dropout` | 0.3 | Regularization in BEV decoder |
| `epochs` | 50 | Total training runs |
| `early_stopping_patience` | 15 | Stop if val IoU doesn't improve for 15 epochs |

#### **Optimization Details**:

- **Optimizer**: Adam (adaptive learning rate)
- **Scheduler**: ReduceLROnPlateau (lower LR if validation loss plateaus)
- **Gradient Clipping**: max_norm=5.0 (prevent NaN from exploding gradients)
- **Checkpointing**: Save best model by validation IoU

**Code Location**: [train.py](train.py)

---

### 5. Evaluation Metrics

#### **Occupancy IoU (Intersection-over-Union)**

Standard metric for segmentation tasks:

```
pred = (sigmoid(logits) > threshold).float()  # threshold = 0.5
intersection = (pred * gt).sum()
union = ((pred + gt) > 0).float().sum()
IoU = intersection / union
```

- **Range**: [0, 1], higher is better
- **Interpretation**: Fraction of spatial area correctly predicted

#### **Distance-Weighted Error (DWE)**

Penalizes mistakes closer to the ego vehicle:

```
for row i:
    dist = |i - center|
    weight[i] = 1.0 / (1.0 + dist)

binary_error = |pred - gt|
weighted_error = mean(binary_error * weight)
```

- **Range**: [0, 1], lower is better
- **Use Case**: Safety-critical — errors near vehicle are more costly than far-field errors

**Code Location**: [utils/metrics.py](utils/metrics.py)

---

## Dataset & Data Processing

### NuScenes Dataset

**What**: Large-scale autonomous driving dataset with:
- 1000 driving scenes (~20 sec each) in Boston, Singapore, Las Vegas
- 6 cameras per vehicle (360° coverage)
- LiDAR point clouds (32 beams)
- Full sensor calibration (intrinsics K, extrinsics E)
- Labeled 3D bounding boxes

**Setup for This Project**:
- Using `nuscenes-mini` version (subset of 10 scenes for quick prototyping)
- Train/Val split: 80/20 of scenes (sorted & seeded for reproducibility)
- **Cameras Used**: CAM_FRONT by default (configurable to use multiple)
- **LiDAR**: Used only for ground-truth occupancy label generation

### Data Augmentation

#### **On PIL Image (before ToTensor)**:

1. **Horizontal Flip** (50% probability):
   - Flip image left-right
   - Mirror BEV grid laterally
   - Adjust principal point of K: `K[0, 2] = W - K[0, 2]`

#### **On PIL Image (photometric)**:

1. **ColorJitter** (brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
2. **RandomGrayscale** (p=0.1)
3. **Resize** to configured resolution (e.g., 448×798)

#### **On Tensor (after ToTensor)**:

1. **RandomErasing** (p=0.2) for robustness to occlusions
2. **Normalize** using ImageNet statistics

**Rationale**: Augmentation improves generalization to diverse scenes and lighting conditions.

**Code Location**: [data/nuscenes_loader.py](data/nuscenes_loader.py)

---

## Model Components

### 1. DepthAnythingEncoder

**File**: [model/depth_anything.py](model/depth_anything.py)

**Inputs**:
- RGB image (B, 3, H_img, W_img)

**Outputs**:
- `depth_dist`: (B, D, fH, fW) — soft depth distribution over D bins
- `context`: (B, C_ctx, fH, fW) — context features for BEV

**Architecture**:
```
Input Image (B, 3, 448, 800)
    ↓
ViT Encoder (patch_size=14)
    ↓ (features at H/14, W/14 = 32×57)
DPT Decoder (upsamples to 32×57)
    ↓
Depth Head: Conv2d → metric depth (B, 1, 32, 57)
Context Head: Conv2d → (B, C_ctx, 32, 57)
    ↓
Depth Conversion: metric → soft distribution
    Depth ∈ [4.0, 50.0]
    D bins: linspace(4.0, 50.0, 112)
    For each pixel:
        d_bin_idx = argmin(|depth - bin_centers|)
        Create one-hot, smooth, apply softmax → P(d)
```

**Key Feature**: The depth distribution is **differentiable**, allowing the depth estimator to improve during training based on downstream occupancy prediction loss (end-to-end learning).

---

### 2. LiftSplat

**File**: [model/lift_splat.py](model/lift_splat.py)

**Inputs**:
- `depth_dist`: (B, D, fH, fW)
- `context`: (B, C_ctx, fH, fW)
- `K`: (B, 3, 3) — intrinsics scaled to feature resolution
- `E`: (B, 4, 4) — camera → ego transformation

**Outputs**:
- `bev_feat`: (B, C_ctx, H_bev, W_bev)

**Implementation Details**:

```python
def forward(depth_dist, context, K, E):
    B, D, fH, fW = depth_dist.shape
    
    # 1. Build pixel grid (cached for efficiency)
    uvh = get_pixel_grid(fH, fW)  # (3, fH*fW) homogeneous coords
    
    # 2. Unproject to camera space
    # K^-1 @ uvh → directions in camera space
    P_cam_dirs = K_inv @ uvh  # (B, 3, fH*fW)
    
    # 3. Lift: scale by each depth bin
    for d_idx in range(D):
        d_prob = depth_dist[:, d_idx, :, :]  # (B, fH, fW)
        d = d_bins[d_idx]
        
        P_cam = d * P_cam_dirs  # (B, 3, fH*fW)
        
        # 4. Transform to ego frame
        P_homo = append_1(P_cam)  # (B, 4, fH*fW)
        P_ego = E @ P_homo  # (B, 4, fH*fW)
        
        # 5. Get x, y (ignore z)
        x_ego = P_ego[0, :, :]  # (B, fH*fW)
        y_ego = P_ego[1, :, :]
        
        # 6. Map to BEV grid
        grid_row = (x_ego - fwd_min) / resolution
        grid_col = (y_ego - lat_min) / resolution
        
        # 7. Gather context and splat into BEV
        ctx = context[:, :, :, :].reshape(B, C, fH*fW)
        
        # Splat: sum into appropriate grid cells
        for row, col in valid_cells:
            bev_feat[:, :, row, col] += ctx[:, :, idx] * d_prob[idx]
    
    return bev_feat  # (B, C_ctx, H_bev, W_bev)
```

**Optimization**: Pixel grid caching (vertices of frustum) to avoid recomputation per forward pass.

---

### 3. BEVDecoder

**File**: [model/bev_decoder.py](model/bev_decoder.py)

**Inputs**:
- `bev_feat`: (B, C_ctx, H_bev, W_bev)

**Outputs**:
- `logits`: (B, 1, H_bev, W_bev) — raw occupancy logits (pre-sigmoid)

**Architecture**:
```
Input BEV Features (B, 64, 100, 100)
    ↓
Conv2d(64→128) + BatchNorm + ReLU + Dropout(0.3)
    ↓ (spatial size unchanged)
Conv2d(128→64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Conv2d(64→32) + BatchNorm + ReLU
    ↓
Conv2d(32→1)  [output head]
    ↓
Logits (B, 1, 100, 100)
```

**Design Rationale**:
- **Lightweight**: Only ~200K parameters — focus on occupancy refinement, not feature extraction
- **No Sigmoid Output**: Returns logits for numerical stability with BCEWithLogitsLoss
- **Spatial Preservation**: No pooling or downsampling — maintains grid resolution
- **Regularization**: Dropout helps prevent overfitting to training distribution

---

## Training & Evaluation

### Basic Training Loop

```python
# Load config
config = yaml.load("configs/default.yaml")

# Build model
model = BEVOccupancyNet(config)

# Build optimizers
optimizer = Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, ...)

# Training loop
for epoch in range(num_epochs):
    # Train
    train_loss, train_iou = train_epoch(model, train_loader, optimizer, ...)
    
    # Validate
    val_loss, val_iou, val_dwe = val_epoch(model, val_loader, ...)
    
    # Checkpoint
    if val_iou > best_iou:
        save_checkpoint(model, optimizer, scheduler, epoch, val_iou)
    
    # Early stopping
    if no_improvement_for_15_epochs:
        break
```

### Hyperparameter Tuning

The config file (`configs/default.yaml`) contains all tunable parameters:

```yaml
data:
  nuscenes_root: "/path/to/nuscenes-mini"
  image_size: [448, 798]
  cameras: ["CAM_FRONT"]

bev:
  lateral_range: [-25.0, 25.0]
  forward_range: [-25.0, 25.0]
  resolution: 0.25  # metres per cell → 200×200 grid

depth:
  min: 4.0
  max: 50.0
  bins: 112

model:
  da2_encoder: "vits"    # or "vitb", "vitl"
  bev_channels: 64
  dropout: 0.3

train:
  batch_size: 4
  lr: 2.0e-4
  epochs: 50
  early_stopping_patience: 15
```

---

## Usage

### Training

```bash
# Train with default config
python train.py

# Train with custom config
python train.py --config configs/custom.yaml

# Resume from checkpoint
python train.py --resume checkpoints/last.pth
```

### Inference

```bash
# Run on validation sample
python infer.py \
  --checkpoint checkpoints/best.pth \
  --sample_idx 0 \
  --save output.png

# With custom config
python infer.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth
```

### Visualization

```bash
# During training, TensorBoard logs are saved to:
tensorboard --logdir logs/

# During validation, sample visualizations are saved to:
logs/viz_val/epoch_*.png
```

---

## File Structure

```
BEV-occupancy/
├── train.py                          # Main training script
├── infer.py                          # Inference & visualization script
├── visualize_predictions.py          # Batch visualization tool
├── requirements.txt                  # Python dependencies
│
├── configs/
│   └── default.yaml                  # Hyperparameter configuration
│
├── data/
│   ├── __init__.py
│   ├── nuscenes_loader.py            # PyTorch Dataset for nuScenes
│   └── bev_gt_generator.py           # LiDAR → BEV ground truth
│
├── model/
│   ├── __init__.py
│   ├── bev_occupancy_net.py          # Full end-to-end model
│   ├── depth_anything.py             # Depth Anything V2 encoder
│   ├── lift_splat.py                 # Camera → BEV transformation
│   ├── bev_decoder.py                # BEV feature → occupancy logits
│   ├── backbone.py                   # (deprecated) ResNet backbone
│   ├── depth_net.py                  # (deprecated) Depth prediction head
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # IoU and distance-weighted error
│   └── visualization.py              # Plotting utilities
│
├── checkpoints/                      # Saved model weights
│   └── best.pth
│
├── logs/
│   ├── events.*                      # TensorBoard trace
│   ├── viz/                          # Training visualizations
│   └── viz_val/                      # Validation visualizations
│
└── README.md                         # (Original) brief README
```

---

## Key Technical Insights

### 1. **End-to-End Learning**
The entire pipeline (depth estimation → lift-splat → occupancy decoding) is differentiable. This allows the depth estimator to specialize for occupancy prediction rather than just canonical depth.

### 2. **Geometric Consistency**
The Lift-Splat module enforces **metric scale consistency**: depth is predicted in meters, properly calibrated with camera intrinsics (K) and extrinsics (E). This ensures the BEV grid aligns with the real world.

### 3. **Efficient Ground Removal**
Traditional fixed-height thresholding fails on slopes. RANSAC plane fitting is robust to road curvature and works across diverse terrain.

### 4. **Distance-Weighted Loss**
Placing more weight on predictions near the vehicle prioritizes safety-critical errors. A mistake 2m in front is more costly than 20m ahead.

### 5. **Feature Reuse**
DA2's ViT embeddings serve dual purposes:
- **Depth**: DPT head predicts metric depth
- **Context**: Same embeddings feed context features to Lift-Splat
This multi-task design improves efficiency and feature quality.

---

## Potential Improvements

1. **Multi-Camera Fusion**: Fuse predictions from multiple cameras (front, sides, back) into a single 360° BEV
2. **Temporal Consistency**: Use optical flow or recurrent layers to smooth predictions across frames
3. **Full Resolution**: Increase BEV grid resolution beyond 200×200 for finer occupancy details
4. **Uncertainty Estimation**: Predict confidence for each grid cell (e.g., Laplace approximation or MC dropout)
5. **Real-Time Deployment**: Optimize for edge devices (ONNX export, TensorRT quantization)
6. **Semantic Occupancy**: Instead of binary occupancy, predict category per cell (car/person/curb/vegetation/etc.)

---

## References

- **Lift, Splat, Shoot** (Li et al., 2020): [arXiv:2008.05711](https://arxiv.org/abs/2008.05711)
- **Depth Anything V2**: [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- **nuScenes Dataset**: [Paper](https://arxiv.org/abs/1903.11027)
- **Vision Transformers** (Dosovitskiy et al., 2020): [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config (default 4)
- Reduce BEV grid resolution (default 0.25m)
- Use smaller DA2 encoder (`"vits"` instead of `"vitb"`)

### Training Not Converging
- Increase learning rate to 1e-3
- Check that augmentation isn't too aggressive
- Verify dataset path and that samples load correctly

### Inference is Slow
- Use `"vits"` (ViT-Small) instead of `"vitl"`
- Reduce image resolution
- Export to ONNX/TensorRT for deployment

---

## Contact & Attribution

This project is inspired by research in monocular 3D perception for autonomous driving. The Depth Anything V2 model is courtesy of the original authors.