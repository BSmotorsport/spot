# train_heatmap_fixed.py

import os
import random
import re
import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler, autocast

# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================
class Config:
    DATASET_PATH = r"E:\BOTB\dataset\aug"
    OUTPUT_DIR = r"./training_output_codexfpn2"
    MODEL_VERSION = "ConvNext_heatmap_combined"
    
    # Image dimensions
    ORIGINAL_WIDTH = 4416
    ORIGINAL_HEIGHT = 3336
    IMAGE_SIZE = 1536
    HEATMAP_SIZE = 256
    
    # Model settings - ACTUALLY USING CONVNEXT NOW
    MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k'  # ConvNeXt Base model
    BATCH_SIZE = 2  # ConvNeXt needs more memory
    EPOCHS = 250
    INITIAL_LR = 3e-4
    FINETUNE_LR = 3e-5
    UNFREEZE_EPOCH = 3
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    WARMUP_EPOCHS = 1
    GRAD_ACCUM_STEPS = 4
    MIN_LR_FACTOR = 0.1
    
    # Loss settings
    WING_W = 5.0
    WING_EPSILON = 0.5
    HEATMAP_SIGMA_START = 10.0  # Start with a broader Gaussian for denser early gradients
    HEATMAP_SIGMA_END = 3.5     # Anneal towards a sharper target later in training
    HEATMAP_SIGMA_MIN = 2.0     # Allow adaptive tuning to push the target sharper when needed
    HEATMAP_SIGMA_DECAY_EPOCHS = 50
    MIXUP_ALPHA = 0.2  # Mixup augmentation strength
    ENABLE_MIXUP = False  # Disable by default for single-keypoint stability

    # Fusion gate tuning
    HEATMAP_CONFIDENCE_THRESHOLD = 0.5   # Confidence needed before heavily trusting heatmap coords
    HEATMAP_CONFIDENCE_SCALE = 1.0  # Linear scaling factor for heatmap fusion weights
    HEATMAP_MIN_FUSION_WEIGHT = 0.1   # Always leave some weight on the regressor branch
    HEATMAP_MIN_FUSION_WEIGHT_MIN = 0.05
    HEATMAP_CONFIDENCE_WARMUP_EPOCHS = 5  # Gradually enable the gate after unfreezing
    HEATMAP_CONFIDENCE_GAMMA = 1.5  # Emphasise confident peaks before handing control to the heatmap
    HEATMAP_CONFIDENCE_DEBUG = False  # Enable to log per-batch confidence stats for tuning

    # Loss weighting
    HEATMAP_LOSS_WEIGHT = 0.5
    HEATMAP_LOSS_WEIGHT_MIN = 0.35
    HEATMAP_LOSS_WEIGHT_MAX = 1.0
    COORD_LOSS_WEIGHT = 0.3
    COORD_LOSS_WEIGHT_MAX = 0.45
    COORD_LOSS_WEIGHT_MIN = 0.05
    PIXEL_COORD_LOSS_WEIGHT = 0.2
    PIXEL_COORD_LOSS_WEIGHT_MAX = 0.35
    PIXEL_COORD_LOSS_WEIGHT_MIN = 0.05
    HEATMAP_CENTER_LOSS_WEIGHT = 0.05  # Encourage the heatmap peak to stay anchored on the target
    HEATMAP_CENTER_TEMPERATURE = 0.6   # Sharper soft-argmax for the center consistency loss
    HEATMAP_CENTER_TEMPERATURE_MIN = 0.35
    PIXEL_LOSS_LOG_SCALE = 75.0  # Damp extreme pixel errors so a bad batch can't destabilise training

    HEATMAP_LABEL_SMOOTHING = 0.01
    HEATMAP_LABEL_SMOOTHING_MIN = 0.002

    # Memory optimization flags
    ENABLE_GRAD_CHECKPOINTING = True
    USE_CHANNELS_LAST = True

    # Adaptive tuning parameters
    ADAPTIVE_TUNING_ENABLED = True
    ADAPTIVE_PATIENCE = 6
    ADAPTIVE_IMPROVEMENT_TOLERANCE = 1.0  # Require at least this MAE gain to reset patience
    ADAPTIVE_BRANCH_MARGIN = 5.0  # Margin (in px) used to judge which branch under-performs
    ADAPTIVE_SIGMA_DECAY = 0.85
    ADAPTIVE_SMOOTHING_DECAY = 0.5
    ADAPTIVE_CONFIDENCE_STEP = 0.05
    ADAPTIVE_FUSION_STEP = 0.02
    ADAPTIVE_LOSS_WEIGHT_STEP = 0.03
    ADAPTIVE_CENTER_TEMP_DECAY = 0.85
    ADAPTIVE_LOGGING = True
    ADAPTIVE_LR_DECAY = 0.6
    ADAPTIVE_LR_COOLDOWN = 3
    ADAPTIVE_MIN_LR = 5e-6

    # Validation settings
    VALIDATE_EVERY_N_EPOCHS = 1
    SAVE_BEST_ONLY = False

    # Loss balancing controller
    LOSS_BALANCER_ENABLED = True
    LOSS_BALANCER_SMOOTHING = 0.25
    LOSS_BALANCER_TOLERANCE = 0.08
    LOSS_BALANCER_STEP = 0.02
    LOSS_BALANCER_LOGGING = True

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(Config.RANDOM_SEED)

# ======================================================================================
# 1a. ADAPTIVE TRAINING STATE
# ======================================================================================


class AdaptiveState:
    """Runtime-adjustable hyperparameters shared across the training pipeline."""

    def __init__(self) -> None:
        self.manual_sigma: float | None = None
        self.current_sigma: float = float(Config.HEATMAP_SIGMA_START)
        self.sigma_floor: float = float(Config.HEATMAP_SIGMA_MIN)
        self.label_smoothing: float = float(Config.HEATMAP_LABEL_SMOOTHING)

        self.heatmap_loss_weight: float = float(Config.HEATMAP_LOSS_WEIGHT)
        self.coord_loss_weight: float = float(Config.COORD_LOSS_WEIGHT)
        self.pixel_coord_loss_weight: float = float(Config.PIXEL_COORD_LOSS_WEIGHT)
        self.center_loss_weight: float = float(Config.HEATMAP_CENTER_LOSS_WEIGHT)
        self.center_temperature: float = float(Config.HEATMAP_CENTER_TEMPERATURE)

        self.heatmap_confidence_threshold: float = float(Config.HEATMAP_CONFIDENCE_THRESHOLD)
        self.heatmap_confidence_gamma: float = float(Config.HEATMAP_CONFIDENCE_GAMMA)
        self.heatmap_min_fusion_weight: float = float(Config.HEATMAP_MIN_FUSION_WEIGHT)
        self.heatmap_confidence_scale: float = float(Config.HEATMAP_CONFIDENCE_SCALE)

    def sigma_for_epoch(self, epoch: int) -> float:
        if Config.HEATMAP_SIGMA_DECAY_EPOCHS <= 0:
            base_sigma = float(Config.HEATMAP_SIGMA_END)
        else:
            progress = min(max(epoch, 0) / Config.HEATMAP_SIGMA_DECAY_EPOCHS, 1.0)
            base_sigma = Config.HEATMAP_SIGMA_START + (
                Config.HEATMAP_SIGMA_END - Config.HEATMAP_SIGMA_START
            ) * progress

        if self.manual_sigma is not None:
            base_sigma = min(base_sigma, self.manual_sigma)

        self.current_sigma = float(max(self.sigma_floor, base_sigma))
        return self.current_sigma

    def apply_to_criterion(self, criterion: "CombinedLoss") -> None:
        criterion.set_weights(
            heatmap_weight=self.heatmap_loss_weight,
            coord_weight=self.coord_loss_weight,
            pixel_coord_weight=self.pixel_coord_loss_weight,
            center_weight=self.center_loss_weight,
            center_temperature=self.center_temperature,
        )

    def state_dict(self) -> dict[str, float | None]:
        return {
            "manual_sigma": self.manual_sigma,
            "current_sigma": self.current_sigma,
            "sigma_floor": self.sigma_floor,
            "label_smoothing": self.label_smoothing,
            "heatmap_loss_weight": self.heatmap_loss_weight,
            "coord_loss_weight": self.coord_loss_weight,
            "pixel_coord_loss_weight": self.pixel_coord_loss_weight,
            "center_loss_weight": self.center_loss_weight,
            "center_temperature": self.center_temperature,
            "heatmap_confidence_threshold": self.heatmap_confidence_threshold,
            "heatmap_confidence_gamma": self.heatmap_confidence_gamma,
            "heatmap_min_fusion_weight": self.heatmap_min_fusion_weight,
            "heatmap_confidence_scale": self.heatmap_confidence_scale,
        }

    def load_state_dict(self, state: dict[str, float | None]) -> None:
        for key, value in state.items():
            if not hasattr(self, key):
                continue
            setattr(self, key, float(value) if value is not None else None)


ADAPTIVE_STATE = AdaptiveState()

# ======================================================================================
# 2. HELPER FUNCTIONS
# ======================================================================================

def _autocast_kwargs_for(device: torch.device | str) -> dict[str, object]:
    """Return safe autocast kwargs for the requested device."""

    device_str = device.type if isinstance(device, torch.device) else str(device)
    device_type = 'cuda' if device_str.startswith('cuda') else 'cpu'

    if device_type == 'cuda' and torch.cuda.is_available():
        return {'device_type': 'cuda', 'dtype': torch.float16}

    return {'device_type': device_type, 'enabled': False}


def parse_filename(filename):
    """Parse filename to extract ball coordinates with validation."""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        
        if len(parts) < 2:
            return None
            
        y_coord = int(parts[-1])
        x_coord = int(parts[-2])
        
        # Validate coordinates are within expected bounds
        if not (0 <= x_coord <= Config.ORIGINAL_WIDTH):
            print(f"Warning: X coordinate out of bounds in {filename}: {x_coord}")
            return None
        if not (0 <= y_coord <= Config.ORIGINAL_HEIGHT):
            print(f"Warning: Y coordinate out of bounds in {filename}: {y_coord}")
            return None
            
        return {'x': x_coord, 'y': y_coord}
        
    except (IndexError, ValueError) as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

def mixup_data(images, heatmaps, coords, alpha=0.2):
    """Apply mixup augmentation and return paired targets for correct loss blending."""
    if alpha > 0:
        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)

        lambda_ = np.random.beta(alpha, alpha)
        lambda_ = max(lambda_, 1 - lambda_)

        mixed_images = lambda_ * images + (1 - lambda_) * images[indices]

        return (
            mixed_images,
            heatmaps,
            heatmaps[indices],
            coords,
            coords[indices],
            lambda_,
        )

    return images, heatmaps, None, coords, None, 1.0

def create_target_heatmap(keypoints, size, sigma=None, smoothing=None):
    """Create Gaussian heatmap for keypoints with proper normalization."""
    if sigma is None:
        sigma = Config.HEATMAP_SIGMA_START

    if smoothing is None:
        smoothing = Config.HEATMAP_LABEL_SMOOTHING

    heatmap = np.zeros((size, size), dtype=np.float32)

    for x, y in keypoints:
        # Skip invalid keypoints
        if x < 0 or x >= size or y < 0 or y >= size:
            continue
        
        # Create coordinate grids
        xx, yy = np.meshgrid(
            np.arange(size, dtype=np.float32),
            np.arange(size, dtype=np.float32)
        )
        
        # Generate Gaussian
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        
        # Use maximum to handle overlapping gaussians (though we only have one keypoint)
        heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize to [0, 1] if there's any signal
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Blend with a very small uniform floor to provide label smoothing
    # while keeping the background pixels below the foreground mask
    # threshold used in the loss.  Distribute the smoothing mass across
    # all pixels so that the effective offset per pixel is tiny.
    smoothing = float(smoothing)
    heatmap = heatmap * (1.0 - smoothing) + smoothing / (size * size)

    return heatmap

def soft_argmax_2d(heatmap, temperature=1.0):
    """Compute soft-argmax for differentiable coordinate extraction."""
    batch_size, _, height, width = heatmap.shape

    # Stabilize logits before applying softmax to avoid numerical issues
    heatmap_float = torch.nan_to_num(
        heatmap.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    heatmap_flat = heatmap_float.view(batch_size, -1)
    max_val, _ = torch.max(heatmap_flat, dim=1, keepdim=True)
    stable_heatmap = heatmap_flat - max_val

    # Apply softmax to get probabilities
    heatmap_probs = F.softmax(stable_heatmap / temperature, dim=1)
    heatmap_probs = heatmap_probs.view(batch_size, 1, height, width)
    
    # Create coordinate grids normalized to [0, 1]
    x_coords = torch.linspace(0, 1, width, device=heatmap.device, dtype=torch.float32)
    y_coords = torch.linspace(0, 1, height, device=heatmap.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)

    # Compute expected coordinates
    x_expected = torch.sum(heatmap_probs * xx, dim=(2, 3))
    y_expected = torch.sum(heatmap_probs * yy, dim=(2, 3))

    coords = torch.stack([x_expected.squeeze(1), y_expected.squeeze(1)], dim=1)
    coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale back to pixel coordinates
    width_range = width - 1 if width > 1 else 0
    height_range = height - 1 if height > 1 else 0
    coords[:, 0] *= width_range
    coords[:, 1] *= height_range

    return coords

def get_coords_from_heatmap(heatmap, method='soft_argmax'):
    """Extract coordinates from heatmap using specified method."""
    if method == 'soft_argmax':
        return soft_argmax_2d(heatmap)
    else:  # argmax method
        batch_size, _, height, width = heatmap.shape
        heatmap_reshaped = heatmap.reshape(batch_size, -1)
        max_indices = torch.argmax(heatmap_reshaped, dim=1)
        preds_y = (max_indices // width).float()
        preds_x = (max_indices % width).float()
        return torch.stack([preds_x, preds_y], dim=1)


def compute_heatmap_fusion_weight(heatmap_logits, epoch=None):
    """Estimate how much trust to place in the heatmap branch."""

    with torch.no_grad():
        logits = heatmap_logits.float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        probs = torch.sigmoid(logits)
        probs = torch.nan_to_num(probs, nan=0.0)

        flattened = probs.flatten(start_dim=1)
        peak = flattened.amax(dim=1)
        mean = flattened.mean(dim=1)

        confidence = (peak - mean) / (peak + 1e-6)
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

        threshold = float(ADAPTIVE_STATE.heatmap_confidence_threshold)
        denom = max(1e-6, 1.0 - threshold)
        if threshold >= 1.0:
            gated = torch.zeros_like(confidence)
        else:
            gated = torch.clamp((confidence - threshold) / denom, min=0.0, max=1.0)

        if ADAPTIVE_STATE.heatmap_confidence_gamma != 1.0:
            gated = gated.pow(ADAPTIVE_STATE.heatmap_confidence_gamma)

        min_weight = float(np.clip(ADAPTIVE_STATE.heatmap_min_fusion_weight, 0.0, 1.0))
        weight = min_weight + (1.0 - min_weight) * gated

        if epoch is not None and Config.HEATMAP_CONFIDENCE_WARMUP_EPOCHS > 0:
            warmup_progress = min(
                max((epoch + 1) / Config.HEATMAP_CONFIDENCE_WARMUP_EPOCHS, 0.0),
                1.0,
            )
            weight = min_weight + (weight - min_weight) * warmup_progress

        if ADAPTIVE_STATE.heatmap_confidence_scale != 1.0:
            weight = weight * ADAPTIVE_STATE.heatmap_confidence_scale

        weight = torch.clamp(weight, min=min_weight, max=1.0)

        if Config.HEATMAP_CONFIDENCE_DEBUG:
            debug_stack = torch.stack((peak, mean, confidence, weight), dim=1)
            debug_cpu = debug_stack.detach().cpu()
            sample_count = debug_cpu.size(0)
            preview_count = min(3, sample_count)

            for idx in range(preview_count):
                peak_v, mean_v, conf_v, weight_v = debug_cpu[idx].tolist()
                print(
                    "[heatmap gate] sample"
                    f" {idx}: peak={peak_v:.3f}, mean={mean_v:.3f}, "
                    f"confidence={conf_v:.3f}, weight={weight_v:.3f}"
                )

            if sample_count > preview_count:
                print(f"[heatmap gate] ... {sample_count - preview_count} more samples in batch")

            batch_means = debug_cpu.mean(dim=0)
            print(
                "[heatmap gate] batch_mean: "
                f"peak={batch_means[0]:.3f}, mean={batch_means[1]:.3f}, "
                f"confidence={batch_means[2]:.3f}, weight={batch_means[3]:.3f}"
            )

    return weight


class AdaptiveTuningController:
    """Dynamic policy that adjusts supervision and fusion settings during training."""

    def __init__(self, state: AdaptiveState) -> None:
        self.state = state
        self.best_val_mae: float = float("inf")
        self.no_improve_epochs: int = 0
        self.last_lr_decay_epoch: int = -1000

    def on_epoch_start(
        self,
        epoch: int,
        dataset: "FootballDataset",
        criterion: "CombinedLoss",
    ) -> float:
        sigma = self.state.sigma_for_epoch(epoch)
        dataset.set_heatmap_sigma(sigma)
        dataset.set_label_smoothing(self.state.label_smoothing)
        self.state.apply_to_criterion(criterion)
        return sigma

    def on_validation_end(
        self,
        epoch: int,
        val_mae: float,
        val_stats: dict[str, float],
        criterion: "CombinedLoss",
        optimizer: optim.Optimizer | None = None,
        scheduler: Optional["WarmupCosineLR"] = None,
    ) -> None:
        if not Config.ADAPTIVE_TUNING_ENABLED:
            return

        if val_mae + float(Config.ADAPTIVE_IMPROVEMENT_TOLERANCE) < self.best_val_mae:
            self.best_val_mae = float(val_mae)
            self.no_improve_epochs = 0
            return

        self.no_improve_epochs += 1
        if self.no_improve_epochs < int(Config.ADAPTIVE_PATIENCE):
            return

        adjustments: list[str] = []
        header_printed = False

        # Push the target supervision sharper and reduce smoothing if plateaus persist.
        new_sigma = max(
            self.state.sigma_floor,
            self.state.current_sigma * float(Config.ADAPTIVE_SIGMA_DECAY),
        )
        if (
            self.state.manual_sigma is None
            or new_sigma < self.state.manual_sigma - 1e-6
        ) and new_sigma < self.state.current_sigma - 1e-6:
            self.state.manual_sigma = new_sigma
            adjustments.append(f"heatmap sigma → {new_sigma:.2f}")

        if self.state.label_smoothing > Config.HEATMAP_LABEL_SMOOTHING_MIN + 1e-6:
            new_smoothing = max(
                Config.HEATMAP_LABEL_SMOOTHING_MIN,
                self.state.label_smoothing * float(Config.ADAPTIVE_SMOOTHING_DECAY),
            )
            if new_smoothing < self.state.label_smoothing - 1e-6:
                self.state.label_smoothing = new_smoothing
                adjustments.append(f"label smoothing → {new_smoothing:.4f}")

        heatmap_mae = float(val_stats.get('heatmap_mae', float('nan')))
        regressor_mae = float(val_stats.get('regressor_mae', float('nan')))

        if not math.isnan(heatmap_mae) and not math.isnan(regressor_mae):
            branch_margin = float(Config.ADAPTIVE_BRANCH_MARGIN)
            if heatmap_mae >= regressor_mae + branch_margin:
                # Heatmap under-performing: favour regressor supervision and gate tightening.
                new_thresh = min(
                    0.99,
                    self.state.heatmap_confidence_threshold
                    + float(Config.ADAPTIVE_CONFIDENCE_STEP),
                )
                if new_thresh > self.state.heatmap_confidence_threshold + 1e-6:
                    self.state.heatmap_confidence_threshold = new_thresh
                    adjustments.append(f"fusion threshold → {new_thresh:.2f}")

                new_min_weight = max(
                    Config.HEATMAP_MIN_FUSION_WEIGHT_MIN,
                    self.state.heatmap_min_fusion_weight
                    - float(Config.ADAPTIVE_FUSION_STEP),
                )
                if new_min_weight < self.state.heatmap_min_fusion_weight - 1e-6:
                    self.state.heatmap_min_fusion_weight = new_min_weight
                    adjustments.append(f"min fusion weight → {new_min_weight:.2f}")

                new_heatmap_weight = max(
                    Config.HEATMAP_LOSS_WEIGHT_MIN,
                    self.state.heatmap_loss_weight
                    - float(Config.ADAPTIVE_LOSS_WEIGHT_STEP),
                )
                if new_heatmap_weight < self.state.heatmap_loss_weight - 1e-6:
                    self.state.heatmap_loss_weight = new_heatmap_weight
                    adjustments.append(f"heatmap loss weight → {new_heatmap_weight:.2f}")

                new_coord_weight = min(
                    Config.COORD_LOSS_WEIGHT_MAX,
                    self.state.coord_loss_weight
                    + float(Config.ADAPTIVE_LOSS_WEIGHT_STEP),
                )
                if new_coord_weight > self.state.coord_loss_weight + 1e-6:
                    self.state.coord_loss_weight = new_coord_weight
                    adjustments.append(f"coord loss weight → {new_coord_weight:.2f}")

                new_pixel_weight = min(
                    Config.PIXEL_COORD_LOSS_WEIGHT_MAX,
                    self.state.pixel_coord_loss_weight
                    + float(Config.ADAPTIVE_LOSS_WEIGHT_STEP),
                )
                if new_pixel_weight > self.state.pixel_coord_loss_weight + 1e-6:
                    self.state.pixel_coord_loss_weight = new_pixel_weight
                    adjustments.append(f"pixel loss weight → {new_pixel_weight:.2f}")

                new_center_temp = max(
                    Config.HEATMAP_CENTER_TEMPERATURE_MIN,
                    self.state.center_temperature
                    * float(Config.ADAPTIVE_CENTER_TEMP_DECAY),
                )
                if new_center_temp < self.state.center_temperature - 1e-6:
                    self.state.center_temperature = new_center_temp
                    adjustments.append(f"center temperature → {new_center_temp:.2f}")

            elif regressor_mae >= heatmap_mae + branch_margin:
                # Regressor lagging: relax fusion gate slightly and favour heatmap loss.
                new_thresh = max(
                    0.2,
                    self.state.heatmap_confidence_threshold
                    - float(Config.ADAPTIVE_CONFIDENCE_STEP) * 0.5,
                )
                if new_thresh < self.state.heatmap_confidence_threshold - 1e-6:
                    self.state.heatmap_confidence_threshold = new_thresh
                    adjustments.append(f"fusion threshold → {new_thresh:.2f}")

                new_min_weight = min(
                    0.5,
                    self.state.heatmap_min_fusion_weight
                    + float(Config.ADAPTIVE_FUSION_STEP),
                )
                if new_min_weight > self.state.heatmap_min_fusion_weight + 1e-6:
                    self.state.heatmap_min_fusion_weight = new_min_weight
                    adjustments.append(f"min fusion weight → {new_min_weight:.2f}")

                new_heatmap_weight = min(
                    1.0,
                    self.state.heatmap_loss_weight
                    + float(Config.ADAPTIVE_LOSS_WEIGHT_STEP),
                )
                if new_heatmap_weight > self.state.heatmap_loss_weight + 1e-6:
                    self.state.heatmap_loss_weight = new_heatmap_weight
                    adjustments.append(f"heatmap loss weight → {new_heatmap_weight:.2f}")

                new_coord_weight = max(
                    0.05,
                    self.state.coord_loss_weight
                    - float(Config.ADAPTIVE_LOSS_WEIGHT_STEP),
                )
                if new_coord_weight < self.state.coord_loss_weight - 1e-6:
                    self.state.coord_loss_weight = new_coord_weight
                    adjustments.append(f"coord loss weight → {new_coord_weight:.2f}")

        if adjustments and Config.ADAPTIVE_LOGGING:
            print("\n🔁 Adaptive tuning triggered:")
            header_printed = True
            for item in adjustments:
                print(f"   • {item}")

        if adjustments:
            self.state.apply_to_criterion(criterion)
            self.no_improve_epochs = 0
            self.best_val_mae = min(self.best_val_mae, float(val_mae))

        # Optionally trigger a gentle learning-rate decay when the plateau persists.
        if (
            optimizer is not None
            and Config.ADAPTIVE_LR_DECAY < 1.0
            and (epoch - self.last_lr_decay_epoch) >= int(Config.ADAPTIVE_LR_COOLDOWN)
        ):
            decay_factor = float(Config.ADAPTIVE_LR_DECAY)
            min_lr = float(Config.ADAPTIVE_MIN_LR)
            lr_changed = False
            new_lrs = []
            for group in optimizer.param_groups:
                current_lr = float(group.get('lr', 0.0))
                if current_lr <= 0.0:
                    new_lrs.append(current_lr)
                    continue
                decayed = max(min_lr, current_lr * decay_factor)
                if decayed < current_lr - 1e-12:
                    lr_changed = True
                group['lr'] = decayed
                new_lrs.append(decayed)
            if lr_changed and scheduler is not None and hasattr(scheduler, "decay_base_lrs"):
                scheduler.decay_base_lrs(decay_factor, min_lr=min_lr)
            if lr_changed:
                self.last_lr_decay_epoch = epoch
                if Config.ADAPTIVE_LOGGING:
                    if not header_printed:
                        print("\n🔁 Adaptive tuning triggered:")
                        header_printed = True
                    pretty = ", ".join(f"{lr:.2e}" for lr in new_lrs)
                    print(f"   • Learning rates decayed → {pretty}")


class LossBalanceController:
    """Balances loss contributions to keep optimisation gradients well-conditioned."""

    def __init__(self, state: AdaptiveState) -> None:
        self.state = state
        self.enabled = bool(Config.LOSS_BALANCER_ENABLED)
        base_total = (
            Config.HEATMAP_LOSS_WEIGHT
            + Config.COORD_LOSS_WEIGHT
            + Config.PIXEL_COORD_LOSS_WEIGHT
        )
        if base_total <= 0:
            base_total = 1.0
        self.target_distribution = {
            'heatmap': Config.HEATMAP_LOSS_WEIGHT / base_total,
            'coord': Config.COORD_LOSS_WEIGHT / base_total,
            'pixel': Config.PIXEL_COORD_LOSS_WEIGHT / base_total,
        }
        self.smoothing = float(Config.LOSS_BALANCER_SMOOTHING)
        self.tolerance = float(Config.LOSS_BALANCER_TOLERANCE)
        self.step = float(Config.LOSS_BALANCER_STEP)
        self.ema_contributions = self.target_distribution.copy()

    def observe(
        self,
        heatmap_loss: float,
        coord_loss: float,
        pixel_loss: float,
        criterion: "CombinedLoss",
    ) -> None:
        if not self.enabled:
            return

        heatmap_weight = float(getattr(criterion, 'heatmap_weight', self.state.heatmap_loss_weight))
        coord_weight = float(getattr(criterion, 'coord_weight', self.state.coord_loss_weight))
        pixel_weight = float(
            getattr(criterion, 'pixel_coord_weight', self.state.pixel_coord_loss_weight)
        )

        weighted_losses = {
            'heatmap': heatmap_weight * heatmap_loss,
            'coord': coord_weight * coord_loss,
            'pixel': pixel_weight * pixel_loss,
        }

        total = weighted_losses['heatmap'] + weighted_losses['coord'] + weighted_losses['pixel']
        if total <= 0:
            return

        contributions = {
            key: value / total for key, value in weighted_losses.items()
        }

        momentum = max(min(self.smoothing, 1.0), 0.0)
        for key, value in contributions.items():
            prev = self.ema_contributions.get(key, value)
            self.ema_contributions[key] = prev * (1.0 - momentum) + value * momentum

        adjustments: list[str] = []

        def _adjust(current, target, weight, lower, upper, label):
            if target < 0:
                return weight
            if current > target + self.tolerance:
                new_weight = max(lower, weight - self.step)
                if new_weight < weight - 1e-6:
                    adjustments.append(f"{label} weight ↓ {new_weight:.2f}")
                return new_weight
            if current + self.tolerance < target:
                new_weight = min(upper, weight + self.step)
                if new_weight > weight + 1e-6:
                    adjustments.append(f"{label} weight ↑ {new_weight:.2f}")
                return new_weight
            return weight

        heatmap_weight = _adjust(
            self.ema_contributions['heatmap'],
            self.target_distribution['heatmap'],
            self.state.heatmap_loss_weight,
            Config.HEATMAP_LOSS_WEIGHT_MIN,
            Config.HEATMAP_LOSS_WEIGHT_MAX,
            "heatmap",
        )

        coord_weight = _adjust(
            self.ema_contributions['coord'],
            self.target_distribution['coord'],
            self.state.coord_loss_weight,
            Config.COORD_LOSS_WEIGHT_MIN,
            Config.COORD_LOSS_WEIGHT_MAX,
            "coord",
        )

        pixel_weight = _adjust(
            self.ema_contributions['pixel'],
            self.target_distribution['pixel'],
            self.state.pixel_coord_loss_weight,
            Config.PIXEL_COORD_LOSS_WEIGHT_MIN,
            Config.PIXEL_COORD_LOSS_WEIGHT_MAX,
            "pixel",
        )

        if adjustments and Config.LOSS_BALANCER_LOGGING:
            print("\n⚖️ Loss balancer adjustments:")
            for item in adjustments:
                print(f"   • {item}")

        if adjustments:
            self.state.heatmap_loss_weight = heatmap_weight
            self.state.coord_loss_weight = coord_weight
            self.state.pixel_coord_loss_weight = pixel_weight
            self.state.apply_to_criterion(criterion)

# ======================================================================================
# 3. DATASET CLASS
# ======================================================================================

class FootballDataset(Dataset):
    def __init__(
        self,
        image_paths,
        transform=None,
        heatmap_size=192,
        augment=True,
        heatmap_sigma=None,
        label_smoothing=None,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma if heatmap_sigma is not None else Config.HEATMAP_SIGMA_START
        self.label_smoothing = (
            Config.HEATMAP_LABEL_SMOOTHING
            if label_smoothing is None
            else float(label_smoothing)
        )
        
        # Pre-filter valid images
        self.valid_paths = []
        for path in image_paths:
            if parse_filename(os.path.basename(path)) is not None:
                self.valid_paths.append(path)
        
        print(f"Dataset initialized with {len(self.valid_paths)}/{len(image_paths)} valid images")
        
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        img_path = self.valid_paths[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse coordinates from filename
        coords = parse_filename(os.path.basename(img_path))
        if coords is None:
            return self.__getitem__((idx + 1) % len(self))
        
        # The image will be resized by the transform, but we need to track 
        # coordinates in the ORIGINAL image space first
        keypoints = [(float(coords['x']), float(coords['y']))]
        
        # Apply transformations (Albumentations will handle coordinate scaling automatically)
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed.get('keypoints', [])

            # Filter/clamp keypoints to valid image region after augmentation
            valid_keypoints = []
            if keypoints:
                max_coord = float(np.nextafter(Config.IMAGE_SIZE, -np.inf))
                for kp in keypoints:
                    if len(kp) < 2:
                        continue
                    x, y = float(kp[0]), float(kp[1])
                    if 0.0 <= x < Config.IMAGE_SIZE and 0.0 <= y < Config.IMAGE_SIZE:
                        clamped_x = min(max(x, 0.0), max_coord)
                        clamped_y = min(max(y, 0.0), max_coord)
                        valid_keypoints.append((clamped_x, clamped_y))
            keypoints = valid_keypoints

            # Skip if augmentation removed the keypoint
            if not keypoints:
                return self.__getitem__((idx + 1) % len(self))
        
        if Config.IMAGE_SIZE <= 1 or self.heatmap_size <= 1:
            raise ValueError("IMAGE_SIZE and heatmap_size must be greater than 1 for coordinate scaling")

        image_to_heatmap = (self.heatmap_size - 1) / (Config.IMAGE_SIZE - 1)
        # Convert keypoints to heatmap space
        heatmap_keypoints = [
            (kp[0] * image_to_heatmap,
             kp[1] * image_to_heatmap)
            for kp in keypoints
        ]

        # Create target heatmap
        target_heatmap = create_target_heatmap(
            heatmap_keypoints,
            self.heatmap_size,
            sigma=self.heatmap_sigma,
            smoothing=self.label_smoothing,
        )

        # Store precise coordinates for direct regression (normalized to [0, 1])
        precise_coords = torch.tensor(keypoints[0], dtype=torch.float32) / (Config.IMAGE_SIZE - 1)

        return image, torch.from_numpy(target_heatmap).unsqueeze(0), precise_coords

    def set_heatmap_sigma(self, sigma):
        self.heatmap_sigma = float(sigma)

    def set_label_smoothing(self, smoothing):
        self.label_smoothing = float(smoothing)


def compute_sigma_for_epoch(epoch):
    """Compatibility wrapper around the adaptive state's sigma schedule."""
    return ADAPTIVE_STATE.sigma_for_epoch(int(epoch))

# ======================================================================================
# 4. LOSS FUNCTIONS
# ======================================================================================

class WingLoss(nn.Module):
    """Wing Loss for robust heatmap regression."""
    def __init__(self, w=5.0, epsilon=0.5):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    """Combined loss for heatmap and coordinate regression."""

    def __init__(
        self,
        heatmap_weight: float | None = None,
        coord_weight: float | None = None,
        pixel_coord_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.coord_loss = nn.SmoothL1Loss()
        self.pixel_coord_weight = float(
            Config.PIXEL_COORD_LOSS_WEIGHT
            if pixel_coord_weight is None
            else float(pixel_coord_weight)
        )
        self.heatmap_weight = float(
            Config.HEATMAP_LOSS_WEIGHT if heatmap_weight is None else float(heatmap_weight)
        )
        self.coord_weight = float(
            Config.COORD_LOSS_WEIGHT if coord_weight is None else float(coord_weight)
        )
        self.heatmap_center_weight = float(Config.HEATMAP_CENTER_LOSS_WEIGHT)
        self.heatmap_center_temperature = max(float(Config.HEATMAP_CENTER_TEMPERATURE), 1e-3)

    def set_weights(
        self,
        *,
        heatmap_weight: float | None = None,
        coord_weight: float | None = None,
        pixel_coord_weight: float | None = None,
        center_weight: float | None = None,
        center_temperature: float | None = None,
    ) -> None:
        if heatmap_weight is not None:
            self.heatmap_weight = float(heatmap_weight)
        if coord_weight is not None:
            self.coord_weight = float(coord_weight)
        if pixel_coord_weight is not None:
            self.pixel_coord_weight = float(pixel_coord_weight)
        if center_weight is not None:
            self.heatmap_center_weight = float(center_weight)
        if center_temperature is not None:
            self.heatmap_center_temperature = max(float(center_temperature), 1e-3)

    def forward(self, pred_heatmaps, target_heatmaps, pred_coords=None, target_coords=None):
        device_type = pred_heatmaps.device.type
        with autocast(device_type=device_type, enabled=False):
            pred_heatmaps_fp32 = pred_heatmaps.to(dtype=torch.float32)
            target_heatmaps_fp32 = target_heatmaps.to(dtype=torch.float32)

            target_coords_fp32 = None
            if target_coords is not None:
                target_coords_fp32 = target_coords.to(dtype=torch.float32)

            # Heatmap logits can overflow to +/-inf when the decoder runs in
            # half precision during fine-tuning.  Sanitise them here so that
            # BCE-with-logits never receives non-finite values, which would
            # otherwise trigger the "Non-finite loss encountered" guard in the
            # training loop.
            pred_heatmaps_fp32 = torch.nan_to_num(
                pred_heatmaps_fp32,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            ).clamp_(-50.0, 50.0)

            target_heatmaps_fp32 = torch.nan_to_num(
                target_heatmaps_fp32,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            # Switch to a logits-based loss that dramatically up-weights pixels near the
            # bright center of the Gaussian target.  This keeps background regions from
            # dominating the objective while still benefiting from numerically stable
            # BCE-with-logits gradients.
            prob = torch.sigmoid(pred_heatmaps_fp32)

            target_clamped = target_heatmaps_fp32.clamp(0.0, 1.0)
            foreground_mask = (target_clamped > 0.01).to(dtype=pred_heatmaps_fp32.dtype)
            background_mask = 1.0 - foreground_mask

            # Build separate modulation terms for foreground and background pixels.  The
            # foreground branch receives a strong boost near the Gaussian peak, while the
            # background keeps a gentle focal-style penalty to suppress spurious blobs.
            bright_weight = 1.0 + 99.0 * target_clamped
            foreground_focus = 1.0 + (1.0 - prob).pow(2.0)
            background_focus = 1.0 + prob.pow(2.0)

            pixel_weight = torch.where(
                foreground_mask > 0.0,
                bright_weight * foreground_focus,
                background_focus,
            ).detach()

            per_pixel_loss = F.binary_cross_entropy_with_logits(
                pred_heatmaps_fp32,
                target_clamped,
                reduction='none',
            )

            foreground_weight = (pixel_weight * foreground_mask).sum().clamp_min(1e-6)
            background_weight = (pixel_weight * background_mask).sum().clamp_min(1e-6)

            foreground_loss = (
                per_pixel_loss * pixel_weight * foreground_mask
            ).sum() / foreground_weight
            background_loss = (
                per_pixel_loss * pixel_weight * background_mask
            ).sum() / background_weight

            h_loss = 0.5 * (foreground_loss + background_loss)

            total_loss = self.heatmap_weight * h_loss
            c_loss = pred_heatmaps_fp32.new_tensor(0.0)
            pixel_loss = pred_heatmaps_fp32.new_tensor(0.0)
            center_loss = pred_heatmaps_fp32.new_tensor(0.0)

            if (
                self.heatmap_center_weight > 0.0
                and target_coords_fp32 is not None
                and pred_heatmaps_fp32.shape[-1] > 1
                and pred_heatmaps_fp32.shape[-2] > 1
            ):
                center_coords = soft_argmax_2d(
                    pred_heatmaps_fp32,
                    temperature=self.heatmap_center_temperature,
                )
                width_range = float(pred_heatmaps_fp32.shape[-1] - 1)
                height_range = float(pred_heatmaps_fp32.shape[-2] - 1)
                target_heatmap_coords = torch.stack(
                    (
                        target_coords_fp32[:, 0] * width_range,
                        target_coords_fp32[:, 1] * height_range,
                    ),
                    dim=1,
                )
                center_loss = F.smooth_l1_loss(
                    center_coords,
                    target_heatmap_coords,
                )
                total_loss = total_loss + self.heatmap_center_weight * center_loss

            if pred_coords is not None and target_coords_fp32 is not None:
                pred_coords_fp32 = pred_coords.to(dtype=torch.float32)
                c_loss = self.coord_loss(pred_coords_fp32, target_coords_fp32)

                if self.coord_weight > 0.0:
                    total_loss = total_loss + self.coord_weight * c_loss

                if self.pixel_coord_weight > 0.0:
                    image_extent = float(Config.IMAGE_SIZE - 1)
                    pred_pixels = pred_coords_fp32 * image_extent
                    target_pixels = target_coords_fp32 * image_extent

                    pixel_abs_error = torch.abs(pred_pixels - target_pixels)
                    pixel_loss = pixel_abs_error.mean()

                    loss_term = pixel_loss
                    log_scale = float(Config.PIXEL_LOSS_LOG_SCALE)
                    if log_scale > 0.0:
                        # Use a log-shaped penalty to prevent extreme outliers from
                        # overwhelming the optimiser.  For large errors the gradient
                        # magnitude now decays roughly as 1 / error, which keeps the
                        # update gentle while still nudging the regressor back towards
                        # the target.
                        loss_term = (
                            torch.log1p(pixel_abs_error / log_scale).mean() * log_scale
                        )

                    total_loss = total_loss + self.pixel_coord_weight * loss_term

        outputs = (total_loss, h_loss, c_loss, pixel_loss)
        if target_coords_fp32 is not None or self.heatmap_center_weight > 0.0:
            outputs = outputs + (center_loss,)

        return outputs


def _make_group_norm(num_channels, max_groups=32):
    """Create a GroupNorm layer with the largest group size that divides the channels."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class WarmupCosineLR:
    """Lightweight warmup + cosine scheduler stepped per optimizer update."""

    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = int(min(warmup_steps, self.total_steps))
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.min_lr_ratio = float(min_lr_ratio)
        self.min_lrs = [lr * self.min_lr_ratio for lr in self.base_lrs]
        self.step_num = 0
        self.last_lrs = self.base_lrs.copy()

    def step(self):
        self.step_num = min(self.step_num + 1, self.total_steps)
        self.last_lrs = []
        for base_lr, min_lr, group in zip(self.base_lrs, self.min_lrs, self.optimizer.param_groups):
            if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * (self.step_num / max(1, self.warmup_steps))
            else:
                progress = 0.0
                if self.total_steps > self.warmup_steps:
                    progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = min_lr + (base_lr - min_lr) * cosine
            group['lr'] = lr
            self.last_lrs.append(lr)
        return self.last_lrs

    def get_last_lr(self):
        return self.last_lrs

    def state_dict(self):
        return {
            'step_num': self.step_num,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'base_lrs': self.base_lrs,
            'min_lrs': self.min_lrs,
            'min_lr_ratio': self.min_lr_ratio,
            'last_lrs': self.last_lrs,
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.total_steps = state_dict['total_steps']
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lrs = state_dict['base_lrs']
        self.min_lrs = state_dict['min_lrs']
        self.min_lr_ratio = state_dict.get('min_lr_ratio', self.min_lr_ratio)
        self.last_lrs = state_dict['last_lrs']
        self._apply_current_lrs()

    def _apply_current_lrs(self):
        # Reapply stored LR values to optimizer parameter groups
        for lr, group in zip(self.last_lrs, self.optimizer.param_groups):
            group['lr'] = lr

    def decay_base_lrs(self, factor: float, min_lr: float = 0.0) -> list[float]:
        """Decay stored base learning rates and re-synchronise the optimizer groups."""

        factor = float(factor)
        if factor >= 1.0:
            return self.get_last_lr()

        min_lr = float(min_lr)
        new_last = []
        for idx, (group, base_lr) in enumerate(
            zip(self.optimizer.param_groups, self.base_lrs)
        ):
            new_base = max(min_lr, base_lr * factor)
            self.base_lrs[idx] = new_base
            self.min_lrs[idx] = max(min_lr, new_base * self.min_lr_ratio)

            current_lr = float(group.get('lr', new_base))
            capped_lr = min(current_lr, new_base)
            capped_lr = max(capped_lr, self.min_lrs[idx])
            group['lr'] = capped_lr
            new_last.append(capped_lr)

        self.last_lrs = new_last
        return new_last


def create_warmup_scheduler(optimizer, remaining_epochs, updates_per_epoch):
    """Utility to build a warmup cosine scheduler for the given training horizon."""
    total_steps = int(max(remaining_epochs * updates_per_epoch, 1))
    warmup_steps = int(min(Config.WARMUP_EPOCHS * updates_per_epoch, total_steps))
    return WarmupCosineLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=Config.MIN_LR_FACTOR,
    )


def build_finetune_optimizer(model):
    """Create an optimizer that decouples backbone and head learning rates."""
    head_params = list(model.head.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters()
        if id(p) not in head_param_ids
    ]

    return optim.AdamW(
        [
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-4},
        ],
        weight_decay=Config.WEIGHT_DECAY,
        eps=1e-8,
    )

# ======================================================================================
# 5. MODEL ARCHITECTURE
# ======================================================================================

class HeatmapHead(nn.Module):
    """FPN-style decoder head for multi-scale feature fusion."""

    def __init__(self, in_channels_list, spatial_sizes, dropout_rate=0.1):
        super().__init__()
        if not isinstance(in_channels_list, (list, tuple)):
            raise ValueError("in_channels_list must be a list or tuple of channel dimensions")

        if not isinstance(spatial_sizes, (list, tuple)):
            raise ValueError("spatial_sizes must be provided for each feature level")

        if len(in_channels_list) != len(spatial_sizes):
            raise ValueError("in_channels_list and spatial_sizes must have the same length")

        self.in_channels_list = list(in_channels_list)
        self.spatial_sizes = list(spatial_sizes)
        self.num_scales = len(self.in_channels_list)

        # Normalize all feature maps to a shared channel width for fusion
        self.fpn_channels = min(256, max(self.in_channels_list))

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, self.fpn_channels, kernel_size=1)
            for channels in self.in_channels_list
        ])

        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(self.fpn_channels),
                nn.GELU(),
            )
            for _ in self.in_channels_list
        ])

        highest_resolution = self.spatial_sizes[0]
        self.decoder = self._build_decoder(highest_resolution, dropout_rate)

        # Coordinate regression uses the deepest semantic feature
        self.coord_proj = nn.Conv2d(self.in_channels_list[-1], self.fpn_channels, kernel_size=1)
        self.coord_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.fpn_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
        )

    def _build_decoder(self, base_spatial_size, dropout_rate):
        layers = [
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(self.fpn_channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
        ]

        current_channels = self.fpn_channels
        current_size = base_spatial_size

        while current_size < Config.HEATMAP_SIZE:
            next_channels = max(current_channels // 2, 64)
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1, bias=False),
                _make_group_norm(next_channels),
                nn.GELU(),
            ])
            current_channels = next_channels
            current_size *= 2

        layers.append(nn.Conv2d(current_channels, 1, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, features):
        if not isinstance(features, (list, tuple)):
            raise ValueError("HeatmapHead expects a list or tuple of backbone features")

        if len(features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature maps but received {len(features)}"
            )


        def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Expected feature tensor but received {type(tensor)}"
                )
            return (
                tensor
                if tensor.is_contiguous(memory_format=torch.contiguous_format)
                else tensor.contiguous()
            )

        fpn_results = [None] * self.num_scales
        prev_feature = None
        for idx in reversed(range(self.num_scales)):
            current_feature = _ensure_contiguous(features[idx])

            lateral = self.lateral_convs[idx](current_feature)
            if prev_feature is not None:
                prev_feature = F.interpolate(
                    prev_feature,
                    size=lateral.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
                lateral = lateral + _ensure_contiguous(prev_feature)

            smoothed = self.output_convs[idx](_ensure_contiguous(lateral))
            fpn_results[idx] = smoothed
            prev_feature = _ensure_contiguous(smoothed)


        fused_high_res = fpn_results[0]

        fused_high_res = torch.nan_to_num(
            fused_high_res,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        decoder_input = fused_high_res.to(dtype=torch.float32)
        device_type = decoder_input.device.type
        with autocast(device_type=device_type, enabled=False):
            heatmap = self.decoder(decoder_input)

        heatmap = torch.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        ).clamp_(-50.0, 50.0)

        if heatmap.shape[-1] != Config.HEATMAP_SIZE:
            heatmap = F.interpolate(
                heatmap,
                size=(Config.HEATMAP_SIZE, Config.HEATMAP_SIZE),
                mode='bilinear',
                align_corners=False,
            )

        deepest_feature = self.coord_proj(features[-1])
        deepest_feature = torch.nan_to_num(
            deepest_feature,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        coords_raw = self.coord_regressor(deepest_feature)
        coords_raw = torch.nan_to_num(
            coords_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        coords = torch.sigmoid(coords_raw)
        coords = torch.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=0.0)

        return heatmap, coords

class FullModel(nn.Module):
    """Complete model with backbone and head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        heatmap, coords = self.head(features)
        return heatmap, coords

def create_model(pretrained=True):
    """Create the full model with proper initialization."""
    print(f"Creating model: {Config.MODEL_NAME}")

    # Use timm for ConvNeXt
    backbone = timm.create_model(
        Config.MODEL_NAME,
        pretrained=pretrained,
        features_only=True,
        out_indices=(0, 1, 2, 3)  # Expose multiple stages for multi-scale fusion
    )

    if Config.ENABLE_GRAD_CHECKPOINTING and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for backbone")

    # Get feature dimensions
    dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    with torch.no_grad():
        features = backbone(dummy_input)
        if not isinstance(features, (list, tuple)):
            features = [features]

        in_channels_list = [feat.shape[1] for feat in features]
        spatial_sizes = [feat.shape[2] for feat in features]

    print(
        "Backbone outputs: "
        + ", ".join(
            f"{c}ch @ {s}x{s}" for c, s in zip(in_channels_list, spatial_sizes)
        )
    )

    # Create head with multi-scale information
    head = HeatmapHead(in_channels_list=in_channels_list, spatial_sizes=spatial_sizes)

    # Combine into full model
    model = FullModel(backbone, head)
    
    # Initialize weights for new layers
    for m in head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    return model

# ======================================================================================
# 6. TRAINING FUNCTIONS
# ======================================================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    epoch,
    scheduler=None,
    grad_accum_steps=1,
    use_mixup=None,
):
    """Train for one epoch with mixed precision, mixup, and gradient accumulation."""

    model.train()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coord_loss = 0.0
    total_pixel_loss = 0.0
    total_center_loss = 0.0

    num_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True)

    accumulation_counter = 0
    has_scaled_grads = False
    processed_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", colour="green")

    if use_mixup is None:
        use_mixup = Config.ENABLE_MIXUP and Config.MIXUP_ALPHA > 0.0

    for batch_idx, (images, target_heatmaps, target_coords) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        if Config.USE_CHANNELS_LAST:
            images = images.to(memory_format=torch.channels_last)
        target_heatmaps = target_heatmaps.to(device)
        target_coords = target_coords.to(device)

        if use_mixup and images.size(0) > 1 and np.random.rand() < 0.5:
            (
                images,
                target_a,
                target_b,
                coords_a,
                coords_b,
                lam,
            ) = mixup_data(images, target_heatmaps, target_coords, alpha=Config.MIXUP_ALPHA)
        else:
            target_a = target_heatmaps
            target_b = None
            coords_a = target_coords
            coords_b = None
            lam = 1.0

        with autocast(**_autocast_kwargs_for(device)):
            pred_heatmaps, pred_coords = model(images)

            if target_b is not None and coords_b is not None:
                loss_a, h_loss_a, c_loss_a, p_loss_a, ctr_loss_a = criterion(
                    pred_heatmaps, target_a, pred_coords, coords_a
                )
                loss_b, h_loss_b, c_loss_b, p_loss_b, ctr_loss_b = criterion(
                    pred_heatmaps, target_b, pred_coords, coords_b
                )
                loss = lam * loss_a + (1 - lam) * loss_b
                h_loss = lam * h_loss_a + (1 - lam) * h_loss_b
                c_loss = lam * c_loss_a + (1 - lam) * c_loss_b
                p_loss = lam * p_loss_a + (1 - lam) * p_loss_b
                center_loss = lam * ctr_loss_a + (1 - lam) * ctr_loss_b
            else:
                loss, h_loss, c_loss, p_loss, center_loss = criterion(
                    pred_heatmaps, target_a, pred_coords, coords_a
                )

        if not torch.isfinite(loss):
            print("\n⚠️ Non-finite loss encountered, skipping batch to protect training stability.")
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0
            has_scaled_grads = False
            continue

        total_loss += loss.item()
        total_heatmap_loss += h_loss.item()
        total_coord_loss += c_loss.item()
        total_pixel_loss += p_loss.item()
        total_center_loss += center_loss.item()

        processed_batches += 1

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        has_scaled_grads = True

        accumulation_counter += 1

        should_step = accumulation_counter >= grad_accum_steps or (batch_idx + 1) == num_batches
        if should_step and accumulation_counter > 0 and has_scaled_grads:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if scheduler is not None:
                current_lrs = scheduler.step()
                if batch_idx == 0:
                    pretty_lrs = ", ".join(f"{lr:.2e}" for lr in current_lrs)
                    print(f"    Scheduler LRs after first step: {pretty_lrs}")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0
            has_scaled_grads = False

        progress_bar.set_postfix(
            {
                'loss': f"{loss.item():.4f}",
                'h_loss': f"{h_loss.item():.4f}",
                'c_loss': f"{c_loss.item():.4f}",
                'p_loss': f"{p_loss.item():.4f}",
                'ctr': f"{center_loss.item():.4f}",
            }
        )

    effective_batches = max(processed_batches, 1)

    avg_loss = total_loss / effective_batches
    avg_h_loss = total_heatmap_loss / effective_batches
    avg_c_loss = total_coord_loss / effective_batches
    avg_p_loss = total_pixel_loss / effective_batches
    avg_ctr_loss = total_center_loss / effective_batches

    return avg_loss, avg_h_loss, avg_c_loss, avg_p_loss, avg_ctr_loss

def validate(model, dataloader, criterion, device, epoch):
    """Validate the model and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_errors = []
    fusion_weight_log = []
    total_pixel_loss = 0.0
    total_center_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}", colour="yellow")

    total_heatmap_abs_error = 0.0
    total_regressor_abs_error = 0.0
    total_coord_elems = 0
    
    with torch.no_grad():
        for images, target_heatmaps, target_coords in progress_bar:
            images = images.to(device, non_blocking=True)
            if Config.USE_CHANNELS_LAST:
                images = images.to(memory_format=torch.channels_last)
            target_heatmaps = target_heatmaps.to(device)
            target_coords = target_coords.to(device)
            
            # Forward pass
            with autocast(**_autocast_kwargs_for(device)):
                pred_heatmaps, pred_coords = model(images)
                loss, _, _, pixel_loss, center_loss = criterion(
                    pred_heatmaps,
                    target_heatmaps,
                    pred_coords,
                    target_coords,
                )

            total_loss += loss.item()
            total_pixel_loss += pixel_loss.item()
            total_center_loss += center_loss.item()
            
            # Extract coordinates from heatmap for evaluation
            heatmap_coords = get_coords_from_heatmap(pred_heatmaps, method='soft_argmax')

            if Config.HEATMAP_SIZE <= 1 or Config.IMAGE_SIZE <= 1:
                raise ValueError("HEATMAP_SIZE and IMAGE_SIZE must be greater than 1 for coordinate scaling")

            heatmap_to_image = (Config.IMAGE_SIZE - 1) / (Config.HEATMAP_SIZE - 1)
            heatmap_coords = heatmap_coords * heatmap_to_image

            pred_coords_pixels = pred_coords.detach() * (Config.IMAGE_SIZE - 1)
            target_coords_pixels = target_coords.detach() * (Config.IMAGE_SIZE - 1)

            total_heatmap_abs_error += torch.abs(heatmap_coords - target_coords_pixels).sum().item()
            total_regressor_abs_error += torch.abs(pred_coords_pixels - target_coords_pixels).sum().item()
            total_coord_elems += heatmap_coords.numel()

            fusion_weights = compute_heatmap_fusion_weight(pred_heatmaps, epoch=epoch)
            fusion_weight_mean = fusion_weights.mean().item()
            fusion_weight_log.append(fusion_weight_mean)
            fusion_weights = fusion_weights.unsqueeze(1)
            final_coords = fusion_weights * heatmap_coords + (1 - fusion_weights) * pred_coords_pixels

            final_coords_cpu = final_coords.cpu()
            target_coords_pixels_cpu = target_coords_pixels.cpu()

            all_preds.append(final_coords_cpu)
            all_targets.append(target_coords_pixels_cpu)

            # Calculate errors
            errors = torch.sqrt(
                (final_coords_cpu[:, 0] - target_coords_pixels_cpu[:, 0])**2 +
                (final_coords_cpu[:, 1] - target_coords_pixels_cpu[:, 1])**2
            )
            all_errors.extend(errors.tolist())

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'w_hm': f"{fusion_weight_mean:.2f}"
            })
    
    # Aggregate metrics
    avg_loss = total_loss / len(dataloader)
    avg_pixel_loss = total_pixel_loss / len(dataloader)
    avg_center_loss = total_center_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calculate MAE and other metrics
    mae = torch.mean(torch.abs(all_preds - all_targets))
    mae_heatmap = total_heatmap_abs_error / max(1, total_coord_elems)
    mae_regressor = total_regressor_abs_error / max(1, total_coord_elems)

    if epoch < 20:
        print(
            f"Branch MAE (epoch {epoch+1}): heatmap-only={mae_heatmap:.2f}px, "
            f"regressor-only={mae_regressor:.2f}px"
        )
    rmse = torch.sqrt(torch.mean((all_preds - all_targets)**2))
    
    # Calculate percentiles
    errors_tensor = torch.tensor(all_errors)
    if len(errors_tensor) > 0:
        percentiles = torch.quantile(errors_tensor, torch.tensor([0.5, 0.75, 0.95]))
        print(f"\nError Percentiles - 50%: {percentiles[0]:.1f}px, "
              f"75%: {percentiles[1]:.1f}px, 95%: {percentiles[2]:.1f}px")
    
    if fusion_weight_log:
        print(f"Average heatmap fusion weight: {np.mean(fusion_weight_log):.2f}")

    return (
        avg_loss,
        mae.item(),
        rmse.item(),
        all_errors,
        {
            'pixel_loss': avg_pixel_loss,
            'center_loss': avg_center_loss,
            'heatmap_mae': mae_heatmap,
            'regressor_mae': mae_regressor,
            'fusion_weight': np.mean(fusion_weight_log) if fusion_weight_log else 0.0,
        },
    )

def save_sample_predictions(model, val_loader, device, epoch, save_dir, num_samples=3):
    """Save sample predictions with heatmap visualizations."""
    model.eval()
    
    # Get a batch
    images, target_heatmaps, target_coords = next(iter(val_loader))
    images = images.to(device)
    target_heatmaps = target_heatmaps.to(device)
    target_coords = target_coords.to(device)
    
    with torch.no_grad():
        pred_heatmaps, pred_coords = model(images)
    
    # Extract coordinates from heatmap
    heatmap_coords = get_coords_from_heatmap(pred_heatmaps, method='soft_argmax')

    if Config.HEATMAP_SIZE <= 1 or Config.IMAGE_SIZE <= 1:
        raise ValueError("HEATMAP_SIZE and IMAGE_SIZE must be greater than 1 for coordinate scaling")

    heatmap_to_image = (Config.IMAGE_SIZE - 1) / (Config.HEATMAP_SIZE - 1)
    heatmap_coords = heatmap_coords * heatmap_to_image

    pred_coords_pixels = pred_coords * (Config.IMAGE_SIZE - 1)
    target_coords_pixels = target_coords * (Config.IMAGE_SIZE - 1)

    fusion_weights = compute_heatmap_fusion_weight(pred_heatmaps, epoch=epoch)
    fusion_weights = fusion_weights.unsqueeze(1)
    final_coords = fusion_weights * heatmap_coords + (1 - fusion_weights) * pred_coords_pixels

    fusion_weights_cpu = fusion_weights.squeeze(1).cpu()
    final_coords_cpu = final_coords.detach().cpu()
    target_coords_cpu = target_coords_pixels.detach().cpu()
    
    # Prepare figure
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image for display
        img = images[i].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        # Plot image with predictions
        axes[i, 0].imshow(img)
        axes[i, 0].scatter(target_coords_cpu[i, 0], target_coords_cpu[i, 1],
                          c='green', s=100, marker='x', linewidths=3, label='GT')
        axes[i, 0].scatter(final_coords_cpu[i, 0], final_coords_cpu[i, 1],
                          c='red', s=100, marker='+', linewidths=3, label='Pred')
        axes[i, 0].set_title(
            f'Image {i+1}: Error={torch.dist(final_coords_cpu[i], target_coords_cpu[i]):.1f}px, '
            f'w_hm={fusion_weights_cpu[i].item():.2f}'
        )
        axes[i, 0].legend()
        axes[i, 0].axis('off')
        
        # Plot target heatmap
        axes[i, 1].imshow(target_heatmaps[i, 0].cpu().numpy(), cmap='hot', interpolation='nearest')
        axes[i, 1].set_title('Target Heatmap')
        axes[i, 1].axis('off')
        
        # Plot predicted heatmap
        # Convert logits to calibrated probabilities before plotting to expose
        # the peak the network has learned rather than raw activation edges.
        pred_hm = torch.sigmoid(pred_heatmaps[i, 0]).cpu().numpy()
        axes[i, 2].imshow(pred_hm, cmap='hot', interpolation='nearest')
        axes[i, 2].set_title(f'Pred Heatmap (max={pred_hm.max():.3f})')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Sample Predictions - Epoch {epoch+1}', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'sample_predictions_epoch_{epoch+1}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

# ======================================================================================
# 7. MAIN TRAINING SCRIPT
# ======================================================================================

def main():
    print("="*80)
    print(f"Football Detection Training Script")
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Image Size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"Heatmap Size: {Config.HEATMAP_SIZE}x{Config.HEATMAP_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. Load and split dataset
    # -------------------------------------------------------------------------
    print("\n📁 Loading dataset...")
    all_files = [
        os.path.join(Config.DATASET_PATH, f) 
        for f in os.listdir(Config.DATASET_PATH) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    print(f"Total image files found: {len(all_files)}")
    
    # Validate files have correct naming format
    valid_files = []
    for f in all_files:
        coords = parse_filename(os.path.basename(f))
        if coords is not None:
            valid_files.append(f)
    
    print(f"Valid files with correct coordinate format: {len(valid_files)}")
    
    # Group files by scene
    scene_groups = defaultdict(list)
    for f in valid_files:
        try:
            # Extract scene ID from filename (adjust pattern as needed)
            base_name = os.path.basename(f)
            first_part = base_name.split('-')[0]
            match = re.search(r'(\d+)', first_part)
            if match:
                scene_id = match.group(1)
                scene_groups[scene_id].append(f)
        except (IndexError, AttributeError):
            continue
    
    # Print dataset statistics
    print(f"Total unique scenes: {len(scene_groups)}")
    print(f"Average images per scene: {len(valid_files) / len(scene_groups):.1f}")
    preview_scenes = list(scene_groups.items())[:5]
    if preview_scenes:
        print("Scene preview (scene_id -> sample filenames):")
        for scene_id, files in preview_scenes:
            sample_files = [os.path.basename(f) for f in files[:3]]
            print(f"  {scene_id}: {sample_files}")
    
    # Split by scenes to avoid data leakage
    scene_ids = list(scene_groups.keys())
    train_scene_ids, val_scene_ids = train_test_split(
        scene_ids, test_size=0.2, random_state=Config.RANDOM_SEED
    )
    
    train_files = [f for sid in train_scene_ids for f in scene_groups[sid]]
    val_files = [f for sid in val_scene_ids for f in scene_groups[sid]]
    
    print(f"Training scenes: {len(train_scene_ids)} ({len(train_files)} images)")
    print(f"Validation scenes: {len(val_scene_ids)} ({len(val_files)} images)")
    
    # -------------------------------------------------------------------------
    # 2. Create data augmentation pipelines
    # -------------------------------------------------------------------------
    try:
        random_resized_crop = A.RandomResizedCrop(
            size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            scale=(0.55, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_CUBIC,
        )

    except (TypeError, ValueError):

        random_resized_crop = A.RandomResizedCrop(
            height=Config.IMAGE_SIZE,
            width=Config.IMAGE_SIZE,
            scale=(0.55, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_CUBIC,
        )

    train_transform = A.Compose([
        A.OneOf([
            random_resized_crop,
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-8, 8),
            shear=(-4, 4),
            fit_output=False,
            cval=0,
            mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=cv2.BORDER_REFLECT_101, p=0.25),
        A.RandomBrightnessContrast(0.25, 0.2, p=0.5),
        A.ColorJitter(0.12, 0.12, 0.12, 0.05, p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
    
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    # -------------------------------------------------------------------------
    # 3. Create datasets and dataloaders
    # -------------------------------------------------------------------------
    train_dataset = FootballDataset(
        train_files,
        transform=train_transform,
        heatmap_size=Config.HEATMAP_SIZE,
        heatmap_sigma=Config.HEATMAP_SIGMA_START,
    )
    val_dataset = FootballDataset(
        val_files,
        transform=val_transform,
        heatmap_size=Config.HEATMAP_SIZE,
        heatmap_sigma=Config.HEATMAP_SIGMA_END,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False  # allow heatmap sigma updates to reach workers each epoch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False
    )

    print(f"\n✅ Dataloaders created successfully")

    updates_per_epoch = max(1, math.ceil(len(train_loader) / Config.GRAD_ACCUM_STEPS))
    
    # -------------------------------------------------------------------------
    # 4. Create model
    # -------------------------------------------------------------------------
    print("\n🏗️ Building model...")
    to_kwargs = {'device': Config.DEVICE}
    if Config.USE_CHANNELS_LAST:
        to_kwargs['memory_format'] = torch.channels_last
    model = create_model(pretrained=True).to(**to_kwargs)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # -------------------------------------------------------------------------
    # 5. Stage 1: Freeze backbone, train head only
    # -------------------------------------------------------------------------
    print("\n🔒 Stage 1: Freezing backbone, training head only")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # -------------------------------------------------------------------------
    # 6. Setup training components
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.INITIAL_LR,
        weight_decay=Config.WEIGHT_DECAY,
        eps=1e-8
    )

    criterion = CombinedLoss().to(Config.DEVICE)
    ADAPTIVE_STATE.apply_to_criterion(criterion)

    scheduler = None

    scaler = GradScaler(enabled=(Config.DEVICE == "cuda"))
    
    # -------------------------------------------------------------------------
    # 7. Initialize tracking variables
    # -------------------------------------------------------------------------
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    start_epoch = 0
    history = {
        'train_loss': [],
        'train_center_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'val_center_loss': [],
        'lr': []
    }
    
    adaptive_controller = AdaptiveTuningController(ADAPTIVE_STATE)
    loss_balancer = LossBalanceController(ADAPTIVE_STATE)

    # -------------------------------------------------------------------------
    # 8. Check for existing checkpoint
    # -------------------------------------------------------------------------
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, f"checkpoint_{Config.MODEL_VERSION}.pth")
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

        def _load_model_weights(model, checkpoint_state):
            """Load checkpoint weights that match the current architecture."""
            model_state = model.state_dict()
            compatible_state = {}
            skipped_keys = []

            for key, value in checkpoint_state.items():
                if key not in model_state:
                    skipped_keys.append((key, "missing in current model"))
                    continue

                if model_state[key].shape != value.shape:
                    skipped_keys.append((key, f"shape mismatch {value.shape} vs {model_state[key].shape}"))
                    continue

                compatible_state[key] = value

            load_result = model.load_state_dict(compatible_state, strict=False)

            if skipped_keys:
                print("\n⚠️ Some checkpoint weights were not loaded due to architecture changes:")
                for key, reason in skipped_keys:
                    print(f"   • {key}: {reason}")

            if load_result.missing_keys:
                print("\nℹ️ Newly initialized parameters:")
                for key in load_result.missing_keys:
                    print(f"   • {key}")

            if load_result.unexpected_keys:
                print("\nℹ️ Unused checkpoint parameters:")
                for key in load_result.unexpected_keys:
                    print(f"   • {key}")

        _load_model_weights(model, checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        if 'history' in checkpoint:
            history = checkpoint['history']
        for key in ('train_center_loss', 'val_center_loss'):
            history.setdefault(key, [])

        adaptive_payload = checkpoint.get('adaptive_state')
        if isinstance(adaptive_payload, dict):
            ADAPTIVE_STATE.load_state_dict(adaptive_payload)
            ADAPTIVE_STATE.apply_to_criterion(criterion)
            adaptive_controller.best_val_mae = best_val_mae

        tuner_payload = checkpoint.get('adaptive_tuner')
        if isinstance(tuner_payload, dict):
            adaptive_controller.best_val_mae = tuner_payload.get('best_val_mae', best_val_mae)
            adaptive_controller.no_improve_epochs = int(
                tuner_payload.get('no_improve_epochs', 0)
            )

        def _safe_load_optimizer(opt, opt_state):
            if not opt_state:
                return
            try:
                opt.load_state_dict(opt_state)
            except (ValueError, RuntimeError) as err:
                print("\n⚠️ Optimizer state could not be fully restored and will be reinitialized:")
                print(f"   • {err}")

        # Check if we should unfreeze based on epoch
        if start_epoch > Config.UNFREEZE_EPOCH:
            print(f"Model was already unfrozen at checkpoint, unfreezing all layers")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = build_finetune_optimizer(model)
            _safe_load_optimizer(optimizer, checkpoint.get('optimizer_state_dict'))
        else:
            # Original frozen state
            _safe_load_optimizer(optimizer, checkpoint.get('optimizer_state_dict'))

        print(f"Resumed from epoch {start_epoch}")

        scheduler_state = checkpoint.get('scheduler_state_dict')
        scheduler_remaining_epochs = 0
        if start_epoch > Config.UNFREEZE_EPOCH:
            scheduler_remaining_epochs = max(Config.EPOCHS - start_epoch, 0)
            if scheduler_remaining_epochs > 0:
                scheduler = create_warmup_scheduler(optimizer, scheduler_remaining_epochs, updates_per_epoch)
            else:
                scheduler = None
        elif start_epoch < Config.UNFREEZE_EPOCH:
            scheduler_remaining_epochs = max(Config.UNFREEZE_EPOCH - start_epoch, 0)
            if scheduler_remaining_epochs > 0:
                scheduler = create_warmup_scheduler(optimizer, scheduler_remaining_epochs, updates_per_epoch)
            else:
                scheduler = None
        else:
            scheduler = None

        if scheduler is not None and scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except (ValueError, RuntimeError) as err:
                print("\n⚠️ Scheduler state could not be restored and will be reinitialized:")
                print(f"   • {err}")
                if scheduler_remaining_epochs > 0:
                    scheduler = create_warmup_scheduler(optimizer, scheduler_remaining_epochs, updates_per_epoch)
                else:
                    scheduler = None

    if scheduler is None:
        if start_epoch < Config.UNFREEZE_EPOCH:
            remaining_epochs = max(Config.UNFREEZE_EPOCH - start_epoch, 0)
        else:
            remaining_epochs = max(Config.EPOCHS - start_epoch, 0)

        if remaining_epochs > 0:
            scheduler = create_warmup_scheduler(optimizer, remaining_epochs, updates_per_epoch)

    # -------------------------------------------------------------------------
    # 9. Main training loop
    # -------------------------------------------------------------------------
    print("\n🚀 Starting training...")
    
    for epoch in range(start_epoch, Config.EPOCHS):
        current_sigma = adaptive_controller.on_epoch_start(epoch, train_dataset, criterion)
        val_dataset.set_heatmap_sigma(current_sigma)
        val_dataset.set_label_smoothing(ADAPTIVE_STATE.label_smoothing)
        if (epoch % 5 == 0) or (epoch < 5):
            print(
                f"Using heatmap sigma {current_sigma:.2f} and smoothing "
                f"{ADAPTIVE_STATE.label_smoothing:.4f} for epoch {epoch+1}"
            )

        # Check if we should unfreeze the backbone
        if epoch == Config.UNFREEZE_EPOCH:
            print("\n" + "="*80)
            print(f"🔓 Epoch {epoch+1}: Unfreezing all layers for fine-tuning")
            print("="*80 + "\n")

            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create new optimizer with all parameters
            optimizer = build_finetune_optimizer(model)

            remaining_epochs = max(Config.EPOCHS - epoch, 0)
            scheduler = create_warmup_scheduler(
                optimizer,
                remaining_epochs,
                updates_per_epoch,
            ) if remaining_epochs > 0 else None

        # Get current learning rate
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        lr_report = ", ".join(
            f"group{i}:{group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)
        )
        print(f"Learning rates: {lr_report}")

        # Training phase
        train_loss, train_h_loss, train_c_loss, train_p_loss, train_ctr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            Config.DEVICE,
            scaler,
            epoch,
            scheduler=scheduler,
            grad_accum_steps=Config.GRAD_ACCUM_STEPS,
        )

        loss_balancer.observe(train_h_loss, train_c_loss, train_p_loss, criterion)

        # Validation phase
        if (epoch + 1) % Config.VALIDATE_EVERY_N_EPOCHS == 0:
            val_loss, val_mae, val_rmse, val_errors, val_stats = validate(
                model, val_loader, criterion, Config.DEVICE, epoch
            )

            # Save metrics
            history['train_loss'].append(train_loss)
            history['train_center_loss'].append(train_ctr_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            history['val_center_loss'].append(val_stats['center_loss'])
            history['lr'].append([group['lr'] for group in optimizer.param_groups])

            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(
                "  Train Loss: "
                f"{train_loss:.4f} (H: {train_h_loss:.4f}, C: {train_c_loss:.4f}, "
                f"Px: {train_p_loss:.4f}, Ctr: {train_ctr_loss:.4f})"
            )
            print(
                "  Val Loss: "
                f"{val_loss:.4f} (Px: {val_stats['pixel_loss']:.4f}, "
                f"Ctr: {val_stats['center_loss']:.4f}, "
                f"HM-MAE: {val_stats['heatmap_mae']:.2f}px, Reg-MAE: {val_stats['regressor_mae']:.2f}px, "
                f"w_hm: {val_stats['fusion_weight']:.2f})"
            )
            print(f"  Val MAE: {val_mae:.2f} pixels")
            print(f"  Val RMSE: {val_rmse:.2f} pixels")

            adaptive_controller.on_validation_end(
                epoch,
                val_mae,
                val_stats,
                criterion,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            # Save sample predictions for visual inspection
            if (epoch + 1) % 5 == 0 or epoch < 25:  # Every 5 epochs or first 5 epochs
                sample_path = save_sample_predictions(
                    model, val_loader, Config.DEVICE, epoch, Config.OUTPUT_DIR
                )
                print(f"  Sample predictions saved to {sample_path}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_val_mae': best_val_mae,
                'best_val_rmse': best_val_rmse,
                'history': history,
                'adaptive_state': ADAPTIVE_STATE.state_dict(),
                'adaptive_tuner': {
                    'best_val_mae': adaptive_controller.best_val_mae,
                    'no_improve_epochs': adaptive_controller.no_improve_epochs,
                },
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_model_path = os.path.join(
                    Config.OUTPUT_DIR, 
                    f"best_model_{Config.MODEL_VERSION}.pth"
                )
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✨ New best model saved! MAE: {best_val_mae:.2f} pixels")
        
        # Early stopping check
        current_lr = min(group['lr'] for group in optimizer.param_groups)
        if current_lr <= 1e-7 and epoch > Config.UNFREEZE_EPOCH + 10:
            print("\n⚠️ Learning rate too small, stopping training")
            break
    
    # -------------------------------------------------------------------------
    # 10. Final summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print(f"Best Validation MAE: {best_val_mae:.2f} pixels")
    print(f"Best Validation RMSE: {best_val_rmse:.2f} pixels")
    print(f"Model saved to: {Config.OUTPUT_DIR}")
    print("="*80)

# ======================================================================================
# 8. ENTRY POINT
# ======================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        raise

