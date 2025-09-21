# train_heatmap_fixed.py

import os
import random
import re
import math
from collections import defaultdict

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
    HEATMAP_SIGMA_END = 4.0     # Anneal towards a sharper target later in training
    HEATMAP_SIGMA_DECAY_EPOCHS = 50
    MIXUP_ALPHA = 0.2  # Mixup augmentation strength
    HEATMAP_CONFIDENCE_THRESHOLD = 0.3  # Confidence needed before trusting heatmap coords
    HEATMAP_CONFIDENCE_SCALE = 1.0  # Linear scaling factor for heatmap fusion weights

    # Memory optimization flags
    ENABLE_GRAD_CHECKPOINTING = True
    USE_CHANNELS_LAST = True
    
    # Validation settings
    VALIDATE_EVERY_N_EPOCHS = 1
    SAVE_BEST_ONLY = False

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
# 2. HELPER FUNCTIONS
# ======================================================================================

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

def create_target_heatmap(keypoints, size, sigma=None):
    """Create Gaussian heatmap for keypoints with proper normalization."""
    if sigma is None:
        sigma = Config.HEATMAP_SIGMA_START
        
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
        
    return heatmap

def soft_argmax_2d(heatmap, temperature=1.0):
    """Compute soft-argmax for differentiable coordinate extraction."""
    batch_size, _, height, width = heatmap.shape
    
    # Apply softmax to get probabilities
    heatmap_flat = heatmap.view(batch_size, -1)
    heatmap_probs = F.softmax(heatmap_flat / temperature, dim=1)
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


def compute_heatmap_fusion_weight(heatmap_logits):
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
        confidence = confidence.clamp_(0.0, 1.0)

        if Config.HEATMAP_CONFIDENCE_SCALE != 1.0:
            confidence = (confidence * Config.HEATMAP_CONFIDENCE_SCALE).clamp_(0.0, 1.0)

    return confidence

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
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma if heatmap_sigma is not None else Config.HEATMAP_SIGMA_START
        
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
        )

        # Store precise coordinates for direct regression (normalized to [0, 1])
        precise_coords = torch.tensor(keypoints[0], dtype=torch.float32) / (Config.IMAGE_SIZE - 1)
        
        return image, torch.from_numpy(target_heatmap).unsqueeze(0), precise_coords

    def set_heatmap_sigma(self, sigma):
        self.heatmap_sigma = float(sigma)


def compute_sigma_for_epoch(epoch):
    """Linearly anneal the target sigma during training for denser early gradients."""
    if Config.HEATMAP_SIGMA_DECAY_EPOCHS <= 0:
        return Config.HEATMAP_SIGMA_END

    progress = min(max(epoch, 0) / Config.HEATMAP_SIGMA_DECAY_EPOCHS, 1.0)
    return (
        Config.HEATMAP_SIGMA_START
        + (Config.HEATMAP_SIGMA_END - Config.HEATMAP_SIGMA_START) * progress
    )

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
    def __init__(self, heatmap_weight=0.7, coord_weight=0.3):
        super().__init__()
        self.coord_loss = nn.SmoothL1Loss()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight

    def forward(self, pred_heatmaps, target_heatmaps, pred_coords=None, target_coords=None):
        device_type = pred_heatmaps.device.type
        with autocast(device_type=device_type, enabled=False):
            pred_heatmaps_fp32 = pred_heatmaps.to(dtype=torch.float32)
            target_heatmaps_fp32 = target_heatmaps.to(dtype=torch.float32)

            # Switch to a logits-based loss that dramatically up-weights pixels near the
            # bright center of the Gaussian target.  This keeps background regions from
            # dominating the objective while still benefiting from numerically stable
            # BCE-with-logits gradients.
            prob = torch.sigmoid(pred_heatmaps_fp32)

            target_clamped = target_heatmaps_fp32.clamp(0.0, 1.0)
            bright_weight = 1.0 + 99.0 * target_clamped
            focal_modulation = torch.where(
                target_clamped > 0.01,
                (1.0 - prob).pow(2.0),
                prob.pow(2.0),
            )
            # Detach the dynamically generated weight tensor.  BCE-with-logits does not
            # support gradient propagation through the ``weight`` argument, so we ensure
            # it remains a statically valued tensor while still reflecting the latest
            # prediction-dependent modulation.
            weight = (bright_weight * (1.0 + focal_modulation)).detach()

            h_loss = F.binary_cross_entropy_with_logits(
                pred_heatmaps_fp32,
                target_clamped,
                weight=weight,
                reduction='mean',
            )

            if pred_coords is not None and target_coords is not None:
                pred_coords_fp32 = pred_coords.to(dtype=torch.float32)
                target_coords_fp32 = target_coords.to(dtype=torch.float32)
                c_loss = self.coord_loss(pred_coords_fp32, target_coords_fp32)
                total_loss = self.heatmap_weight * h_loss + self.coord_weight * c_loss
            else:
                c_loss = pred_heatmaps_fp32.new_tensor(0.0)
                total_loss = self.heatmap_weight * h_loss

        return total_loss, h_loss, c_loss


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
        self.min_lrs = [lr * min_lr_ratio for lr in self.base_lrs]
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
            'last_lrs': self.last_lrs,
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.total_steps = state_dict['total_steps']
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lrs = state_dict['base_lrs']
        self.min_lrs = state_dict['min_lrs']
        self.last_lrs = state_dict['last_lrs']
        self._apply_current_lrs()

    def _apply_current_lrs(self):
        # Reapply stored LR values to optimizer parameter groups
        for lr, group in zip(self.last_lrs, self.optimizer.param_groups):
            group['lr'] = lr


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
        heatmap = self.decoder(fused_high_res)

        if heatmap.shape[-1] != Config.HEATMAP_SIZE:
            heatmap = F.interpolate(
                heatmap,
                size=(Config.HEATMAP_SIZE, Config.HEATMAP_SIZE),
                mode='bilinear',
                align_corners=False,
            )

        deepest_feature = self.coord_proj(features[-1])
        coords_raw = self.coord_regressor(deepest_feature)
        coords = torch.sigmoid(coords_raw)

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
    use_mixup=True,
):
    """Train for one epoch with mixed precision, mixup, and gradient accumulation."""

    model.train()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coord_loss = 0.0

    num_batches = len(dataloader)
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", colour="green")

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

        with autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=torch.float16):
            pred_heatmaps, pred_coords = model(images)

            if target_b is not None and coords_b is not None:
                loss_a, h_loss_a, c_loss_a = criterion(pred_heatmaps, target_a, pred_coords, coords_a)
                loss_b, h_loss_b, c_loss_b = criterion(pred_heatmaps, target_b, pred_coords, coords_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                h_loss = lam * h_loss_a + (1 - lam) * h_loss_b
                c_loss = lam * c_loss_a + (1 - lam) * c_loss_b
            else:
                loss, h_loss, c_loss = criterion(pred_heatmaps, target_a, pred_coords, coords_a)

        if not torch.isfinite(loss):
            print("\n⚠️ Non-finite loss encountered, skipping batch to protect training stability.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        total_loss += loss.item()
        total_heatmap_loss += h_loss.item()
        total_coord_loss += c_loss.item()

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches
        if should_step:
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

        progress_bar.set_postfix(
            {
                'loss': f"{loss.item():.4f}",
                'h_loss': f"{h_loss.item():.4f}",
                'c_loss': f"{c_loss.item():.4f}",
            }
        )

    avg_loss = total_loss / num_batches
    avg_h_loss = total_heatmap_loss / num_batches
    avg_c_loss = total_coord_loss / num_batches

    return avg_loss, avg_h_loss, avg_c_loss

def validate(model, dataloader, criterion, device, epoch):
    """Validate the model and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_errors = []
    fusion_weight_log = []
    
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
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=torch.float16):
                pred_heatmaps, pred_coords = model(images)
                loss, _, _ = criterion(
                    pred_heatmaps, target_heatmaps,
                    pred_coords, target_coords
                )
            
            total_loss += loss.item()
            
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

            fusion_weights = compute_heatmap_fusion_weight(pred_heatmaps)
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

    return avg_loss, mae.item(), rmse.item(), all_errors

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

    fusion_weights = compute_heatmap_fusion_weight(pred_heatmaps)
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
    train_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=8, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=0.3),
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

    criterion = CombinedLoss(heatmap_weight=0.7, coord_weight=0.3).to(Config.DEVICE)

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
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'lr': []
    }
    
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
        current_sigma = compute_sigma_for_epoch(epoch)
        train_dataset.set_heatmap_sigma(current_sigma)
        if (epoch % 5 == 0) or (epoch < 5):
            print(f"Using heatmap sigma {current_sigma:.2f} for epoch {epoch+1}")

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
        train_loss, train_h_loss, train_c_loss = train_one_epoch(
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
        
        # Validation phase
        if (epoch + 1) % Config.VALIDATE_EVERY_N_EPOCHS == 0:
            val_loss, val_mae, val_rmse, val_errors = validate(
                model, val_loader, criterion, Config.DEVICE, epoch
            )

            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            history['lr'].append([group['lr'] for group in optimizer.param_groups])
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (H: {train_h_loss:.4f}, C: {train_c_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_mae:.2f} pixels")
            print(f"  Val RMSE: {val_rmse:.2f} pixels")
            
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
                'history': history
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

