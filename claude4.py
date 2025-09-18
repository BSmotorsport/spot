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
    OUTPUT_DIR = r"./training_output_convnext"
    MODEL_VERSION = "ConvNext_heatmap_combined"
    
    # Image dimensions
    ORIGINAL_WIDTH = 4416
    ORIGINAL_HEIGHT = 3336
    IMAGE_SIZE = 1536
    HEATMAP_SIZE = 384
    
    # Model settings - ACTUALLY USING CONVNEXT NOW
    MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k'  # ConvNeXt Base model
    BATCH_SIZE = 2  # ConvNeXt needs more memory
    EPOCHS = 250
    INITIAL_LR = 3e-4
    FINETUNE_LR = 3e-5
    UNFREEZE_EPOCH = 10
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
    HEATMAP_SIGMA = 1.5  # Tighter gaussian for more precision
    MIXUP_ALPHA = 0.2  # Mixup augmentation strength
    
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
        sigma = Config.HEATMAP_SIGMA
        
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
    coords[:, 0] *= width
    coords[:, 1] *= height
    
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

# ======================================================================================
# 3. DATASET CLASS
# ======================================================================================

class FootballDataset(Dataset):
    def __init__(self, image_paths, transform=None, heatmap_size=192, augment=True):
        self.image_paths = image_paths
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.augment = augment
        
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
            
            # Skip if augmentation removed the keypoint
            if not keypoints:
                return self.__getitem__((idx + 1) % len(self))
        
        # Convert keypoints to heatmap space
        heatmap_keypoints = [
            (kp[0] * (self.heatmap_size / Config.IMAGE_SIZE),
             kp[1] * (self.heatmap_size / Config.IMAGE_SIZE))
            for kp in keypoints
        ]
        
        # Create target heatmap
        target_heatmap = create_target_heatmap(heatmap_keypoints, self.heatmap_size)
        
        # Store precise coordinates for direct regression
        precise_coords = torch.tensor(keypoints[0], dtype=torch.float32)
        
        return image, torch.from_numpy(target_heatmap).unsqueeze(0), precise_coords

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
        # Use BCE with logits to preserve gradients on sparse heatmaps
        self.heatmap_loss = nn.BCEWithLogitsLoss()
        # Keep SmoothL1 for coordinates
        self.coord_loss = nn.SmoothL1Loss()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight

    def forward(self, pred_heatmaps, target_heatmaps, pred_coords=None, target_coords=None):
        h_loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)

        if pred_coords is not None and target_coords is not None:
            c_loss = self.coord_loss(pred_coords, target_coords)
            return self.heatmap_weight * h_loss + self.coord_weight * c_loss, h_loss, c_loss

        return h_loss, h_loss, torch.tensor(0.0)


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

# ======================================================================================
# 5. MODEL ARCHITECTURE
# ======================================================================================

class HeatmapHead(nn.Module):
    """Decoder head for heatmap generation with skip connections."""
    def __init__(self, in_features, spatial_size=24, dropout_rate=0.1):
        super().__init__()
        self.spatial_size = spatial_size
        self.in_features = in_features

        # Calculate upsampling factor needed
        # From spatial_size to Config.HEATMAP_SIZE (384)
        self.scale_factor = Config.HEATMAP_SIZE // spatial_size

        # Adjust channels based on input features
        mid_channels = min(512, in_features // 2)

        half_channels = max(mid_channels // 2, 128)
        quarter_channels = max(half_channels // 2, 64)

        # Progressive upsampling decoder with GroupNorm for small batches
        self.decoder = nn.Sequential(
            nn.Conv2d(in_features, mid_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(mid_channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, half_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(half_channels),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(half_channels, quarter_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(quarter_channels),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(quarter_channels, 64, kernel_size=3, padding=1, bias=False),
            _make_group_norm(64),
            nn.GELU(),

            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Coordinate regression head
        self.coord_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # Output x, y coordinates
        )
    
    def forward(self, x):
        # Store original shape for coord regression
        orig_x = x
        
        # Reshape if necessary (from patches to spatial)
        if len(x.shape) == 3:  # (B, H*W, C)
            B, L, C = x.shape
            H = W = int(np.sqrt(L))
            x = x.transpose(1, 2).view(B, C, H, W)
        elif len(x.shape) == 4:
            # Already in (B, C, H, W) format
            B, C, H, W = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Generate heatmap
        heatmap = self.decoder(x)
        
        # Ensure heatmap is the correct size
        if heatmap.shape[-1] != Config.HEATMAP_SIZE:
            heatmap = F.interpolate(
                heatmap, 
                size=(Config.HEATMAP_SIZE, Config.HEATMAP_SIZE),
                mode='bilinear',
                align_corners=False
            )
        
        # Predict coordinates directly for combined loss
        # Use the original feature for coordinate regression
        if len(orig_x.shape) == 3:
            B, L, C = orig_x.shape
            H = W = int(np.sqrt(L))
            orig_x = orig_x.transpose(1, 2).view(B, C, H, W)
        
        coords_raw = self.coord_regressor(orig_x)
        
        # Apply sigmoid to bound coordinates to [0, 1] then scale to image size
        # This ensures coordinates are always in valid range
        coords = torch.sigmoid(coords_raw) * Config.IMAGE_SIZE
        
        return heatmap, coords

class FullModel(nn.Module):
    """Complete model with backbone and head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        features = self.backbone(x)
        # Use the last feature map
        last_features = features[-1] if isinstance(features, list) else features
        heatmap, coords = self.head(last_features)
        return heatmap, coords

def create_model(pretrained=True):
    """Create the full model with proper initialization."""
    print(f"Creating model: {Config.MODEL_NAME}")
    
    # Use timm for ConvNeXt
    backbone = timm.create_model(
        Config.MODEL_NAME,
        pretrained=pretrained,
        features_only=True,
        out_indices=[-1]  # Get only the last feature map
    )
    
    # Get feature dimensions
    dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    with torch.no_grad():
        features = backbone(dummy_input)
        if isinstance(features, list):
            in_features = features[-1].shape[1]
            spatial_size = features[-1].shape[2]
        else:
            in_features = features.shape[1]
            spatial_size = features.shape[2]
    
    print(f"Backbone output: {in_features} channels, {spatial_size}x{spatial_size} spatial")
    
    # Create head with correct spatial size
    head = HeatmapHead(in_features, spatial_size=spatial_size)
    
    # Combine into full model
    model = FullModel(backbone, head)
    
    # Initialize weights for new layers
    for m in head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        target_coords = target_coords.to(device)

        if use_mixup and np.random.rand() < 0.5:
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
                scheduler.step()

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
    
    progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}", colour="yellow")
    
    with torch.no_grad():
        for images, target_heatmaps, target_coords in progress_bar:
            images = images.to(device)
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
            
            # Scale heatmap coordinates back to image size
            heatmap_coords = heatmap_coords * (Config.IMAGE_SIZE / Config.HEATMAP_SIZE)
            
            # Use combination of heatmap and direct predictions
            final_coords = 0.6 * heatmap_coords.cpu() + 0.4 * pred_coords.cpu()
            
            all_preds.append(final_coords)
            all_targets.append(target_coords.cpu())
            
            # Calculate errors
            errors = torch.sqrt(
                (final_coords[:, 0] - target_coords.cpu()[:, 0])**2 +
                (final_coords[:, 1] - target_coords.cpu()[:, 1])**2
            )
            all_errors.extend(errors.tolist())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Aggregate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calculate MAE and other metrics
    mae = torch.mean(torch.abs(all_preds - all_targets))
    rmse = torch.sqrt(torch.mean((all_preds - all_targets)**2))
    
    # Calculate percentiles
    errors_tensor = torch.tensor(all_errors)
    if len(errors_tensor) > 0:
        percentiles = torch.quantile(errors_tensor, torch.tensor([0.5, 0.75, 0.95]))
        print(f"\nError Percentiles - 50%: {percentiles[0]:.1f}px, "
              f"75%: {percentiles[1]:.1f}px, 95%: {percentiles[2]:.1f}px")
    
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
    heatmap_coords = heatmap_coords * (Config.IMAGE_SIZE / Config.HEATMAP_SIZE)
    
    # Combined prediction
    final_coords = 0.6 * heatmap_coords + 0.4 * pred_coords
    # Move coords to CPU once for visualization to avoid device mismatches
    final_coords_cpu = final_coords.detach().cpu()
    target_coords_cpu = target_coords.detach().cpu()
    
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
            f'Image {i+1}: Error={torch.dist(final_coords_cpu[i], target_coords_cpu[i]):.1f}px'
        )
        axes[i, 0].legend()
        axes[i, 0].axis('off')
        
        # Plot target heatmap
        axes[i, 1].imshow(target_heatmaps[i, 0].cpu().numpy(), cmap='hot', interpolation='nearest')
        axes[i, 1].set_title('Target Heatmap')
        axes[i, 1].axis('off')
        
        # Plot predicted heatmap
        pred_hm = pred_heatmaps[i, 0].cpu().numpy()
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
        A.Rotate(limit=10, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
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
        heatmap_size=Config.HEATMAP_SIZE
    )
    val_dataset = FootballDataset(
        val_files, 
        transform=val_transform, 
        heatmap_size=Config.HEATMAP_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=Config.NUM_WORKERS > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=Config.NUM_WORKERS > 0
    )

    print(f"\n✅ Dataloaders created successfully")

    updates_per_epoch = max(1, math.ceil(len(train_loader) / Config.GRAD_ACCUM_STEPS))
    
    # -------------------------------------------------------------------------
    # 4. Create model
    # -------------------------------------------------------------------------
    print("\n🏗️ Building model...")
    model = create_model(pretrained=True).to(Config.DEVICE)
    
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

    criterion = CombinedLoss(heatmap_weight=0.7, coord_weight=0.3)

    scheduler = None
    
    scaler = GradScaler()
    
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
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        if 'history' in checkpoint:
            history = checkpoint['history']

        # Check if we should unfreeze based on epoch
        if start_epoch > Config.UNFREEZE_EPOCH:
            print(f"Model was already unfrozen at checkpoint, unfreezing all layers")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.FINETUNE_LR,
                weight_decay=Config.WEIGHT_DECAY,
                eps=1e-8
            )
            # Now load optimizer state after creating with all parameters
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Original frozen state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Resumed from epoch {start_epoch}")

        scheduler_state = checkpoint.get('scheduler_state_dict')
        if start_epoch > Config.UNFREEZE_EPOCH:
            remaining_epochs = max(Config.EPOCHS - start_epoch, 0)
            scheduler = create_warmup_scheduler(optimizer, remaining_epochs, updates_per_epoch) if remaining_epochs > 0 else None
        elif start_epoch < Config.UNFREEZE_EPOCH:
            remaining_epochs = max(Config.UNFREEZE_EPOCH - start_epoch, 0)
            scheduler = create_warmup_scheduler(optimizer, remaining_epochs, updates_per_epoch) if remaining_epochs > 0 else None
        else:
            scheduler = None

        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

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
        # Check if we should unfreeze the backbone
        if epoch == Config.UNFREEZE_EPOCH:
            print("\n" + "="*80)
            print(f"🔓 Epoch {epoch+1}: Unfreezing all layers for fine-tuning")
            print("="*80 + "\n")

            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True

            # Create new optimizer with all parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.FINETUNE_LR,
                weight_decay=Config.WEIGHT_DECAY,
                eps=1e-8
            )

            remaining_epochs = max(Config.EPOCHS - epoch, 0)
            scheduler = create_warmup_scheduler(
                optimizer,
                remaining_epochs,
                updates_per_epoch,
            ) if remaining_epochs > 0 else None

        # Get current learning rate
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

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
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (H: {train_h_loss:.4f}, C: {train_c_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_mae:.2f} pixels")
            print(f"  Val RMSE: {val_rmse:.2f} pixels")
            
            # Save sample predictions for visual inspection
            if (epoch + 1) % 5 == 0 or epoch < 5:  # Every 5 epochs or first 5 epochs
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
        current_lr = optimizer.param_groups[0]['lr']
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