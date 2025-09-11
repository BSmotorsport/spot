import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import argparse
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import warnings
import time
import subprocess
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Performance optimizations and reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def set_seed(seed=42, deterministic=True):
    """Set consistent seeds and performance settings"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# Configuration
ORIGINAL_IMAGE_SIZE = (4416, 3336)
IMAGE_SIZE = (2048, 2048)
HEATMAP_SIZE = 256  # Output heatmap resolution for sub-pixel accuracy
CACHE_DIR = "cache_2048x2048"

class SSDImageCache:
    """SSD cache for 2048×2048 images with coordinate preservation"""
    
    def __init__(self, cache_dir=CACHE_DIR, original_size=ORIGINAL_IMAGE_SIZE):
        self.cache_dir = cache_dir
        self.original_size = original_size
        self.mapping_file = os.path.join(cache_dir, "filename_mapping.json")
        self.coordinate_file = os.path.join(cache_dir, "coordinates.json")
        self.filename_map = {}
        self.coordinates = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_mappings()
    
    def _load_mappings(self):
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    self.filename_map = json.load(f)
                print(f"Loaded {len(self.filename_map)} cached filename mappings")
            except Exception as e:
                print(f"Error loading cache mapping: {e}")
        
        if os.path.exists(self.coordinate_file):
            try:
                with open(self.coordinate_file, 'r') as f:
                    self.coordinates = json.load(f)
                print(f"Loaded {len(self.coordinates)} cached coordinates")
            except Exception as e:
                print(f"Error loading coordinates: {e}")
    
    def _save_mappings(self):
        """Save mappings with batching to reduce I/O"""
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.filename_map, f)
            with open(self.coordinate_file, 'w') as f:
                json.dump(self.coordinates, f)
        except Exception as e:
            print(f"Error saving cache mappings: {e}")
    
    def get_cache_path(self, filename, target_size):
        size_str = f"{target_size[0]}x{target_size[1]}"
        cache_filename = f"{size_str}_{filename}"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        map_key = f"{filename}_{size_str}"
        self.filename_map[map_key] = cache_filename
        
        if len(self.filename_map) % 50 == 0:
            self._save_mappings()
        
        return cache_path
    
    def extract_coordinates(self, filename):
        if filename in self.coordinates:
            return self.coordinates[filename]
        
        stem = os.path.splitext(filename)[0]
        # Remove augmentation suffixes (_DC, _LDC, _XDC)
        base_stem = re.sub(r'_[XLA]*DC$', '', stem, flags=re.IGNORECASE)
        
        # Extract coordinates from format like ADC0122-1812-1424
        coord_match = re.search(r'-(\d{3,4})-(\d{3,4})$', base_stem)
        if coord_match:
            x = float(coord_match.group(1))
            y = float(coord_match.group(2))
            coords = [x, y]
            self.coordinates[filename] = coords
            return coords
        else:
            # Try alternative patterns for different formats
            coord_match = re.search(r'(\d{3,4})-(\d{3,4})$', base_stem)
            if coord_match:
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                coords = [x, y]
                self.coordinates[filename] = coords
                return coords
            raise ValueError(f"Could not parse coordinates from filename: {filename}")
    
    def get_cached_image(self, original_path, target_size):
        filename = os.path.basename(original_path)
        map_key = f"{filename}_{target_size[0]}x{target_size[1]}"
        
        if map_key in self.filename_map:
            cache_filename = self.filename_map[map_key]
            cache_path = os.path.join(self.cache_dir, cache_filename)
            if os.path.exists(cache_path):
                with Image.open(cache_path) as img:
                    image = img.convert('RGB')
                    coords = self.extract_coordinates(filename)
                    return image, coords, filename
        
        cache_path = self.get_cache_path(filename, target_size)
        with Image.open(original_path) as img:
            original = img.convert('RGB')
            resized = original.resize(target_size, Image.Resampling.LANCZOS)
            
            coords = self.extract_coordinates(filename)
            resized.save(cache_path, 'JPEG', quality=85)
            self._save_mappings()
            
            return resized, coords, filename
    
    def create_full_cache(self, dataset_dir):
        print("Creating image cache for faster training...")
        
        all_files = []
        exclude_dirs = {'checkpoints', 'sample_predictions', 'training_plots', 'cache', '.git', '__pycache__'}
        
        for root, dirs, files in os.walk(dataset_dir):
            root_path = Path(root)
            if any(exclude in str(root_path).lower() for exclude in exclude_dirs):
                continue
                
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    if not rel_path.startswith(f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_"):
                        all_files.append(rel_path)
        
        total_files = len(all_files)
        for i, rel_path in enumerate(all_files):
            if i % 100 == 0:
                print(f"Caching images: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            original_path = os.path.join(dataset_dir, rel_path)
            try:
                self.get_cached_image(original_path, IMAGE_SIZE)
            except Exception as e:
                print(f"Error caching {rel_path}: {e}")
        
        print(f"Cache creation complete! {total_files} images cached at {IMAGE_SIZE}")

def group_key_from_name(fname: str) -> str:
    """Extract base image key from filename, excluding augmentation suffixes."""
    stem = os.path.splitext(fname)[0]
    
    # Remove augmentation suffixes (_DC, _LDC, _XDC)
    stem = re.sub(r'_[XLA]*DC\d+$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'_[XLA]*DC$', '', stem, flags=re.IGNORECASE)
    
    # Extract DC number pattern
    dc_match = re.search(r'(DC\d{4})', stem, flags=re.IGNORECASE)
    if dc_match:
        return dc_match.group(1).upper()
    
    return stem.upper()

def build_grouped_split_files(image_folder: str, seed: int, val_frac: float):
    all_files = []
    exclude_dirs = {'checkpoints', 'sample_predictions', 'training_plots', 'cache', '.git', '__pycache__'}
    
    for root, dirs, files in os.walk(image_folder):
        root_path = Path(root)
        if any(exclude in root_path.parts for exclude in exclude_dirs):
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if not file.startswith(f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_"):
                    all_files.append(os.path.relpath(os.path.join(root, file), image_folder))
    all_files = sorted(all_files)
    
    groups = {}
    for f in all_files:
        g = group_key_from_name(os.path.basename(f))
        groups.setdefault(g, []).append(f)
    
    print(f"[split] total_images={len(all_files)} | groups={len(groups)} | mean_per_group={np.mean([len(v) for v in groups.values()]):.2f}")
    
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(val_frac * len(keys))))
    val_keys = set(keys[:n_val])
    
    train_files = [f for k in keys if k not in val_keys for f in groups[k]]
    val_files = [f for k in val_keys for f in groups[k]]
    print(f"[split] train_imgs={len(train_files)} | val_imgs={len(val_files)}")
    return train_files, val_files

def create_gaussian_heatmap(coords_norm, heatmap_size=HEATMAP_SIZE, sigma=2.0):
    """Create Gaussian heatmap for given normalized coordinates"""
    heatmap = np.zeros((heatmap_size, heatmap_size))
    
    # Convert normalized coords to heatmap coordinates
    x = int(coords_norm[0] * heatmap_size)
    y = int(coords_norm[1] * heatmap_size)
    
    # Ensure coordinates are within bounds
    x = np.clip(x, 0, heatmap_size - 1)
    y = np.clip(y, 0, heatmap_size - 1)
    
    # Create Gaussian centered at (x, y)
    for i in range(max(0, x-10), min(heatmap_size, x+11)):
        for j in range(max(0, y-10), min(heatmap_size, y+11)):
            dist_sq = (i - x) ** 2 + (j - y) ** 2
            heatmap[j, i] = np.exp(-dist_sq / (2 * sigma ** 2))
    
    return heatmap

class HighResSpotBallDataset(Dataset):
    def __init__(self, image_folder: str, file_list: List[str], 
                 image_size: Tuple[int, int] = IMAGE_SIZE,
                 original_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE,
                 heatmap_size: int = HEATMAP_SIZE):
        self.image_folder = image_folder
        self.file_list = file_list
        self.image_size = image_size
        self.original_size = original_size
        self.heatmap_size = heatmap_size
        self.data_dir = image_folder
        self.cache = SSDImageCache()
        
        # Pre-compute normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Pre-compute transform pipeline for efficiency
        self.transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        original_path = os.path.join(self.image_folder, filename)
        
        # Use cached image and coordinates
        image, coords, _ = self.cache.get_cached_image(original_path, self.image_size)
        
        # Apply pre-computed transform
        image = self.transform_pipeline(image)
        
        # Normalize coordinates consistently
        x_norm = coords[0] / ORIGINAL_IMAGE_SIZE[0]
        y_norm = coords[1] / ORIGINAL_IMAGE_SIZE[1]
        
        # Create Gaussian heatmap for training
        heatmap = create_gaussian_heatmap([x_norm, y_norm], self.heatmap_size)
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        
        # Also return normalized coords for direct regression loss
        coords_tensor = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        
        return image, heatmap, coords_tensor, filename

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on relevant regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        
    def forward(self, x):
        attn = torch.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return x * attn

class CoordConv2d(nn.Module):
    """Add coordinate channels to help model understand spatial locations"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        batch, _, h, w = x.size()
        
        # Create coordinate channels
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(batch, 1, 1, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(batch, 1, h, 1)
        
        # Concatenate coordinate channels
        x = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(x)

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
    
    def forward(self, features):
        # Build top-down path
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down path with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        
        # Apply output convolutions
        outputs = [conv(lateral) for conv, lateral in zip(self.output_convs, laterals)]
        
        return outputs

class EnhancedSpotBallModel(nn.Module):
    def __init__(self, backbone_name='convnext_xlarge.fb_in22k_ft_in1k', pretrained=True, heatmap_size=HEATMAP_SIZE):
        super().__init__()
        self.heatmap_size = heatmap_size
        
        # Create backbone - using ConvNeXt-XL for better features
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        
        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
        
        # FPN for multi-scale features
        fpn_dim = 256
        self.fpn = FPN(self.feature_dims[-3:], fpn_dim)  # Use last 3 feature levels
        
        # Spatial attention modules for each FPN level
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention(fpn_dim) for _ in range(3)
        ])
        
        # Heatmap prediction head with CoordConv
        self.heatmap_head = nn.Sequential(
            CoordConv2d(fpn_dim * 3, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )
        
        # Direct regression head for auxiliary loss
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_dim * 3, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'stages.0' in name or 'stages.1' in name:
                param.requires_grad = False
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Apply FPN
        fpn_features = self.fpn(features[-3:])
        
        # Apply spatial attention
        attended_features = [attn(feat) for attn, feat in zip(self.spatial_attentions, fpn_features)]
        
        # Resize all features to same size and concatenate
        target_size = attended_features[0].shape[2:]
        resized_features = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) 
                          for f in attended_features]
        combined_features = torch.cat(resized_features, dim=1)
        
        # Generate heatmap
        heatmap = self.heatmap_head(combined_features)
        heatmap = F.interpolate(heatmap, size=(self.heatmap_size, self.heatmap_size), 
                               mode='bilinear', align_corners=False)
        
        # Also generate direct regression for auxiliary loss
        coords = self.regression_head(combined_features)
        
        return heatmap.squeeze(1), coords

def soft_argmax_2d(heatmap, temperature=1.0):
    """Extract coordinates from heatmap using differentiable soft-argmax"""
    batch_size, height, width = heatmap.shape
    
    # Apply temperature scaling
    heatmap = heatmap / temperature
    
    # Normalize to probability distribution
    heatmap = heatmap.view(batch_size, -1)
    heatmap = F.softmax(heatmap, dim=1)
    heatmap = heatmap.view(batch_size, height, width)
    
    # Create coordinate grids
    x_coords = torch.linspace(0, 1, width, device=heatmap.device)
    y_coords = torch.linspace(0, 1, height, device=heatmap.device)
    
    # Compute expected coordinates
    x_exp = (heatmap.sum(dim=1) * x_coords).sum(dim=1)
    y_exp = (heatmap.sum(dim=2) * y_coords).sum(dim=1)
    
    return torch.stack([x_exp, y_exp], dim=1)

class CombinedLoss(nn.Module):
    """Combined loss for heatmap and direct regression"""
    def __init__(self, heatmap_weight=1.0, regression_weight=0.5, coordinate_weight=0.3):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.regression_weight = regression_weight
        self.coordinate_weight = coordinate_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, heatmap_pred, coords_pred, heatmap_gt, coords_gt):
        # Heatmap loss (BCE)
        heatmap_loss = self.bce(heatmap_pred, heatmap_gt)
        
        # Direct regression loss
        regression_loss = self.mse(coords_pred, coords_gt)
        
        # Extract coordinates from predicted heatmap
        heatmap_coords = soft_argmax_2d(torch.sigmoid(heatmap_pred))
        coordinate_loss = self.mse(heatmap_coords, coords_gt)
        
        # Add regularization for out-of-bounds predictions
        oob_penalty = (torch.relu(coords_pred - 1) + torch.relu(-coords_pred)).mean()
        
        total_loss = (self.heatmap_weight * heatmap_loss + 
                     self.regression_weight * regression_loss +
                     self.coordinate_weight * coordinate_loss +
                     0.1 * oob_penalty)
        
        return total_loss, {
            'heatmap': heatmap_loss.item(),
            'regression': regression_loss.item(),
            'coordinate': coordinate_loss.item(),
            'out_of_bounds': oob_penalty.item()
        }

def calculate_pixel_error(predictions, targets, original_size):
    """Calculate pixel error in original image coordinates"""
    pred_x = torch.clamp(predictions[:, 0], 0, 1) * original_size[0]
    pred_y = torch.clamp(predictions[:, 1], 0, 1) * original_size[1]
    
    target_x = torch.clamp(targets[:, 0], 0, 1) * original_size[0]
    target_y = torch.clamp(targets[:, 1], 0, 1) * original_size[1]

    return torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)

def calculate_percentile_errors(errors):
    """Calculate detailed error percentiles"""
    return {
        'p50': np.percentile(errors, 50),
        'p75': np.percentile(errors, 75),
        'p90': np.percentile(errors, 90),
        'p95': np.percentile(errors, 95),
        'p99': np.percentile(errors, 99),
        'mean': np.mean(errors),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors)
    }

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    checkpoint_dir="./checkpoints",
    start_epoch=0,
    optimizer_state=None,
    scheduler_state=None,
    warmup_epochs=2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Warmup scheduler
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    
    # Main scheduler - ReduceLROnPlateau
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    # Load optimizer and scheduler states if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        main_scheduler.load_state_dict(scheduler_state)
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = CombinedLoss()
    best_val_error = float("inf")
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [], 
        "train_error": [], "val_error": [],
        "learning_rate": [], "loss_components": []
    }

    # Mixed precision scaler
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    epochs_no_improve = 0
    early_stop_patience = 15

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses, train_pixel_errors = [], []
        loss_components_epoch = {'heatmap': [], 'regression': [], 'coordinate': [], 'out_of_bounds': []}
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch_idx, (images, heatmaps_gt, coords_gt, filenames) in enumerate(train_bar):
            images = images.to(device)
            heatmaps_gt = heatmaps_gt.to(device)
            coords_gt = coords_gt.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    heatmap_pred, coords_pred = model(images)
                    loss, components = criterion(heatmap_pred, coords_pred, heatmaps_gt, coords_gt)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                heatmap_pred, coords_pred = model(images)
                loss, components = criterion(heatmap_pred, coords_pred, heatmaps_gt, coords_gt)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Extract coordinates from heatmap for error calculation
            with torch.no_grad():
                pred_coords = soft_argmax_2d(torch.sigmoid(heatmap_pred))
            
            pixel_error = calculate_pixel_error(pred_coords, coords_gt, ORIGINAL_IMAGE_SIZE)
            train_pixel_errors.extend(pixel_error.detach().cpu().numpy())
            train_losses.append(loss.item())
            
            # Store loss components
            for key, value in components.items():
                loss_components_epoch[key].append(value)
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Error': f'{np.mean(train_pixel_errors):.2f}px'
            })
        
# Validation
        model.eval()
        val_losses, val_pixel_errors = [], []
        all_val_predictions = []
        
        with torch.no_grad():
            for images, heatmaps_gt, coords_gt, filenames in val_loader:
                images = images.to(device)
                heatmaps_gt = heatmaps_gt.to(device)
                coords_gt = coords_gt.to(device)
                
                if scaler and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        heatmap_pred, coords_pred = model(images)
                        loss, _ = criterion(heatmap_pred, coords_pred, heatmaps_gt, coords_gt)
                else:
                    heatmap_pred, coords_pred = model(images)
                    loss, _ = criterion(heatmap_pred, coords_pred, heatmaps_gt, coords_gt)
                
                val_losses.append(loss.item())
                
                # Extract coordinates from heatmap
                pred_coords = soft_argmax_2d(torch.sigmoid(heatmap_pred))
                pixel_error = calculate_pixel_error(pred_coords, coords_gt, ORIGINAL_IMAGE_SIZE)
                val_pixel_errors.extend(pixel_error.detach().cpu().numpy())
                
                # Store predictions for visualization
                for i in range(len(filenames)):
                    all_val_predictions.append({
                        'filename': filenames[i],
                        'pred': pred_coords[i].cpu().numpy(),
                        'gt': coords_gt[i].cpu().numpy(),
                        'error': pixel_error[i].item()
                    })
        
        avg_train_error = np.mean(train_pixel_errors)
        avg_val_error = np.mean(val_pixel_errors)
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Calculate detailed error metrics
        val_error_stats = calculate_percentile_errors(val_pixel_errors)
        
        # Store metrics in history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_error'].append(avg_train_error)
        history['val_error'].append(avg_val_error)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['loss_components'].append({k: np.mean(v) for k, v in loss_components_epoch.items()})
        
        # Track validation loss improvements
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train Error: {avg_train_error:.2f}px | Val Error: {avg_val_error:.2f}px")
        print(f"  Val Error Stats - P50: {val_error_stats['p50']:.1f}px | P90: {val_error_stats['p90']:.1f}px | P95: {val_error_stats['p95']:.1f}px")
        print(f"  Loss Components - Heatmap: {np.mean(loss_components_epoch['heatmap']):.4f} | Regression: {np.mean(loss_components_epoch['regression']):.4f}")
        
        # Create visualizations
        try:
            create_training_plots(history, checkpoint_dir, epoch)
            
            if val_loader is not None and len(all_val_predictions) > 0:
                # Create sample predictions
                sample_predictions = sorted(all_val_predictions, key=lambda x: x['error'])[:6]
                create_prediction_visualizations(
                    sample_predictions, val_loader.dataset, 
                    checkpoint_dir, epoch, device
                )
                
                # Create error distribution plot
                create_error_distribution(val_pixel_errors, checkpoint_dir, epoch)
                
        except Exception as e:
            print(f"Could not create plots: {e}")
        
        # Save checkpoints
        if avg_val_error < best_val_error:
            old_best = best_val_error
            best_val_error = avg_val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': main_scheduler.state_dict(),
                'best_val_error': best_val_error,
                'val_error_stats': val_error_stats,
                'history': history
            }, os.path.join(checkpoint_dir, 'checkpoint_best.pth'))
            print(f"🎯 NEW BEST! Val error improved from {old_best:.2f} to {best_val_error:.2f}px")
        else:
            print(f"📊 Val error: {avg_val_error:.2f}px (best: {best_val_error:.2f}px)")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': main_scheduler.state_dict(),
            'best_val_error': best_val_error,
            'history': history
        }, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
        
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(avg_val_loss)
        
        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"⛔ Early stopping: no improvement for {early_stop_patience} epochs")
            break
    
    return model, history

def create_training_plots(history, checkpoint_dir, epoch):
    """Create and save training progress plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plots
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title(f'Training Progress - Epoch {epoch+1}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pixel error plots
    ax2.plot(epochs, history['train_error'], 'b-', label='Train Error', linewidth=2)
    ax2.plot(epochs, history['val_error'], 'r-', label='Val Error', linewidth=2)
    ax2.set_title('Pixel Error Progress')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pixel Error (px)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Loss components
    if history['loss_components']:
        components = history['loss_components'][-1]
        labels = list(components.keys())
        values = list(components.values())
        ax4.bar(labels, values, color=['blue', 'green', 'orange', 'red'])
        ax4.set_title('Loss Components (Latest Epoch)')
        ax4.set_ylabel('Loss Value')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    os.makedirs(os.path.join(checkpoint_dir, 'training_plots'), exist_ok=True)
    plt.savefig(os.path.join(checkpoint_dir, 'training_plots', f'progress_epoch_{epoch+1:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_prediction_visualizations(predictions, dataset, checkpoint_dir, epoch, device):
    """Create visualizations of predictions with heatmaps"""
    sample_dir = os.path.join(checkpoint_dir, 'sample_predictions')
    os.makedirs(sample_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, pred_data in enumerate(predictions[:6]):
        if idx >= len(axes):
            break
        
        try:
            # Load original image
            original_path = os.path.join(dataset.data_dir, pred_data['filename'])
            image, _, _ = dataset.cache.get_cached_image(original_path, dataset.image_size)
            
            # Convert predictions to pixel coordinates
            pred_px = pred_data['pred'] * np.array([ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]])
            gt_px = pred_data['gt'] * np.array([ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]])
            
            # Scale to display image size
            scale_x = dataset.image_size[0] / ORIGINAL_IMAGE_SIZE[0]
            scale_y = dataset.image_size[1] / ORIGINAL_IMAGE_SIZE[1]
            
            # Display
            axes[idx].imshow(np.array(image))
            axes[idx].scatter(gt_px[0] * scale_x, gt_px[1] * scale_y, 
                            c='green', s=100, marker='o', label='GT', alpha=0.8)
            axes[idx].scatter(pred_px[0] * scale_x, pred_px[1] * scale_y, 
                            c='red', s=100, marker='x', label='Pred', linewidth=2)
            
            axes[idx].set_title(f'Error: {pred_data["error"]:.1f}px')
            axes[idx].legend(loc='upper right')
            axes[idx].axis('off')
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[idx].axis('off')
    
    plt.suptitle(f'Sample Predictions - Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, f'predictions_epoch_{epoch+1:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_error_distribution(errors, checkpoint_dir, epoch):
    """Create error distribution histogram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}px')
    ax1.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.1f}px')
    ax1.set_xlabel('Pixel Error')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax2.plot(sorted_errors, cumulative, linewidth=2)
    ax2.set_xlabel('Pixel Error')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative Error Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Mark key percentiles
    for p in [50, 75, 90, 95]:
        val = np.percentile(errors, p)
        ax2.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax2.text(val, p, f'P{p}: {val:.0f}px', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.join(checkpoint_dir, 'training_plots'), exist_ok=True)
    plt.savefig(os.path.join(checkpoint_dir, 'training_plots', f'error_dist_epoch_{epoch+1:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Spot-the-Ball Model with Heatmap Output')
    parser.add_argument('--mode', choices=['train', 'inference', 'cache'], required=True,
                        help='Mode: train, inference, or cache creation')
    parser.add_argument('--data_dir', type=str, default='E:/botb/dataset/aug',
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save/load checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (use 1 for high resolution)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for inference mode')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image for inference mode')
    
    args = parser.parse_args()
    
    if args.mode == 'cache':
        cache = SSDImageCache()
        cache.create_full_cache(args.data_dir)
        print("Cache creation completed!")
        
    elif args.mode == 'train':
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Using checkpoint directory: {checkpoint_dir}")
        
        set_seed(42, deterministic=True)
        
        # Load or create train/val split
        split_file = os.path.join(args.checkpoint_dir, 'train_val_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            train_files, val_files = split_data['train_files'], split_data['val_files']
            print(f"✅ Loaded existing split: {len(train_files)} train, {len(val_files)} val")
        else:
            train_files, val_files = build_grouped_split_files(
                args.data_dir, seed=42, val_frac=0.15
            )
            with open(split_file, 'w') as f:
                json.dump({
                    'train_files': train_files,
                    'val_files': val_files,
                    'seed': 42,
                    'val_frac': 0.15
                }, f, indent=2)
            print(f"✅ Created and saved new split: {len(train_files)} train, {len(val_files)} val")
        
        # Create datasets
        train_dataset = HighResSpotBallDataset(args.data_dir, train_files)
        val_dataset = HighResSpotBallDataset(args.data_dir, val_files)
        
        # Create data loaders
        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        
        # Create model
        model = EnhancedSpotBallModel()
        
        # Load checkpoint if resuming
        start_epoch = 0
        optimizer_state = None
        scheduler_state = None
        
        if args.resume:
            checkpoint_path = args.resume
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_path)
            
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    optimizer_state = checkpoint.get('optimizer_state_dict', None)
                    scheduler_state = checkpoint.get('scheduler_state_dict', None)
                    print(f"✅ Resuming from epoch {start_epoch}")
                    print(f"   Best val error: {checkpoint.get('best_val_error', 'N/A'):.2f}px")
                except Exception as e:
                    print(f"❌ Error loading checkpoint: {e}")
                    start_epoch = 0
            else:
                print(f"❌ Checkpoint not found: {checkpoint_path}")
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
        )
        
        print("Training completed!")
        
    elif args.mode == 'inference':
        if not args.image:
            print("Error: --image is required for inference mode")
            return
            
        checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
            
        print("Running inference...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = EnhancedSpotBallModel()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        # Load and process image
        dataset = HighResSpotBallDataset(args.data_dir, [args.image])
        image, _, coords_gt, filename = dataset[0]
        image = image.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            heatmap_pred, coords_pred = model(image)
            # Extract coordinates from heatmap
            pred_coords = soft_argmax_2d(torch.sigmoid(heatmap_pred))
            pred_coords = pred_coords.cpu().numpy()[0]
        
        # Scale to original coordinates
        pred_original = pred_coords * np.array([ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]])
        gt_original = coords_gt.numpy() * np.array([ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1]])
        
        error = np.linalg.norm(pred_original - gt_original)
        
        print(f"\nResults for {args.image}:")
        print(f"  Predicted: ({pred_original[0]:.1f}, {pred_original[1]:.1f})")
        print(f"  Ground Truth: ({gt_original[0]:.1f}, {gt_original[1]:.1f})")
        print(f"  Error: {error:.1f}px")

if __name__ == "__main__":
    main()