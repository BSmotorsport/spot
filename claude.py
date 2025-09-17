# train_heatmap_fixed.py

import os
import random
import re
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import csv
import pandas as pd
from torch.amp import GradScaler, autocast

# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================
class Config:
    DATASET_PATH = r"E:\BOTB\dataset\aug"
    OUTPUT_DIR = r"./training_output1536"
    MODEL_VERSION = "SwinV2_heatmap_combined"
    
    # Image dimensions
    ORIGINAL_WIDTH = 4416
    ORIGINAL_HEIGHT = 3336
    IMAGE_SIZE = 1536
    HEATMAP_SIZE = 384
    
    # Model settings
    MODEL_NAME = 'resnet50'  # Much simpler model, better for small datasets
    BATCH_SIZE = 4  # Can use larger batch with smaller model
    EPOCHS = 250
    INITIAL_LR = 5e-5  # More conservative LR
    FINETUNE_LR = 5e-6
    UNFREEZE_EPOCH = 10  # Unfreeze earlier since ResNet trains faster
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    
    # Loss settings
    WING_W = 5.0
    WING_EPSILON = 0.5
    HEATMAP_SIGMA = 2.5  # Increased from 2.0 for larger Gaussian targets
    
    # Validation settings
    VALIDATE_EVERY_N_EPOCHS = 1
    SAVE_BEST_ONLY = False  # Set to True to save space

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
    """
    Compute soft-argmax for differentiable coordinate extraction.
    Returns normalized coordinates in [0, 1] range.
    """
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
        # Use MSE for more stable training initially
        self.heatmap_loss = nn.MSELoss()
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
        # From spatial_size to Config.HEATMAP_SIZE (192)
        self.scale_factor = Config.HEATMAP_SIZE // spatial_size  # Should be 8 for 24->192
        
        # Adjust channels based on input features (ResNet50 has 2048, Swin has different)
        mid_channels = min(512, in_features // 2)
        
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            # Initial projection
            nn.Conv2d(in_features, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Upsample by 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, mid_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels//2),
            nn.ReLU(inplace=True),
            
            # Upsample by 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels//2, mid_channels//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels//4),
            nn.ReLU(inplace=True),
            
            # Upsample by 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels//4, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final heatmap generation
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
            # Removed Sigmoid - let the loss function handle normalization
        )
        
        # Coordinate regression head - simpler for ResNet
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
        last_features = features[-1]
        heatmap, coords = self.head(last_features)
        return heatmap, coords

def create_model(pretrained=True):
    """Create the full model with proper initialization."""
    print(f"Creating model: {Config.MODEL_NAME}")
    
    if 'resnet' in Config.MODEL_NAME.lower():
        # Special handling for ResNet models
        import torchvision.models as models
        
        # Get the ResNet model
        if Config.MODEL_NAME == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
        elif Config.MODEL_NAME == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
        elif Config.MODEL_NAME == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {Config.MODEL_NAME}")
        
        # Remove the final FC layer and avgpool to keep spatial dimensions
        modules = list(base_model.children())[:-2]
        backbone = nn.Sequential(*modules)
        
        # Create a feature extraction wrapper that returns a list (to match timm interface)
        class ResNetBackbone(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                features = self.backbone(x)
                return [features]  # Return as list to match timm interface

            def train(self, mode: bool = True):
                super().train(mode)
                self.backbone.train(mode)
                return self

            def eval(self):
                return self.train(False)

        backbone = ResNetBackbone(backbone)
        
        # Get feature dimensions
        dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        with torch.no_grad():
            features = backbone(dummy_input)
            in_features = features[0].shape[1]
            spatial_size = features[0].shape[2]
        
        print(f"ResNet backbone output: {in_features} channels, {spatial_size}x{spatial_size} spatial")
        
    else:
        # Original timm-based model creation
        backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            img_size=Config.IMAGE_SIZE
        )
        
        # Get feature dimensions
        dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        with torch.no_grad():
            features = backbone(dummy_input)
            in_features = features[-1].shape[1]
            spatial_size = features[-1].shape[2]
        
        print(f"Backbone output: {in_features} channels, {spatial_size}x{spatial_size} spatial")
    
    # Verify spatial dimensions match expectations
    expected_spatial = Config.IMAGE_SIZE // 32  # Typical for both ResNet and Swin
    if spatial_size != expected_spatial:
        print(f"Note: Expected {expected_spatial}x{expected_spatial}, got {spatial_size}x{spatial_size}")
    
    # Create head with correct spatial size
    head = HeatmapHead(in_features, spatial_size=spatial_size)
    
    # Combine into full model
    model = FullModel(backbone, head)
    
    # Initialize weights for new layers
    for m in head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
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

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    """Train for one epoch with mixed precision."""
    model.train()
    if not any(param.requires_grad for param in model.backbone.parameters()):
        model.backbone.eval()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coord_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", colour="green")
    
    for batch_idx, (images, target_heatmaps, target_coords) in enumerate(progress_bar):
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        target_coords = target_coords.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=torch.float16):
            pred_heatmaps, pred_coords = model(images)
            loss, h_loss, c_loss = criterion(
                pred_heatmaps, target_heatmaps,
                pred_coords, target_coords
            )
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update statistics
        total_loss += loss.item()
        total_heatmap_loss += h_loss.item()
        total_coord_loss += c_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'h_loss': f"{h_loss.item():.4f}",
            'c_loss': f"{c_loss.item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_h_loss = total_heatmap_loss / len(dataloader)
    avg_c_loss = total_coord_loss / len(dataloader)
    
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

def find_lr(model, train_loader, criterion, device, num_iter=100, start_lr=1e-7, end_lr=1):
    """Learning rate finder using exponential schedule."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=start_lr)
    
    lr_multiplier = (end_lr / start_lr) ** (1 / num_iter)
    lrs, losses = [], []
    best_loss = float('inf')
    
    progress_bar = tqdm(enumerate(train_loader), total=min(num_iter, len(train_loader)),
                        desc="Finding LR", colour="cyan")
    
    for i, (images, target_heatmaps, target_coords) in progress_bar:
        if i >= num_iter:
            break
            
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        target_coords = target_coords.to(device)
        
        optimizer.zero_grad()
        
        pred_heatmaps, pred_coords = model(images)
        loss, _, _ = criterion(pred_heatmaps, target_heatmaps, pred_coords, target_coords)
        
        # Check for divergence
        if i > 10 and loss.item() > 4 * best_loss:
            print("Loss diverging, stopping LR finder")
            break
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        losses.append(loss.item())
        lrs.append(optimizer.param_groups[0]['lr'])
        
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        optimizer.param_groups[0]['lr'] *= lr_multiplier
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 
                                  'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    
    # Find suggested LR (steepest negative gradient)
    if len(losses) > 10:
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients[5:-5]) + 5
        suggested_lr = lrs[min_gradient_idx]
        plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        print(f"\nSuggested learning rate: {suggested_lr:.2e}")
    
    lr_plot_path = os.path.join(Config.OUTPUT_DIR, "lr_finder_plot.png")
    plt.savefig(lr_plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"LR finder plot saved to {lr_plot_path}")

# ======================================================================================
# 7. VISUALIZATION AND LOGGING
# ======================================================================================

def save_training_plots(history, save_path):
    """Create comprehensive training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss over Epochs', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error (pixels)', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (pixels)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE plot
    axes[1, 0].plot(history['val_rmse'], label='Validation RMSE', color='green', linewidth=2)
    axes[1, 0].set_title('Root Mean Square Error (pixels)', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE (pixels)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate', color='red', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_error_distribution(errors, epoch, save_dir):
    """Save error distribution analysis."""
    errors = np.array(errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}px')
    axes[0].axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.1f}px')
    axes[0].set_xlabel('Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution - Epoch {epoch+1}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1].plot(sorted_errors, cumulative, linewidth=2)
    axes[1].set_xlabel('Error (pixels)')
    axes[1].set_ylabel('Cumulative Percentage (%)')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Add percentile lines
    for p in [50, 75, 90, 95]:
        val = np.percentile(errors, p)
        axes[1].axvline(val, linestyle='--', alpha=0.5, label=f'{p}%: {val:.1f}px')
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'error_distribution_epoch_{epoch+1}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

# ======================================================================================
# 8. MAIN TRAINING SCRIPT
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
    
    # Verify augmentation pattern (should be ~6 images per scene)
    scene_counts = [len(files) for files in scene_groups.values()]
    print(f"Images per scene - Min: {min(scene_counts)}, Max: {max(scene_counts)}, Mean: {np.mean(scene_counts):.1f}")
    
    # Split by scenes to avoid data leakage
    scene_ids = list(scene_groups.keys())
    train_scene_ids, val_scene_ids = train_test_split(
        scene_ids, test_size=0.2, random_state=Config.RANDOM_SEED
    )
    
    train_files = [f for sid in train_scene_ids for f in scene_groups[sid]]
    val_files = [f for sid in val_scene_ids for f in scene_groups[sid]]
    
    print(f"Training scenes: {len(train_scene_ids)} ({len(train_files)} images)")
    print(f"Validation scenes: {len(val_scene_ids)} ({len(val_files)} images)")
    
    # Check for potential data issues
    if len(train_files) < 100:
        print("⚠️ Warning: Very small training set. Consider adding more data.")
    if len(scene_groups) < 50:
        print("⚠️ Warning: Limited number of unique scenes. Model may overfit.")
    
    # -------------------------------------------------------------------------
    # 2. Create data augmentation pipelines
    # -------------------------------------------------------------------------
    train_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Rotate(limit=10, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.GaussNoise(p=0.3),  # Fixed: removed var_limit parameter
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
    model.backbone.eval()
    
    # -------------------------------------------------------------------------
    # 6. Optional: Run learning rate finder
    # -------------------------------------------------------------------------
    if False:  # Set to True to run LR finder
        print("\n🔍 Running learning rate finder...")
        temp_criterion = CombinedLoss()
        find_lr(model, train_loader, temp_criterion, Config.DEVICE)
        print("Check the LR finder plot and update Config.INITIAL_LR accordingly")
        return
    
    # -------------------------------------------------------------------------
    # 7. Setup training components
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.INITIAL_LR,
        weight_decay=Config.WEIGHT_DECAY,
        eps=1e-8
    )
    
    criterion = CombinedLoss(heatmap_weight=0.7, coord_weight=0.3)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    scaler = GradScaler()
    
    # -------------------------------------------------------------------------
    # 8. Initialize tracking variables
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
    # 9. Check for existing checkpoint
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
            model.backbone.train()
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
            model.backbone.eval()
        
        print(f"Resumed from epoch {start_epoch}")
    
    # -------------------------------------------------------------------------
    # 10. Main training loop
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
            model.backbone.train()

            # Create new optimizer with all parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.FINETUNE_LR,
                weight_decay=Config.WEIGHT_DECAY,
                eps=1e-8
            )
            
            # Reset scheduler
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-7
            )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        print(f"Learning rate: {current_lr:.2e}")
        
        # Training phase
        train_loss, train_h_loss, train_c_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE, scaler, epoch
        )
        
        # Validation phase
        if (epoch + 1) % Config.VALIDATE_EVERY_N_EPOCHS == 0:
            val_loss, val_mae, val_rmse, val_errors = validate(
                model, val_loader, criterion, Config.DEVICE, epoch
            )
            
            # Update scheduler based on validation MAE
            scheduler.step(val_mae)
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (H: {train_h_loss:.4f}, C: {train_c_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_mae:.2f} pixels")
            print(f"  Val RMSE: {val_rmse:.2f} pixels")
            
            # Save error distribution plot
            if len(val_errors) > 0:
                save_error_distribution(val_errors, epoch, Config.OUTPUT_DIR)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
            
            # Save training plots
            if len(history['train_loss']) > 1:
                plot_path = os.path.join(
                    Config.OUTPUT_DIR,
                    f"training_plots_{Config.MODEL_VERSION}.png"
                )
                save_training_plots(history, plot_path)
        
        # Early stopping check
        if current_lr <= 1e-7 and epoch > Config.UNFREEZE_EPOCH + 10:
            print("\n⚠️ Learning rate too small, stopping training")
            break
    
    # -------------------------------------------------------------------------
    # 11. Final summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print(f"Best Validation MAE: {best_val_mae:.2f} pixels")
    print(f"Best Validation RMSE: {best_val_rmse:.2f} pixels")
    print(f"Model saved to: {Config.OUTPUT_DIR}")
    print("="*80)

# ======================================================================================
# 9. ENTRY POINT
# ======================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        raise