# train_missing_ball_prediction.py

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
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================
class Config:
    DATASET_PATH = r"E:\BOTB\dataset\aug"
    OUTPUT_DIR = r"./training_output_context"
    MODEL_VERSION = "ContextAware_MissingBall_v1"
    
    # Image dimensions
    ORIGINAL_WIDTH = 4416
    ORIGINAL_HEIGHT = 3336
    IMAGE_SIZE = 1536  # Higher res to capture player details
    HEATMAP_SIZE = 384
    
    # Model settings - Using a stronger backbone for context understanding
    MODEL_NAME = 'convnext_large.fb_in22k_ft_in1k'  # Larger model for better context
    BATCH_SIZE = 1  # Reduced due to larger model
    EPOCHS = 300
    INITIAL_LR = 2e-5
    FINETUNE_LR = 5e-7
    UNFREEZE_EPOCH = 15
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    
    # Loss settings
    HEATMAP_SIGMA = 2.0  # Slightly larger for uncertainty
    MIXUP_ALPHA = 0.1  # Less mixup for context preservation
    
    # Context attention settings
    USE_SELF_ATTENTION = True
    ATTENTION_HEADS = 8
    
    # Validation settings
    VALIDATE_EVERY_N_EPOCHS = 1

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Set random seeds
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
    """Parse filename to extract ball coordinates."""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        
        if len(parts) < 2:
            return None
            
        y_coord = int(parts[-1])
        x_coord = int(parts[-2])
        
        if not (0 <= x_coord <= Config.ORIGINAL_WIDTH):
            return None
        if not (0 <= y_coord <= Config.ORIGINAL_HEIGHT):
            return None
            
        return {'x': x_coord, 'y': y_coord}
        
    except (IndexError, ValueError):
        return None

def create_target_heatmap(keypoints, size, sigma=None):
    """Create Gaussian heatmap for keypoints."""
    if sigma is None:
        sigma = Config.HEATMAP_SIGMA
        
    heatmap = np.zeros((size, size), dtype=np.float32)
    
    for x, y in keypoints:
        if x < 0 or x >= size or y < 0 or y >= size:
            continue
        
        xx, yy = np.meshgrid(
            np.arange(size, dtype=np.float32),
            np.arange(size, dtype=np.float32)
        )
        
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    return heatmap

def soft_argmax_2d(heatmap, temperature=1.0):
    """Compute soft-argmax for differentiable coordinate extraction."""
    batch_size, _, height, width = heatmap.shape
    
    heatmap_flat = heatmap.view(batch_size, -1)
    heatmap_probs = F.softmax(heatmap_flat / temperature, dim=1)
    heatmap_probs = heatmap_probs.view(batch_size, 1, height, width)
    
    x_coords = torch.linspace(0, 1, width, device=heatmap.device)
    y_coords = torch.linspace(0, 1, height, device=heatmap.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)

    x_expected = torch.sum(heatmap_probs * xx, dim=(2, 3))
    y_expected = torch.sum(heatmap_probs * yy, dim=(2, 3))
    
    coords = torch.stack([x_expected.squeeze(1), y_expected.squeeze(1)], dim=1)
    coords[:, 0] *= width
    coords[:, 1] *= height
    
    return coords

# ======================================================================================
# 3. DATASET CLASS
# ======================================================================================

class MissingBallDataset(Dataset):
    """Dataset for missing ball prediction task with safer error handling."""
    
    def __init__(self, image_paths, transform=None, heatmap_size=384):
        self.image_paths = image_paths
        self.transform = transform
        self.heatmap_size = heatmap_size
        
        self.valid_paths = []
        for path in image_paths:
            if parse_filename(os.path.basename(path)) is not None:
                self.valid_paths.append(path)
        
        if len(self.valid_paths) == 0:
            raise ValueError("No valid images found in dataset. Check filename format.")
        
        print(f"Dataset: {len(self.valid_paths)}/{len(image_paths)} valid images")
        
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        # Try multiple times to get a valid sample
        max_attempts = min(10, len(self.valid_paths))
        
        for attempt in range(max_attempts):
            img_path = self.valid_paths[(idx + attempt) % len(self.valid_paths)]
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            coords = parse_filename(os.path.basename(img_path))
            if coords is None:
                continue
            
            # The coordinates represent where the ball SHOULD be (not visible)
            keypoints = [(float(coords['x']), float(coords['y']))]
            
            if self.transform:
                transformed = self.transform(image=image, keypoints=keypoints)
                image = transformed['image']
                keypoints = transformed.get('keypoints', [])
                
                if not keypoints:
                    continue
            
            heatmap_keypoints = [
                (kp[0] * (self.heatmap_size / Config.IMAGE_SIZE),
                 kp[1] * (self.heatmap_size / Config.IMAGE_SIZE))
                for kp in keypoints
            ]
            
            target_heatmap = create_target_heatmap(heatmap_keypoints, self.heatmap_size)
            precise_coords = torch.tensor(keypoints[0], dtype=torch.float32)
            
            return image, torch.from_numpy(target_heatmap).unsqueeze(0), precise_coords
        
        # If all attempts failed, raise an error
        raise RuntimeError(f"Could not load a valid sample after {max_attempts} attempts")

# ======================================================================================
# 4. CONTEXT-AWARE MODEL ARCHITECTURE
# ======================================================================================

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant regions."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ContextAggregator(nn.Module):
    """Aggregates context from multiple scales with self-attention."""
    
    def __init__(self, in_channels_list, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = len(in_channels_list)
        
        # Project each scale to same dimension
        self.projections = nn.ModuleList()
        for in_channels in in_channels_list:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim, kernel_size=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Multi-head self-attention for context understanding
        # Fix: Use correct dimension for concatenated features
        if Config.USE_SELF_ATTENTION:
            self.attention_proj = nn.Conv2d(embed_dim * self.num_scales, embed_dim, kernel_size=1)
            self.self_attention = nn.MultiheadAttention(
                embed_dim, 
                num_heads=Config.ATTENTION_HEADS,
                dropout=0.1,
                batch_first=True
            )
        
        # Spatial attention for each scale
        self.spatial_attentions = nn.ModuleList(
            [SpatialAttention() for _ in in_channels_list]
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_scales, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """Process multi-scale features with attention mechanisms."""
        
        attended_features = []
        
        # Project and apply spatial attention
        for feat, proj, spatial_att in zip(features, self.projections, self.spatial_attentions):
            # Project to common dimension
            proj_feat = proj(feat)
            
            # Apply spatial attention
            spatial_weights = spatial_att(proj_feat)
            attended_feat = proj_feat * spatial_weights
            
            attended_features.append(attended_feat)
        
        # Resize all features to same spatial size (use the middle scale)
        target_size = attended_features[len(attended_features)//2].shape[-2:]
        resized_features = []
        
        for feat in attended_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)
        
        # Apply self-attention across spatial positions if enabled
        if Config.USE_SELF_ATTENTION:
            # Concatenate all scales
            concat_features = torch.cat(resized_features, dim=1)  # [B, C*num_scales, H, W]
            
            # Project to embed_dim for attention
            concat_features = self.attention_proj(concat_features)  # [B, embed_dim, H, W]
            B, C, H, W = concat_features.shape
            
            # Reshape for attention: [B, H*W, C]
            feat_flat = concat_features.flatten(2).transpose(1, 2)
            
            # Apply self-attention
            attn_out, _ = self.self_attention(feat_flat, feat_flat, feat_flat)
            
            # Reshape back: [B, C, H, W]
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
            
            # Expand back to full concatenated size for fusion
            fused = self.context_fusion(torch.cat([attn_out] + resized_features[1:], dim=1))
        else:
            # Simple concatenation and fusion
            concat_features = torch.cat(resized_features, dim=1)
            fused = self.context_fusion(concat_features)
        
        return fused, resized_features

class MissingBallPredictor(nn.Module):
    """Predicts missing ball location from contextual features."""
    
    def __init__(self, in_channels=512, hidden_dim=256):
        super().__init__()
        
        # Heatmap generation branch
        self.heatmap_branch = nn.Sequential(
            # Initial processing
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Upsampling layers
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim//4, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final heatmap
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Direct coordinate regression branch (as auxiliary)
        self.coord_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Uncertainty estimation
        self.uncertainty_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output uncertainty score [0, 1]
        )
    
    def forward(self, context_features):
        # Generate heatmap
        heatmap = self.heatmap_branch(context_features)
        
        # Resize to target size
        if heatmap.shape[-1] != Config.HEATMAP_SIZE:
            heatmap = F.interpolate(
                heatmap, 
                size=(Config.HEATMAP_SIZE, Config.HEATMAP_SIZE),
                mode='bilinear', 
                align_corners=False
            )
        
        # Predict coordinates
        coords_raw = self.coord_branch(context_features)
        coords = torch.sigmoid(coords_raw) * Config.IMAGE_SIZE
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_branch(context_features)
        
        return heatmap, coords, uncertainty

class ContextAwareBallModel(nn.Module):
    """Complete model for missing ball prediction using scene context."""
    
    def __init__(self, backbone_name='convnext_large.fb_in22k_ft_in1k'):
        super().__init__()
        
        # Backbone for feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # Multi-scale features
        )
        
        # Get feature dimensions
        dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_channels_list = [f.shape[1] for f in features]
            print(f"Backbone features: {in_channels_list}")
        
        # Context aggregation with attention
        self.context_aggregator = ContextAggregator(in_channels_list, embed_dim=512)
        
        # Ball prediction head
        self.predictor = MissingBallPredictor(in_channels=512)
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Aggregate context with attention
        context, scale_features = self.context_aggregator(features)
        
        # Predict missing ball location
        heatmap, coords, uncertainty = self.predictor(context)
        
        return heatmap, coords, uncertainty

# ======================================================================================
# 5. LOSS FUNCTIONS
# ======================================================================================

class ContextAwareLoss(nn.Module):
    """Loss function that considers prediction uncertainty with proper per-sample weighting."""
    
    def __init__(self, heatmap_weight=0.6, coord_weight=0.3, uncertainty_weight=0.1):
        super().__init__()
        # Use reduction='none' for per-sample weighting
        self.heatmap_loss = nn.MSELoss(reduction='none')
        self.coord_loss = nn.SmoothL1Loss(reduction='none')
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, pred_heatmaps, target_heatmaps, pred_coords, target_coords, uncertainty):
        # Calculate per-sample losses
        h_loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)
        h_loss = h_loss.mean(dim=[1, 2, 3])  # Reduce spatial dims, keep batch
        
        c_loss = self.coord_loss(pred_coords, target_coords)
        c_loss = c_loss.mean(dim=1)  # Reduce coord dim, keep batch
        
        # Calculate actual error for uncertainty calibration
        with torch.no_grad():
            coord_error = torch.norm(pred_coords - target_coords, dim=1, keepdim=True)
            normalized_error = torch.tanh(coord_error / 100.0)  # Normalize to [0, 1]
        
        # Uncertainty should correlate with actual error
        u_loss = F.mse_loss(uncertainty, normalized_error)
        
        # Weight losses by inverse uncertainty (more certain predictions get higher weight)
        confidence = 1.0 - uncertainty.squeeze()
        weighted_h_loss = h_loss * confidence
        weighted_c_loss = c_loss * confidence
        
        # Average after weighting
        total_loss = (self.heatmap_weight * weighted_h_loss.mean() + 
                     self.coord_weight * weighted_c_loss.mean() + 
                     self.uncertainty_weight * u_loss)
        
        return total_loss, h_loss.mean(), c_loss.mean(), u_loss

# ======================================================================================
# 6. TRAINING FUNCTIONS
# ======================================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    """Train for one epoch with proper AMP handling."""
    model.train()
    total_loss = 0.0
    total_h_loss = 0.0
    total_c_loss = 0.0
    total_u_loss = 0.0
    
    # Check if AMP should be used
    use_amp = device == "cuda" and torch.cuda.is_available()
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, (images, target_heatmaps, target_coords) in enumerate(progress_bar):
        images = images.to(device)
        target_heatmaps = target_heatmaps.to(device)
        target_coords = target_coords.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_heatmaps, pred_coords, uncertainty = model(images)
                
                loss, h_loss, c_loss, u_loss = criterion(
                    pred_heatmaps, target_heatmaps, 
                    pred_coords, target_coords,
                    uncertainty
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU training without AMP
            pred_heatmaps, pred_coords, uncertainty = model(images)
            
            loss, h_loss, c_loss, u_loss = criterion(
                pred_heatmaps, target_heatmaps, 
                pred_coords, target_coords,
                uncertainty
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_h_loss += h_loss.item()
        total_c_loss += c_loss.item()
        total_u_loss += u_loss.item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'h': f"{h_loss.item():.4f}",
            'c': f"{c_loss.item():.4f}",
            'u': f"{u_loss.item():.4f}"
        })
    
    n = len(dataloader)
    return total_loss/n, total_h_loss/n, total_c_loss/n, total_u_loss/n

def validate(model, dataloader, criterion, device, epoch):
    """Validate the model with proper AMP handling."""
    model.eval()
    total_loss = 0.0
    all_errors = []
    all_uncertainties = []
    
    # Check if AMP should be used
    use_amp = device == "cuda" and torch.cuda.is_available()
    
    progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")
    
    with torch.no_grad():
        for images, target_heatmaps, target_coords in progress_bar:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)
            target_coords = target_coords.to(device)
            
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred_heatmaps, pred_coords, uncertainty = model(images)
                    loss, _, _, _ = criterion(
                        pred_heatmaps, target_heatmaps,
                        pred_coords, target_coords,
                        uncertainty
                    )
            else:
                pred_heatmaps, pred_coords, uncertainty = model(images)
                loss, _, _, _ = criterion(
                    pred_heatmaps, target_heatmaps,
                    pred_coords, target_coords,
                    uncertainty
                )
            
            total_loss += loss.item()
            
            # Get final prediction from heatmap
            heatmap_coords = soft_argmax_2d(pred_heatmaps)
            heatmap_coords = heatmap_coords * (Config.IMAGE_SIZE / Config.HEATMAP_SIZE)
            
            # Weighted combination based on uncertainty
            weight = (1.0 - uncertainty).squeeze()
            final_coords = weight.unsqueeze(1) * heatmap_coords + (1 - weight.unsqueeze(1)) * pred_coords
            
            errors = torch.sqrt(
                (final_coords[:, 0] - target_coords[:, 0])**2 +
                (final_coords[:, 1] - target_coords[:, 1])**2
            )
            
            all_errors.extend(errors.cpu().tolist())
            all_uncertainties.extend(uncertainty.cpu().squeeze().tolist())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    mae = np.mean(all_errors)
    rmse = np.sqrt(np.mean(np.square(all_errors)))
    avg_uncertainty = np.mean(all_uncertainties)
    
    # Calculate correlation between error and uncertainty
    if len(all_errors) > 1:
        correlation = np.corrcoef(all_errors, all_uncertainties)[0, 1]
        print(f"  Error-Uncertainty Correlation: {correlation:.3f}")
    
    return avg_loss, mae, rmse, all_errors, avg_uncertainty

# ======================================================================================
# 7. MAIN TRAINING SCRIPT
# ======================================================================================

def main():
    print("="*80)
    print("Missing Ball Prediction - Context-Aware Model")
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Using Self-Attention: {Config.USE_SELF_ATTENTION}")
    print("="*80)
    
    # Load dataset
    all_files = [
        os.path.join(Config.DATASET_PATH, f) 
        for f in os.listdir(Config.DATASET_PATH) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    valid_files = []
    for f in all_files:
        if parse_filename(os.path.basename(f)) is not None:
            valid_files.append(f)
    
    print(f"Valid files: {len(valid_files)}/{len(all_files)}")
    
    # Split dataset by scenes
    scene_groups = defaultdict(list)
    for f in valid_files:
        base_name = os.path.basename(f)
        first_part = base_name.split('-')[0]
        match = re.search(r'(\d+)', first_part)
        if match:
            scene_id = match.group(1)
            scene_groups[scene_id].append(f)
    
    scene_ids = list(scene_groups.keys())
    train_scene_ids, val_scene_ids = train_test_split(
        scene_ids, test_size=0.2, random_state=Config.RANDOM_SEED
    )
    
    train_files = [f for sid in train_scene_ids for f in scene_groups[sid]]
    val_files = [f for sid in val_scene_ids for f in scene_groups[sid]]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create transforms - preserve aspect ratio
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=Config.IMAGE_SIZE, 
            min_width=Config.IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        ),
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
    
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=Config.IMAGE_SIZE,
            min_width=Config.IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy'))),
        A.HorizontalFlip(p=0.3),  # Less flipping to preserve scene structure
        A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),  # Smaller rotation
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
    
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    # Create datasets
    train_dataset = MissingBallDataset(
        train_files, transform=train_transform, heatmap_size=Config.HEATMAP_SIZE
    )
    val_dataset = MissingBallDataset(
        val_files, transform=val_transform, heatmap_size=Config.HEATMAP_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print("\nBuilding context-aware model...")
    model = ContextAwareBallModel(Config.MODEL_NAME).to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Setup training
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.INITIAL_LR, weight_decay=Config.WEIGHT_DECAY
    )
    
    criterion = ContextAwareLoss(
        heatmap_weight=0.6, 
        coord_weight=0.3, 
        uncertainty_weight=0.1
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-8
    )
    scaler = GradScaler()
    
    best_val_mae = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(Config.EPOCHS):
        if epoch == Config.UNFREEZE_EPOCH:
            print(f"\nUnfreezing backbone at epoch {epoch+1}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=Config.FINETUNE_LR, 
                weight_decay=Config.WEIGHT_DECAY
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-9
            )
        
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_h, train_c, train_u = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE, scaler, epoch
        )
        
        scheduler.step()
        
        if (epoch + 1) % Config.VALIDATE_EVERY_N_EPOCHS == 0:
            val_loss, val_mae, val_rmse, val_errors, avg_uncertainty = validate(
                model, val_loader, criterion, Config.DEVICE, epoch
            )
            
            print(f"  Train Loss: {train_loss:.4f} (H:{train_h:.4f} C:{train_c:.4f} U:{train_u:.4f})")
            print(f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.2f}px, RMSE: {val_rmse:.2f}px")
            print(f"  Avg Uncertainty: {avg_uncertainty:.3f}")
            
            # Calculate percentiles
            if len(val_errors) > 0:
                percentiles = np.percentile(val_errors, [50, 75, 95])
                print(f"  Error Percentiles - 50%: {percentiles[0]:.1f}px, "
                      f"75%: {percentiles[1]:.1f}px, 95%: {percentiles[2]:.1f}px")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_mae': best_val_mae,
                    'val_rmse': val_rmse
                }, os.path.join(Config.OUTPUT_DIR, f"best_model_{Config.MODEL_VERSION}.pth"))
                print(f"  ✓ New best model saved! MAE: {best_val_mae:.2f}px")
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mae': val_mae
                }, os.path.join(Config.OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pth"))
    
    print(f"\nTraining complete! Best MAE: {best_val_mae:.2f}px")
    print(f"Model saved to: {Config.OUTPUT_DIR}")

# ======================================================================================
# 8. INFERENCE UTILITIES
# ======================================================================================

def visualize_prediction(model, image_path, device, save_path=None):
    """Visualize model prediction with uncertainty."""
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image_rgb)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmap, coords, uncertainty = model(img_tensor)
    
    # Get prediction from heatmap
    heatmap_coords = soft_argmax_2d(heatmap)
    heatmap_coords = heatmap_coords * (Config.IMAGE_SIZE / Config.HEATMAP_SIZE)
    
    # Weighted combination
    weight = (1.0 - uncertainty).squeeze()
    final_coords = weight * heatmap_coords + (1 - weight) * coords
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with prediction
    axes[0].imshow(image_rgb)
    x, y = final_coords[0].cpu().numpy()
    axes[0].scatter(x * (image.shape[1] / Config.IMAGE_SIZE), 
                    y * (image.shape[0] / Config.IMAGE_SIZE),
                    c='red', s=100, marker='x', linewidths=3)
    axes[0].set_title(f'Prediction (Uncertainty: {uncertainty[0].item():.3f})')
    axes[0].axis('off')
    
    # Heatmap
    hm = heatmap[0, 0].cpu().numpy()
    axes[1].imshow(hm, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Predicted Heatmap')
    axes[1].axis('off')
    
    # Uncertainty visualization
    axes[2].bar(['Confidence', 'Uncertainty'], 
                [1 - uncertainty[0].item(), uncertainty[0].item()],
                color=['green', 'red'])
    axes[2].set_ylim([0, 1])
    axes[2].set_title('Prediction Confidence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()
    
    return final_coords[0].cpu().numpy(), uncertainty[0].item()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

# ======================================================================================
# 9. ANALYSIS UTILITIES
# ======================================================================================

def analyze_errors_by_uncertainty(model, dataloader, device, num_bins=5):
    """Analyze how prediction errors correlate with uncertainty."""
    model.eval()
    
    all_errors = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, target_heatmaps, target_coords in tqdm(dataloader, desc="Analyzing"):
            images = images.to(device)
            target_coords = target_coords.to(device)
            
            pred_heatmaps, pred_coords, uncertainty = model(images)
            
            heatmap_coords = soft_argmax_2d(pred_heatmaps)
            heatmap_coords = heatmap_coords * (Config.IMAGE_SIZE / Config.HEATMAP_SIZE)
            
            weight = (1.0 - uncertainty).squeeze()
            final_coords = weight * heatmap_coords + (1 - weight) * pred_coords
            
            errors = torch.sqrt(
                (final_coords[:, 0] - target_coords[:, 0])**2 +
                (final_coords[:, 1] - target_coords[:, 1])**2
            )
            
            all_errors.extend(errors.cpu().tolist())
            all_uncertainties.extend(uncertainty.cpu().squeeze().tolist())
    
    # Bin by uncertainty
    all_errors = np.array(all_errors)
    all_uncertainties = np.array(all_uncertainties)
    
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    mean_errors = []
    std_errors = []
    counts = []
    
    for i in range(num_bins):
        mask = (all_uncertainties >= bins[i]) & (all_uncertainties < bins[i+1])
        bin_errors = all_errors[mask]
        
        if len(bin_errors) > 0:
            mean_errors.append(np.mean(bin_errors))
            std_errors.append(np.std(bin_errors))
            counts.append(len(bin_errors))
        else:
            mean_errors.append(0)
            std_errors.append(0)
            counts.append(0)
    
    # Plot analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error vs Uncertainty
    axes[0].errorbar(bin_centers, mean_errors, yerr=std_errors, 
                     fmt='o-', capsize=5, capthick=2)
    axes[0].set_xlabel('Uncertainty')
    axes[0].set_ylabel('Mean Error (pixels)')
    axes[0].set_title('Error vs Model Uncertainty')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of uncertainties
    axes[1].hist(all_uncertainties, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Uncertainty')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Prediction Uncertainties')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation
    correlation = np.corrcoef(all_errors, all_uncertainties)[0, 1]
    print(f"Error-Uncertainty Correlation: {correlation:.3f}")
    
    return all_errors, all_uncertainties