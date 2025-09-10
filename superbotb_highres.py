import os
import re
import time
import itertools
import warnings
import hashlib
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Suppress xFormers warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Group regex pattern - matches DC followed by 4 digits, with optional prefixes
GROUP_RE = re.compile(r'([XLA]*DC\d{4})', flags=re.IGNORECASE)

# Cache configuration
CACHE_CONFIG = {
    'dinov2': (1400, 1400),    # DINOv2 ViT-Giant
    'sam': (1024, 1024),       # SAM ViT-Huge
    'convnext': (1024, 1024),  # ConvNeXt V2 XXL
}

class SSDImageCache:
    """Cache resized images on SSD to speed up training"""
    def __init__(self, cache_dir="E:/botb/dataset/aug/cache", original_size=(4416, 3336)):
        self.cache_dir = cache_dir
        self.original_size = original_size
        self.mapping_file = os.path.join(cache_dir, "filename_mapping.json")
        self.filename_map = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing mapping if available
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    self.filename_map = json.load(f)
                print(f"Loaded {len(self.filename_map)} cached filename mappings")
            except Exception as e:
                print(f"Error loading cache mapping: {e}")
                self.filename_map = {}
        
    def get_cache_path(self, filename, target_size, quality='high'):
        """Generate cache filename based on original name and target size"""
        size_str = f"{target_size[0]}x{target_size[1]}"
        # Use a hash to avoid filename length issues
        size_hash = hashlib.md5(f"{filename}_{size_str}_{quality}".encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{size_hash}.jpg")
        
        # Store mapping between original filename and cache filename
        map_key = f"{filename}_{size_str}"
        self.filename_map[map_key] = size_hash + ".jpg"
        
        # Save mapping periodically
        if len(self.filename_map) % 100 == 0:
            self._save_mapping()
            
        return cache_path
    
    def _save_mapping(self):
        """Save the filename mapping to disk"""
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.filename_map, f)
        except Exception as e:
            print(f"Error saving cache mapping: {e}")
    
    def get_cached_image(self, original_path, target_size, quality='high'):
        """Retrieve cached image or create if doesn't exist"""
        filename = os.path.basename(original_path)
        size_str = f"{target_size[0]}x{target_size[1]}"
        map_key = f"{filename}_{size_str}"
        
        # Check if we have this file in our mapping
        if map_key in self.filename_map:
            cache_filename = self.filename_map[map_key]
            cache_path = os.path.join(self.cache_dir, cache_filename)
            if os.path.exists(cache_path):
                return Image.open(cache_path), filename
        
        # If not in mapping or file doesn't exist, create new cache entry
        cache_path = self.get_cache_path(filename, target_size, quality)
        
        # Create and cache the resized image
        original = Image.open(original_path).convert('RGB')
        
        # Calculate resize dimensions preserving aspect ratio
        original_width, original_height = original.size
        target_width, target_height = target_size
        
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize image
        resized_image = original.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded_image.paste(resized_image, (paste_x, paste_y))
        
        # Save as JPEG with moderate quality to reduce file size
        padded_image.save(cache_path, 'JPEG', quality=85)
        
        # Save mapping after creating new entries
        self._save_mapping()
        
        return padded_image, filename

def create_full_cache(dataset_dir, cache_dir="E:/botb/dataset/aug/cache"):
    """Pre-process all images into cached resolutions"""
    print("Creating image cache for faster training...")
    # Recursively walk through all subdirectories
    all_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get path relative to dataset_dir
                rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                all_files.append(rel_path)
    cache = SSDImageCache(cache_dir)
    
    total_files = len(all_files)
    for i, filename in enumerate(all_files):
        if i % 100 == 0:
            print(f"Caching images: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
        original_path = os.path.join(dataset_dir, filename)
        
        for model_name, target_size in CACHE_CONFIG.items():
            cache.get_cached_image(original_path, target_size)
    
    print(f"Cache creation complete! {total_files} images cached at multiple resolutions.")

def group_key_from_name(fname: str) -> str:
    """Extract the group key from filename based on the exact format:
       Group by DCwwYY part (week and year identifier), ignoring any prefixes
    """
    stem = os.path.splitext(fname)[0]
    
    # Extract the DCwwYY part (e.g., DC1524 from DC1524-1234-123)
    # This works with any prefix (L, A, X, XL, XA)
    dc_match = re.search(r'([XLA]*DC\d{4})', stem, flags=re.IGNORECASE)
    
    if dc_match:
        # Extract just the DCwwYY part, removing any prefix
        full_match = dc_match.group(1).upper()
        # Strip any prefixes (L, A, X, XL, XA) to get just the DCwwYY part
        base_key = re.search(r'(DC\d{4})', full_match).group(1)
        return base_key
    else:
        # Fallback - should not happen with your described format
        return stem.upper()

def build_grouped_split_files(image_folder: str, seed: int, val_frac: float, expected_group_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
    # Recursively walk through all subdirectories to find all image files
    all_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get path relative to image_folder
                rel_path = os.path.relpath(os.path.join(root, file), image_folder)
                all_files.append(rel_path)
    all_files = sorted(all_files)
    groups: Dict[str, List[str]] = {}
    for f in all_files:
        g = group_key_from_name(f)
        groups.setdefault(g, []).append(f)
    
    family_sizes = [len(v) for v in groups.values()]
    print(f"[split] total_images={len(all_files)} | groups={len(groups)} | mean_per_group={np.mean(family_sizes):.2f} | min/max={min(family_sizes)}/{max(family_sizes)}")
    
    if expected_group_size is not None:
        bad = [k for k, v in groups.items() if len(v) != expected_group_size]
        if bad: 
            print(f"[split][warn] {len(bad)} families not size {expected_group_size}:")
            for i, k in enumerate(bad[:5]):
                print(f"  Group {k}: {len(groups[k])} images (expected {expected_group_size})")
            if len(bad) > 5:
                print(f"  ... and {len(bad) - 5} more")

    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(val_frac * len(keys))))
    val_keys = set(keys[:n_val])

    train_files = list(itertools.chain.from_iterable(groups[k] for k in keys if k not in val_keys))
    val_files   = list(itertools.chain.from_iterable(groups[k] for k in val_keys))
    print(f"[split] train_imgs={len(train_files)} | val_imgs={len(val_files)} (val_frac={val_frac})")
    return train_files, val_files

class CachedSpotBallDataset(Dataset):
    def __init__(self, image_folder: str, file_list: List[str], cache_dir="E:/botb/dataset/aug/cache", 
                 target_size=(1400, 1400), original_size=(4416, 3336)):
        self.image_folder = image_folder
        self.file_list = file_list
        self.target_size = target_size
        self.original_size = original_size
        self.cache = SSDImageCache(cache_dir, original_size)
        
    def parse_coordinates_from_filename(self, filename: str) -> Tuple[float, float]:
        """Extract coordinates from filename based on the format:
           [prefix]DCwwYY-XXX[X]-YY[Y] where:
           - prefix can be L, A, X, XL, XA (or none)
           - ww is week number
           - YY is year
           - X coordinate can be 3 or 4 digits
           - Y coordinate can be 3 or 4 digits
        """
        stem = os.path.splitext(filename)[0]
        
        # Extract coordinates using a more flexible pattern
        # Match pattern like: DC1524-1234-123 or DC1524-123-1234 or any combination of digits
        # This handles prefixes like L, A, X, XL, XA
        coord_match = re.search(r'[XLA]*DC\d{4}-(\d+)-(\d+)$', stem, re.IGNORECASE)
        
        if coord_match:
            x = float(coord_match.group(1))
            y = float(coord_match.group(2))
            
            # Normalize coordinates to [0, 1] based on original image size
            x_norm = x / self.original_size[0]
            y_norm = y / self.original_size[1]
            
            return x_norm, y_norm
        else:
            raise ValueError(f"Could not parse coordinates from filename: {filename}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        image_path = os.path.join(self.image_folder, filename)
        
        # Use cached image instead of resizing on-the-fly
        # The get_cached_image function now returns both the image and the original filename
        image, original_filename = self.cache.get_cached_image(image_path, self.target_size)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize for pre-trained models
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Get coordinates from the ORIGINAL filename, not the cache filename
        x_norm, y_norm = self.parse_coordinates_from_filename(filename)
        coordinates = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        
        return image_tensor, coordinates, filename

class TriModalFusion(nn.Module):
    def __init__(self, dinov2_dim=1536, sam_dim=256, convnext_dim=2816):
        super().__init__()
        
        # Project each modality to common dimension
        self.dinov2_proj = nn.Sequential(
            nn.Linear(dinov2_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.sam_proj = nn.Sequential(
            nn.Linear(sam_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.convnext_proj = nn.Sequential(
            nn.Linear(convnext_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True, dropout=0.1)
        
        # Final fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, dinov2_feat, sam_feat, convnext_feat):
        # Project to common space
        dinov2_proj = self.dinov2_proj(dinov2_feat)
        sam_proj = self.sam_proj(sam_feat)
        convnext_proj = self.convnext_proj(convnext_feat)
        
        # Stack for cross-attention
        features = torch.stack([dinov2_proj, sam_proj, convnext_proj], dim=1)
        
        # Cross-modal attention
        attended_features, _ = self.cross_attention(features, features, features)
        
        # Flatten and fuse
        flattened = attended_features.flatten(start_dim=1)
        fused = self.fusion_layers(flattened)
        
        return fused

class DualModalFusion(nn.Module):
    def __init__(self, dinov2_dim=1536, sam_dim=256):
        super().__init__()
        
        # Project each modality to common dimension
        self.dinov2_proj = nn.Sequential(
            nn.Linear(dinov2_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.sam_proj = nn.Sequential(
            nn.Linear(sam_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True, dropout=0.1)
        
        # Final fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, dinov2_feat, sam_feat):
        # Project to common space
        dinov2_proj = self.dinov2_proj(dinov2_feat)
        sam_proj = self.sam_proj(sam_feat)
        
        # Stack for cross-attention
        features = torch.stack([dinov2_proj, sam_proj], dim=1)
        
        # Cross-modal attention
        attended_features, _ = self.cross_attention(features, features, features)
        
        # Flatten and fuse
        flattened = attended_features.flatten(start_dim=1)
        fused = self.fusion_layers(flattened)
        
        return fused

class BigGunsSpotBallModel(nn.Module):
    def __init__(self, original_image_size=(4416, 3336)):
        super().__init__()
        
        # Store original image dimensions for coordinate scaling
        self.original_image_size = original_image_size
        
        # Load DINOv2 ViT-Giant - frozen
        print("Loading DINOv2 ViT-Giant...")
        self.dinov2_giant = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.dinov2_giant, 'set_grad_checkpointing'):
            self.dinov2_giant.set_grad_checkpointing(True)
        for param in self.dinov2_giant.parameters():
            param.requires_grad = False
            
        # Load SAM ViT-Huge encoder - frozen
        print("Loading SAM ViT-Huge encoder...")
        from segment_anything import sam_model_registry
        self.sam_encoder = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").image_encoder
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.sam_encoder, 'set_grad_checkpointing'):
            self.sam_encoder.set_grad_checkpointing(True)
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
            
        # Load ConvNeXt V2 - frozen
        print("Loading ConvNeXt V2...")
        try:
            # Try to load from timm
            self.convnext = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_512', pretrained=True)
            
            # Test the model to get the actual feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 1024, 1024)
                features = self.convnext.forward_features(dummy_input)
                
                # Handle different output formats
                if isinstance(features, tuple):
                    features = features[0]
                
                # Get feature dimension
                if len(features.shape) == 4:  # [B, C, H, W]
                    self.convnext_dim = features.shape[1]
                elif len(features.shape) == 3:  # [B, L, C]
                    self.convnext_dim = features.shape[2]
                elif len(features.shape) == 2:  # [B, C]
                    self.convnext_dim = features.shape[1]
                else:
                    raise ValueError(f"Unexpected ConvNeXt feature shape: {features.shape}")
                
                print(f"Detected ConvNeXt feature dimension: {self.convnext_dim}")
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.convnext, 'set_grad_checkpointing'):
                self.convnext.set_grad_checkpointing(True)
            for param in self.convnext.parameters():
                param.requires_grad = False
                
            print("ConvNeXt V2 loaded successfully!")
            self.use_convnext = True
        except Exception as e:
            print(f"Failed to load ConvNeXt V2: {e}")
            print("Continuing without ConvNeXt...")
            self.use_convnext = False
            
        # Feature dimensions
        self.dinov2_dim = 1536  # ViT-Giant
        self.sam_dim = 256     # SAM feature dimension after pooling
        
        # Choose the appropriate fusion module based on available models
        if self.use_convnext:
            # Fusion for all three models
            self.feature_fusion = TriModalFusion(
                dinov2_dim=self.dinov2_dim,
                sam_dim=self.sam_dim, 
                convnext_dim=self.convnext_dim
            )
        else:
            # Fusion for just DINOv2 and SAM
            self.feature_fusion = DualModalFusion(
                dinov2_dim=self.dinov2_dim,
                sam_dim=self.sam_dim
            )
        
        # Multi-head ensemble prediction
        self.prediction_heads = nn.ModuleList([
            self._create_prediction_head(512) for _ in range(5)
        ])
        
        # Uncertainty weighting
        self.uncertainty_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5),
            nn.Softmax(dim=1)
        )
        
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def _create_prediction_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # x, y coordinates
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract features from all big models (no gradients)
        with torch.no_grad():
            # DINOv2 features - best for scene understanding
            # Input is already at target size from the cached dataset
            dinov2_features = self.dinov2_giant(x)
            
            # SAM features - best for spatial relationships
            # Resize to 1024x1024 for SAM
            x_sam = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
            sam_features = self.sam_encoder(x_sam)
            # Proper pooling for SAM features
            sam_features = F.adaptive_avg_pool2d(sam_features, (1, 1)).squeeze(-1).squeeze(-1)
            
            # ConvNeXt features if available
            if self.use_convnext:
                # Resize to 1024x1024 for ConvNeXt
                x_convnext = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
                # Process through ConvNeXt
                convnext_features = self.convnext.forward_features(x_convnext)
                
                # Handle different output formats from ConvNeXt
                if isinstance(convnext_features, tuple):
                    convnext_features = convnext_features[0]  # Take the main feature tensor
                
                # Global average pooling if needed
                if len(convnext_features.shape) > 2:
                    # If features are [B, C, H, W] format
                    if len(convnext_features.shape) == 4:
                        convnext_features = F.adaptive_avg_pool2d(convnext_features, (1, 1)).flatten(1)
                    # If features are [B, L, C] format (transformer-like)
                    elif len(convnext_features.shape) == 3:
                        convnext_features = convnext_features.mean(dim=1)
        
        # Fuse features based on available models
        if self.use_convnext:
            fused_features = self.feature_fusion(dinov2_features, sam_features, convnext_features)
        else:
            fused_features = self.feature_fusion(dinov2_features, sam_features)
        
        # Multi-head predictions
        predictions = [head(fused_features) for head in self.prediction_heads]
        
        # Uncertainty-weighted ensemble
        uncertainties = self.uncertainty_head(fused_features)
        
        # Final prediction
        ensemble_pred = sum(w.unsqueeze(-1) * pred for w, pred in zip(uncertainties.T, predictions))
        
        return ensemble_pred, predictions, uncertainties

class SpotBallLoss(nn.Module):
    def __init__(self, original_image_size=(4416, 3336)):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss(delta=0.1)
        self.original_width, self.original_height = original_image_size
        
    def forward(self, ensemble_pred, individual_preds, uncertainties, targets):
        # Convert normalized coordinates to pixel coordinates for loss calculation
        # Use original image dimensions for loss calculation
        pixel_pred = ensemble_pred * torch.tensor([self.original_width, self.original_height], device=device)
        pixel_target = targets * torch.tensor([self.original_width, self.original_height], device=device)
        
        # Primary coordinate losses
        mse_loss = self.mse_loss(pixel_pred, pixel_target)
        l1_loss = self.l1_loss(pixel_pred, pixel_target)
        huber_loss = self.huber_loss(pixel_pred, pixel_target)
        
        # Ensemble consistency loss
        consistency_loss = 0
        for pred in individual_preds:
            pred_pixel = pred * torch.tensor([self.original_width, self.original_height], device=device)
            consistency_loss += self.mse_loss(pred_pixel, pixel_pred.detach())
        consistency_loss /= len(individual_preds)
        
        # Uncertainty regularization
        uncertainty_reg = torch.mean(torch.var(uncertainties, dim=1))
        
        # Combined loss
        total_loss = (
            0.4 * mse_loss + 
            0.3 * l1_loss + 
            0.2 * huber_loss + 
            0.05 * consistency_loss + 
            0.05 * uncertainty_reg
        )
        
        # Calculate pixel error for monitoring
        pixel_error = torch.norm(pixel_pred - pixel_target, dim=1).mean()
        
        return total_loss, {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'huber': huber_loss.item(),
            'pixel_error': pixel_error.item(),
            'consistency': consistency_loss.item()
        }

def cleanup_checkpoints(checkpoint_dir, keep_every=10, keep_latest=2, current_epoch=None):
    """Clean up old checkpoints to save disk space"""
    # Get all epoch checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    
    # Extract epoch numbers
    epoch_nums = []
    for f in checkpoint_files:
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', f)
        if match:
            epoch_nums.append(int(match.group(1)))
    
    # Sort epochs
    epoch_nums.sort()
    
    # Determine which epochs to keep
    keep_epochs = set()
    
    # Keep every Nth epoch
    for epoch in epoch_nums:
        if epoch % keep_every == 0:
            keep_epochs.add(epoch)
    
    # Keep latest N epochs
    if current_epoch is not None:
        for i in range(keep_latest):
            if current_epoch - i >= 0:
                keep_epochs.add(current_epoch - i)
    
    # Delete checkpoints not in keep_epochs
    for epoch in epoch_nums:
        if epoch not in keep_epochs:
            file_to_delete = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                print(f"Deleted old checkpoint: {file_to_delete}")

def plot_training_history(history, save_path):
    """Plot and save training history"""
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot losses
    axs[0].plot(history['train_losses'], label='Train Loss')
    axs[0].plot(history['val_losses'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot pixel errors
    axs[1].plot(history['train_pixel_errors'], label='Train Pixel Error')
    axs[1].plot(history['val_pixel_errors'], label='Validation Pixel Error')
    axs[1].set_title('Training and Validation Pixel Error')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Pixel Error')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot learning rate
    axs[2].plot(history['learning_rates'])
    axs[2].set_title('Learning Rate')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Learning Rate')
    axs[2].set_yscale('log')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_prediction(model, val_dataset, epoch, output_dir, original_image_size=(4416, 3336)):
    """Create a visualization of model prediction on a sample validation image"""
    original_width, original_height = original_image_size
    
    # Select a random sample from validation set
    idx = np.random.randint(0, len(val_dataset))
    image_tensor, target_coords, filename = val_dataset[idx]
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            ensemble_pred, individual_preds, uncertainties = model(image_tensor.unsqueeze(0).to(device))
    
    # Convert normalized coordinates to pixel coordinates
    # Always use original image dimensions for visualization
    pred_x = ensemble_pred[0, 0].item() * original_width
    pred_y = ensemble_pred[0, 1].item() * original_height
    target_x = target_coords[0].item() * original_width
    target_y = target_coords[1].item() * original_height
    
    # Get individual predictions
    individual_x = [pred[0, 0].item() * original_width for pred in individual_preds]
    individual_y = [pred[0, 1].item() * original_height for pred in individual_preds]
    
    # Convert tensor to image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_np = (image_tensor * std + mean).permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(image_np)
    
    # Calculate scale factor for circle sizes
    # This adjusts the circle radius based on the displayed image size vs original image size
    scale_factor = min(image_np.shape[0]/original_height, image_np.shape[1]/original_width)
    
    # Plot ground truth - scale coordinates to visualization size
    vis_target_x = target_x * image_np.shape[1] / original_width
    vis_target_y = target_y * image_np.shape[0] / original_height
    gt_circle = Circle((vis_target_x, vis_target_y), 
                      radius=30 * scale_factor, 
                      color='green', fill=False, linewidth=3, label='Ground Truth')
    plt.gca().add_patch(gt_circle)
    
    # Plot ensemble prediction - scale coordinates to visualization size
    vis_pred_x = pred_x * image_np.shape[1] / original_width
    vis_pred_y = pred_y * image_np.shape[0] / original_height
    pred_circle = Circle((vis_pred_x, vis_pred_y), 
                        radius=30 * scale_factor, 
                        color='red', fill=False, linewidth=3, label='Prediction')
    plt.gca().add_patch(pred_circle)
    
    # Plot individual predictions - scale coordinates to visualization size
    for i, (x, y) in enumerate(zip(individual_x, individual_y)):
        vis_x = x * image_np.shape[1] / original_width
        vis_y = y * image_np.shape[0] / original_height
        circle = Circle((vis_x, vis_y), 
                       radius=15 * scale_factor, 
                       color=f'C{i+2}', fill=False, linewidth=2, alpha=0.7)
        plt.gca().add_patch(circle)
    
    # Calculate error using original pixel coordinates
    error = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
    
    plt.title(f"Epoch {epoch} - Pixel Error: {error:.2f} - File: {os.path.basename(filename)}")
    plt.legend(loc='upper right')
    plt.axis('off')
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"val_sample_epoch_{epoch}.png"), bbox_inches='tight')
    plt.close()
    
    return error

def train_model(resume_from=None, original_image_size=(4416, 3336), 
             data_dir="E:/botb/dataset/aug", 
             checkpoint_dir="E:/botb/dataset/aug/superbotb/checkpoints"):
    """
    Train the spot-the-ball model with checkpoint saving and resuming capabilities
    
    Args:
        resume_from: Path to checkpoint file to resume training from (optional)
        original_image_size: Original dimensions of the images (width, height)
        data_dir: Path to dataset directory
        checkpoint_dir: Directory to save checkpoints
    """
    # Dataset setup
    data_folder = data_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if cache exists, if not create it
    if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        create_full_cache(data_folder, cache_dir)
    
    train_files, val_files = build_grouped_split_files(
        data_folder, 
        seed=42, 
        val_frac=0.15,  # 15% for validation
        expected_group_size=6  # 6 augmented versions per base image
    )
    
    # Create datasets with cached images
    train_dataset = CachedSpotBallDataset(
        data_folder, 
        train_files, 
        cache_dir=cache_dir,
        target_size=CACHE_CONFIG['dinov2'],  # Use DINOv2 size as default
        original_size=original_image_size
    )
    
    val_dataset = CachedSpotBallDataset(
        data_folder, 
        val_files, 
        cache_dir=cache_dir,
        target_size=CACHE_CONFIG['dinov2'],  # Use DINOv2 size as default
        original_size=original_image_size
    )
    
    # Create data loaders - with workers=0 as requested
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Small batch size for memory
        shuffle=True, 
        num_workers=0,  # No workers as requested
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,  # No workers as requested
        pin_memory=True
    )
    
    # Initialize model
    model = BigGunsSpotBallModel(original_image_size=original_image_size).to(device)
    
    # Loss and optimizer
    criterion = SpotBallLoss(original_image_size=original_image_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler()
    
    # Training state variables
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_pixel_errors': [],
        'val_pixel_errors': [],
        'learning_rates': []
    }
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load scaler state if it exists
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        # Resume training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        
        # Load training history if it exists
        if 'training_history' in checkpoint:
            training_history = checkpoint['training_history']
            
        print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    
    # Training loop
    max_epochs = 200
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        train_pixel_errors = []
        
        for batch_idx, (images, targets, filenames) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                ensemble_pred, individual_preds, uncertainties = model(images)
                loss, metrics = criterion(ensemble_pred, individual_preds, uncertainties, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            train_pixel_errors.append(metrics['pixel_error'])
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Pixel Error: {metrics['pixel_error']:.2f}")
        
        # Validation
        model.eval()
        val_losses = []
        val_pixel_errors = []
        
        with torch.no_grad():
            for images, targets, filenames in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                with torch.amp.autocast(device_type='cuda'):
                    ensemble_pred, individual_preds, uncertainties = model(images)
                    loss, metrics = criterion(ensemble_pred, individual_preds, uncertainties, targets)
                
                val_losses.append(loss.item())
                val_pixel_errors.append(metrics['pixel_error'])
        
        # Calculate averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_pixel_error = np.mean(train_pixel_errors)
        avg_val_pixel_error = np.mean(val_pixel_errors)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update training history
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['train_pixel_errors'].append(avg_train_pixel_error)
        training_history['val_pixel_errors'].append(avg_val_pixel_error)
        training_history['learning_rates'].append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Pixel Error: {avg_train_pixel_error:.2f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Pixel Error: {avg_val_pixel_error:.2f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Visualize a sample prediction
        vis_dir = os.path.join(checkpoint_dir, "visualizations")
        sample_error = visualize_prediction(model, val_dataset, epoch, vis_dir, original_image_size)
        print(f"  Sample visualization saved with error: {sample_error:.2f} pixels")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint only every 5 epochs (except best and latest)
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'training_history': training_history,
                'val_loss': avg_val_loss,
                'val_pixel_error': avg_val_pixel_error
            }, checkpoint_path)

        # Always save latest checkpoint (overwrite)
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'training_history': training_history,
            'val_loss': avg_val_loss,
            'val_pixel_error': avg_val_pixel_error
        }, latest_path)
        
        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'training_history': training_history,
                'val_loss': avg_val_loss,
                'val_pixel_error': avg_val_pixel_error
            }, best_path)
            
            print(f"  New best model saved! Val Pixel Error: {avg_val_pixel_error:.2f}")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{patience} epochs")
            
        # Clean up old checkpoints (keep only every 10th epoch + latest 2)
        cleanup_checkpoints(checkpoint_dir, keep_every=10, keep_latest=2, current_epoch=epoch)
            
        if patience_counter >= patience:
            print(f"Early stopping after {epoch} epochs")
            break
    
    # Save final training history plot
    plot_training_history(training_history, os.path.join(checkpoint_dir, "training_history.png"))
    
    print("Training completed!")
    return model

def inference(image_path, checkpoint_path, output_dir="E:/botb/dataset/aug/superbotb/results", original_image_size=(4416, 3336)):
    """Run inference on a single image using a trained model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = BigGunsSpotBallModel(original_image_size=original_image_size).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize with padding to maintain aspect ratio
    target_width, target_height = CACHE_CONFIG['dinov2']
    
    # Calculate resize dimensions preserving aspect ratio
    original_width, original_height = image.size
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(padded_image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Run inference
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            ensemble_pred, individual_preds, uncertainties = model(image_tensor.unsqueeze(0).to(device))
    
    # Convert normalized prediction back to pixel coordinates
    # Use original image dimensions for coordinate scaling
    pred_x = ensemble_pred[0, 0].item() * original_image_size[0]
    pred_y = ensemble_pred[0, 1].item() * original_image_size[1]
    
    # Get individual predictions for visualization
    individual_x = [pred[0, 0].item() * original_image_size[0] for pred in individual_preds]
    individual_y = [pred[0, 1].item() * original_image_size[1] for pred in individual_preds]
    
    # Visualize prediction
    plt.figure(figsize=(15, 12))
    plt.imshow(np.array(image))
    
    # Calculate scale factor for circle sizes
    scale_factor = min(image.height/original_image_size[1], image.width/original_image_size[0])
    
    # Plot ensemble prediction
    main_circle = Circle((pred_x, pred_y), radius=30, color='red', fill=False, linewidth=3, label='Ensemble Prediction')
    plt.gca().add_patch(main_circle)
    
    # Plot individual predictions
    for i, (x, y) in enumerate(zip(individual_x, individual_y)):
        circle = Circle((x, y), radius=15, color=f'C{i+1}', fill=False, linewidth=2, label=f'Head {i+1}')
        plt.gca().add_patch(circle)
    
    plt.title(f"Spot the Ball Prediction")
    plt.legend(loc='upper right')
    plt.axis('off')
    
    # Save result
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.', '_prediction.'))
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Prediction: X={pred_x:.2f}, Y={pred_y:.2f}")
    print(f"Result saved to {output_path}")
    
    return pred_x, pred_y

def verify_dataset_filenames(data_folder):
    """Verify that filenames in the dataset match the expected format and test grouping"""
    print("Verifying dataset filenames...")
    # Recursively walk through all subdirectories
    all_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get path relative to data_folder
                rel_path = os.path.relpath(os.path.join(root, file), data_folder)
                all_files.append(rel_path)
    
    # Expected format: [prefix]DCwwYY-XXX[X]-YY[Y]
    valid_pattern = re.compile(r'[XLA]*DC\d{4}-\d+-\d+\.(jpg|jpeg|png)$', re.IGNORECASE)
    
    valid_files = 0
    invalid_files = []
    
    for filename in all_files:
        if valid_pattern.match(filename):
            valid_files += 1
        else:
            invalid_files.append(filename)
    
    print(f"Found {valid_files} valid filenames out of {len(all_files)} total files")
    
    if invalid_files:
        print(f"Found {len(invalid_files)} invalid filenames:")
        for i, f in enumerate(invalid_files[:10]):
            print(f"  {i+1}. {f}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    
    # Check for coordinate extraction
    test_dataset = CachedSpotBallDataset(data_folder, all_files[:5])
    print("\nTesting coordinate extraction from filenames:")
    for i, filename in enumerate(all_files[:5]):
        try:
            x_norm, y_norm = test_dataset.parse_coordinates_from_filename(filename)
            x_pixel = x_norm * 4416
            y_pixel = y_norm * 3336
            print(f"  {i+1}. {filename} → ({x_pixel:.1f}, {y_pixel:.1f})")
        except Exception as e:
            print(f"  {i+1}. {filename} → ERROR: {str(e)}")
    
    # Test grouping
    print("\nTesting image grouping:")
    groups = {}
    for f in all_files:
        g = group_key_from_name(f)
        groups.setdefault(g, []).append(f)
    
    # Print a few example groups
    print(f"Found {len(groups)} unique groups")
    for i, (group_key, files) in enumerate(list(groups.items())[:3]):
        print(f"\nGroup {i+1}: {group_key} - {len(files)} files")
        for j, f in enumerate(files[:6]):
            print(f"  {j+1}. {f}")
        if len(files) > 6:
            print(f"  ... and {len(files) - 6} more")
    
    # Check for expected group size
    expected_group_size = 6
    bad_groups = [k for k, v in groups.items() if len(v) != expected_group_size]
    if bad_groups:
        print(f"\nFound {len(bad_groups)} groups that don't have exactly {expected_group_size} images:")
        for i, k in enumerate(bad_groups[:5]):
            print(f"  Group {k}: {len(groups[k])} images")
        if len(bad_groups) > 5:
            print(f"  ... and {len(bad_groups) - 5} more")
    else:
        print(f"\nAll groups have exactly {expected_group_size} images as expected!")
    
    return valid_files, invalid_files, groups

def download_convnext_weights():
    """Download ConvNeXt V2 XXL weights if not available"""
    try:
        import timm
        print("Checking if ConvNeXt V2 XXL is available...")
        model = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_512', pretrained=True)
        print("ConvNeXt V2 XXL weights already available!")
    except Exception as e:
        print(f"ConvNeXt V2 XXL not available in timm: {e}")
        print("Attempting to download weights manually...")
        
        # Create directory for weights
        os.makedirs("model_weights", exist_ok=True)
        
        # Download weights using wget
        import subprocess
        try:
            subprocess.run([
                "wget", 
                "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt",
                "-O", 
                "model_weights/convnextv2_huge_22k_512_ema.pt"
            ], check=True)
            print("ConvNeXt V2 XXL weights downloaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download weights: {e}")
            print("Please download the weights manually from: https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SuperBotB Spot the Ball Model with ConvNeXt V2')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'cache', 'verify', 'download'],
                        help='Mode: train, inference, cache, verify, or download')
    parser.add_argument('--data_dir', type=str, default='E:/botb/dataset/aug',
                        help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='E:/botb/dataset/aug/superbotb/checkpoints',
                        help='Directory to save/load checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--image', type=str, default=None, help='Path to image for inference')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for inference')
    parser.add_argument('--cache_only', action='store_true', help='Only create the cache without training')
    parser.add_argument('--create_cache', action='store_true', help='Create cache before training')
    
    args = parser.parse_args()
    
    # Original image dimensions
    original_image_size = (4416, 3336)
    
    if args.mode == 'download':
        download_convnext_weights()
    elif args.mode == 'verify':
        # Verify dataset filenames
        verify_dataset_filenames(args.data_dir)
    elif args.mode == 'cache' or args.cache_only:
        # Create cache directory
        cache_dir = os.path.join(args.data_dir, "cache")
        create_full_cache(args.data_dir, cache_dir)
        
        if args.cache_only:
            print("Cache creation completed. Exiting.")
            exit(0)
    elif args.mode == 'train':
        # Create cache if requested
        if args.create_cache:
            cache_dir = os.path.join(args.data_dir, "cache")
            create_full_cache(args.data_dir, cache_dir)
        
        # Update train_model to use the provided arguments
        train_model(
            resume_from=args.resume, 
            original_image_size=original_image_size,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir
        )
    elif args.mode == 'inference':
        if args.image is None or args.checkpoint is None:
            parser.error("Inference mode requires --image and --checkpoint")
        inference(args.image, args.checkpoint, original_image_size=original_image_size)