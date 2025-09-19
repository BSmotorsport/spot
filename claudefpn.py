# train_missing_ball_clean.py
"""
Context-aware model for predicting missing ball location from player behavior.
This is a clean, verified working version without syntax errors.
"""

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
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    DATASET_PATH = r"E:\BOTB\dataset\aug"
    OUTPUT_DIR = r"./training_output_claudefpn"
    MODEL_VERSION = "ContextAware_MissingBall_v1"
    
    # Image dimensions
    ORIGINAL_WIDTH = 4416
    ORIGINAL_HEIGHT = 3336
    IMAGE_SIZE = 1536
    HEATMAP_SIZE = 384
    
    # Model settings
    MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k'
    BATCH_SIZE = 1
    EPOCHS = 100
    INITIAL_LR = 3e-5
    FINETUNE_LR = 1e-6
    UNFREEZE_EPOCH = 10
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42
    
    # Loss settings
    HEATMAP_SIGMA = 2.0
    USE_SELF_ATTENTION = True
    ATTENTION_HEADS = 8

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(Config.RANDOM_SEED)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_filename(filename):
    """Parse ball coordinates from filename."""
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        if len(parts) < 2:
            return None
        y_coord = int(parts[-1])
        x_coord = int(parts[-2])
        if 0 <= x_coord <= Config.ORIGINAL_WIDTH and 0 <= y_coord <= Config.ORIGINAL_HEIGHT:
            return {'x': x_coord, 'y': y_coord}
        return None
    except (IndexError, ValueError):
        return None

def create_target_heatmap(keypoints, size, sigma=None):
    """Create Gaussian heatmap."""
    if sigma is None:
        sigma = Config.HEATMAP_SIGMA
    heatmap = np.zeros((size, size), dtype=np.float32)
    for x, y in keypoints:
        if 0 <= x < size and 0 <= y < size:
            xx, yy = np.meshgrid(np.arange(size), np.arange(size))
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            heatmap = np.maximum(heatmap, gaussian)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

def soft_argmax_2d(heatmap):
    """Extract coordinates from heatmap."""
    B, _, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    heatmap_probs = F.softmax(heatmap_flat, dim=1).view(B, 1, H, W)
    
    x_coords = torch.linspace(0, 1, W, device=heatmap.device)
    y_coords = torch.linspace(0, 1, H, device=heatmap.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    xx = xx.unsqueeze(0).unsqueeze(0)
    yy = yy.unsqueeze(0).unsqueeze(0)
    
    x_expected = torch.sum(heatmap_probs * xx, dim=(2, 3)).squeeze(1)
    y_expected = torch.sum(heatmap_probs * yy, dim=(2, 3)).squeeze(1)
    
    coords = torch.stack([x_expected * W, y_expected * H], dim=1)
    return coords

# ============================================================================
# DATASET
# ============================================================================
class MissingBallDataset(Dataset):
    def __init__(self, image_paths, transform=None, heatmap_size=384):
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.valid_paths = [p for p in image_paths if parse_filename(os.path.basename(p))]
        if not self.valid_paths:
            raise ValueError("No valid images found")
        print(f"Dataset: {len(self.valid_paths)} valid images")
    
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        for attempt in range(10):
            img_path = self.valid_paths[(idx + attempt) % len(self.valid_paths)]
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            coords = parse_filename(os.path.basename(img_path))
            if coords is None:
                continue
            
            keypoints = [(float(coords['x']), float(coords['y']))]
            
            if self.transform:
                transformed = self.transform(image=image, keypoints=keypoints)
                image = transformed['image']
                keypoints = transformed.get('keypoints', [])
                if not keypoints:
                    continue
            
            heatmap_keypoints = [(kp[0] * self.heatmap_size / Config.IMAGE_SIZE,
                                 kp[1] * self.heatmap_size / Config.IMAGE_SIZE) for kp in keypoints]
            
            target_heatmap = create_target_heatmap(heatmap_keypoints, self.heatmap_size)
            precise_coords = torch.tensor(keypoints[0], dtype=torch.float32)
            
            return image, torch.from_numpy(target_heatmap).unsqueeze(0).float(), precise_coords
        
        raise RuntimeError(f"Could not load valid sample after 10 attempts")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class ContextAggregator(nn.Module):
    def __init__(self, in_channels_list, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = len(in_channels_list)
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention() for _ in in_channels_list
        ])
        
        if Config.USE_SELF_ATTENTION:
            self.attention_proj = nn.Conv2d(embed_dim * self.num_scales, embed_dim, 1)
            self.self_attention = nn.MultiheadAttention(
                embed_dim, Config.ATTENTION_HEADS, dropout=0.1, batch_first=True
            )
        
        self.context_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_scales, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        attended_features = []
        for feat, proj, spatial_att in zip(features, self.projections, self.spatial_attentions):
            proj_feat = proj(feat)
            spatial_weights = spatial_att(proj_feat)
            attended_features.append(proj_feat * spatial_weights)
        
        target_size = attended_features[len(attended_features)//2].shape[-2:]
        resized = [F.interpolate(f, target_size, mode='bilinear', align_corners=False) 
                   if f.shape[-2:] != target_size else f for f in attended_features]
        
        concat = torch.cat(resized, dim=1)
        
        if Config.USE_SELF_ATTENTION:
            concat = self.attention_proj(concat)
            B, C, H, W = concat.shape
            feat_flat = concat.flatten(2).transpose(1, 2)
            attn_out, _ = self.self_attention(feat_flat, feat_flat, feat_flat)
            concat = attn_out.transpose(1, 2).reshape(B, C, H, W)
            concat = torch.cat([concat] + resized[1:], dim=1)
        
        return self.context_fusion(concat), resized

class MissingBallPredictor(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        self.coord_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        self.uncertainty_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context):
        heatmap = self.heatmap_branch(context)
        if heatmap.shape[-1] != Config.HEATMAP_SIZE:
            heatmap = F.interpolate(heatmap, (Config.HEATMAP_SIZE, Config.HEATMAP_SIZE), 
                                  mode='bilinear', align_corners=False)
        coords = torch.sigmoid(self.coord_branch(context)) * Config.IMAGE_SIZE
        uncertainty = self.uncertainty_branch(context)
        return heatmap, coords, uncertainty

class ContextAwareBallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=True, features_only=True, out_indices=[0, 1, 2, 3]
        )
        dummy = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        with torch.no_grad():
            features = self.backbone(dummy)
            in_channels_list = [f.shape[1] for f in features]
        
        self.context_aggregator = ContextAggregator(in_channels_list, 512)
        self.predictor = MissingBallPredictor(512)
    
    def forward(self, x):
        features = self.backbone(x)
        context, _ = self.context_aggregator(features)
        return self.predictor(context)

# ============================================================================
# LOSS FUNCTION
# ============================================================================
class ContextAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.heatmap_loss = nn.MSELoss(reduction='none')
        self.coord_loss = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred_hm, target_hm, pred_coords, target_coords, uncertainty):
        h_loss = self.heatmap_loss(pred_hm, target_hm).mean([1, 2, 3])
        c_loss = self.coord_loss(pred_coords, target_coords).mean(1)
        
        with torch.no_grad():
            error = torch.norm(pred_coords - target_coords, dim=1, keepdim=True)
            norm_error = torch.tanh(error / 100.0)
        
        u_loss = F.mse_loss(uncertainty, norm_error)
        confidence = (1.0 - uncertainty.squeeze(-1)).clamp(min=0.1)
        
        total = 0.6 * (h_loss * confidence).mean() + 0.3 * (c_loss * confidence).mean() + 0.1 * u_loss
        return total, h_loss.mean(), c_loss.mean(), u_loss

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    use_amp = device == "cuda"
    total_loss = 0.0
    
    for images, target_hm, target_coords in tqdm(loader, desc=f"Epoch {epoch+1}"):
        images, target_hm, target_coords = images.to(device), target_hm.to(device), target_coords.to(device)
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_hm, pred_coords, uncertainty = model(images)
                loss, _, _, _ = criterion(pred_hm, target_hm, pred_coords, target_coords, uncertainty)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_hm, pred_coords, uncertainty = model(images)
            loss, _, _, _ = criterion(pred_hm, target_hm, pred_coords, target_coords, uncertainty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, epoch):
    model.eval()
    use_amp = device == "cuda"
    errors = []
    
    with torch.no_grad():
        for images, target_hm, target_coords in tqdm(loader, desc=f"Val {epoch+1}"):
            images, target_hm, target_coords = images.to(device), target_hm.to(device), target_coords.to(device)
            
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred_hm, pred_coords, uncertainty = model(images)
            else:
                pred_hm, pred_coords, uncertainty = model(images)
            
            hm_coords = soft_argmax_2d(pred_hm) * (Config.IMAGE_SIZE / Config.HEATMAP_SIZE)
            weight = (1.0 - uncertainty.squeeze(-1)).clamp(min=0.1)
            if weight.dim() == 0:
                weight = weight.unsqueeze(0)
            final_coords = weight.unsqueeze(1) * hm_coords + (1 - weight.unsqueeze(1)) * pred_coords
            
            error = torch.sqrt(((final_coords - target_coords)**2).sum(1))
            errors.extend(error.cpu().tolist())
    
    return np.mean(errors), np.sqrt(np.mean(np.square(errors)))

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"Missing Ball Prediction Training")
    print(f"Device: {Config.DEVICE}")
    
    # Load files
    all_files = [os.path.join(Config.DATASET_PATH, f) for f in os.listdir(Config.DATASET_PATH)
                 if f.lower().endswith(('.jpg', '.png'))]
    valid_files = [f for f in all_files if parse_filename(os.path.basename(f))]
    
    # Split by scene
    scene_groups = defaultdict(list)
    for f in valid_files:
        match = re.search(r'(\d+)', os.path.basename(f).split('-')[0])
        if match:
            scene_groups[match.group(1)].append(f)
    
    scene_ids = list(scene_groups.keys())
    train_ids, val_ids = train_test_split(scene_ids, test_size=0.2, random_state=42)
    
    train_files = [f for sid in train_ids for f in scene_groups[sid]]
    val_files = [f for sid in val_ids for f in scene_groups[sid]]
    
    # Create transforms
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
        A.PadIfNeeded(Config.IMAGE_SIZE, Config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
    
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
        A.PadIfNeeded(Config.IMAGE_SIZE, Config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    # Create datasets and loaders
    train_dataset = MissingBallDataset(train_files, train_transform, Config.HEATMAP_SIZE)
    val_dataset = MissingBallDataset(val_files, val_transform, Config.HEATMAP_SIZE)
    
    train_loader = DataLoader(train_dataset, Config.BATCH_SIZE, shuffle=True, 
                             num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, Config.BATCH_SIZE, shuffle=False,
                           num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Create model
    model = ContextAwareBallModel().to(Config.DEVICE)
    
    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=Config.INITIAL_LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = ContextAwareLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler() if Config.DEVICE == "cuda" else None
    
    best_mae = float('inf')
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        if epoch == Config.UNFREEZE_EPOCH:
            print("Unfreezing backbone")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=Config.FINETUNE_LR, 
                                   weight_decay=Config.WEIGHT_DECAY)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler, epoch)
        mae, rmse = validate(model, val_loader, criterion, Config.DEVICE, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "best_model.pth"))
        
        scheduler.step()
    
    print(f"Training complete. Best MAE: {best_mae:.2f}")

if __name__ == '__main__':
    main()