import os
import re
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import argparse
from pathlib import Path
import json
import hashlib
from typing import List, Tuple, Optional

# Configuration
ORIGINAL_IMAGE_SIZE = (4416, 3336)
IMAGE_SIZE = (2048, 2048)
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
                image = Image.open(cache_path).convert('RGB')
                coords = self.coordinates[filename]
                return image, coords, filename
        
        cache_path = self.get_cache_path(filename, target_size)
        original = Image.open(original_path).convert('RGB')
        resized = original.resize(target_size, Image.Resampling.LANCZOS)
        
        coords = self.extract_coordinates(filename)
        resized.save(cache_path, 'JPEG', quality=85)
        self._save_mappings()
        
        return resized, coords, filename
    
    def create_full_cache(self, dataset_dir):
        print("Creating image cache for faster training...")
        
        all_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
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
    stem = os.path.splitext(fname)[0]
    dc_match = re.search(r'([XLA]*DC\d{4})', stem, flags=re.IGNORECASE)
    
    if dc_match:
        full_match = dc_match.group(1).upper()
        base_key = re.search(r'(DC\d{4})', full_match).group(1)
        return base_key
    return stem.upper()

def build_grouped_split_files(image_folder: str, seed: int, val_frac: float, expected_group_size=None):
    all_files = []
    for root, dirs, files in os.walk(image_folder):
        # Skip cache directory
        if CACHE_DIR in root:
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Skip cached images (they start with size prefix)
                if not file.startswith(f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_"):
                    all_files.append(os.path.relpath(os.path.join(root, file), image_folder))
    all_files = sorted(all_files)
    
    groups = {}
    for f in all_files:
        g = group_key_from_name(os.path.basename(f))
        groups.setdefault(g, []).append(f)
    
    print(f"[split] total_images={len(all_files)} | groups={len(groups)} | mean_per_group={np.mean([len(v) for v in groups.values()]):.2f}")
    
    keys = list(groups.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)
    n_val = max(1, int(round(val_frac * len(keys))))
    val_keys = set(keys[:n_val])
    
    train_files = [f for k in keys if k not in val_keys for f in groups[k]]
    val_files = [f for k in val_keys for f in groups[k]]
    print(f"[split] train_imgs={len(train_files)} | val_imgs={len(val_files)}")
    return train_files, val_files

class HighResSpotBallDataset(Dataset):
    def __init__(self, image_folder: str, file_list: List[str], 
                 image_size: Tuple[int, int] = IMAGE_SIZE,
                 original_size: Tuple[int, int] = ORIGINAL_IMAGE_SIZE):
        self.image_folder = image_folder
        self.file_list = file_list
        self.image_size = image_size
        self.original_size = original_size
        self.cache = SSDImageCache()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        original_path = os.path.join(self.image_folder, filename)
        
        # Use cached image and coordinates
        image, coords, _ = self.cache.get_cached_image(original_path, self.image_size)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize coordinates
        x_norm = coords[0] / self.original_size[0]
        y_norm = coords[1] / self.original_size[1]
        
        return image, torch.tensor([x_norm, y_norm], dtype=torch.float32), filename

class ConvNeXtSpotModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
        
        for name, param in self.backbone.named_parameters():
            if 'stages.0' in name or 'stages.1' in name:
                param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def calculate_pixel_error(predictions, targets, original_size):
    pred_x = predictions[:, 0] * original_size[0]
    pred_y = predictions[:, 1] * original_size[1]
    target_x = targets[:, 0] * original_size[0]
    target_y = targets[:, 1] * original_size[1]
    return torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, checkpoint_dir='./checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    best_val_error = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_losses, train_pixel_errors = [], []
        
        for images, targets, filenames in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            pixel_error = calculate_pixel_error(predictions, targets, ORIGINAL_IMAGE_SIZE)
            train_pixel_errors.extend(pixel_error.cpu().numpy())
            train_losses.append(loss.item())
        
        model.eval()
        val_losses, val_pixel_errors = [], []
        
        with torch.no_grad():
            for images, targets, filenames in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_losses.append(loss.item())
                
                pixel_error = calculate_pixel_error(predictions, targets, ORIGINAL_IMAGE_SIZE)
                val_pixel_errors.extend(pixel_error.cpu().numpy())
        
        avg_train_error = np.mean(train_pixel_errors)
        avg_val_error = np.mean(val_pixel_errors)
        
        print(f"Epoch {epoch+1}: Train Error={avg_train_error:.2f} | Val Error={avg_val_error:.2f}")
        
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_error': best_val_error
            }, os.path.join(checkpoint_dir, 'checkpoint_best.pth'))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
        
        scheduler.step()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='High-Resolution Spot-the-Ball Model with SSD Cache')
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
    
    args = parser.parse_args()
    
    if args.mode == 'cache':
        # Create SSD cache for 2048×2048 images
        cache = SSDImageCache()
        cache.create_full_cache(args.data_dir)
        print("Cache creation completed!")
        
    elif args.mode == 'train':
        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Build train/val splits
        train_files, val_files = build_grouped_split_files(
            args.data_dir, 
            seed=42, 
            val_frac=0.15,
            expected_group_size=6  # 6 augmented versions per base image
        )
        
        # Create datasets using SSD cache
        train_dataset = HighResSpotBallDataset(args.data_dir, train_files)
        val_dataset = HighResSpotBallDataset(args.data_dir, val_files)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Create model
        model = ConvNeXtSpotModel(pretrained=True)
        
        # Load checkpoint if resuming
        if args.resume and os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Resuming training...")
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print("Training completed!")

if __name__ == "__main__":
    main()