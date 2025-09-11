import os
import re
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
from typing import List, Tuple, Optional
from tqdm import tqdm
import warnings
import time
import subprocess
import shutil
import torch.utils.benchmark as benchmark

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Performance optimizations and reproducibility
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set random seeds for reproducibility
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
        self._save_pending = False
        
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
            # Skip excluded directories
            root_path = Path(root)
            if any(exclude in str(root_path).lower() for exclude in exclude_dirs):
                continue
                
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    # Skip cached images (they start with size prefix)
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
    
    # Fallback to cleaned stem
    return stem.upper()

def build_grouped_split_files(image_folder: str, seed: int, val_frac: float):
    all_files = []
    exclude_dirs = {'checkpoints', 'sample_predictions', 'training_plots', 'cache', '.git', '__pycache__'}
    
    for root, dirs, files in os.walk(image_folder):
        # Skip excluded directories
        root_path = Path(root)
        if any(exclude in root_path.parts for exclude in exclude_dirs):
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
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
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
        self.data_dir = image_folder  # Add data_dir attribute for visualization compatibility
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
        
        # Apply pre-computed transform (more efficient)
        image = self.transform_pipeline(image)
        
        # Normalize coordinates consistently
        x_norm = coords[0] / ORIGINAL_IMAGE_SIZE[0]
        y_norm = coords[1] / ORIGINAL_IMAGE_SIZE[1]
        
        return image, torch.tensor([x_norm, y_norm], dtype=torch.float32), filename

class ConvNeXtSpotModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=pretrained, num_classes=0)
        # ConvNeXt Large has 1536 features (avoiding memory spike during init)
        self.feature_dim = 1536
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
    """Calculate pixel error in original image coordinates with bounds checking"""
    # Ensure predictions are within [0,1] range
    pred_x = torch.clamp(predictions[:, 0], 0, 1) * original_size[0]
    pred_y = torch.clamp(predictions[:, 1], 0, 1) * original_size[1]
    
    # Ensure targets are within [0,1] range
    target_x = torch.clamp(targets[:, 0], 0, 1) * original_size[0]
    target_y = torch.clamp(targets[:, 1], 0, 1) * original_size[1]

    return torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)


def profile_bottlenecks(model, train_loader, device, num_batches: int = 20):
    """Rudimentary profiling for data loading and model forward time."""
    model.eval()

    load_times = []
    data_iter = iter(train_loader)
    for _ in range(num_batches):
        start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        load_times.append(time.perf_counter() - start)

    images, targets, _ = batch
    images = images.to(device)

    timer = benchmark.Timer(
        stmt="model(images)",
        globals={"model": model, "images": images},
    )
    result = timer.timeit(20)

    print(f"Avg data loading time: {np.mean(load_times):.4f}s")
    print(
        f"Forward pass: {result.mean:.4f}s ± {result.stddev:.4f}s (n={result.number})"
    )

    if shutil.which("nvidia-smi"):
        try:
            print("nvidia-smi utilization:")
            subprocess.run(["nvidia-smi"], check=False)
        except Exception as e:
            print(f"Failed to run nvidia-smi: {e}")
    else:
        print("nvidia-smi not found; install NVIDIA drivers to profile GPU utilization.")

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
    scheduler_type="auto",
    lr_factor=0.1,
    lr_patience=5,
    early_stop_patience=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    auto_mode = scheduler_type == "auto"
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )
        current_scheduler = "plateau"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        current_scheduler = "cosine"

    # Load optimizer and scheduler states if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.HuberLoss(delta=0.5)  # More aggressive outlier handling
    best_val_error = float("inf")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_error": [], "val_error": []}

    # Mixed precision scaler (single declaration)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    epochs_no_improve = 0
    if auto_mode and early_stop_patience is None:
        early_stop_patience = lr_patience * 2

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses, train_pixel_errors = [], []
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch_idx, (images, targets, filenames) in enumerate(train_bar):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if scaler and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
            
            pixel_error = calculate_pixel_error(predictions, targets, ORIGINAL_IMAGE_SIZE)
            train_pixel_errors.extend(pixel_error.detach().cpu().numpy())
            train_losses.append(loss.item())
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Error': f'{np.mean(train_pixel_errors):.2f}px'
            })
        
        model.eval()
        val_losses, val_pixel_errors = [], []
        
        with torch.no_grad():
            for images, targets, filenames in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                if scaler and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        predictions = model(images)
                        loss = criterion(predictions, targets)
                else:
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                val_losses.append(loss.item())
                
                pixel_error = calculate_pixel_error(predictions, targets, ORIGINAL_IMAGE_SIZE)
                val_pixel_errors.extend(pixel_error.detach().cpu().numpy())
                
                # Let PyTorch handle memory management automatically
                # torch.cuda.empty_cache()  # Removed - let PyTorch manage memory
        
        avg_train_error = np.mean(train_pixel_errors)
        avg_val_error = np.mean(val_pixel_errors)
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Store metrics in history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_error'].append(avg_train_error)
        history['val_error'].append(avg_val_error)

        # Track learning rate and scheduler type
        history.setdefault('learning_rate', []).append(optimizer.param_groups[0]['lr'])
        history.setdefault('scheduler', []).append(current_scheduler)

        # Track validation loss improvements for scheduling/early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Error: {avg_train_error:.2f}px | Val Error: {avg_val_error:.2f}px")
        
        # Create visualizations
        try:
            create_training_plots(history, checkpoint_dir, epoch)
            
            # Create sample predictions for sanity checking
            if val_loader is not None and len(val_loader.dataset) > 0:
                # Get sample files for visualization
                sample_files = []
                dataset = val_loader.dataset
                indices = np.random.choice(len(dataset), min(6, len(dataset)), replace=False)
                for idx in indices:
                    _, _, filename = dataset[idx]
                    sample_files.append(filename)
                
                create_prediction_visualizations(
                    model, val_loader.dataset, sample_files, device, 
                    checkpoint_dir, epoch, original_size=(4416, 3336)
                )
                
                # Create individual sample predictions for detailed sanity checking
                create_detailed_sample_predictions(model, val_loader, device, checkpoint_dir, epoch)
                
        except Exception as e:
            print(f"Could not create plots: {e}")
        
        if avg_val_error < best_val_error:
            old_best = best_val_error
            best_val_error = avg_val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_error': best_val_error
            }, os.path.join(checkpoint_dir, 'checkpoint_best.pth'))
            print(f"🎯 NEW BEST! Val error improved from {old_best:.2f} to {best_val_error:.2f}px")
        else:
            print(f"📊 Val error: {avg_val_error:.2f}px (best: {best_val_error:.2f}px)")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_error': best_val_error
        }, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
        
        # Scheduler handling and early stopping
        if auto_mode and current_scheduler == 'cosine' and epochs_no_improve >= lr_patience:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_factor, patience=lr_patience
            )
            current_scheduler = 'plateau'
            epochs_no_improve = 0
            print(f"🔄 Switching scheduler to ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience})")

        if current_scheduler == 'plateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        if auto_mode and current_scheduler == 'plateau' and epochs_no_improve >= early_stop_patience:
            print(f"⏹ Early stopping: no improvement for {early_stop_patience} epochs under ReduceLROnPlateau")
            break

    return model, history

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
    parser.add_argument('--scheduler', choices=['cosine', 'plateau', 'auto'], default='auto',
                        help='Learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Epochs with no improvement before reducing LR')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='Factor for ReduceLROnPlateau')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Patience for early stopping (auto mode sets 2×lr_patience if omitted)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for inference mode')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image for inference mode')
    parser.add_argument('--profile', action='store_true',
                        help='Profile data loading and model to find bottlenecks')
    
    args = parser.parse_args()
    
    if args.mode == 'cache':
        # Create SSD cache for 2048×2048 images
        cache = SSDImageCache()
        cache.create_full_cache(args.data_dir)
        print("Cache creation completed!")
        
    elif args.mode == 'train':
        # Create checkpoint directory
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Using checkpoint directory: {checkpoint_dir}")
        
        # Build train/val splits
        set_seed(42, deterministic=True)
        
        # Load or create train/val split lists
        split_file = os.path.join(args.checkpoint_dir, 'train_val_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            train_files, val_files = split_data['train_files'], split_data['val_files']
            print(f"✅ Loaded existing split: {len(train_files)} train, {len(val_files)} val")
        else:
            # Create new split
            train_files, val_files = build_grouped_split_files(
                args.data_dir, 
                seed=42, 
                val_frac=0.15,
            )
            # Save split for consistency across runs
            with open(split_file, 'w') as f:
                json.dump({
                    'train_files': train_files,
                    'val_files': val_files,
                    'seed': 42,
                    'val_frac': 0.15
                }, f, indent=2)
            print(f"✅ Created and saved new split: {len(train_files)} train, {len(val_files)} val")
        
        # Create datasets using SSD cache
        train_dataset = HighResSpotBallDataset(args.data_dir, train_files)
        val_dataset = HighResSpotBallDataset(args.data_dir, val_files)
        
        # Create data loaders with memory-efficient settings
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True, persistent_workers=True,
                                prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True,
                               prefetch_factor=2)

        # Create model
        model = ConvNeXtSpotModel(pretrained=True)

        if args.profile:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            profile_bottlenecks(model, train_loader, device)

        # Optional torch.compile for performance (commented out for compatibility)
        # if torch.cuda.is_available():
        #     model = torch.compile(model, mode='reduce-overhead')
        
        
        # Load checkpoint if resuming
        start_epoch = 0
        optimizer_state = None
        scheduler_state = None
        
        if args.resume:
            # Handle both absolute and relative paths
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
                    print(f"✅ Resuming training from epoch {start_epoch}")
                    print(f"   - Last saved epoch: {checkpoint.get('epoch', 0)}")
                    print(f"   - Best validation error: {checkpoint.get('best_val_error', 'N/A')}")
                except Exception as e:
                    print(f"❌ Error loading checkpoint: {e}")
                    print("🔄 Starting training from epoch 0 instead")
                    start_epoch = 0
                    optimizer_state = None
                    scheduler_state = None
            else:
                print(f"❌ Checkpoint file not found: {checkpoint_path}")
                print("🔄 Starting training from epoch 0")
        else:
            print("🔄 Starting training from epoch 0 (no resume specified)")

        # Train model
        model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            scheduler_type=args.scheduler,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            early_stop_patience=args.early_stop_patience,
        )
        
        print("Training completed!")

    elif args.mode == 'inference':
        # Inference mode
        if not args.image:
            print("Error: --image is required for inference mode")
            return
            
        checkpoint_path = args.checkpoint if args.checkpoint else 'checkpoint_best.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return
            
        print("Running inference...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = ConvNeXtSpotModel(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dataset for single image
        dataset = HighResSpotBallDataset(args.data_dir, [args.image])
        image, _, filename = dataset[0]
        image = image.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred_coords = model(image)
            pred_coords = pred_coords.cpu().numpy()[0]
        
        # Scale to original coordinates
        pred_original = pred_coords * np.array([4416, 3336])
        
        # Display result
        print(f"Predicted coordinates for {args.image}:")
        print(f"  Normalized: ({pred_coords[0]:.4f}, {pred_coords[1]:.4f})")
        print(f"  Original (4416x3336): ({pred_original[0]:.1f}, {pred_original[1]:.1f})")
        print("Inference completed!")

def create_training_plots(history, checkpoint_dir, epoch):
    """Create and save training progress plots"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title(f'Training Progress - Epoch {epoch}')
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
    lr_history = history.get('learning_rate', [1e-4] * len(epochs))
    ax3.plot(epochs, lr_history, 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)
    
    # Error distribution histogram
    if len(history['val_error']) > 0:
        ax4.hist(history['val_error'][-min(50, len(history['val_error'])):], bins=20, alpha=0.7, color='orange')
        ax4.set_title('Validation Error Distribution')
        ax4.set_xlabel('Pixel Error (px)')
    else:
        ax4.text(0.5, 0.5, 'No data yet', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'training_progress_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_prediction_visualizations(model, val_dataset, sample_files, device, checkpoint_dir, epoch, original_size):
    """Create visualizations of predictions on sample images"""
    import matplotlib.pyplot as plt
    
    if len(sample_files) == 0:
        return
    
    n_samples = min(6, len(sample_files))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    model.eval()
    
    for i, sample_file in enumerate(sample_files[:n_samples]):
        if i >= len(axes):
            break
            
        try:
            sample_path = os.path.join(val_dataset.image_folder, sample_file)
            image, true_coords, _ = val_dataset.cache.get_cached_image(sample_path, val_dataset.image_size)
            image_tensor = val_dataset.transform_pipeline(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(image_tensor)
            
            pred_coords = pred[0].cpu().numpy()
            pred_coords[0] *= original_size[0]
            pred_coords[1] *= original_size[1]
            
            # Display original image (resized)
            img_np = np.array(image)
            axes[i].imshow(img_np)
            axes[i].scatter([true_coords[0] * val_dataset.image_size[0] / original_size[0]], 
                           [true_coords[1] * val_dataset.image_size[1] / original_size[1]], 
                           c='red', s=100, marker='o', label='True')
            axes[i].scatter([pred_coords[0] * val_dataset.image_size[0] / original_size[0]], 
                           [pred_coords[1] * val_dataset.image_size[1] / original_size[1]], 
                           c='green', s=100, marker='x', label='Pred')
            
            error = np.sqrt((pred_coords[0]-true_coords[0])**2 + (pred_coords[1]-true_coords[1])**2)
            axes[i].set_title(f'{os.path.basename(sample_file)}\nError: {error:.1f}px')
            axes[i].legend()
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'Validation Predictions - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, f'predictions_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_detailed_sample_predictions(model, val_loader, device, checkpoint_dir, epoch):
    """Create clean sample predictions showing GT vs Pred on original image."""
    try:
        import matplotlib.pyplot as plt
        
        model.eval()
        sample_dir = os.path.join(checkpoint_dir, 'sample_predictions')
        os.makedirs(sample_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get 3 random validation samples for clean visualization
            num_samples = min(3, len(val_loader.dataset))
            indices = np.random.choice(len(val_loader.dataset), num_samples, replace=False)
            
            for sample_idx, dataset_idx in enumerate(indices):
                # Get image and ground truth
                image, gt_coords_normalized, original_filename = val_loader.dataset[dataset_idx]
                image = image.unsqueeze(0).to(device)
                
                # Get prediction
                pred_coords_normalized = model(image)
                pred_coords_normalized = pred_coords_normalized.cpu().numpy()[0]
                
                # Scale to original coordinates
                gt_original = np.array(gt_coords_normalized) * np.array([4416, 3336])
                pred_original = np.array(pred_coords_normalized) * np.array([4416, 3336])
                
                # Calculate error
                pixel_error = np.linalg.norm(gt_original - pred_original)
                
                # Create clean single visualization
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Load original image
                original_path = os.path.join(val_loader.dataset.data_dir, original_filename)
                original_img = Image.open(original_path)
                original_img = original_img.resize((4416, 3336))
                
                # Display original image with clear GT and Pred
                ax.imshow(original_img)
                ax.scatter(gt_original[0], gt_original[1], c='green', s=150, marker='o', 
                          label='Ground Truth')
                ax.scatter(pred_original[0], pred_original[1], c='red', s=150, marker='x', 
                          label='Prediction')
                
                # Add clean text with coordinates and error
                coord_text = f"GT: ({gt_original[0]:.1f}, {gt_original[1]:.1f})\n"
                coord_text += f"Pred: ({pred_original[0]:.1f}, {pred_original[1]:.1f})\n"
                coord_text += f"Error: {pixel_error:.1f}px"
                
                ax.text(0.02, 0.98, coord_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'Epoch {epoch+1}: {original_filename}\nGT vs Pred (Error: {pixel_error:.1f}px)', 
                           fontsize=14, fontweight='bold')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Save clean visualization
                clean_filename = f'sample_epoch_{epoch+1:03d}_{original_filename.replace(".jpg", "")}.png'
                clean_path = os.path.join(sample_dir, clean_filename)
                plt.savefig(clean_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved clean sample: {clean_filename}")
                
    except Exception as e:
        print(f"Error creating samples: {e}")

if __name__ == "__main__":
    main()
