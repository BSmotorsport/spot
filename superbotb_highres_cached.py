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
import hashlib
from typing import List, Tuple, Optional
from tqdm import tqdm

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

def build_grouped_split_files(image_folder: str, seed: int, val_frac: float, expected_group_size=None):
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
        self.data_dir = image_folder  # Add data_dir attribute for visualization compatibility
        self.cache = SSDImageCache()
        
        # Pre-compute normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        original_path = os.path.join(self.image_folder, filename)
        
        # Use cached image and coordinates
        image, coords, _ = self.cache.get_cached_image(original_path, self.image_size)
        
        # Convert to tensor and normalize with proper preprocessing
        image_np = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        
        # Use pre-computed normalization tensors
        image = (image - self.mean) / self.std
        
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

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, checkpoint_dir='./checkpoints', start_epoch=0, optimizer_state=None, scheduler_state=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.backbone, 'set_grad_checkpointing'):
        model.backbone.set_grad_checkpointing(True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Load optimizer and scheduler states if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    best_val_error = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': []}
    
    # Mixed precision scaler (single declaration)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses, train_pixel_errors = [], []
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        train_losses, train_pixel_errors = [], []
        
        for images, targets, filenames in train_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
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
        
        # Store learning rate for plots
        history.setdefault('learning_rate', []).append(scheduler.get_last_lr()[0])
        
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
            best_val_error = avg_val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_error': best_val_error
            }, os.path.join(checkpoint_dir, 'checkpoint_best.pth'))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
        
        scheduler.step()
    
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image for inference mode')
    
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
        train_files, val_files = build_grouped_split_files(
            args.data_dir, 
            seed=42, 
            val_frac=0.15,
            expected_group_size=6  # 6 augmented versions per base image
        )
        
        # Create datasets using SSD cache
        train_dataset = HighResSpotBallDataset(args.data_dir, train_files)
        val_dataset = HighResSpotBallDataset(args.data_dir, val_files)
        
        # Create data loaders with memory-efficient settings
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        
        # Create model
        model = ConvNeXtSpotModel(pretrained=True)
        
        
        # Load checkpoint if resuming
        start_epoch = 0
        optimizer_state = None
        scheduler_state = None
        
        if args.resume and os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            scheduler_state = checkpoint.get('scheduler_state_dict', None)
            print(f"Resuming training from epoch {start_epoch}")

        # Train model
        model, history = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state
        )
        
        print("Training completed!")

    elif args.mode == 'inference':
        # Inference mode
        if not args.image:
            print("Error: --image is required for inference mode")
            return
            
        checkpoint_path = args.checkpoint if args.checkpoint else checkpoint_best.pth
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return
            
        print("Running inference...")
        device = torch.device(cuda if torch.cuda.is_available() else cpu)
        
        # Load model
        model = ConvNeXtSpotModel(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint[model_state_dict])
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
            image_tensor = val_dataset.transform(image).unsqueeze(0).to(device)
            
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

def create_training_plots(history, checkpoint_dir, epoch):
    """Create and save training progress plots"""
    try:
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
            ax4.set_ylabel('Frequency')
        else:
            ax4.text(0.5, 0.5, 'No data yet', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, f'training_progress_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("Matplotlib not available - skipping plots")

def create_prediction_visualizations(model, val_dataset, sample_files, device, checkpoint_dir, epoch, original_size):
    """Create visualizations of predictions on sample images"""
    try:
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
                image_tensor = val_dataset.transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(image_tensor)
                
                pred_coords = pred[0].cpu().numpy()
                pred_coords[0] *= original_size[0]
                pred_coords[1] *= original_size[1]
                
                # Display image
                img_np = np.array(image)
                axes[i].imshow(img_np)
                
                # Scale coordinates for display
                display_scale_x = val_dataset.image_size[0] / original_size[0]
                display_scale_y = val_dataset.image_size[1] / original_size[1]
                
                axes[i].scatter([true_coords[0] * display_scale_x], 
                               [true_coords[1] * display_scale_y], 
                               c='red', s=100, marker='o', label='True')
                axes[i].scatter([pred_coords[0] * display_scale_x], 
                               [pred_coords[1] * display_scale_y], 
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
    except ImportError:
        print("Matplotlib not available - skipping visualizations")

def create_detailed_sample_predictions(model, val_loader, device, checkpoint_dir, epoch):
    """Create detailed sample predictions for sanity checking GT vs Pred."""
    try:
        import matplotlib.pyplot as plt
        
        model.eval()
        sample_dir = os.path.join(checkpoint_dir, 'sample_predictions')
        os.makedirs(sample_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get 3 random validation samples for detailed analysis
            num_samples = min(3, len(val_loader.dataset))
            indices = np.random.choice(len(val_loader.dataset), num_samples, replace=False)
            
            for sample_idx, dataset_idx in enumerate(indices):
                # Get image and ground truth
                image, gt_coords_normalized, original_filename = val_loader.dataset[dataset_idx]
                image = image.unsqueeze(0).to(device)
                
                # Get prediction
                pred_coords_normalized = model(image)
                pred_coords_normalized = pred_coords_normalized.cpu().numpy()[0]
                
                # Scale back to original coordinates
                gt_original = np.array(gt_coords_normalized) * np.array([4416, 3336])
                pred_original = np.array(pred_coords_normalized) * np.array([4416, 3336])
                
                # Calculate errors
                pixel_error = np.linalg.norm(gt_original - pred_original)
                normalized_error = np.linalg.norm(np.array(gt_coords_normalized) - np.array(pred_coords_normalized))
                
                # Create detailed visualization
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Load original image
                original_path = os.path.join(val_loader.dataset.data_dir, original_filename)
                original_img = Image.open(original_path)
                original_img = original_img.resize((4416, 3336))
                
                # Plot 1: Original image with GT and Pred
                axes[0, 0].imshow(original_img)
                axes[0, 0].scatter(gt_original[0], gt_original[1], c='green', s=200, marker='o', 
                                 edgecolors='white', linewidth=2, label='Ground Truth')
                axes[0, 0].scatter(pred_original[0], pred_original[1], c='red', s=200, marker='x', 
                                 edgecolors='white', linewidth=2, label='Prediction')
                axes[0, 0].set_title(f'Sample {sample_idx+1}: {original_filename}', fontsize=12, fontweight='bold')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Save detailed prediction
                detailed_filename = f'detailed_sample_epoch_{epoch+1:03d}_{sample_idx+1}_{original_filename.replace(".jpg", "")}.png'
                detailed_path = os.path.join(sample_dir, detailed_filename)
                plt.savefig(detailed_path, dpi=200, bbox_inches='tight')
                plt.close()
                
                # Save detailed text report
                report_filename = f'detailed_sample_epoch_{epoch+1:03d}_{sample_idx+1}_{original_filename.replace(".jpg", "")}.txt'
                report_path = os.path.join(sample_dir, report_filename)
                
                with open(report_path, 'w') as f:
                    f.write(f"Image: {original_filename}\n")
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write(f"GT: ({gt_original[0]:.2f}, {gt_original[1]:.2f})\n")
                    f.write(f"Pred: ({pred_original[0]:.2f}, {pred_original[1]:.2f})\n")
                    f.write(f"Pixel Error: {pixel_error:.2f}\n")
                
                print(f"Saved detailed sample: {detailed_filename}")
                
    except Exception as e:
        print(f"Error creating detailed samples: {e}")

if __name__ == "__main__":
    main()