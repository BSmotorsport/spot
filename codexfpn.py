import math
import os
import random
import re
from typing import Optional, Sequence, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================


class Config:
    """Centralised hyperparameters for the coordinate regression pipeline."""

    # Paths
    DATASET_PATH: str = r"E:\\BOTB\\dataset\\aug"
    OUTPUT_DIR: str = "./training_output_coordinate"
    MODEL_VERSION: str = "coordinate_regression_v1"

    # Image geometry
    ORIGINAL_WIDTH: int = 4416
    ORIGINAL_HEIGHT: int = 3336
    IMAGE_SIZE: int = 1536

    # Optimisation
    MODEL_NAME: str = "convnext_base.fb_in22k_ft_in1k"
    BATCH_SIZE: int = 2
    EPOCHS: int = 120
    INITIAL_LR: float = 3e-4
    BACKBONE_LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_WORKERS: int = 4
    RANDOM_SEED: int = 42
    GRAD_ACCUM_STEPS: int = 4
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    AMP: bool = torch.cuda.is_available()
    USE_CHANNELS_LAST: bool = True

    NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Training stages
    FREEZE_BACKBONE_EPOCHS: int = 2

    # Loss
    PIXEL_LOSS_WEIGHT: float = 1.0

    # Validation utilities
    VALIDATE_EVERY: int = 1
    NUM_VAL_SAMPLES: int = 6

    # Checkpointing
    CHECKPOINT_FILENAME: str = "coordinate_regressor.pth"


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ======================================================================================
# 2. REPRODUCIBILITY HELPERS
# ======================================================================================


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seeds(Config.RANDOM_SEED)


# ======================================================================================
# 3. AUGMENTATION UTILITIES
# ======================================================================================


def instantiate_albumentations_transform(
    transform_cls,
    common_kwargs: dict,
    candidate_kwargs_list: Sequence[dict],
):
    """Instantiate an Albumentations transform while ignoring stale kwargs."""

    signature = inspect_signature(transform_cls)
    accepts_var_kwargs = signature is None
    allowed_keys = None if accepts_var_kwargs else set(signature)

    errors: list[str] = []
    attempted = []

    for candidate in candidate_kwargs_list:
        attempted.append(dict(candidate))
        filtered = candidate
        if allowed_keys is not None:
            filtered = {k: v for k, v in candidate.items() if k in allowed_keys}
            if candidate and not filtered:
                continue
        try:
            return transform_cls(**common_kwargs, **filtered)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue

    try:
        return transform_cls(**common_kwargs)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        attempted.append({})
        raise TypeError(
            f"Unable to instantiate {transform_cls.__name__}; attempted {attempted}; errors: {errors}"
        ) from exc


def inspect_signature(transform_cls):
    """Lightweight helper that returns constructor argument names."""

    try:
        import inspect

        signature = inspect.signature(transform_cls.__init__)
    except (TypeError, ValueError):
        return None

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None

    return {name for name in params if name != "self"}


# ======================================================================================
# 4. DATASET
# ======================================================================================


def parse_filename(filename: str) -> Optional[dict]:
    """Parse ``filename`` (without extension) and extract pixel coordinates."""

    base = os.path.splitext(filename)[0]
    parts = base.split("-")
    if len(parts) < 2:
        return None
    try:
        y_coord = int(parts[-1])
        x_coord = int(parts[-2])
    except ValueError:
        return None

    if not (0 <= x_coord <= Config.ORIGINAL_WIDTH and 0 <= y_coord <= Config.ORIGINAL_HEIGHT):
        return None

    return {"x": float(x_coord), "y": float(y_coord)}


class FootballDataset(Dataset):
    """Dataset that returns an augmented frame and the hidden-ball coordinates."""

    def __init__(
        self,
        image_paths: Sequence[str],
        transform: Optional[A.BasicTransform] = None,
        augment: bool = True,
    ) -> None:
        self.transform = transform
        self.augment = augment

        self.valid_paths: list[str] = []
        for path in image_paths:
            if parse_filename(os.path.basename(path)) is not None:
                self.valid_paths.append(path)

        print(
            f"Dataset initialised with {len(self.valid_paths)}/{len(image_paths)} files containing coordinates"
        )

    def __len__(self) -> int:
        return len(self.valid_paths)

    def __getitem__(self, idx: int):
        img_path = self.valid_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        coords = parse_filename(os.path.basename(img_path))
        if coords is None:
            raise ValueError(f"Filename {img_path} does not contain coordinates")

        keypoints = [(coords["x"], coords["y"])]
        if self.transform is not None:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed.get("keypoints", [])

            if not keypoints:
                # Albumentations can drop keypoints when they leave the frame. To
                # keep batches stable we simply return the original sample.
                return self.__getitem__((idx + 1) % len(self.valid_paths))

            clamped = []
            limit = float(np.nextafter(Config.IMAGE_SIZE, -np.inf))
            for kp in keypoints:
                x, y = kp[:2]
                if 0.0 <= x < Config.IMAGE_SIZE and 0.0 <= y < Config.IMAGE_SIZE:
                    clamped.append((min(max(x, 0.0), limit), min(max(y, 0.0), limit)))
            if not clamped:
                return self.__getitem__((idx + 1) % len(self.valid_paths))
            keypoints = clamped

        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        tensor_image = image
        if isinstance(tensor_image, np.ndarray):
            tensor_image = torch.from_numpy(tensor_image.transpose(2, 0, 1))
        elif torch.is_tensor(tensor_image):
            tensor_image = tensor_image.clone()
        else:
            tensor_image = torch.as_tensor(tensor_image)

        if tensor_image.dtype.is_floating_point:
            tensor_image = tensor_image.to(dtype=torch.float32)
            max_val = tensor_image.max().item()
            min_val = tensor_image.min().item()
            if min_val >= 0.0 and max_val > 1.0:
                tensor_image = tensor_image / 255.0
        else:
            tensor_image = tensor_image.to(dtype=torch.float32) / 255.0

        if tensor_image.shape[-2:] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            tensor_image = torch.nn.functional.interpolate(
                tensor_image.unsqueeze(0),
                size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        image_extent = float(Config.IMAGE_SIZE - 1)
        keypoint = torch.tensor(keypoints[0], dtype=torch.float32)
        normalised = keypoint / image_extent
        normalised = normalised.clamp(0.0, 1.0)

        return tensor_image, normalised, img_path


# ======================================================================================
# 5. MODEL
# ======================================================================================


class CoordinateRegressionModel(nn.Module):
    """Backbone + attention-pooled head that regresses normalised coordinates."""

    def __init__(self, backbone_name: str, pretrained: bool = True, hidden_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise AttributeError("Backbone does not expose num_features")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if features.dim() == 3:
            # Some transformer-style backbones return (B, N, C). Reshape back to
            # a spatial map by assuming a square token layout.
            b, n, c = features.shape
            side = int(math.sqrt(n))
            features = features.transpose(1, 2).reshape(b, c, side, side)

        avg = self.avg_pool(features)
        maxv = self.max_pool(features)
        pooled = torch.cat([avg, maxv], dim=1).flatten(1)
        logits = self.head(pooled)
        return torch.sigmoid(logits)


def create_model(pretrained: bool = True) -> CoordinateRegressionModel:
    model = CoordinateRegressionModel(Config.MODEL_NAME, pretrained=pretrained)
    return model


# ======================================================================================
# 6. LOSS & METRICS
# ======================================================================================


class CoordinateLoss(nn.Module):
    """Smooth L1 + pixel MAE encourages precise coordinate regression."""

    def __init__(self, pixel_weight: float = 1.0) -> None:
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.pixel_weight = float(pixel_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        smooth = self.smooth_l1(pred, target)

        image_extent = max(float(Config.IMAGE_SIZE - 1), 1.0)
        pixel_errors = torch.abs(pred - target) * image_extent
        pixel_mae = pixel_errors.mean()

        loss = smooth + self.pixel_weight * (pixel_mae / image_extent)
        return loss, smooth.detach(), pixel_mae.detach()


# ======================================================================================
# 7. TRAINING UTILITIES
# ======================================================================================


def prepare_dataloaders(
    train_files: Sequence[str],
    val_files: Sequence[str],
    train_transform: A.BasicTransform,
    val_transform: A.BasicTransform,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = FootballDataset(train_files, transform=train_transform)
    val_dataset = FootballDataset(val_files, transform=val_transform, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def compute_pixel_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    image_extent = max(float(Config.IMAGE_SIZE - 1), 1.0)
    diff = (pred - target).detach()
    errors = torch.abs(diff) * image_extent
    mae = errors.mean().item()
    rmse = torch.sqrt(((diff * image_extent) ** 2).mean()).item()
    return mae, rmse


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CoordinateLoss,
    scaler: GradScaler,
    epoch: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_coord = 0.0
    total_pixel = 0.0

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"Train {epoch+1}", colour="green")
    accumulation = 0

    for images, targets, _ in progress:
        images = images.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True)
        if Config.USE_CHANNELS_LAST:
            images = images.to(memory_format=torch.channels_last)

        with autocast(device_type="cuda" if Config.DEVICE.startswith("cuda") else "cpu", enabled=Config.AMP):
            preds = model(images)
            loss, coord_loss, pixel_mae = criterion(preds, targets)

        scaled_loss = loss / Config.GRAD_ACCUM_STEPS
        scaler.scale(scaled_loss).backward()
        accumulation += 1

        total_loss += loss.item()
        total_coord += coord_loss.item()
        total_pixel += pixel_mae.item()

        if accumulation >= Config.GRAD_ACCUM_STEPS:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pix": f"{pixel_mae.item():.2f}",
        })

    if accumulation > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    batches = max(len(loader), 1)
    return total_loss / batches, total_coord / batches, total_pixel / batches


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CoordinateLoss,
    epoch: int,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_coord = 0.0
    total_pixel = 0.0

    progress = tqdm(loader, desc=f"Val {epoch+1}", colour="yellow")

    with torch.no_grad():
        for images, targets, _ in progress:
            images = images.to(Config.DEVICE, non_blocking=True)
            targets = targets.to(Config.DEVICE, non_blocking=True)
            if Config.USE_CHANNELS_LAST:
                images = images.to(memory_format=torch.channels_last)

            with autocast(device_type="cuda" if Config.DEVICE.startswith("cuda") else "cpu", enabled=Config.AMP):
                preds = model(images)
                loss, coord_loss, pixel_mae = criterion(preds, targets)

            total_loss += loss.item()
            total_coord += coord_loss.item()
            total_pixel += pixel_mae.item()

    batches = max(len(loader), 1)
    avg_loss = total_loss / batches
    avg_coord = total_coord / batches
    avg_pixel = total_pixel / batches
    return avg_loss, avg_coord, avg_pixel


def save_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    save_dir: str,
    max_samples: int = 6,
    metrics: Optional[dict] = None,
) -> str:
    model.eval()
    iterator = iter(loader)
    images, targets, paths = next(iterator)
    images = images.to(Config.DEVICE)
    if Config.USE_CHANNELS_LAST:
        images = images.to(memory_format=torch.channels_last)

    with torch.no_grad():
        preds = model(images).cpu()

    targets = targets.cpu()
    image_extent = float(Config.IMAGE_SIZE - 1)
    count = min(max_samples, images.size(0))

    fig, axes = plt.subplots(count, 1, figsize=(6, 4 * count))
    if count == 1:
        axes = [axes]

    mean = torch.tensor(Config.NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.NORMALIZE_STD).view(3, 1, 1)

    for idx in range(count):
        img = images[idx].detach().cpu().to(dtype=torch.float32)
        img = img * std + mean
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0.0, 1.0)

        ax = axes[idx]
        ax.imshow(img_np)

        tgt = targets[idx] * image_extent
        prd = preds[idx] * image_extent
        pixel_error = float(math.dist(prd.tolist(), tgt.tolist()))

        ax.scatter([tgt[0]], [tgt[1]], c="lime", marker="o", label="target")
        ax.scatter([prd[0]], [prd[1]], c="red", marker="x", label="pred")
        filename = os.path.basename(paths[idx])
        ax.set_title(f"{filename} | Δpx: {pixel_error:.2f}")
        ax.axis("off")
        ax.legend(loc="upper right")

        annotation = (
            f"Target: ({tgt[0]:.1f}, {tgt[1]:.1f})\n"
            f"Pred:   ({prd[0]:.1f}, {prd[1]:.1f})\n"
            f"Δpx:    {pixel_error:.2f}"
        )
        ax.text(
            0.02,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.6},
        )

    summary_lines: list[str] = []
    if metrics:
        epoch_num = metrics.get("epoch")
        total_epochs = metrics.get("epochs")
        if epoch_num is not None and total_epochs is not None:
            summary_lines.append(f"Epoch {epoch_num}/{total_epochs}")
        if "train_loss" in metrics and "train_pixel" in metrics:
            summary_lines.append(
                f"Train Loss: {metrics['train_loss']:.4f} | Train MAE: {metrics['train_pixel']:.2f} px"
            )
        if "train_coord" in metrics:
            summary_lines.append(f"Train Coord MAE: {metrics['train_coord']:.4f}")
        if "val_loss" in metrics and "val_pixel" in metrics:
            summary_lines.append(
                f"Val Loss: {metrics['val_loss']:.4f} | Val MAE: {metrics['val_pixel']:.2f} px"
            )
        if "val_coord" in metrics:
            summary_lines.append(f"Val Coord MAE: {metrics['val_coord']:.4f}")
        if "best_val_mae" in metrics:
            summary_lines.append(f"Best Val MAE: {metrics['best_val_mae']:.2f} px")

    if summary_lines:
        fig.suptitle("\n".join(summary_lines), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    output_path = os.path.join(save_dir, f"val_samples_epoch_{epoch+1}.png")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    best_val_mae: float,
    history: dict,
) -> str:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": best_val_mae,
        "history": history,
        "scaler": scaler.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path = os.path.join(Config.OUTPUT_DIR, Config.CHECKPOINT_FILENAME)
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
) -> Tuple[int, float, dict]:
    path = os.path.join(Config.OUTPUT_DIR, Config.CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        return 0, float("inf"), {"train": [], "val": [], "val_mae": []}

    payload = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    epoch = payload.get("epoch", 0) + 1
    best_val = payload.get("best_val_mae", float("inf"))
    history = payload.get("history", {"train": [], "val": [], "val_mae": []})
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch, best_val, history


# ======================================================================================
# 8. MAIN TRAINING ROUTINE
# ======================================================================================


def build_transforms() -> Tuple[A.BasicTransform, A.BasicTransform]:
    random_resized_crop_signature = inspect_signature(A.RandomResizedCrop)
    random_resized_crop_candidates: list[dict] = []
    if random_resized_crop_signature is None:
        # Prefer modern ``size`` signatures but keep height/width as a last resort.
        random_resized_crop_candidates.extend(
            [
                {"size": Config.IMAGE_SIZE},
                {"size": (Config.IMAGE_SIZE, Config.IMAGE_SIZE)},
                {"size": [Config.IMAGE_SIZE, Config.IMAGE_SIZE]},
                {"height": Config.IMAGE_SIZE, "width": Config.IMAGE_SIZE},
            ]
        )
    else:
        if "size" in random_resized_crop_signature:
            random_resized_crop_candidates.extend(
                [
                    {"size": Config.IMAGE_SIZE},
                    {"size": (Config.IMAGE_SIZE, Config.IMAGE_SIZE)},
                    {"size": [Config.IMAGE_SIZE, Config.IMAGE_SIZE]},
                ]
            )
        if {"height", "width"}.issubset(random_resized_crop_signature):
            random_resized_crop_candidates.append(
                {"height": Config.IMAGE_SIZE, "width": Config.IMAGE_SIZE}
            )

    random_resized_crop = instantiate_albumentations_transform(
        A.RandomResizedCrop,
        dict(scale=(0.55, 1.0), ratio=(0.75, 1.33), interpolation=cv2.INTER_CUBIC),
        random_resized_crop_candidates,
    )

    affine = instantiate_albumentations_transform(
        A.Affine,
        dict(scale=(0.9, 1.1), rotate=(-8, 8), shear=(-4, 4), fit_output=False, p=0.4),
        [
            {"cval": 0, "mode": cv2.BORDER_REFLECT_101},
            {"value": 0, "border_mode": cv2.BORDER_REFLECT_101},
            {},
        ],
    )

    perspective = instantiate_albumentations_transform(
        A.Perspective,
        dict(scale=(0.02, 0.05), keep_size=True, p=0.25),
        [
            {"pad_mode": cv2.BORDER_REFLECT_101},
            {"border_mode": cv2.BORDER_REFLECT_101},
            {},
        ],
    )

    train_transform = A.Compose(
        [
            A.OneOf([random_resized_crop, A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE)], p=1.0),
            A.HorizontalFlip(p=0.5),
            affine,
            perspective,
            A.RandomBrightnessContrast(0.25, 0.2, p=0.5),
            A.ColorJitter(0.12, 0.12, 0.12, 0.05, p=0.3),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )

    val_transform = A.Compose(
        [
            A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return train_transform, val_transform


def group_files_by_scene(files: Sequence[str]) -> dict:
    grouped: dict[str, list[str]] = {}
    for file in files:
        base = os.path.basename(file)
        first = base.split("-")[0]
        match = re.search(r"(\d+)", first)
        if match:
            grouped.setdefault(match.group(1), []).append(file)
    return grouped


def gather_dataset_files() -> Tuple[Sequence[str], Sequence[str]]:
    files = [
        os.path.join(Config.DATASET_PATH, f)
        for f in os.listdir(Config.DATASET_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    valid = [f for f in files if parse_filename(os.path.basename(f)) is not None]
    grouped = group_files_by_scene(valid)

    print(f"Total valid frames: {len(valid)} across {len(grouped)} scenes")

    scene_ids = list(grouped.keys())
    train_ids, val_ids = train_test_split(scene_ids, test_size=0.2, random_state=Config.RANDOM_SEED)

    train_files = [f for sid in train_ids for f in grouped[sid]]
    val_files = [f for sid in val_ids for f in grouped[sid]]

    print(f"Train scenes: {len(train_ids)} ({len(train_files)} frames)")
    print(f"Val scenes:   {len(val_ids)} ({len(val_files)} frames)")
    return train_files, val_files


def main() -> None:
    print("=" * 80)
    print("Coordinate Regression Training")
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print("=" * 80)

    train_files, val_files = gather_dataset_files()
    train_transform, val_transform = build_transforms()
    train_loader, val_loader = prepare_dataloaders(train_files, val_files, train_transform, val_transform)

    model = create_model(pretrained=True).to(Config.DEVICE)

    head_params = list(model.head.parameters()) + list(model.avg_pool.parameters()) + list(model.max_pool.parameters())

    if Config.FREEZE_BACKBONE_EPOCHS > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.BACKBONE_LR},
            {"params": head_params, "lr": Config.INITIAL_LR},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.BACKBONE_LR * 0.1)

    criterion = CoordinateLoss(pixel_weight=Config.PIXEL_LOSS_WEIGHT)
    scaler = GradScaler(enabled=Config.AMP)

    start_epoch, best_val_mae, history = load_checkpoint(model, optimizer, scheduler, scaler)

    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        if Config.FREEZE_BACKBONE_EPOCHS and epoch == Config.FREEZE_BACKBONE_EPOCHS:
            print("\nUnfreezing backbone for fine-tuning")
            for param in model.backbone.parameters():
                param.requires_grad = True

        train_loss, train_coord, train_pixel = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        scheduler.step()

        val_results: Optional[dict] = None

        if (epoch + 1) % Config.VALIDATE_EVERY == 0:
            val_loss, val_coord, val_pixel = validate(model, val_loader, criterion, epoch)
            history.setdefault("train", []).append(train_loss)
            history.setdefault("val", []).append(val_loss)
            history.setdefault("val_mae", []).append(val_pixel)

            val_results = {
                "loss": val_loss,
                "coord": val_coord,
                "pixel": val_pixel,
            }

            if val_pixel < best_val_mae:
                prev_best = best_val_mae
                best_val_mae = val_pixel
                save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_val_mae, history)
                if math.isinf(prev_best):
                    print(f"Best model updated: val MAE {val_pixel:.2f} px (first checkpoint)")
                else:
                    print(
                        "Best model updated: val MAE "
                        f"{val_pixel:.2f} px (improved {prev_best - best_val_mae:.2f} px)"
                    )

            metrics_payload = {
                "epoch": epoch + 1,
                "epochs": Config.EPOCHS,
                "train_loss": train_loss,
                "train_coord": train_coord,
                "train_pixel": train_pixel,
                "val_loss": val_loss,
                "val_coord": val_coord,
                "val_pixel": val_pixel,
                "best_val_mae": best_val_mae,
            }

            sample_path = save_sample_predictions(
                model,
                val_loader,
                epoch,
                Config.OUTPUT_DIR,
                Config.NUM_VAL_SAMPLES,
                metrics=metrics_payload,
            )
            print(f"Validation samples saved to {sample_path}")

        else:
            history.setdefault("train", []).append(train_loss)

        lr_values = [group["lr"] for group in optimizer.param_groups]
        lr_text = ", ".join(f"{lr:.2e}" for lr in lr_values)
        print(
            "Train -> Loss: "
            f"{train_loss:.4f} | Coord MAE: {train_coord:.4f} | Pixel MAE: {train_pixel:.2f} px"
            f" | LRs: {lr_text}"
        )

        if val_results is not None:
            delta_pixel = val_results["pixel"] - train_pixel
            print(
                "Val   -> Loss: "
                f"{val_results['loss']:.4f} | Coord MAE: {val_results['coord']:.4f} | "
                f"Pixel MAE: {val_results['pixel']:.2f} px (Δ vs train: {delta_pixel:+.2f} px) | "
                f"Best MAE: {best_val_mae:.2f} px"
            )
        else:
            print("Val   -> Skipped (waiting for scheduled evaluation)")

    final_path = save_checkpoint(Config.EPOCHS - 1, model, optimizer, scheduler, scaler, best_val_mae, history)
    print(f"Training completed. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
