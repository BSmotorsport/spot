"""Training script for predicting the location of an occluded football.

This module prepares a heatmap-regression pipeline tailored to the weekly
"Ball on the Ball" (BOTB) challenge.  The dataset consists of 4K photographs
where the ball has been digitally removed; the ground-truth coordinates are
encoded in each filename.  Every scene appears with a handful of deterministic
augmentations (e.g. left-right flips) so the script ensures that all augmented
variants for a scene stay on the same side of the train/validation split.

The script deliberately avoids performing any heavy training-time work when it
is imported.  Instead, configuration is controlled via command-line arguments
and a :class:`TrainingConfig` dataclass.  The high-level flow is:

1. Discover image files within the dataset directory and parse the encoded
   metadata (scene identifier, x/y location, whether the file represents a
   mirrored variant, etc.).
2. Split the data into train/validation partitions grouped by the scene
   identifier so that no validation image shares content with the training set.
3. Construct :class:`torch.utils.data.Dataset` implementations that resize the
   native 4416×3336 photographs into a square training canvas, generate
   supervision heatmaps, and retain bookkeeping metadata required to map model
   predictions back to the original resolution.
4. Instantiate a lightweight heatmap-regression model (backed by a ConvNeXt
   feature extractor via ``timm``) and iterate through a standard PyTorch
   training loop with mixed-precision support.

The resulting file focuses on establishing a solid training scaffold—the
architecture, loss, logging, and validation metrics are all encapsulated in a
single place so subsequent experiments (e.g. plugging in gaze cues or a
transformer-based reasoning head) can reuse the same foundations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class TrainingConfig:
    """Runtime configuration for the training loop."""

    dataset_dir: Path = Path(r"E:\BOTB\dataset_augmented")
    output_dir: Path = Path(r"E:\BOTB\new_codex")

    # Image / heatmap geometry
    input_size: int = 1024
    heatmap_size: int = 256
    heatmap_sigma: float = 2.5

    # Optimisation parameters
    batch_size: int = 4
    num_epochs: int = 150
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0

    # Loss weighting
    heatmap_fg_weight: float = 1.0
    heatmap_bg_weight: float = 1.0

    # Data loading
    num_workers: int = 6
    val_fraction: float = 0.2
    random_seed: int = 1337

    # Model
    backbone_name: str = "convnext_base.fb_in22k_ft_in1k"
    pretrained: bool = True

    # Training utilities
    log_every: int = 50
    checkpoint_interval: int = 1
    amp: bool = torch.cuda.is_available()

    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Dataset utilities
# --------------------------------------------------------------------------------------


FILENAME_PATTERN = re.compile(
    r"(?P<prefix>[A-Z]*)?(?P<core>DC(?P<scene>\d{4})-(?P<x>\d+)-(?P<y>\d+))",
    re.IGNORECASE,
)


@dataclasses.dataclass
class ImageRecord:
    """Metadata for a single image in the BOTB dataset."""

    path: Path
    scene_id: str
    x: float
    y: float
    prefix: str


def discover_dataset_images(dataset_dir: Path) -> List[ImageRecord]:
    """Walk ``dataset_dir`` and parse metadata from BOTB filenames."""

    records: List[ImageRecord] = []
    for path in sorted(dataset_dir.rglob("*.jpg")):
        match = FILENAME_PATTERN.search(path.stem)
        if match is None:
            raise ValueError(
                f"Could not parse BOTB metadata from filename '{path.name}'."
            )
        prefix = (match.group("prefix") or "").upper()
        scene_id = match.group("scene")
        x = float(match.group("x"))
        y = float(match.group("y"))
        records.append(ImageRecord(path=path, scene_id=scene_id, x=x, y=y, prefix=prefix))

    if not records:
        raise RuntimeError(
            f"No .jpg images were discovered under '{dataset_dir}'. Check the path."
        )

    return records


def group_split(
    records: Sequence[ImageRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split records into train/validation by scene identifier."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be within (0, 1)")

    groups = [rec.scene_id for rec in records]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )

    indices = np.arange(len(records))
    train_idx, val_idx = next(splitter.split(indices, groups=groups, y=None))
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    return train_records, val_records


def _prepare_canvas(
    image: np.ndarray,
    original_xy: Tuple[float, float],
    input_size: int,
) -> tuple[np.ndarray, Tuple[float, float], float, float, float]:
    """Resize and letterbox the image to a square training canvas."""

    orig_h, orig_w = image.shape[:2]
    scale = input_size / max(orig_w, orig_h)
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((input_size, input_size, 3), dtype=resized.dtype)
    pad_x = (input_size - scaled_w) // 2
    pad_y = (input_size - scaled_h) // 2
    canvas[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = resized

    x, y = original_xy
    transformed_xy = (x * scale + pad_x, y * scale + pad_y)
    return canvas, transformed_xy, scale, pad_x, pad_y


def _maybe_horizontal_flip(
    image: np.ndarray,
    xy: Tuple[float, float],
    p: float = 0.5,
) -> tuple[np.ndarray, Tuple[float, float]]:
    """Randomly mirror the square image (and associated keypoint)."""

    if random.random() > p:
        return image, xy

    flipped = cv2.flip(image, 1)
    h, w = flipped.shape[:2]
    x, y = xy
    flipped_xy = (w - 1 - x, y)
    return flipped, flipped_xy


def gaussian_heatmap(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered on (``center_x``, ``center_y``)."""

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    heatmap = np.exp(
        -((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2.0 * sigma**2)
    )
    heatmap /= heatmap.max(initial=1e-6)
    return heatmap


class BotbBallDataset(Dataset):
    """PyTorch dataset that produces training samples for heatmap regression."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        config: TrainingConfig,
        *,
        is_train: bool,
        color_transform: A.BasicTransform | None = None,
    ) -> None:
        self.records = list(records)
        self.config = config
        self.is_train = is_train
        self.color_transform = color_transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image '{record.path}'")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Geometry: resize+pad to a square canvas.
        canvas, transformed_xy, scale, pad_x, pad_y = _prepare_canvas(
            image, (record.x, record.y), self.config.input_size
        )

        # Optional geometric augmentation (horizontal flip).
        if self.is_train:
            canvas, transformed_xy = _maybe_horizontal_flip(canvas, transformed_xy, p=0.5)

        # Appearance augmentation after geometry is fixed.
        if self.color_transform is not None:
            augmented = self.color_transform(image=canvas)
            canvas = augmented["image"]

        canvas = canvas.astype(np.float32) / 255.0
        canvas = (canvas - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        canvas_tensor = torch.from_numpy(canvas.transpose(2, 0, 1))

        heatmap_scale_x = self.config.heatmap_size / self.config.input_size
        heatmap_scale_y = self.config.heatmap_size / self.config.input_size
        cx = np.clip(transformed_xy[0] * heatmap_scale_x, 0, self.config.heatmap_size - 1)
        cy = np.clip(transformed_xy[1] * heatmap_scale_y, 0, self.config.heatmap_size - 1)
        heatmap = gaussian_heatmap(
            self.config.heatmap_size,
            self.config.heatmap_size,
            center_x=cx,
            center_y=cy,
            sigma=self.config.heatmap_sigma,
        )
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        sample = {
            "image": canvas_tensor,
            "heatmap": heatmap_tensor,
            "input_xy": torch.tensor(transformed_xy, dtype=torch.float32),
            "original_xy": torch.tensor((record.x, record.y), dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "pad": torch.tensor((pad_x, pad_y), dtype=torch.float32),
            "scene_id": record.scene_id,
            "path": str(record.path),
        }
        return sample


# --------------------------------------------------------------------------------------
# Model definition
# --------------------------------------------------------------------------------------


class HeatmapRegressor(nn.Module):
    """A thin wrapper around a classification backbone that emits heatmaps."""

    def __init__(self, backbone_name: str, heatmap_size: int, pretrained: bool) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=heatmap_size * heatmap_size,
            drop_rate=0.1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(images)
        heatmap = logits.view(-1, 1, self.heatmap_size, self.heatmap_size)
        return heatmap


# --------------------------------------------------------------------------------------
# Training / evaluation helpers
# --------------------------------------------------------------------------------------


def compute_metrics(
    outputs: torch.Tensor,
    batch: dict[str, torch.Tensor | list[str] | list],
    config: TrainingConfig,
) -> dict[str, float]:
    """Derive pixel-level metrics from raw model outputs."""

    with torch.no_grad():
        probs = torch.sigmoid(outputs)
        b, _, h, w = probs.shape
        flat = probs.view(b, -1)
        indices = torch.argmax(flat, dim=1)
        pred_y = (indices // w).float() + 0.5
        pred_x = (indices % w).float() + 0.5

        # Convert to the training canvas coordinate system.
        pred_x_canvas = pred_x * (config.input_size / config.heatmap_size)
        pred_y_canvas = pred_y * (config.input_size / config.heatmap_size)

        pad = batch["pad"].to(outputs.device)
        scale = batch["scale"].to(outputs.device)
        original_pred_x = (pred_x_canvas - pad[:, 0]) / scale
        original_pred_y = (pred_y_canvas - pad[:, 1]) / scale

        original_xy = batch["original_xy"].to(outputs.device)
        error = torch.sqrt(
            (original_pred_x - original_xy[:, 0]) ** 2
            + (original_pred_y - original_xy[:, 1]) ** 2
        )

        metrics = {
            "pixel_mae": error.mean().item(),
            "pixel_median": error.median().item(),
        }

        return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    scaler: GradScaler | None,
    metrics: dict[str, float],
) -> None:
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": dataclasses.asdict(config),
        "metrics": metrics,
        "loss_weights": {
            "heatmap_fg_weight": config.heatmap_fg_weight,
            "heatmap_bg_weight": config.heatmap_bg_weight,
        },
    }
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()

    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
    torch.save(state, checkpoint_path)


def create_dataloaders(
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    color_transform = A.Compose(
        [
            A.OneOf(
                [
                    A.ColorJitter(0.1, 0.1, 0.1, 0.05, p=1.0),
                    A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
                ],
                p=0.9,
            ),
            A.ISONoise(color_shift=(0.0, 0.02), intensity=(0.0, 0.02), p=0.2),
        ]
    )

    train_dataset = BotbBallDataset(
        train_records,
        config,
        is_train=True,
        color_transform=color_transform,
    )
    val_dataset = BotbBallDataset(
        val_records,
        config,
        is_train=False,
        color_transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    scaler: GradScaler | None,
    config: TrainingConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_meter = 0.0
    weight_meter = 0.0

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    progress = tqdm(dataloader, desc=f"Train {epoch:03d}")
    device = config.device()
    autocast_device_type = device.type

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device)
        targets = batch["heatmap"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=autocast_device_type, enabled=config.amp):
            outputs = model(images)
            loss_map = criterion(outputs, targets)
            weights = (
                targets * config.heatmap_fg_weight
                + (1.0 - targets) * config.heatmap_bg_weight
            )
            weighted_loss = loss_map * weights
            weight_sum = weights.sum()
            loss = weighted_loss.sum() / torch.clamp(weight_sum, min=1e-6)

        if scaler is not None and config.amp:
            scaler.scale(loss).backward()
            if config.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            optimizer.step()

        loss_meter += weighted_loss.sum().detach().item()
        weight_meter += weight_sum.detach().item()

        if step % config.log_every == 0:
            avg_loss = loss_meter / max(weight_meter, 1e-12)
            progress.set_postfix({"loss": avg_loss})

    avg_loss = loss_meter / max(weight_meter, 1e-12)
    return {"loss": avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    loss_meter = 0.0
    weight_meter = 0.0
    pixel_errors: List[float] = []
    device = config.device()

    for batch in tqdm(dataloader, desc="Validate"):
        images = batch["image"].to(device)
        targets = batch["heatmap"].to(device)

        outputs = model(images)
        loss_map = criterion(outputs, targets)
        weights = (
            targets * config.heatmap_fg_weight
            + (1.0 - targets) * config.heatmap_bg_weight
        )
        weighted_loss = loss_map * weights
        weight_sum = weights.sum()
        loss = weighted_loss.sum() / torch.clamp(weight_sum, min=1e-6)

        metrics = compute_metrics(outputs, batch, config)
        pixel_errors.append(metrics["pixel_mae"])

        loss_meter += weighted_loss.sum().item()
        weight_meter += weight_sum.item()

    avg_loss = loss_meter / max(weight_meter, 1e-12)
    avg_pixel_error = float(np.mean(pixel_errors)) if pixel_errors else float("nan")
    return {"val_loss": avg_loss, "val_pixel_mae": avg_pixel_error}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="BOTB heatmap training script")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset folder")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument(
        "--backbone", type=str, default=None, help="timm backbone name"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation fraction (grouped by scene)",
    )
    parser.add_argument(
        "--heatmap-fg-weight",
        type=float,
        default=None,
        help="Foreground weight for the heatmap loss",
    )
    parser.add_argument(
        "--heatmap-bg-weight",
        type=float,
        default=None,
        help="Background weight for the heatmap loss",
    )
    parser.add_argument(
        "--config-dump",
        action="store_true",
        help="Print resolved configuration and exit",
    )

    args = parser.parse_args()

    config = TrainingConfig()
    if args.dataset is not None:
        config.dataset_dir = args.dataset
    if args.output is not None:
        config.output_dir = args.output
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.backbone is not None:
        config.backbone_name = args.backbone
    if args.val_fraction is not None:
        config.val_fraction = args.val_fraction
    if args.heatmap_fg_weight is not None:
        config.heatmap_fg_weight = args.heatmap_fg_weight
    if args.heatmap_bg_weight is not None:
        config.heatmap_bg_weight = args.heatmap_bg_weight

    if args.config_dump:
        print(json.dumps(dataclasses.asdict(config), indent=2, default=str))
        raise SystemExit(0)

    return config


def main() -> None:  # pragma: no cover - CLI entry point
    config = parse_args()
    setup_seed(config.random_seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering dataset under: {config.dataset_dir}")
    records = discover_dataset_images(config.dataset_dir)
    train_records, val_records = group_split(
        records, val_fraction=config.val_fraction, seed=config.random_seed
    )
    print(
        f"Discovered {len(records)} images across {len({r.scene_id for r in records})} scenes"
    )
    print(f"Training samples: {len(train_records)} | Validation samples: {len(val_records)}")

    train_loader, val_loader = create_dataloaders(train_records, val_records, config)

    model = HeatmapRegressor(
        config.backbone_name, config.heatmap_size, pretrained=config.pretrained
    )
    model.to(config.device())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = GradScaler(enabled=config.amp)

    history: list[dict[str, float]] = []

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, scaler, config, epoch
        )
        val_metrics = validate(model, val_loader, config)

        combined = {**train_metrics, **val_metrics, "epoch": epoch}
        combined["heatmap_fg_weight"] = config.heatmap_fg_weight
        combined["heatmap_bg_weight"] = config.heatmap_bg_weight
        history.append(combined)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_pixel_mae={val_metrics['val_pixel_mae']:.2f}"
        )

        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, config, scaler, val_metrics)

    history_path = config.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()

