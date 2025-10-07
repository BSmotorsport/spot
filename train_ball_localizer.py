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
import copy
import dataclasses
import inspect
import json
import random
import re
import pathlib
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    heatmap_sigma: float = 12.0

    # Optimisation parameters
    batch_size: int = 4
    num_epochs: int = 150
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6

    # Loss weighting (targets blend linearly between bg/fg weights)
    heatmap_fg_weight: float = 3.0
    heatmap_bg_weight: float = 1.0

    # Data loading
    num_workers: int = 6
    val_fraction: float = 0.2
    random_seed: int = 1337

    # Geometric augmentation
    enable_spatial_aug: bool = False
    spatial_flip_prob: float = 0.5
    spatial_shift_limit: float = 0.02
    spatial_scale_limit: float = 0.05
    spatial_rotate_limit: float = 5.0
    spatial_shift_scale_prob: float = 0.75
    spatial_crop_prob: float = 0.2
    spatial_crop_scale_min: float = 0.9
    spatial_crop_scale_max: float = 1.0

    # Model
    backbone_name: str = "convnext_base.fb_in22k_ft_in1k"
    pretrained: bool = True
    backbone_freeze_epochs: int = 0
    backbone_unfreeze_warmup: int = 0
    backbone_lr_multiplier: float = 0.1
    decoder_dropout: float = 0.0

    # Training utilities
    log_every: int = 50
    validation_sample_count: int = 16
    amp: bool = torch.cuda.is_available()
    resume_from: Path | None = None
    ema_decay: float = 0.999
    ema_start_epoch: int = 1

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


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_dataset_images(dataset_dir: Path) -> List[ImageRecord]:
    """Walk ``dataset_dir`` and parse metadata from BOTB filenames."""

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' does not exist."
        )
    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset path '{dataset_dir}' is not a directory."
        )

    records: List[ImageRecord] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
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
            "No supported images were discovered under "
            f"'{dataset_dir}'. Check the path and file extensions."
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


def gaussian_heatmap(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    sigma: float,
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered on (``center_x``, ``center_y``)."""

    # ``numpy`` defaults to ``float64`` for scalar inputs which caused the
    # generated heatmaps to become ``float64`` tensors when converted with
    # ``torch.from_numpy``.  The training loop expects ``float32`` tensors so we
    # explicitly perform the computation in single precision.
    center_x32 = np.float32(center_x)
    center_y32 = np.float32(center_y)
    sigma32 = np.float32(sigma)

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)[:, None]
    two_sigma_sq = np.float32(2.0) * sigma32 * sigma32
    heatmap = np.exp(-((xs - center_x32) ** 2 + (ys - center_y32) ** 2) / two_sigma_sq)
    return heatmap.astype(np.float32)


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
        self._enable_spatial_aug = self.is_train and self.config.enable_spatial_aug

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def _build_spatial_transform(self, height: int, width: int) -> A.Compose:
        transforms: list[A.BasicTransform] = []
        if self.config.spatial_flip_prob > 0.0:
            transforms.append(A.HorizontalFlip(p=self.config.spatial_flip_prob))
        if self.config.spatial_shift_scale_prob > 0.0:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=self.config.spatial_shift_limit,
                    scale_limit=self.config.spatial_scale_limit,
                    rotate_limit=self.config.spatial_rotate_limit,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=self.config.spatial_shift_scale_prob,
                )
            )
        if (
            self.config.spatial_crop_prob > 0.0
            and self.config.spatial_crop_scale_min < self.config.spatial_crop_scale_max
        ):
            transforms.append(
                A.RandomResizedCrop(
                    height=height,
                    width=width,
                    scale=(
                        self.config.spatial_crop_scale_min,
                        self.config.spatial_crop_scale_max,
                    ),
                    ratio=(1.0, 1.0),
                    p=self.config.spatial_crop_prob,
                )
            )

        if not transforms:
            transforms.append(A.NoOp())

        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image '{record.path}'")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        keypoint_xy = (float(record.x), float(record.y))
        if self._enable_spatial_aug:
            h, w = image.shape[:2]
            spatial_transform = self._build_spatial_transform(h, w)
            augmented = spatial_transform(image=image, keypoints=[keypoint_xy])
            image = augmented["image"]
            if not augmented["keypoints"]:
                # Fallback: retain the original coordinates if augmentation dropped them.
                keypoint_xy = (float(record.x), float(record.y))
            else:
                aug_x, aug_y = augmented["keypoints"][0]
                keypoint_xy = (float(aug_x), float(aug_y))

        # Geometry: resize+pad to a square canvas.  Some BOTB files represent
        # pre-flipped imagery (the filename already encodes the mirrored
        # coordinates) so the default behaviour avoids additional geometric
        # augmentations.  When enabled via :class:`TrainingConfig` the optional
        # spatial jitter runs *before* the resize so bookkeeping stays
        # consistent.
        canvas, transformed_xy, scale, pad_x, pad_y = _prepare_canvas(
            image, keypoint_xy, self.config.input_size
        )

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
        sigma = float(self.config.heatmap_sigma)
        heatmap = gaussian_heatmap(
            self.config.heatmap_size,
            self.config.heatmap_size,
            center_x=cx,
            center_y=cy,
            sigma=sigma,
        )
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        sample = {
            "image": canvas_tensor,
            "heatmap": heatmap_tensor,
            "input_xy": torch.tensor(transformed_xy, dtype=torch.float32),
            "original_xy": torch.tensor(keypoint_xy, dtype=torch.float32),
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
    """ConvNeXt-style backbone with a lightweight FPN decoder for heatmaps."""

    def __init__(
        self,
        backbone_name: str,
        heatmap_size: int,
        pretrained: bool,
        decoder_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.decoder_dropout = float(decoder_dropout)

        # ``features_only`` exposes intermediate resolution feature maps instead of
        # the classification logits (which would collapse spatial information).
        # ``timm`` models expose varying numbers of intermediate feature maps so we
        # adaptively select the deepest four stages (or all of them if fewer are
        # available) without re-instantiating the backbone.  This avoids asking the
        # factory for non-existent ``out_indices`` which previously triggered an
        # ``IndexError`` on backbones that only expose four feature maps (e.g.
        # ConvNeXt).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        feature_channels_all = self.backbone.feature_info.channels()
        if not feature_channels_all:
            raise RuntimeError(
                f"Backbone '{backbone_name}' did not expose any feature maps."
            )

        self.feature_slice_start = max(0, len(feature_channels_all) - 4)
        feature_channels = feature_channels_all[self.feature_slice_start :]
        fpn_channels = 256

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(ch, fpn_channels, kernel_size=1) for ch in feature_channels
        )
        def make_decoder_block() -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            if self.decoder_dropout > 0.0:
                layers.append(nn.Dropout2d(self.decoder_dropout))
            return nn.Sequential(*layers)

        self.output_convs = nn.ModuleList(
            make_decoder_block() for _ in feature_channels
        )

        head_layers: list[nn.Module] = [
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if self.decoder_dropout > 0.0:
            head_layers.append(nn.Dropout(self.decoder_dropout))
        head_layers.append(nn.Conv2d(fpn_channels, 1, kernel_size=1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if self.feature_slice_start:
            features = features[self.feature_slice_start :]

        # Build a top-down feature pyramid so the heatmap prediction retains
        # spatial detail.  ``features`` is ordered from high-to-low resolution.
        pyramid = None
        for feat, lateral, output in zip(reversed(features), reversed(self.lateral_convs), reversed(self.output_convs)):
            lateral_feat = lateral(feat)
            if pyramid is None:
                pyramid = output(lateral_feat)
            else:
                upsampled = F.interpolate(pyramid, size=lateral_feat.shape[-2:], mode="bilinear", align_corners=False)
                pyramid = output(lateral_feat + upsampled)

        assert pyramid is not None  # pragma: no cover - defensive

        heatmap = self.head(pyramid)
        heatmap = F.interpolate(
            heatmap,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False,
        )
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
        pred_y = (indices // w).float()
        pred_x = (indices % w).float()

        # Convert to the training canvas coordinate system.  ``+ 0.5`` ensures
        # the decoded point corresponds to the centre of the winning heatmap
        # cell rather than its top-left corner (see dataset encoding).  The
        # dataset clamps keypoints to the heatmap boundaries before rendering,
        # so boundary indices already represent the centre of the outermost
        # cell—skip the offset there to avoid nudging edge predictions inward.
        stride = config.input_size / config.heatmap_size
        interior_x = (pred_x > 0.0) & (pred_x < float(w - 1))
        interior_y = (pred_y > 0.0) & (pred_y < float(h - 1))
        pred_x_canvas = (pred_x + interior_x.float() * 0.5) * stride
        pred_y_canvas = (pred_y + interior_y.float() * 0.5) * stride

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
            "pixel_errors": error.detach().cpu().tolist(),
        }

        return metrics


class ValidationSampleExporter:
    """Persist visualisations of validation predictions for debugging."""

    def __init__(
        self,
        output_dir: Path,
        config: TrainingConfig,
        max_samples: int,
    ) -> None:
        self.max_samples = max(0, int(max_samples))
        self.saved = 0
        self.config = config
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.output_dir = output_dir
        if self.max_samples > 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_batch(self, batch: dict[str, torch.Tensor | list[str]], probs: torch.Tensor) -> None:
        if self.saved >= self.max_samples or self.max_samples == 0:
            return

        images = batch["image"]
        targets = batch["heatmap"]
        input_xy = batch["input_xy"]
        original_xy = batch["original_xy"]
        pads = batch["pad"]
        scales = batch["scale"]
        paths = batch["path"]
        batch_size = int(images.shape[0])
        remaining = self.max_samples - self.saved
        export_count = min(batch_size, remaining)

        probs = probs.detach().cpu()

        for idx in range(export_count):
            image_tensor = images[idx]
            image = self._denormalize_image(image_tensor)

            pred_heatmap = probs[idx, 0].numpy()
            target_heatmap = targets[idx, 0].detach().cpu().numpy()

            pred_canvas_xy_display = self._decode_heatmap(pred_heatmap)
            pred_canvas_xy_for_error = pred_canvas_xy_display

            gt_canvas_xy = input_xy[idx].detach().cpu().numpy()
            pad = pads[idx].detach().cpu().numpy()
            scale = float(scales[idx].detach().cpu().item())
            gt_original_xy = original_xy[idx].detach().cpu().numpy()
            pred_original_xy = (
                (pred_canvas_xy_for_error[0] - pad[0]) / scale,
                (pred_canvas_xy_for_error[1] - pad[1]) / scale,
            )
            pixel_error = float(
                np.linalg.norm(np.array(pred_original_xy) - gt_original_xy)
            )

            annotated = self._annotate_points(
                image.copy(), pred_canvas_xy_display, gt_canvas_xy
            )
            pred_overlay = self._heatmap_overlay(image.copy(), pred_heatmap)
            pred_overlay = self._annotate_points(
                pred_overlay, pred_canvas_xy_display, gt_canvas_xy
            )
            target_overlay = self._heatmap_overlay(image.copy(), target_heatmap)
            target_overlay = self._annotate_points(
                target_overlay, gt_canvas_xy, gt_canvas_xy
            )

            combined = np.concatenate(
                [annotated, pred_overlay, target_overlay], axis=1
            )

            info_text = (
                f"pred canvas=({pred_canvas_xy_display[0]:.1f}, "
                f"{pred_canvas_xy_display[1]:.1f}) "
                f"| gt canvas=({gt_canvas_xy[0]:.1f}, {gt_canvas_xy[1]:.1f}) "
                f"| pixel err={pixel_error:.2f}"
            )
            cv2.putText(
                combined,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            path_value = paths[idx]
            if isinstance(path_value, (list, tuple)):
                path_value = path_value[0]
            stem = Path(path_value).stem
            output_path = self.output_dir / f"{self.saved:03d}_{stem}.png"
            cv2.imwrite(str(output_path), combined)
            self.saved += 1

            if self.saved >= self.max_samples:
                break

    def flush(self) -> None:
        """Flush any buffered outputs to disk.

        The exporter writes images immediately when ``save_batch`` is called, so
        there is no buffered state to clear.  The method exists to maintain API
        compatibility with callers that expect a ``flush`` hook.
        """

        # Intentionally a no-op.  Keeping the method allows callers to invoke
        # ``flush`` unconditionally without needing to guard against an
        # ``AttributeError`` when validation is configured to export samples.
        return

    def _denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().permute(1, 2, 0).numpy()
        array = array * self._std + self._mean
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    def _heatmap_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.nan_to_num(heatmap, nan=0.0)
        max_val = float(np.max(heatmap))
        if max_val > 1e-6:
            normalized = heatmap / max_val
        else:
            normalized = heatmap
        resized = cv2.resize(
            normalized.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        overlay = cv2.applyColorMap(
            np.clip(resized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    def _annotate_points(
        self,
        image: np.ndarray,
        pred_canvas_xy: tuple[float, float],
        gt_canvas_xy: tuple[float, float],
    ) -> np.ndarray:
        pred_point = (int(round(pred_canvas_xy[0])), int(round(pred_canvas_xy[1])))
        gt_point = (int(round(gt_canvas_xy[0])), int(round(gt_canvas_xy[1])))
        cv2.circle(image, gt_point, 8, (0, 255, 0), 2)
        cv2.circle(image, pred_point, 8, (0, 0, 255), 2)
        return image

    def _decode_heatmap(self, heatmap: np.ndarray) -> tuple[float, float]:
        h, w = heatmap.shape
        flat_idx = int(np.argmax(heatmap))
        y = float(flat_idx // w)
        x = float(flat_idx % w)
        scale = self.config.input_size / self.config.heatmap_size
        offset_x = 0.5 if 0.0 < x < (w - 1) else 0.0
        offset_y = 0.5 if 0.0 < y < (h - 1) else 0.0
        return (x + offset_x) * scale, (y + offset_y) * scale


class ModelEma:
    """Maintain an exponential moving average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        *,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie within (0, 1)")

        self.decay = float(decay)
        self.device = device
        self.ema_model = copy.deepcopy(model)
        target_device = device if device is not None else next(model.parameters()).device
        self.ema_model.to(device=target_device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1.0 - self.decay
                )
            for ema_buffer, model_buffer in zip(
                self.ema_model.buffers(), model.buffers()
            ):
                ema_buffer.copy_(model_buffer)

    def state_dict(self) -> dict:
        return {"model": self.ema_model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if "model" in state:
            self.ema_model.load_state_dict(state["model"])
        else:
            self.ema_model.load_state_dict(state)


def split_backbone_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Separate backbone parameters from the rest of the model."""

    backbone_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        backbone_params = list(backbone.parameters())
        backbone_param_ids = {id(param) for param in backbone_params}
        other_params = [
            param
            for param in model.parameters()
            if id(param) not in backbone_param_ids
        ]
    else:
        other_params = list(model.parameters())

    return backbone_params, other_params


def maybe_freeze_backbone(model: nn.Module, freeze_epochs: int) -> bool:
    """Freeze the backbone parameters when a freeze schedule is requested."""

    if freeze_epochs > 0 and hasattr(model, "backbone"):
        model.backbone.requires_grad_(False)
        return True
    return False


def find_param_group_for_params(
    optimizer: torch.optim.Optimizer, param_ids: set[int]
) -> dict | None:
    """Locate the parameter group that owns the provided parameter ids."""

    for group in optimizer.param_groups:
        if any(id(param) in param_ids for param in group["params"]):
            return group
    return None


def compute_backbone_lr_scale(config: TrainingConfig, epoch: int) -> float:
    """Compute the learning rate scaling factor for the backbone at ``epoch``."""

    if config.backbone_freeze_epochs <= 0:
        return 1.0
    if epoch <= config.backbone_freeze_epochs:
        return 0.0

    warmup = max(config.backbone_unfreeze_warmup, 0)
    if warmup <= 0:
        return 1.0

    progress = min(epoch - config.backbone_freeze_epochs, warmup)
    return progress / warmup


def reset_param_group_lrs_to_base(optimizer: torch.optim.Optimizer) -> None:
    """Restore parameter group learning rates to their stored base values."""

    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", group["lr"])
        group["lr"] = base_lr


def store_param_group_lrs_as_base(optimizer: torch.optim.Optimizer) -> None:
    """Record the current learning rates as the new base for each param group."""

    for group in optimizer.param_groups:
        group["base_lr"] = group.get("lr", group.get("base_lr", 0.0))


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    config: TrainingConfig,
    scaler: GradScaler | None,
    metrics: dict[str, float],
    best_val_loss: float,
    is_best: bool,
    ema: ModelEma | None,
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
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if ema is not None:
        state["ema_state"] = ema.state_dict()

    last_path = checkpoint_dir / "last.pth"
    torch.save(state, last_path)

    if is_best:
        best_path = checkpoint_dir / "best.pth"
        torch.save(state, best_path)


def resolve_resume_checkpoint(config: TrainingConfig) -> Path | None:
    if config.resume_from is not None:
        return config.resume_from

    default_path = config.output_dir / "checkpoints" / "last.pth"
    if default_path.exists():
        return default_path
    return None


def resume_from_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    config: TrainingConfig,
    ema: ModelEma | None,
) -> tuple[int, float]:
    checkpoint_path = resolve_resume_checkpoint(config)
    if checkpoint_path is None:
        return 1, float("inf")

    print(f"Resuming from checkpoint: {checkpoint_path}")
    if hasattr(torch, "serialization") and hasattr(pathlib, "WindowsPath"):
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([pathlib.WindowsPath])
    load_kwargs = {"map_location": "cpu"}
    load_signature = inspect.signature(torch.load)
    if "weights_only" in load_signature.parameters:
        load_kwargs["weights_only"] = False
    state = torch.load(checkpoint_path, **load_kwargs)

    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    device = config.device()
    for value in optimizer.state.values():
        for key, tensor in value.items():
            if isinstance(tensor, torch.Tensor):
                value[key] = tensor.to(device=device, non_blocking=device.type == "cuda")

    if scaler is not None and "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])
    if scheduler is not None and "scheduler_state" in state:
        scheduler.load_state_dict(state["scheduler_state"])
    if ema is not None and "ema_state" in state:
        ema.load_state_dict(state["ema_state"])

    start_epoch = int(state.get("epoch", 0)) + 1
    best_val_loss = float(state.get("best_val_loss", float("inf")))
    if not np.isfinite(best_val_loss):
        metrics = state.get("metrics", {})
        best_val_loss = float(metrics.get("val_loss", float("inf")))

    return max(start_epoch, 1), best_val_loss


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

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def compute_weighted_loss_components(
    loss_map: torch.Tensor, targets: torch.Tensor, config: TrainingConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the combined loss alongside the foreground/background terms."""

    # ``loss_map`` arrives in the autocast dtype (float16 on CUDA).  Upcasting to
    # ``float32`` ensures the reductions are numerically stable without breaking
    # the gradient graph.
    loss_map_f32 = loss_map.to(dtype=torch.float32)
    targets_f32 = targets.to(dtype=torch.float32)

    bg_mask = torch.clamp(1.0 - targets_f32, min=0.0, max=1.0)

    fg_norm = targets_f32.sum(dtype=torch.float32).clamp_min(1e-12)
    bg_norm = bg_mask.sum(dtype=torch.float32).clamp_min(1e-12)

    fg_loss = (loss_map_f32 * targets_f32).sum(dtype=torch.float32) / fg_norm
    bg_loss = (loss_map_f32 * bg_mask).sum(dtype=torch.float32) / bg_norm

    loss = config.heatmap_fg_weight * fg_loss + config.heatmap_bg_weight * bg_loss
    return loss, fg_loss, bg_loss


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    scaler: GradScaler | None,
    config: TrainingConfig,
    epoch: int,
    ema: ModelEma | None,
) -> dict[str, float]:
    model.train()
    loss_meter = 0.0
    fg_loss_meter = 0.0
    bg_loss_meter = 0.0
    weight_meter = 0.0
    pixel_error_sum = 0.0
    pixel_error_count = 0
    pixel_error_values: list[float] = []

    criterion = nn.MSELoss(reduction="none")
    progress = tqdm(dataloader, desc=f"Train {epoch:03d}")
    device = config.device()
    autocast_device_type = device.type
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    update_ema = ema is not None and epoch >= config.ema_start_epoch

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device)
        targets = batch["heatmap"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=autocast_device_type, enabled=config.amp):
            outputs = model(images)
            loss_map = criterion(outputs, targets)

        metrics = compute_metrics(outputs, batch, config)
        batch_pixel_errors = metrics.get("pixel_errors", [])
        if batch_pixel_errors:
            pixel_error_sum += float(sum(batch_pixel_errors))
            pixel_error_count += len(batch_pixel_errors)
            pixel_error_values.extend(float(err) for err in batch_pixel_errors)

        loss, fg_loss, bg_loss = compute_weighted_loss_components(
            loss_map, targets, config
        )

        if scaler is not None and config.amp:
            scaler.scale(loss).backward()
            if config.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, config.gradient_clip_norm
                    )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.gradient_clip_norm is not None:
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, config.gradient_clip_norm
                    )
            optimizer.step()

        if update_ema:
            ema.update(model)

        batch_size = images.size(0)
        loss_meter += loss.detach().item() * batch_size
        fg_loss_meter += fg_loss.detach().item() * batch_size
        bg_loss_meter += bg_loss.detach().item() * batch_size
        weight_meter += batch_size

        if step % config.log_every == 0:
            avg_loss = loss_meter / max(weight_meter, 1e-12)
            running_metrics: dict[str, float] = {"loss": avg_loss}
            running_metrics["loss_fg"] = fg_loss.detach().item()
            running_metrics["loss_bg"] = bg_loss.detach().item()
            if pixel_error_count > 0:
                running_metrics["pixel_mae"] = pixel_error_sum / pixel_error_count
                running_metrics["pixel_median"] = float(
                    np.median(pixel_error_values)
                )
            progress.set_postfix(running_metrics)

    avg_loss = loss_meter / max(weight_meter, 1e-12)
    avg_fg_loss = fg_loss_meter / max(weight_meter, 1e-12)
    avg_bg_loss = bg_loss_meter / max(weight_meter, 1e-12)
    avg_pixel_mae = (
        pixel_error_sum / pixel_error_count if pixel_error_count > 0 else float("nan")
    )
    avg_pixel_median = (
        float(np.median(pixel_error_values)) if pixel_error_values else float("nan")
    )
    return {
        "loss": avg_loss,
        "loss_fg": avg_fg_loss,
        "loss_bg": avg_bg_loss,
        "pixel_mae": avg_pixel_mae,
        "pixel_median": avg_pixel_median,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    *,
    epoch: int | None = None,
    export_samples: bool = True,
) -> dict[str, float]:
    model.eval()
    criterion = nn.MSELoss(reduction="none")

    loss_meter = 0.0
    fg_loss_meter = 0.0
    bg_loss_meter = 0.0
    weight_meter = 0.0
    pixel_errors: List[float] = []
    device = config.device()

    sample_exporter: ValidationSampleExporter | None = None
    if export_samples and config.validation_sample_count > 0:
        export_dir = config.output_dir / "validation_samples"
        if epoch is not None:
            export_dir = export_dir / f"epoch_{epoch:03d}"
        sample_exporter = ValidationSampleExporter(
            export_dir, config, config.validation_sample_count
        )

    for batch in tqdm(dataloader, desc="Validate"):
        images = batch["image"].to(device)
        targets = batch["heatmap"].to(device)

        outputs = model(images)
        loss_map = criterion(outputs, targets)
        loss, fg_loss, bg_loss = compute_weighted_loss_components(
            loss_map, targets, config
        )

        metrics = compute_metrics(outputs, batch, config)
        pixel_errors.extend(metrics.get("pixel_errors", []))

        batch_size = images.size(0)
        loss_meter += loss.detach().item() * batch_size
        fg_loss_meter += fg_loss.detach().item() * batch_size
        bg_loss_meter += bg_loss.detach().item() * batch_size
        weight_meter += batch_size

        if sample_exporter is not None:
            probs = torch.sigmoid(outputs)
            sample_exporter.save_batch(batch, probs)

    avg_loss = loss_meter / max(weight_meter, 1e-12)
    avg_fg_loss = fg_loss_meter / max(weight_meter, 1e-12)
    avg_bg_loss = bg_loss_meter / max(weight_meter, 1e-12)
    avg_pixel_error = float(np.mean(pixel_errors)) if pixel_errors else float("nan")
    if sample_exporter is not None:
        sample_exporter.flush()
    return {
        "val_loss": avg_loss,
        "val_loss_fg": avg_fg_loss,
        "val_loss_bg": avg_bg_loss,
        "val_pixel_mae": avg_pixel_error,
    }


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def validate_config(config: TrainingConfig) -> None:
    if config.input_size <= 0:
        raise ValueError("input_size must be positive")
    if config.heatmap_size <= 0:
        raise ValueError("heatmap_size must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be within (0, 1)")
    if config.heatmap_size > config.input_size:
        raise ValueError("heatmap_size cannot exceed input_size")
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path '{config.output_dir}' exists but is not a directory"
        )
    if config.lr_scheduler_patience < 0:
        raise ValueError("lr_scheduler_patience must be non-negative")
    if not 0.0 < config.lr_scheduler_factor < 1.0:
        raise ValueError("lr_scheduler_factor must be within (0, 1)")
    if config.lr_scheduler_min_lr < 0.0:
        raise ValueError("lr_scheduler_min_lr must be non-negative")
    if config.backbone_freeze_epochs < 0:
        raise ValueError("backbone_freeze_epochs must be non-negative")
    if config.backbone_unfreeze_warmup < 0:
        raise ValueError("backbone_unfreeze_warmup must be non-negative")
    if config.decoder_dropout < 0.0:
        raise ValueError("decoder_dropout must be non-negative")
    if config.decoder_dropout >= 1.0:
        raise ValueError("decoder_dropout must be less than 1.0")
    if config.backbone_lr_multiplier <= 0.0:
        raise ValueError("backbone_lr_multiplier must be positive")
    if config.ema_decay < 0.0 or config.ema_decay >= 1.0:
        raise ValueError("ema_decay must be within [0, 1)")
    if config.ema_start_epoch < 1:
        raise ValueError("ema_start_epoch must be at least 1")


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
        "--backbone-freeze-epochs",
        type=int,
        default=None,
        help="Number of initial epochs to keep the backbone frozen",
    )
    parser.add_argument(
        "--backbone-unfreeze-warmup",
        type=int,
        default=None,
        help="Epochs to linearly warm up the backbone learning rate after unfreezing",
    )
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=None,
        help="Multiplier applied to the learning rate for backbone parameters (default: 0.1)",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=None,
        help="Dropout rate applied in the decoder head (default: 0.0)",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optimizer weight decay")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation fraction (grouped by scene)",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=None,
        help="ReduceLROnPlateau patience before reducing the learning rate",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=None,
        help="Multiplicative factor for ReduceLROnPlateau (default: 0.5)",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=None,
        help="Lower bound for the learning rate in the scheduler",
    )
    parser.add_argument(
        "--heatmap-fg-weight",
        type=float,
        default=None,
        help="Foreground weight for the heatmap loss (default: 5.0)",
    )
    parser.add_argument(
        "--heatmap-bg-weight",
        type=float,
        default=None,
        help="Background weight for the heatmap loss (default: 1.0)",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--input-size", type=int, default=None, help="Square input resolution")
    parser.add_argument("--heatmap-size", type=int, default=None, help="Predicted heatmap resolution")
    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=None,
        help="Gaussian sigma used to render heatmaps (default: 6.0)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--val-sample-count",
        type=int,
        default=None,
        help="Number of validation samples to export as visualisations",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision even when CUDA is available",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="EMA decay factor (set to 0 to disable EMA tracking)",
    )
    parser.add_argument(
        "--ema-start-epoch",
        type=int,
        default=None,
        help="Epoch to begin EMA updates (default: 1)",
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
    if args.backbone_freeze_epochs is not None:
        config.backbone_freeze_epochs = args.backbone_freeze_epochs
    if args.backbone_unfreeze_warmup is not None:
        config.backbone_unfreeze_warmup = args.backbone_unfreeze_warmup
    if args.backbone_lr_multiplier is not None:
        config.backbone_lr_multiplier = args.backbone_lr_multiplier
    if args.decoder_dropout is not None:
        config.decoder_dropout = args.decoder_dropout
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.lr_scheduler_patience is not None:
        config.lr_scheduler_patience = args.lr_scheduler_patience
    if args.lr_scheduler_factor is not None:
        config.lr_scheduler_factor = args.lr_scheduler_factor
    if args.lr_scheduler_min_lr is not None:
        config.lr_scheduler_min_lr = args.lr_scheduler_min_lr
    if args.val_fraction is not None:
        config.val_fraction = args.val_fraction
    if args.heatmap_fg_weight is not None:
        config.heatmap_fg_weight = args.heatmap_fg_weight
    if args.heatmap_bg_weight is not None:
        config.heatmap_bg_weight = args.heatmap_bg_weight
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.seed is not None:
        config.random_seed = args.seed
    if args.input_size is not None:
        config.input_size = args.input_size
    if args.heatmap_size is not None:
        config.heatmap_size = args.heatmap_size
    if args.heatmap_sigma is not None:
        config.heatmap_sigma = args.heatmap_sigma
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.val_sample_count is not None:
        config.validation_sample_count = args.val_sample_count
    if args.no_amp:
        config.amp = False
    if args.ema_decay is not None:
        config.ema_decay = args.ema_decay
    if args.ema_start_epoch is not None:
        config.ema_start_epoch = args.ema_start_epoch

    if args.config_dump:
        print(json.dumps(dataclasses.asdict(config), indent=2, default=str))
        raise SystemExit(0)

    return config


def main() -> None:  # pragma: no cover - CLI entry point
    config = parse_args()
    validate_config(config)
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
        config.backbone_name,
        config.heatmap_size,
        pretrained=config.pretrained,
        decoder_dropout=config.decoder_dropout,
    )
    model.to(config.device())

    ema: ModelEma | None = None
    if 0.0 < config.ema_decay < 1.0:
        ema = ModelEma(model, config.ema_decay, device=config.device())

    backbone_frozen = maybe_freeze_backbone(model, config.backbone_freeze_epochs)
    backbone_params, other_params = split_backbone_parameters(model)
    optimizer_param_groups: list[dict[str, object]] = []
    if other_params:
        optimizer_param_groups.append({"params": other_params})
    if backbone_params:
        backbone_lr = config.learning_rate * config.backbone_lr_multiplier
        optimizer_param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if not optimizer_param_groups:
        raise RuntimeError("No parameters were found for optimisation")

    optimizer = torch.optim.AdamW(
        optimizer_param_groups, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])

    backbone_param_ids = {id(param) for param in backbone_params}
    backbone_group = (
        find_param_group_for_params(optimizer, backbone_param_ids)
        if backbone_param_ids
        else None
    )
    if backbone_group is not None:
        backbone_group.setdefault("base_lr", backbone_group["lr"])
        if backbone_frozen:
            backbone_group["lr"] = 0.0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_scheduler_patience,
        factor=config.lr_scheduler_factor,
        min_lr=config.lr_scheduler_min_lr,
    )
    scaler = GradScaler(enabled=config.amp)

    history_path = config.output_dir / "training_history.json"
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            history: list[dict[str, float]] = json.load(f)
    else:
        history = []

    history_best_candidates = [
        float(entry.get("val_loss", float("inf"))) for entry in history
    ]
    best_val_loss = min(history_best_candidates, default=float("inf"))

    start_epoch, checkpoint_best = resume_from_checkpoint(
        model, optimizer, scaler, scheduler, config, ema
    )
    if backbone_param_ids:
        backbone_group = find_param_group_for_params(optimizer, backbone_param_ids)
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
    if backbone_group is not None and hasattr(model, "backbone"):
        if config.backbone_freeze_epochs > 0 and start_epoch <= config.backbone_freeze_epochs:
            model.backbone.requires_grad_(False)
            backbone_frozen = True
            backbone_group["lr"] = 0.0
        else:
            model.backbone.requires_grad_(True)
            backbone_frozen = False
    if np.isfinite(checkpoint_best):
        best_val_loss = min(best_val_loss, checkpoint_best)

    if start_epoch > 1:
        history = [
            entry for entry in history if int(entry.get("epoch", 0)) < start_epoch
        ]

    for epoch in range(start_epoch, config.num_epochs + 1):
        if backbone_group is not None and hasattr(model, "backbone"):
            lr_scale = compute_backbone_lr_scale(config, epoch)
            base_lr = backbone_group.get("base_lr", backbone_group["lr"])
            if lr_scale <= 0.0:
                if not backbone_frozen:
                    model.backbone.requires_grad_(False)
                    backbone_frozen = True
                backbone_group["lr"] = 0.0
            else:
                if backbone_frozen:
                    model.backbone.requires_grad_(True)
                    backbone_frozen = False
                backbone_group["lr"] = base_lr * lr_scale
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, scaler, config, epoch, ema
        )

        use_ema_for_eval = ema is not None and epoch >= config.ema_start_epoch
        eval_model = ema.ema_model if use_ema_for_eval else model
        val_metrics = validate(
            eval_model,
            val_loader,
            config,
            epoch=epoch,
            export_samples=True,
        )
        if use_ema_for_eval:
            raw_val_metrics = validate(
                model,
                val_loader,
                config,
                epoch=None,
                export_samples=False,
            )
            val_metrics.update({f"raw_{k}": v for k, v in raw_val_metrics.items()})

        combined = {"epoch": epoch}
        combined.update({f"train_{k}": v for k, v in train_metrics.items()})
        combined.update(val_metrics)
        combined["heatmap_fg_weight"] = config.heatmap_fg_weight
        combined["heatmap_bg_weight"] = config.heatmap_bg_weight
        history.append(combined)

        message = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_pixel_mae={train_metrics['pixel_mae']:.2f} | "
            f"train_pixel_median={train_metrics['pixel_median']:.2f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_pixel_mae={val_metrics['val_pixel_mae']:.2f}"
        )
        if "raw_val_loss" in val_metrics and "raw_val_pixel_mae" in val_metrics:
            message += (
                f" | raw_val_loss={val_metrics['raw_val_loss']:.4f}"
                f" | raw_val_pixel_mae={val_metrics['raw_val_pixel_mae']:.2f}"
            )
        print(message)

        current_val_loss = float(val_metrics.get("val_loss", float("inf")))
        is_best = current_val_loss < best_val_loss
        reset_param_group_lrs_to_base(optimizer)
        if scheduler is not None and np.isfinite(current_val_loss):
            scheduler.step(current_val_loss)
        store_param_group_lrs_as_base(optimizer)
        if is_best:
            best_val_loss = current_val_loss

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            scaler,
            val_metrics,
            best_val_loss,
            is_best,
            ema,
        )

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()

