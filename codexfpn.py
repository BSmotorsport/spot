import csv
import json
import math
import os
import random
import re
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================


class Config:
    """Centralised hyperparameters for the coordinate regression pipeline.

    ``MODEL_NAME`` must be one of :pyattr:`BACKBONE_CHOICES`.  The curated set
    mixes the original ConvNeXt baseline with global-context transformer models
    (ViT/Swin) and lighter CNN alternatives (EfficientNet) so experiments can
    quickly compare architectural families.  The helper :py:meth:`image_size`
    adapts preprocessing to the recommended resolution for the selected
    backbone.
    """

    # Paths
    DATASET_PATH: str = r"E:\\BOTB\\dataset\\aug"
    OUTPUT_DIR: str = "./training_output_coordinate"
    MODEL_VERSION: str = "coordinate_regression_v1"

    # Image geometry
    ORIGINAL_WIDTH: int = 4416
    ORIGINAL_HEIGHT: int = 3336
    DEFAULT_IMAGE_SIZE: int = 1536

    BACKBONE_CHOICES: dict[str, dict[str, object]] = {
        "convnext_base.fb_in22k_ft_in1k": {
            "image_size": 1536,
            "summary": "Baseline ConvNeXt (strong CNN inductive biases)",
        },
        "vit_base_patch16_384": {
            "image_size": 384,
            "summary": "Vision Transformer with global token mixing",
        },
        "swin_base_patch4_window12_384": {
            "image_size": 384,
            "summary": "Swin Transformer with shifted windows",
        },
        "efficientnet_b4": {
            "image_size": 380,
            "summary": "Lightweight CNN with squeeze-excite blocks",
        },
    }

    # Optimisation
    MODEL_NAME: str = "convnext_base.fb_in22k_ft_in1k"
    BATCH_SIZE: int = 2
    EPOCHS: int = 200
    INITIAL_LR: float = 1e-4
    BACKBONE_LR: float = 2e-5

    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 4
    RANDOM_SEED: int = 42
    GRAD_ACCUM_STEPS: int = 8
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    AMP: bool = torch.cuda.is_available()
    USE_CHANNELS_LAST: bool = True

    DROP_PATH_RATE: float = 0.1
    HEAD_DROPOUT: float = 0.15

    MIXUP_ALPHA: float = 0.2
    CUTMIX_ALPHA: float = 0.0
    MIXUP_PROB: float = 0.15
    MIXUP_SWITCH_PROB: float = 0.0

    MIXUP_MODE: str = "batch"
    MIXUP_START_EPOCH: int = 30


    LR_WARMUP_EPOCHS: int = 10
    LR_WARMUP_START_FACTOR: float = 0.2
    AUG_WARMUP_EPOCHS: int = 20

    NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Training stages
    FREEZE_BACKBONE_EPOCHS: int = 0

    # Loss
    PIXEL_LOSS_WEIGHT: float = 0.4
    LOSS_MIN_VARIANCE: float = 0.01
    LOSS_MAX_VARIANCE: float = 10.0
    LOSS_VARIANCE_REG_WEIGHT: float = 0.01
    USE_UNCERTAINTY: bool = False

    # Context metadata / auxiliary cues
    CONTEXT_METADATA_ENABLED: bool = False
    CONTEXT_METADATA_SUFFIX: str = ".context.json"
    CONTEXT_VECTOR_SIZE: int = 128
    CONTEXT_DEFAULT_FILL_VALUE: float = 0.0
    CONTEXT_HIDDEN_DIM: int = 256
    USE_CONTEXT_BRANCH: bool = False

    # Validation utilities
    VALIDATE_EVERY: int = 1
    NUM_VAL_SAMPLES: int = 6

    # Checkpointing
    CHECKPOINT_FILENAME: str = "coordinate_regressor.pth"

    @classmethod
    def validate_model_name(cls, name: Optional[str] = None) -> str:
        model_name = name or cls.MODEL_NAME
        if model_name not in cls.BACKBONE_CHOICES:
            available = ", ".join(sorted(cls.BACKBONE_CHOICES))
            raise ValueError(
                f"Unknown MODEL_NAME '{model_name}'. Available options: {available}"
            )
        return model_name

    @classmethod
    def image_size(cls, model_name: Optional[str] = None) -> int:
        resolved = cls.validate_model_name(model_name)
        metadata = cls.BACKBONE_CHOICES.get(resolved, {})
        return int(metadata.get("image_size", cls.DEFAULT_IMAGE_SIZE))

    @classmethod
    def describe_backbones(cls) -> str:
        """Return a human-readable summary of available backbones."""

        lines = []
        for key, meta in cls.BACKBONE_CHOICES.items():
            summary = meta.get("summary", "")
            size = meta.get("image_size", cls.DEFAULT_IMAGE_SIZE)
            lines.append(f"- {key}: {summary} (recommended {size}px square)")
        return "\n".join(lines)


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


def _context_metadata_path(image_path: str) -> str:
    suffix = Config.CONTEXT_METADATA_SUFFIX
    base, ext = os.path.splitext(image_path)
    if suffix.startswith("."):
        return base + suffix
    return image_path + suffix


def _flatten_numeric_structure(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).ravel().tolist()
    if isinstance(value, (list, tuple, set)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_structure(item))
        return flattened
    if isinstance(value, dict):
        flattened: list[float] = []
        for key in sorted(value):
            flattened.extend(_flatten_numeric_structure(value[key]))
        return flattened
    return []


def _read_context_metadata(meta_path: str) -> list[float]:
    try:
        extension = os.path.splitext(meta_path)[1].lower()
        if extension == ".json":
            with open(meta_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return _flatten_numeric_structure(payload)
        if extension == ".npz":
            data = np.load(meta_path)
            flattened: list[float] = []
            for key in sorted(data.files):
                flattened.extend(_flatten_numeric_structure(data[key]))
            return flattened
        array = np.load(meta_path, allow_pickle=False)
        return _flatten_numeric_structure(array)
    except FileNotFoundError:
        return []
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"⚠️ Could not parse context metadata from {meta_path}: {exc}")
        return []


def get_context_features(image_path: str) -> Optional[torch.Tensor]:
    if not Config.CONTEXT_METADATA_ENABLED:
        return None

    vector_length = max(int(Config.CONTEXT_VECTOR_SIZE), 0)
    if vector_length == 0:
        return None

    meta_path = _context_metadata_path(image_path)
    values = _read_context_metadata(meta_path)

    if not values:
        if Config.CONTEXT_DEFAULT_FILL_VALUE == 0.0:
            return torch.zeros(vector_length, dtype=torch.float32)
        fill = float(Config.CONTEXT_DEFAULT_FILL_VALUE)
        return torch.full((vector_length,), fill, dtype=torch.float32)

    truncated = values[:vector_length]
    context_tensor = torch.zeros(vector_length, dtype=torch.float32)
    context_tensor[: len(truncated)] = torch.tensor(truncated, dtype=torch.float32)
    return context_tensor


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
        image_size = Config.image_size()
        if self.transform is not None:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed.get("keypoints", [])

            if not keypoints:
                # Albumentations can drop keypoints when they leave the frame. To
                # keep batches stable we simply return the original sample.
                return self.__getitem__((idx + 1) % len(self.valid_paths))

            clamped = []
            limit = float(np.nextafter(image_size, -np.inf))
            for kp in keypoints:
                x, y = kp[:2]
                if 0.0 <= x < image_size and 0.0 <= y < image_size:
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

        if tensor_image.shape[-2:] != (image_size, image_size):
            tensor_image = torch.nn.functional.interpolate(
                tensor_image.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        image_extent = float(image_size - 1)
        keypoint = torch.tensor(keypoints[0], dtype=torch.float32)
        normalised = keypoint / image_extent
        normalised = normalised.clamp(0.0, 1.0)

        context_tensor = get_context_features(img_path)

        return tensor_image, normalised, context_tensor, img_path


# ======================================================================================
# 5. MODEL
# ======================================================================================


class GeM(nn.Module):
    """Generalised mean pooling with learnable exponent."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        p = self.p.clamp(min=self.eps)
        x = x.pow(p)
        _, _, h, w = x.shape
        pooled = F.avg_pool2d(x, kernel_size=(h, w))
        return pooled.pow(1.0 / p)


class CoordinateRegressionModel(nn.Module):
    """Backbone + attention-pooled head that regresses normalised coordinates."""

    def __init__(self, backbone_name: str, pretrained: bool = True, hidden_dim: int = 512):
        super().__init__()
        self.backbone_name = Config.validate_model_name(backbone_name)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            raise AttributeError("Backbone does not expose num_features")

        reduction_channels = max(feature_dim // 8, 1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, reduction_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gem_pool = GeM()

        self.context_dim = 0
        self.context_hidden_dim = 0
        self.context_encoder: Optional[nn.Module] = None
        if Config.USE_CONTEXT_BRANCH and Config.CONTEXT_VECTOR_SIZE > 0:
            self.context_dim = int(Config.CONTEXT_VECTOR_SIZE)
            self.context_hidden_dim = int(Config.CONTEXT_HIDDEN_DIM)
            self.context_encoder = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, self.context_hidden_dim),
                nn.GELU(),
                nn.Dropout(Config.HEAD_DROPOUT),
            )

        head_input_dim = feature_dim * 3 + self.context_hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Dropout(Config.HEAD_DROPOUT),
            nn.Linear(head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.HEAD_DROPOUT),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(x)
        if features.dim() == 3:
            # Some transformer-style backbones return (B, N, C). Reshape back to
            # a spatial map by assuming a square token layout.  Drop class or
            # distillation tokens until we obtain a square number of patches.
            b, n, c = features.shape
            spatial_tokens = features
            side = int(math.sqrt(n))
            if side * side != n:
                trimmed = False
                for drop in (1, 2):
                    candidate = n - drop
                    if candidate <= 0:
                        continue
                    side_candidate = int(math.sqrt(candidate))
                    if side_candidate * side_candidate == candidate:
                        spatial_tokens = features[:, drop:, :]
                        side = side_candidate
                        trimmed = True
                        break
                if not trimmed and side * side != n:
                    raise ValueError(
                        f"Unable to reshape transformer tokens from {self.backbone_name}: "
                        f"{n} tokens do not form a square map"
                    )
            features = spatial_tokens.transpose(1, 2).reshape(b, c, side, side)

        attn = self.spatial_attention(features)
        features = features * attn

        avg = self.avg_pool(features)
        maxv = self.max_pool(features)
        gem = self.gem_pool(features)
        pooled = torch.cat([avg, maxv, gem], dim=1).flatten(1)

        fused = pooled
        if self.context_encoder is not None:
            batch_size = pooled.size(0)
            if context is None:
                fill = float(Config.CONTEXT_DEFAULT_FILL_VALUE)
                context = torch.full(
                    (batch_size, self.context_dim),
                    fill,
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
            else:
                if context.dim() == 1:
                    context = context.unsqueeze(0)
                if context.size(0) != batch_size:
                    raise ValueError(
                        "Context tensor batch dimension does not match images"
                    )
                if context.size(-1) > self.context_dim:
                    context = context[:, : self.context_dim]
                elif context.size(-1) < self.context_dim:
                    pad = torch.full(
                        (batch_size, self.context_dim - context.size(-1)),
                        float(Config.CONTEXT_DEFAULT_FILL_VALUE),
                        device=context.device,
                        dtype=context.dtype,
                    )
                    context = torch.cat([context, pad], dim=-1)
                context = context.to(device=pooled.device, dtype=pooled.dtype)

            context_emb = self.context_encoder(context)
            fused = torch.cat([pooled, context_emb], dim=1)

        output = self.head(fused)
        coords = torch.sigmoid(output[:, :2])
        if Config.USE_UNCERTAINTY:
            log_uncertainty = output[:, 2:]
            max_log = float(math.log(Config.LOSS_MAX_VARIANCE)) if Config.LOSS_MAX_VARIANCE > 0 else 10.0
            min_log = float(math.log(Config.LOSS_MIN_VARIANCE)) if Config.LOSS_MIN_VARIANCE > 0 else -10.0
            uncertainty = torch.exp(log_uncertainty.clamp(min=min_log, max=max_log))
        else:
            uncertainty = torch.ones_like(coords)
        return coords, uncertainty


def create_model(pretrained: bool = True) -> CoordinateRegressionModel:
    backbone_name = Config.validate_model_name()
    model = CoordinateRegressionModel(backbone_name, pretrained=pretrained)
    return model


# ======================================================================================
# 6. LOSS & METRICS
# ======================================================================================


class CoordinateLoss(nn.Module):
    """Smooth L1 + pixel MAE encourages precise coordinate regression.

    When the model predicts per-axis uncertainties, the loss down-weights the
    contribution of samples with high variance while regularising the
    uncertainty via a logarithmic penalty.  The class still accepts raw
    coordinate tensors for backwards compatibility with lighter baselines and
    unit tests.
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        min_variance: float = 1e-4,
        max_variance: float = 10.0,
        variance_reg_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.pixel_weight = float(pixel_weight)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.variance_reg_weight = float(max(variance_reg_weight, 0.0))

    def forward(self, pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], target: torch.Tensor):
        coords: torch.Tensor
        uncertainty: Optional[torch.Tensor]
        if isinstance(pred, tuple):
            coords, uncertainty = pred
        else:
            coords = pred
            uncertainty = None

        coords = coords.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        smooth = self.smooth_l1(coords, target)
        smooth_mean = smooth.mean()

        coord_term: torch.Tensor
        if uncertainty is not None:
            variance = uncertainty.to(dtype=torch.float32)
            variance = variance.clamp(min=self.min_variance, max=self.max_variance)
            scaled = (smooth / variance).mean() + torch.log(variance).mean()
            if self.variance_reg_weight > 0.0:
                variance_reg = self.variance_reg_weight * (variance.reciprocal()).mean()
                coord_term = scaled + variance_reg
            else:
                coord_term = scaled
        else:
            coord_term = smooth_mean

        image_extent = max(float(Config.image_size() - 1), 1.0)
        extent_tensor = coords.new_tensor([
            max(float(Config.ORIGINAL_WIDTH - 1), 1.0),
            max(float(Config.ORIGINAL_HEIGHT - 1), 1.0),
        ])

        pixel_errors_original = torch.abs(coords - target) * extent_tensor
        pixel_mae_original = pixel_errors_original.mean()

        pixel_errors_resized = torch.abs(coords - target) * image_extent
        pixel_mae_resized = pixel_errors_resized.mean()

        loss = coord_term + self.pixel_weight * (pixel_mae_resized / image_extent)
        return (
            loss,
            smooth_mean.detach(),
            pixel_mae_original.detach(),
        )


# ======================================================================================
# 7. TRAINING UTILITIES
# ======================================================================================


def build_train_loader(
    train_files: Sequence[str],
    train_transform: A.BasicTransform,
) -> DataLoader:
    train_dataset = FootballDataset(train_files, transform=train_transform)

    return DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )


def build_val_loader(
    val_files: Sequence[str],
    val_transform: A.BasicTransform,
) -> DataLoader:
    val_dataset = FootballDataset(val_files, transform=val_transform, augment=False)

    return DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def prepare_dataloaders(
    train_files: Sequence[str],
    val_files: Sequence[str],
    train_transform: A.BasicTransform,
    val_transform: A.BasicTransform,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = build_train_loader(train_files, train_transform)
    val_loader = build_val_loader(val_files, val_transform)
    return train_loader, val_loader


class MixupCutmix:
    """Apply mixup or cutmix augmentation to batches when enabled."""

    def __init__(
        self,
        mixup_alpha: float,
        cutmix_alpha: float,
        prob: float,
        switch_prob: float,
        mode: str = "batch",
    ) -> None:
        self.mixup_alpha = max(mixup_alpha, 0.0)
        self.cutmix_alpha = max(cutmix_alpha, 0.0)
        self.prob = max(prob, 0.0)
        self.switch_prob = min(max(switch_prob, 0.0), 1.0)
        self.mode = mode

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.size(0) < 2:
            return images, targets
        if random.random() >= self.prob:
            return images, targets

        use_cutmix = False
        if self.cutmix_alpha > 0.0 and self.mixup_alpha > 0.0:
            use_cutmix = random.random() < self.switch_prob
        elif self.cutmix_alpha > 0.0:
            use_cutmix = True

        if use_cutmix:
            return self._apply_cutmix(images, targets)
        if self.mixup_alpha > 0.0:
            return self._apply_mixup(images, targets)
        return images, targets

    def _sample_lambda(self, alpha: float) -> float:
        lam = np.random.beta(alpha, alpha)
        return float(np.clip(lam, 0.0, 1.0))

    def _apply_mixup(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = self._sample_lambda(self.mixup_alpha)
        index = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[index]
        mixed_targets = lam * targets + (1.0 - lam) * targets[index]
        return mixed_images, mixed_targets

    def _apply_cutmix(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = self._sample_lambda(self.cutmix_alpha)
        index = torch.randperm(images.size(0), device=images.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape, lam)
        if bbx1 >= bbx2 or bby1 >= bby2:
            return self._apply_mixup(images, targets) if self.mixup_alpha > 0.0 else (images, targets)

        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        shuffled_targets = targets[index]

        width = images.size(-1)
        height = images.size(-2)
        scale = torch.tensor(
            [max(width - 1, 1), max(height - 1, 1)],
            dtype=targets.dtype,
            device=targets.device,
        )

        primary_pixels = targets * scale
        secondary_pixels = shuffled_targets * scale

        def _in_bbox(pixel_coords: torch.Tensor) -> torch.Tensor:
            return (
                (pixel_coords[:, 0] >= bbx1)
                & (pixel_coords[:, 0] < bbx2)
                & (pixel_coords[:, 1] >= bby1)
                & (pixel_coords[:, 1] < bby2)
            )

        primary_in_patch = _in_bbox(primary_pixels)
        secondary_in_patch = _in_bbox(secondary_pixels)

        mixed_targets = targets.clone()

        # When the original keypoint lies inside the pasted region but the replacement sample
        # does not provide a visible keypoint, revert the patch for those batch elements to
        # avoid mismatched labels.
        invalid_mask = primary_in_patch & ~secondary_in_patch
        if torch.any(invalid_mask):
            mixed_images[invalid_mask, :, bby1:bby2, bbx1:bbx2] = images[
                invalid_mask, :, bby1:bby2, bbx1:bbx2
            ]
            primary_in_patch = primary_in_patch & ~invalid_mask
            secondary_in_patch = secondary_in_patch & ~invalid_mask

        replace_mask = primary_in_patch & secondary_in_patch
        mixed_targets[replace_mask] = shuffled_targets[replace_mask]
        return mixed_images, mixed_targets

    @staticmethod
    def _rand_bbox(size: Sequence[int], lam: float) -> Tuple[int, int, int, int]:
        _, _, height, width = size
        cut_ratio = math.sqrt(max(1.0 - lam, 0.0))
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = random.randint(0, max(width - 1, 0))
        cy = random.randint(0, max(height - 1, 0))

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        return x1, y1, x2, y2


def create_mixup_cutmix_fn() -> Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    if Config.MIXUP_PROB <= 0.0:
        return None
    if Config.MIXUP_ALPHA <= 0.0 and Config.CUTMIX_ALPHA <= 0.0:
        return None
    return MixupCutmix(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=Config.MIXUP_SWITCH_PROB,
        mode=Config.MIXUP_MODE,
    )


def compute_pixel_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    scale = pred.new_tensor(
        [
            max(float(Config.ORIGINAL_WIDTH - 1), 1.0),
            max(float(Config.ORIGINAL_HEIGHT - 1), 1.0),
        ]
    )
    diff = (pred - target).detach()
    scaled_diff = diff * scale
    mae = scaled_diff.abs().mean().item()
    rmse = torch.sqrt((scaled_diff ** 2).mean()).item()
    return mae, rmse


def compute_percentile_errors(
    errors: Union[torch.Tensor, Sequence[float]],
    percentiles: Sequence[float] = (50, 75, 90, 95),
) -> Dict[str, float]:
    """Return key percentiles from a collection of pixel errors."""

    if isinstance(errors, torch.Tensor):
        values = errors.detach().flatten().to(dtype=torch.float32)
    else:
        values = torch.tensor(list(errors), dtype=torch.float32)

    if values.numel() == 0:
        return {f"p{int(p)}": float("nan") for p in percentiles}

    array = values.cpu().numpy()
    return {f"p{int(p)}": float(np.percentile(array, p)) for p in percentiles}


def save_validation_csv(
    rows: Sequence[Dict[str, Union[str, float, int]]],
    epoch: int,
    output_dir: str,
) -> Optional[str]:
    if not rows:
        return None

    os.makedirs(output_dir, exist_ok=True)
    filename = f"val_pixel_errors_epoch_{epoch + 1:03d}.csv"
    path = os.path.join(output_dir, filename)

    fieldnames = [
        "epoch",
        "image_path",
        "target_x",
        "target_y",
        "pred_x",
        "pred_y",
        "error_x",
        "error_y",
        "error_euclidean",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CoordinateLoss,
    scaler: GradScaler,
    epoch: int,

    mixup_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,

) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_coord = 0.0
    total_pixel = 0.0

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"Train {epoch+1}", colour="green")
    accumulation = 0

    for batch in progress:
        if len(batch) == 4:
            images, targets, context, _ = batch
        else:
            images, targets, _ = batch
            context = None
        images = images.to(Config.DEVICE, non_blocking=True)
        targets = targets.to(Config.DEVICE, non_blocking=True)
        if context is not None:
            context = context.to(Config.DEVICE, non_blocking=True)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        if Config.USE_CHANNELS_LAST:
            images = images.to(memory_format=torch.channels_last)

        with autocast(device_type="cuda" if Config.DEVICE.startswith("cuda") else "cpu", enabled=Config.AMP):
            coords, uncertainty = model(images, context)
            loss, coord_loss, pixel_mae = criterion((coords, uncertainty), targets)

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
    csv_dir: Optional[str] = None,
) -> Tuple[float, float, float, Dict[str, float], Optional[str]]:
    model.eval()
    total_loss = 0.0
    total_coord = 0.0
    total_pixel = 0.0
    pixel_errors: list[float] = []
    csv_rows: list[Dict[str, Union[str, float, int]]] = []

    progress = tqdm(loader, desc=f"Val {epoch+1}", colour="yellow")

    all_preds: list[torch.Tensor] = []

    base_scale = torch.tensor(
        [
            max(float(Config.ORIGINAL_WIDTH - 1), 1.0),
            max(float(Config.ORIGINAL_HEIGHT - 1), 1.0),
        ],
        device=Config.DEVICE,
    )

    with torch.no_grad():
        for batch in progress:
            if len(batch) == 4:
                images, targets, context, paths = batch
            else:
                images, targets, paths = batch
                context = None
            images = images.to(Config.DEVICE, non_blocking=True)
            targets = targets.to(Config.DEVICE, non_blocking=True)
            if context is not None:
                context = context.to(Config.DEVICE, non_blocking=True)
            if Config.USE_CHANNELS_LAST:
                images = images.to(memory_format=torch.channels_last)

            with autocast(device_type="cuda" if Config.DEVICE.startswith("cuda") else "cpu", enabled=Config.AMP):
                coords, uncertainty = model(images, context)
                loss, coord_loss, pixel_mae = criterion((coords, uncertainty), targets)

            total_loss += loss.item()
            total_coord += coord_loss.item()
            total_pixel += pixel_mae.item()

            all_preds.append(coords.detach().cpu())

            scale = base_scale.to(device=coords.device, dtype=coords.dtype)
            diff = coords - targets
            scaled_diff = diff * scale
            batch_errors = torch.linalg.vector_norm(scaled_diff, dim=-1)
            if batch_errors.ndim == 0:
                batch_errors = batch_errors.unsqueeze(0)
            pixel_errors.extend(batch_errors.detach().cpu().tolist())

            if csv_dir is not None:
                preds_original = (coords * scale).detach().cpu()
                targets_original = (targets * scale).detach().cpu()
                per_axis_errors = scaled_diff.abs().detach().cpu()
                euclidean_errors = batch_errors.detach().cpu()

                for idx, path in enumerate(paths):
                    csv_rows.append(
                        {
                            "epoch": epoch + 1,
                            "image_path": path,
                            "target_x": float(targets_original[idx, 0]),
                            "target_y": float(targets_original[idx, 1]),
                            "pred_x": float(preds_original[idx, 0]),
                            "pred_y": float(preds_original[idx, 1]),
                            "error_x": float(per_axis_errors[idx, 0]),
                            "error_y": float(per_axis_errors[idx, 1]),
                            "error_euclidean": float(euclidean_errors[idx]),
                        }
                    )

            percentiles = compute_percentile_errors(pixel_errors)
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "pix": f"{pixel_mae.item():.2f}",
                    **{k: f"{v:.1f}" for k, v in percentiles.items()},
                }
            )

    batches = max(len(loader), 1)
    avg_loss = total_loss / batches
    avg_coord = total_coord / batches
    avg_pixel = total_pixel / batches
    percentile_metrics = compute_percentile_errors(pixel_errors)

    csv_path = None
    if csv_dir is not None:
        csv_path = save_validation_csv(csv_rows, epoch, csv_dir)

    if all_preds:
        stacked = torch.cat(all_preds, dim=0)
        pred_mean = stacked.mean(dim=0)
        pred_std = stacked.std(dim=0)
        print(
            "Prediction stats - Mean: "
            f"{pred_mean.tolist()} Std: {pred_std.tolist()}"
        )
        if float(pred_std.mean()) < 0.05:
            print("⚠️ Warning: Predictions may have collapsed to a single point")

    return avg_loss, avg_coord, avg_pixel, percentile_metrics, csv_path


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
    batch = next(iterator)
    if len(batch) == 4:
        images, targets, context, paths = batch
    else:
        images, targets, paths = batch
        context = None
    images = images.to(Config.DEVICE)
    if context is not None:
        context = context.to(Config.DEVICE)
    if Config.USE_CHANNELS_LAST:
        images = images.to(memory_format=torch.channels_last)

    with torch.no_grad():
        preds, uncertainties = model(images, context)

    preds = preds.cpu()
    uncertainties = uncertainties.cpu()
    targets = targets.cpu()
    original_scale = torch.tensor(
        [
            max(float(Config.ORIGINAL_WIDTH - 1), 1.0),
            max(float(Config.ORIGINAL_HEIGHT - 1), 1.0),
        ],
        dtype=torch.float32,
    ).to(dtype=targets.dtype)
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

        display_scale = torch.tensor(
            [
                max(float(img_np.shape[1] - 1), 1.0),
                max(float(img_np.shape[0] - 1), 1.0),
            ],
            dtype=torch.float32,
        ).to(dtype=targets.dtype)

        ax = axes[idx]
        ax.imshow(img_np)

        tgt_display = targets[idx] * display_scale
        prd_display = preds[idx] * display_scale
        tgt = targets[idx] * original_scale
        prd = preds[idx] * original_scale
        unc = uncertainties[idx]
        sigma = torch.sqrt(unc.clamp(min=1e-6)) * original_scale
        pixel_error = float(math.dist(prd.tolist(), tgt.tolist()))

        ax.scatter([tgt_display[0]], [tgt_display[1]], c="lime", marker="o", label="target")
        ax.scatter([prd_display[0]], [prd_display[1]], c="red", marker="x", label="pred")
        ax.set_xlim(0.0, float(display_scale[0].item()))
        ax.set_ylim(float(display_scale[1].item()), 0.0)
        filename = os.path.basename(paths[idx])
        ax.set_title(f"{filename} | Δpx: {pixel_error:.2f}")
        ax.axis("off")
        ax.legend(loc="upper right")

        annotation = (
            f"Target: ({tgt[0]:.1f}, {tgt[1]:.1f})\n"
            f"Pred:   ({prd[0]:.1f}, {prd[1]:.1f})\n"
            f"σ:      ({sigma[0]:.1f}, {sigma[1]:.1f}) px\n"
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
        if metrics.get("val_percentiles"):
            percentile_summary = " | ".join(
                f"{key.upper()}: {value:.1f} px" for key, value in metrics["val_percentiles"].items()
            )
            summary_lines.append(f"Val Percentiles -> {percentile_summary}")
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


def build_transforms() -> dict:
    image_size = Config.image_size()
    random_resized_crop_signature = inspect_signature(A.RandomResizedCrop)
    random_resized_crop_candidates: list[dict] = []
    if random_resized_crop_signature is None:
        # Prefer modern ``size`` signatures but keep height/width as a last resort.
        random_resized_crop_candidates.extend(
            [
                {"size": image_size},
                {"size": (image_size, image_size)},
                {"size": [image_size, image_size]},
                {"height": image_size, "width": image_size},
            ]
        )
    else:
        if "size" in random_resized_crop_signature:
            random_resized_crop_candidates.extend(
                [
                    {"size": image_size},
                    {"size": (image_size, image_size)},
                    {"size": [image_size, image_size]},
                ]
            )
        if {"height", "width"}.issubset(random_resized_crop_signature):
            random_resized_crop_candidates.append(
                {"height": image_size, "width": image_size}
            )

    random_resized_crop = instantiate_albumentations_transform(
        A.RandomResizedCrop,
        dict(scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=cv2.INTER_CUBIC),

        random_resized_crop_candidates,
    )

    affine = instantiate_albumentations_transform(
        A.Affine,
        dict(scale=(0.95, 1.05), rotate=(-4, 4), shear=(-2, 2), fit_output=False, p=0.2),
        [
            {"cval": 0, "mode": cv2.BORDER_REFLECT_101},
            {"value": 0, "border_mode": cv2.BORDER_REFLECT_101},
            {},
        ],
    )

    perspective = instantiate_albumentations_transform(
        A.Perspective,
        dict(scale=(0.01, 0.03), keep_size=True, p=0.1),
        [
            {"pad_mode": cv2.BORDER_REFLECT_101},
            {"border_mode": cv2.BORDER_REFLECT_101},
            {},
        ],
    )

    gauss_noise = instantiate_albumentations_transform(
        A.GaussNoise,
        dict(p=0.15),
        [
            {"var_limit": (0.0, 10.0)},
            {"sigma_limit": (0.0, math.sqrt(10.0))},
        ],
    )
    if hasattr(gauss_noise, "var_limit"):
        gauss_noise.var_limit = (0.0, 6.0)
    elif hasattr(gauss_noise, "sigma_limit"):
        gauss_noise.sigma_limit = (0.0, math.sqrt(6.0))

    train_transform = A.Compose(
        [
            A.OneOf([random_resized_crop, A.Resize(image_size, image_size)], p=0.75),
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            affine,
            perspective,
            A.RandomBrightnessContrast(0.1, 0.1, p=0.35),
            A.ColorJitter(0.08, 0.08, 0.08, 0.04, p=0.2),
            A.HueSaturationValue(12, 18, 12, p=0.35),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),

                ],
                p=0.2,
            ),
            gauss_noise,
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )

    warmup_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.1, p=0.35),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return {
        "train": train_transform,
        "warmup": warmup_transform,
        "val": val_transform,
    }


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
    print(f"Input resolution: {Config.image_size()} px")
    print(
        "Learning rates: head={head:.2e}, backbone={backbone:.2e}".format(
            head=Config.INITIAL_LR, backbone=Config.BACKBONE_LR
        )
    )
    print(
        "Cosine annealing eta_min (backbone scale): {eta:.2e}".format(
            eta=Config.BACKBONE_LR * 0.1
        )
    )
    print("Backbone registry (set Config.MODEL_NAME to switch):")
    print(Config.describe_backbones())
    print("=" * 80)

    train_files, val_files = gather_dataset_files()
    transforms = build_transforms()
    full_train_transform = transforms.get("train")
    warmup_transform = transforms.get("warmup")
    val_transform = transforms.get("val")

    if full_train_transform is None or val_transform is None:
        raise ValueError("build_transforms() must provide 'train' and 'val' transforms")

    model = create_model(pretrained=True).to(Config.DEVICE)

    head_params = (
        list(model.head.parameters())
        + list(model.avg_pool.parameters())
        + list(model.max_pool.parameters())
        + list(model.gem_pool.parameters())
    )

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

    warmup_epochs = max(Config.LR_WARMUP_EPOCHS, 0)
    schedulers = []
    milestones = []
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(Config.LR_WARMUP_START_FACTOR, 0.0),
            total_iters=warmup_epochs,
        )
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_epochs)

    cosine_epochs = max(Config.EPOCHS - warmup_epochs, 1)
    min_backbone_lr = Config.BACKBONE_LR * 0.1
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_backbone_lr,
    )
    schedulers.append(cosine_scheduler)

    if len(schedulers) == 1:
        scheduler = cosine_scheduler
    else:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )

    criterion = CoordinateLoss(
        pixel_weight=Config.PIXEL_LOSS_WEIGHT,
        min_variance=Config.LOSS_MIN_VARIANCE,
        max_variance=Config.LOSS_MAX_VARIANCE,
        variance_reg_weight=Config.LOSS_VARIANCE_REG_WEIGHT,
    )
    scaler = GradScaler(enabled=Config.AMP)

    start_epoch, best_val_mae, history = load_checkpoint(model, optimizer, scheduler, scaler)

    use_warmup_aug = Config.AUG_WARMUP_EPOCHS > 0 and warmup_transform is not None
    if use_warmup_aug and start_epoch < Config.AUG_WARMUP_EPOCHS:
        train_loader = build_train_loader(train_files, warmup_transform)
        augmentation_swapped = False
    else:
        train_loader = build_train_loader(train_files, full_train_transform)
        augmentation_swapped = True

    val_loader = build_val_loader(val_files, val_transform)
    mixup_cutmix_fn = create_mixup_cutmix_fn()
    mixup_active = False

    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        if use_warmup_aug and not augmentation_swapped and epoch >= Config.AUG_WARMUP_EPOCHS:
            print("\nEnabling full augmentation pipeline")
            train_loader = build_train_loader(train_files, full_train_transform)
            augmentation_swapped = True

        if (
            mixup_cutmix_fn is not None
            and not mixup_active
            and (epoch + 1) >= Config.MIXUP_START_EPOCH
        ):
            print("\nEnabling mixup augmentation")
            mixup_active = True

        if Config.FREEZE_BACKBONE_EPOCHS and epoch == Config.FREEZE_BACKBONE_EPOCHS:
            print("\nUnfreezing backbone for fine-tuning")
            for param in model.backbone.parameters():
                param.requires_grad = True

        train_loss, train_coord, train_pixel = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            epoch,
            mixup_fn=mixup_cutmix_fn if mixup_active else None,

        )
        scheduler.step()

        val_results: Optional[dict] = None

        if (epoch + 1) % Config.VALIDATE_EVERY == 0:
            (
                val_loss,
                val_coord,
                val_pixel,
                val_percentiles,
                csv_path,
            ) = validate(
                model,
                val_loader,
                criterion,
                epoch,
                csv_dir=Config.OUTPUT_DIR,
            )
            history.setdefault("train", []).append(train_loss)
            history.setdefault("val", []).append(val_loss)
            history.setdefault("val_mae", []).append(val_pixel)
            history.setdefault("val_pixel_percentiles", []).append(val_percentiles)

            val_results = {
                "loss": val_loss,
                "coord": val_coord,
                "pixel": val_pixel,
                "percentiles": val_percentiles,
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
                "val_percentiles": val_percentiles,
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
            if csv_path:
                print(f"Validation metrics CSV saved to {csv_path}")

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
            percentile_text = " | ".join(
                f"{key.upper()}: {value:.1f} px" for key, value in val_results["percentiles"].items()
            )
            print(
                "Val   -> Loss: "
                f"{val_results['loss']:.4f} | Coord MAE: {val_results['coord']:.4f} | "
                f"Pixel MAE: {val_results['pixel']:.2f} px (Δ vs train: {delta_pixel:+.2f} px) | "
                f"{percentile_text} | Best MAE: {best_val_mae:.2f} px"
            )
        else:
            print("Val   -> Skipped (waiting for scheduled evaluation)")

    final_path = save_checkpoint(Config.EPOCHS - 1, model, optimizer, scheduler, scaler, best_val_mae, history)
    print(f"Training completed. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
