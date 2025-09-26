"""Utility to synthesise context sidecar files for the football dataset.

The coordinate regression model can consume optional numeric cues stored next to
image files (``*.context.json`` by default).  This script builds simple global
image descriptors so experiments can exercise the context branch without
external preprocessing pipelines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from codexfpn import Config


def _config_value(name: str, default):
    """Return ``Config.<name>`` if available, otherwise ``default``.

    The training configuration historically shipped without the context-specific
    attributes used by this utility.  Older checkpoints therefore lack
    ``CONTEXT_METADATA_SUFFIX`` and companions which breaks standalone
    invocation of the script.  Using ``getattr`` keeps the defaults in this
    module while still honouring project overrides when present.
    """

    return getattr(Config, name, default)


IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def discover_images(root: Path, recursive: bool = True) -> Iterable[Path]:
    """Yield image paths under ``root`` that match supported extensions."""

    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")

    for path in iterator:
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            yield path


def normalise_histogram(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / total).astype(np.float32)


def compute_context_vector(image: np.ndarray, bins_per_channel: int = 16) -> np.ndarray:
    """Derive a numeric descriptor from ``image`` suitable for sidecar export."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a colour image in BGR format")

    features: list[float] = []
    float_image = image.astype(np.float32)

    # Global colour statistics (mean and standard deviation per channel).
    channel_means = float_image.mean(axis=(0, 1))
    channel_stds = float_image.std(axis=(0, 1))
    features.extend(channel_means.tolist())
    features.extend(channel_stds.tolist())

    # Grayscale intensity distribution and spread.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)
    features.append(float(gray_float.mean()))
    features.append(float(gray_float.std()))
    features.append(float(np.min(gray_float)))
    features.append(float(np.max(gray_float)))
    features.append(float(np.percentile(gray_float, 75.0)))
    features.append(float(np.percentile(gray_float, 25.0)))

    # Edge density as a proxy for scene complexity.
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    features.append(edge_density)

    # Colour histograms for each channel.
    for channel_index in range(3):
        hist = cv2.calcHist(
            [image],
            [channel_index],
            None,
            [bins_per_channel],
            [0, 256],
        )
        features.extend(normalise_histogram(hist).flatten().tolist())

    return np.asarray(features, dtype=np.float32)


def adjust_vector_length(
    vector: Sequence[float],
    vector_size: int,
    fill_value: float,
) -> np.ndarray:
    """Pad or truncate ``vector`` to ``vector_size``."""

    array = np.asarray(vector, dtype=np.float32)
    if vector_size <= 0:
        return array

    if array.size >= vector_size:
        return array[:vector_size]

    padded = np.full(vector_size, fill_value, dtype=np.float32)
    padded[: array.size] = array
    return padded


def write_sidecar(path: Path, values: Sequence[float], suffix: str) -> None:
    """Persist ``values`` as a JSON array next to ``path`` using ``suffix``."""

    base = path.with_suffix("")
    if suffix.startswith("."):
        target = base.with_suffix(suffix)
    else:
        target = Path(str(path) + suffix)

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump([float(v) for v in values], handle, ensure_ascii=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic context metadata sidecars (JSON arrays) for images "
            "so that the optional context branch can be exercised."
        )
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory that contains the training frames.",
    )
    parser.add_argument(
        "--suffix",
        default=_config_value("CONTEXT_METADATA_SUFFIX", ".context.json"),
        help="Filename suffix used for the generated sidecar files.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=_config_value("CONTEXT_VECTOR_SIZE", 128),
        help="Length of the context vector expected by the model.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=_config_value("CONTEXT_DEFAULT_FILL_VALUE", 0.0),
        help="Pad shorter vectors with this value.",
    )
    parser.add_argument(
        "--bins-per-channel",
        type=int,
        default=16,
        help="Number of histogram bins per colour channel.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top-level directory for images.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    image_dir = args.image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")

    recursive = not args.non_recursive
    generated = 0

    for image_path in discover_images(image_dir, recursive=recursive):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"⚠️ Skipping unreadable image: {image_path}")
            continue

        vector = compute_context_vector(image, bins_per_channel=args.bins_per_channel)
        vector = adjust_vector_length(vector, args.vector_size, args.fill_value)
        write_sidecar(image_path, vector, args.suffix)
        generated += 1

    print(f"Generated {generated} context sidecar file(s) in {image_dir}")
    return 0


if __name__ == "__main__":
    parser = build_parser()
    exit(main(parser.parse_args()))
