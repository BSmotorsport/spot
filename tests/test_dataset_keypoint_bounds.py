import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("albumentations")

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from codexfpn import Config, FootballDataset


def _write_dummy_image(path):
    image = np.zeros((Config.ORIGINAL_HEIGHT, Config.ORIGINAL_WIDTH, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_rotated_sample_targets_stay_in_range(tmp_path):
    center_x = Config.ORIGINAL_WIDTH // 2
    center_y = Config.ORIGINAL_HEIGHT // 2
    edge_x = Config.ORIGINAL_WIDTH - 1
    edge_y = Config.ORIGINAL_HEIGHT // 2

    center_path = tmp_path / f"scene-center-{center_x}-{center_y}.jpg"
    edge_path = tmp_path / f"scene-edge-{edge_x}-{edge_y}.jpg"

    _write_dummy_image(center_path)
    _write_dummy_image(edge_path)

    transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Rotate(limit=(45, 45), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    dataset = FootballDataset(
        [str(center_path), str(edge_path)],
        transform=transform,
    )

    image_tensor, precise_coords, context, path = dataset[0]

    assert image_tensor.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert precise_coords.shape == (2,)
    assert context is None
    assert path.endswith(".jpg")

    assert torch.all(precise_coords >= 0.0)
    assert torch.all(precise_coords < 1.0)

    image_coords = precise_coords * (Config.IMAGE_SIZE - 1)
    assert torch.all(image_coords >= 0.0)
    assert torch.all(image_coords <= Config.IMAGE_SIZE - 1)

    # The second sample should fall back to the first if its keypoint is dropped
    image_tensor_2, precise_coords_2, context_2, _ = dataset[1]
    assert torch.allclose(image_tensor, image_tensor_2)
    assert torch.allclose(precise_coords, precise_coords_2)
    assert context_2 is None
