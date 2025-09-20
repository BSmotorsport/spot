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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    dataset = FootballDataset(
        [str(center_path), str(edge_path)],
        transform=transform,
        heatmap_size=Config.HEATMAP_SIZE,
        heatmap_sigma=Config.HEATMAP_SIGMA_START,
    )

    image_tensor, target_heatmap, precise_coords = dataset[0]

    assert image_tensor.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert target_heatmap.shape == (1, Config.HEATMAP_SIZE, Config.HEATMAP_SIZE)
    assert precise_coords.shape == (2,)

    assert torch.all(precise_coords >= 0.0)
    assert torch.all(precise_coords < 1.0)

    image_coords = precise_coords * (Config.IMAGE_SIZE - 1)
    heatmap_coords = precise_coords * (Config.HEATMAP_SIZE - 1)

    assert torch.all(image_coords >= 0.0)
    assert torch.all(image_coords <= Config.IMAGE_SIZE - 1)
    assert torch.all(heatmap_coords >= 0.0)
    assert torch.all(heatmap_coords <= Config.HEATMAP_SIZE - 1)

    heatmap_np = target_heatmap.squeeze(0).numpy()
    peak_index = np.argmax(heatmap_np)
    peak_y, peak_x = np.unravel_index(peak_index, heatmap_np.shape)

    assert abs(peak_x - heatmap_coords[0].item()) <= 2.0
    assert abs(peak_y - heatmap_coords[1].item()) <= 2.0

    # The second sample should fall back to the first if its keypoint is dropped
    image_tensor_2, target_heatmap_2, precise_coords_2 = dataset[1]
    assert torch.allclose(image_tensor, image_tensor_2)
    assert torch.allclose(target_heatmap, target_heatmap_2)
    assert torch.allclose(precise_coords, precise_coords_2)
