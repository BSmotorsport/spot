import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from train_ball_localizer import TrainingConfig, ValidationSampleExporter, compute_metrics


def test_compute_metrics_returns_unbiased_coordinates():
    config = TrainingConfig(input_size=256, heatmap_size=4)

    outputs = torch.zeros((1, 1, 4, 4))
    outputs[0, 0, 1, 2] = 10.0

    batch = {
        "pad": torch.zeros(1, 2),
        "scale": torch.ones(1),
        "original_xy": torch.tensor([[160.0, 96.0]]),
    }

    metrics = compute_metrics(outputs, batch, config)

    assert metrics["pixel_mae"] == pytest.approx(0.0)
    assert metrics["pixel_median"] == pytest.approx(0.0)
    assert metrics["pixel_errors"] == [0.0]


def test_compute_metrics_preserves_boundary_predictions():
    config = TrainingConfig(input_size=256, heatmap_size=4)

    outputs = torch.zeros((1, 1, 4, 4))
    outputs[0, 0, 0, 0] = 10.0

    batch = {
        "pad": torch.zeros(1, 2),
        "scale": torch.ones(1),
        "original_xy": torch.tensor([[0.0, 0.0]]),
    }

    metrics = compute_metrics(outputs, batch, config)

    assert metrics["pixel_mae"] == pytest.approx(0.0)
    assert metrics["pixel_median"] == pytest.approx(0.0)
    assert metrics["pixel_errors"] == [0.0]


def test_decode_heatmap_matches_encoding(tmp_path):
    config = TrainingConfig(input_size=256, heatmap_size=4)
    exporter = ValidationSampleExporter(tmp_path, config, max_samples=0)

    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[1, 2] = 1.0

    decoded_x, decoded_y = exporter._decode_heatmap(heatmap)

    assert decoded_x == pytest.approx(160.0)
    assert decoded_y == pytest.approx(96.0)


def test_decode_heatmap_respects_boundary_cells(tmp_path):
    config = TrainingConfig(input_size=256, heatmap_size=4)
    exporter = ValidationSampleExporter(tmp_path, config, max_samples=0)

    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[0, 0] = 1.0

    decoded_x, decoded_y = exporter._decode_heatmap(heatmap)

    assert decoded_x == pytest.approx(0.0)
    assert decoded_y == pytest.approx(0.0)
