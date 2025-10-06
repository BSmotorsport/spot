import pytest

torch = pytest.importorskip("torch")

from train_ball_localizer import TrainingConfig, compute_weighted_loss_components


def test_weighted_loss_float32_reduction():
    config = TrainingConfig(
        heatmap_fg_weight=10_000.0,
        heatmap_bg_weight=0.0,
    )

    loss_map = torch.full((1, 1, 32, 32), 100.0, dtype=torch.float16)
    targets = torch.ones_like(loss_map)

    loss, _, weight_sum = compute_weighted_loss_components(loss_map, targets, config)

    assert torch.isfinite(loss), "Normalised loss should remain finite in float32"
    assert torch.isfinite(weight_sum), "Weight reduction should occur in float32"
    assert loss.dtype == torch.float32
    assert weight_sum.dtype == torch.float32
