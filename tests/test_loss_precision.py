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

    loss, fg_loss, bg_loss = compute_weighted_loss_components(
        loss_map, targets, config
    )

    for component in (loss, fg_loss, bg_loss):
        assert torch.isfinite(component), "Loss components should remain finite"
        assert component.dtype == torch.float32
