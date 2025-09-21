import pytest

torch = pytest.importorskip("torch")

from codexfpn import CombinedLoss


def _create_gaussian_heatmap(size: int = 32, sigma: float = 2.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    gaussian = torch.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    gaussian /= gaussian.max()
    return gaussian.unsqueeze(0).unsqueeze(0)


def test_combined_loss_pushes_max_probability_above_half():
    torch.manual_seed(0)

    target = _create_gaussian_heatmap()
    logits = torch.nn.Parameter(torch.zeros_like(target))
    optimizer = torch.optim.Adam([logits], lr=0.1)
    criterion = CombinedLoss(heatmap_weight=1.0, coord_weight=0.0)

    # Run a miniature training loop that mimics a single epoch worth of updates.
    for _ in range(120):
        optimizer.zero_grad()
        total_loss, h_loss, c_loss, pixel_loss = criterion(logits, target)
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(logits)

    assert probs.max().item() > 0.5, "weighted BCE should push the peak probability well above 0.5"
    assert pixel_loss.item() == pytest.approx(0.0), "pixel loss should be zero when coordinates are absent"
