import pytest

torch = pytest.importorskip("torch")

from codexfpn import Config, CoordinateLoss


def test_coordinate_loss_guides_predictions_towards_target():
    torch.manual_seed(0)

    target = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
    preds = torch.nn.Parameter(torch.zeros_like(target))

    optimizer = torch.optim.SGD([preds], lr=0.1)
    criterion = CoordinateLoss(pixel_weight=0.5)

    for _ in range(120):
        optimizer.zero_grad()
        loss, coord_loss, pixel_mae = criterion(preds, target)
        loss.backward()
        optimizer.step()

    assert torch.allclose(preds.detach(), target, atol=0.05)
    assert coord_loss.item() < 0.01
    assert pixel_mae.item() < (Config.IMAGE_SIZE * 0.1)
