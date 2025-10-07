import pytest

torch = pytest.importorskip("torch")

from train_ball_localizer import ModelEma


def test_model_ema_tracks_parameters_and_buffers():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=True),
        torch.nn.BatchNorm1d(4),
    )
    ema = ModelEma(model, decay=0.5)

    initial_params = [param.detach().clone() for param in ema.ema_model.parameters()]

    for param in model.parameters():
        param.data.add_(1.0)
    for buffer in model.buffers():
        buffer.add_(0.25)

    ema.update(model)

    for ema_param, model_param, initial in zip(
        ema.ema_model.parameters(), model.parameters(), initial_params
    ):
        expected = initial * 0.5 + model_param.detach() * 0.5
        assert torch.allclose(ema_param, expected)

    for ema_buffer, buffer in zip(ema.ema_model.buffers(), model.buffers()):
        assert torch.allclose(ema_buffer, buffer)

    state = ema.state_dict()
    clone = ModelEma(model, decay=0.5)
    clone.load_state_dict(state)
    for ema_param, clone_param in zip(
        ema.ema_model.parameters(), clone.ema_model.parameters()
    ):
        assert torch.allclose(ema_param, clone_param)
