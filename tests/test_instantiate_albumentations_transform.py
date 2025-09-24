import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("albumentations")

from codexfpn import instantiate_albumentations_transform


def test_skips_candidate_when_all_kwargs_filtered():
    class DummyTransform:
        def __init__(self, *, border_mode=None, value=None):
            self.border_mode = border_mode
            self.value = value

    common_kwargs = {"p": 0.5}
    candidates = [
        {"mode": "legacy", "cval": 123},
        {"border_mode": "reflect", "value": 0},
    ]

    transform = instantiate_albumentations_transform(
        DummyTransform,
        common_kwargs,
        candidates,
    )

    assert isinstance(transform, DummyTransform)
    assert transform.border_mode == "reflect"
    assert transform.value == 0


def test_returns_common_kwargs_when_candidate_empty():
    class DummyTransform:
        def __init__(self, *, foo=0):
            self.foo = foo

    transform = instantiate_albumentations_transform(
        DummyTransform,
        {"foo": 10},
        [{}],
    )

    assert transform.foo == 10
