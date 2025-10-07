import pytest

pytest.importorskip("albumentations")

import albumentations as A

from train_ball_localizer import _coarse_dropout_kwargs


def test_coarse_dropout_kwargs_legacy_signature(monkeypatch):
    class LegacyCoarseDropout:
        def __init__(
            self,
            *,
            p,
            max_holes,
            min_holes=None,
            max_height,
            min_height=None,
            max_width,
            min_width=None,
            fill_value=0,
        ):
            self.p = p
            self.max_holes = max_holes
            self.min_holes = min_holes
            self.max_height = max_height
            self.min_height = min_height
            self.max_width = max_width
            self.min_width = min_width
            self.fill_value = fill_value

    monkeypatch.setattr(A, "CoarseDropout", LegacyCoarseDropout)

    kwargs = _coarse_dropout_kwargs(
        p=0.5,
        max_holes=4,
        min_holes=2,
        min_size=10,
        max_size=20,
        image_height=100,
        image_width=200,
    )

    transform = A.CoarseDropout(**kwargs)

    assert transform.max_height == 20
    assert transform.min_height == 10
    assert isinstance(transform.max_height, int)
    assert isinstance(transform.min_height, int)
    assert transform.max_width == 20
    assert transform.min_width == 10
    assert isinstance(transform.max_width, int)
    assert isinstance(transform.min_width, int)
    assert transform.fill_value == 0


def test_coarse_dropout_kwargs_fractional_signature(monkeypatch):
    class FractionalCoarseDropout:
        def __init__(
            self,
            *,
            p,
            holes_number_range=(0, 0),
            hole_height_range=(0.0, 0.0),
            hole_width_range=(0.0, 0.0),
            fill_value=0,
        ):
            self.p = p
            self.holes_number_range = holes_number_range
            self.hole_height_range = hole_height_range
            self.hole_width_range = hole_width_range
            self.fill_value = fill_value

    monkeypatch.setattr(A, "CoarseDropout", FractionalCoarseDropout)

    kwargs = _coarse_dropout_kwargs(
        p=0.25,
        max_holes=3,
        min_holes=1,
        min_size=10,
        max_size=20,
        image_height=100,
        image_width=200,
    )

    transform = A.CoarseDropout(**kwargs)

    assert transform.holes_number_range == (1, 3)
    assert transform.hole_height_range == pytest.approx((0.1, 0.2))
    assert transform.hole_width_range == pytest.approx((0.05, 0.1))
    for value in transform.hole_height_range + transform.hole_width_range:
        assert 0.0 < value <= 1.0
    assert transform.fill_value == 0
