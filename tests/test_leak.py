"""Light Leak: measured on the clear film around the frame and taken out as added light."""

from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.leak.logic import LEAK_GAMMA, apply_leak, corner_indices, default_points, leak_token, sides
from negpy.features.leak.models import LEAK_SIDE_POINTS, LeakConfig

_BASE = np.array([0.5, 0.3, 0.08], dtype=np.float32)
_FRAME = (0.05, 0.05, 0.95, 0.95)


def _leak_profile(h: int, w: int) -> np.ndarray:
    """A leak strongest at the top-left, in units of the base fog exposure, per channel."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    fall = np.exp(-3.0 * (y / h)) * (1.0 - 0.5 * x / w)
    return (fall - fall.min())[..., None] * np.array([1.5, 0.1, 1.2])


def _scan(h: int = 120, w: int = 180, picture: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(clean scan, leaked scan): exposure x over the base fog, printed as t = base * x^-gamma."""
    x = np.ones((h, w, 3))
    if picture:
        x[30:90, 40:140] = 40.0  # a bright subject, dense on the negative
        x[50:70, 60:120] = 4.0
    clean = (_BASE * x ** (-LEAK_GAMMA)).astype(np.float32)
    leaked = (_BASE * (x + _leak_profile(h, w)) ** (-LEAK_GAMMA)).astype(np.float32)
    return clean, leaked


def _on(**kw) -> LeakConfig:
    return LeakConfig(leak_apply=True, leak_points=default_points(_FRAME), **kw)


def test_default_points_make_a_closed_clockwise_loop():
    pts = default_points(_FRAME)
    assert len(pts) == 4 * (LEAK_SIDE_POINTS + 1)
    tl, tr, br, bl = corner_indices(len(pts))
    assert [pts[i] for i in (tl, tr, br, bl)] == [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]
    top, bottom, left, right = sides(pts)
    assert top[0] == left[0] == (0.05, 0.05)
    assert bottom[0] == left[-1] == (0.05, 0.95)
    assert right[-1] == bottom[-1] == (0.95, 0.95)


def test_inactive_leaves_the_image_and_the_hash_alone():
    _, leaked = _scan()
    for conf in (LeakConfig(), LeakConfig(leak_points=default_points(_FRAME)), _on(leak_strength=0.0)):
        assert apply_leak(leaked, conf) is leaked
        assert leak_token(conf) == ""
    assert leak_token(_on()) != leak_token(_on(leak_strength=0.5))
    moved = list(default_points(_FRAME))
    moved[1] = (0.3, 0.04)
    assert leak_token(_on()) != leak_token(replace(_on(), leak_points=tuple(moved)))


def test_a_leak_on_the_film_edge_comes_out_of_the_whole_frame():
    clean, leaked = _scan()
    out = apply_leak(leaked, _on())
    before = np.abs(np.log10(leaked / clean)).max()
    after = np.abs(np.log10(out / clean))
    assert before > 0.2
    # Inside the measuring frame the thin film, where a leak shows most, returns to the clear base.
    assert after[8:30, 8:172].max() < 0.02
    # A dense subject barely moves.
    assert np.abs(np.log10(out[35:45, 45:55] / leaked[35:45, 45:55])).max() < 0.02


def test_strength_scales_the_correction():
    clean, leaked = _scan(picture=False)
    full = np.log10(apply_leak(leaked, _on()) / leaked)[5:15].mean()
    half = np.log10(apply_leak(leaked, _on(leak_strength=0.5)) / leaked)[5:15].mean()
    assert 0.3 * full < half < 0.7 * full


def test_a_point_on_a_perforation_does_not_break_the_reading():
    clean, leaked = _scan(picture=False)
    holed = leaked.copy()
    holed[2:10, 60:70] = 1.0  # bare light through a perforation, under one top point
    pts = list(default_points((0.05, 0.05, 0.95, 0.95)))
    a = apply_leak(leaked, _on())
    b = apply_leak(holed, LeakConfig(leak_apply=True, leak_points=tuple(pts)))
    assert np.abs(np.log10(a[20:] / b[20:])).max() < 0.01


def test_the_config_round_trips_through_the_flat_dict():
    conf = replace(WorkspaceConfig(), leak=_on(leak_strength=1.2))
    back = WorkspaceConfig.from_flat_dict(conf.to_dict())
    assert back.leak == conf.leak
    assert isinstance(back.leak.leak_points[0], tuple)
