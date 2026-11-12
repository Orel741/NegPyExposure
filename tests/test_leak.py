"""Light Leak: measured on the clear film around the frame and taken out as added light."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.logic import map_coords_to_geometry
from negpy.features.leak.logic import (
    LEAK_GAMMA,
    anchored_frame,
    apply_leak,
    corner_indices,
    default_points,
    leak_token,
    sides,
    source_points,
)
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


def _on(points=None, **kw) -> WorkspaceConfig:
    leak = LeakConfig(leak_apply=True, leak_frame=default_points(_FRAME) if points is None else points, **kw)
    return replace(WorkspaceConfig(), leak=leak)


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
    off = replace(WorkspaceConfig(), leak=LeakConfig(leak_frame=default_points(_FRAME)))
    for conf in (WorkspaceConfig(), off, _on(leak_strength=0.0)):
        assert apply_leak(leaked, conf) is leaked
        assert leak_token(conf) == ""
    assert leak_token(_on()) != leak_token(_on(leak_strength=0.5))
    moved = list(default_points(_FRAME))
    moved[1] = (0.3, 0.04)
    assert leak_token(_on()) != leak_token(_on(points=tuple(moved)))
    rotated = _on()
    rotated = replace(rotated, geometry=replace(rotated.geometry, fine_rotation=1.0))
    assert leak_token(_on()) != leak_token(rotated)


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
    b = apply_leak(holed, _on(points=tuple(pts)))
    assert np.abs(np.log10(a[20:] / b[20:])).max() < 0.01


def test_the_config_round_trips_through_the_flat_dict():
    conf = _on(leak_strength=1.2)
    conf = replace(conf, leak=replace(conf.leak, leak_anchor=(0.1, 0.1, 0.9, 0.9)))
    back = WorkspaceConfig.from_flat_dict(conf.to_dict())
    assert back.leak == conf.leak
    assert isinstance(back.leak.leak_frame[0], tuple)


def test_a_raw_point_frame_from_before_the_crop_space_is_dropped():
    flat = {**WorkspaceConfig().to_dict(), "leak_points": [[0.1, 0.1]] * 16, "leak_apply": True}
    assert WorkspaceConfig.from_flat_dict(flat).leak.leak_frame == ()


def test_the_frame_follows_each_frame_crop():
    leak = LeakConfig(leak_frame=default_points((0.08, 0.08, 0.92, 0.92)), leak_anchor=(0.1, 0.1, 0.9, 0.9))
    moved = anchored_frame(leak, (0.2, 0.15, 0.6, 0.95))
    tl, tr, br, bl = corner_indices(len(moved))
    assert moved[tl] == pytest.approx((0.19, 0.13))
    assert moved[br] == pytest.approx((0.61, 0.97))
    # With no anchor or no crop, the frame stays where it was placed.
    assert anchored_frame(leak, None) == leak.leak_frame
    assert anchored_frame(replace(leak, leak_anchor=()), (0.2, 0.15, 0.6, 0.95)) == leak.leak_frame


@pytest.mark.parametrize(
    "geo",
    [
        {},
        {"rotation": 1, "fine_rotation": 2.5},
        {"rotation": 3, "flip_horizontal": True, "converge_v": 4.0},
        {"fine_rotation": -1.5, "distortion_k1": 0.03, "converge_h": -3.0},
    ],
)
def test_source_points_invert_the_geometry(geo):
    geometry = replace(WorkspaceConfig().geometry, **geo)
    shape = (400, 600)
    targets = default_points((0.1, 0.12, 0.9, 0.88))
    for (tx, ty), (sx, sy) in zip(targets, source_points(targets, geometry, shape)):
        back = map_coords_to_geometry(
            sx,
            sy,
            shape,
            geometry.rotation,
            geometry.fine_rotation,
            geometry.flip_horizontal,
            geometry.flip_vertical,
            distortion_k1=geometry.distortion_k1,
            converge_v=geometry.converge_v,
            converge_h=geometry.converge_h,
        )
        assert back == pytest.approx((tx, ty), abs=1e-5)


def test_moving_the_frame_keeps_the_correction_off():
    from negpy.desktop.controller import AppController

    applied = []
    ctrl = AppController.__new__(AppController)
    off = replace(WorkspaceConfig(), leak=LeakConfig(leak_apply=False, leak_frame=default_points(_FRAME)))
    off = replace(off, geometry=replace(off.geometry, crop_rect=(0.1, 0.1, 0.9, 0.9)))
    ctrl.state = SimpleNamespace(config=off)
    ctrl.apply_config = lambda conf, **_kw: applied.append(conf)
    ctrl.set_leak_frame(default_points((0.06, 0.06, 0.94, 0.94)))
    assert applied[-1].leak.leak_apply is False
    assert applied[-1].leak.leak_anchor == (0.1, 0.1, 0.9, 0.9)
