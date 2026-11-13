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
    idle_points,
    leak_token,
    sides,
    source_points,
)
from negpy.features.leak.models import (
    LEAK_SIDE_POINTS,
    LEAK_SIDES,
    LEAK_SIDES_ALL,
    LEAK_SIDES_LEFT_RIGHT,
    LEAK_SIDES_TOP_BOTTOM,
    LeakConfig,
)

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


def _band_scan(across: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(clean, leaked): a leak that runs the frame's height over its left part (its width over the
    top part with *across*) and misses the film gap that the other pair of sides sits on."""
    h, w = 120, 180
    leak = np.zeros((h, w))
    leak[:, :140] = 1.0
    leak[15:105, :15] = 0.0
    if across:
        leak = leak.T
    leak = leak[..., None] * np.array([1.5, 0.1, 1.2])
    clean = np.broadcast_to(_BASE, leak.shape).astype(np.float32)
    leaked = (_BASE * (1.0 + leak) ** (-LEAK_GAMMA)).astype(np.float32)
    return clean, leaked


def _center_error(out: np.ndarray, clean: np.ndarray) -> float:
    h, w = out.shape[:2]
    win = (slice(h * 3 // 8, h * 5 // 8), slice(w * 3 // 8, w * 5 // 8))
    return float(np.abs(np.log10(out[win] / clean[win])).max())


@pytest.mark.parametrize("mode, across", [(LEAK_SIDES_TOP_BOTTOM, False), (LEAK_SIDES_LEFT_RIGHT, True)])
def test_a_pair_of_sides_takes_out_a_leak_the_other_sides_miss(mode, across):
    clean, leaked = _band_scan(across=across)
    all_sides = _center_error(apply_leak(leaked, _on()), clean)
    pair = _center_error(apply_leak(leaked, _on(leak_sides=mode)), clean)
    # Four sides average in the clean gap and leave part of the leak in the middle.
    assert all_sides > 0.05
    assert pair < 0.01


def test_an_unused_side_on_bare_light_does_not_set_the_base():
    clean, leaked = _band_scan()
    lit = leaked.copy()
    lit[15:105, :15] = 1.0  # the left side sits off the film
    a = apply_leak(leaked, _on(leak_sides=LEAK_SIDES_TOP_BOTTOM))
    b = apply_leak(lit, _on(leak_sides=LEAK_SIDES_TOP_BOTTOM))
    assert np.abs(np.log10(a[:, 30:150] / b[:, 30:150])).max() < 1e-6
    assert _center_error(a, clean) < 0.01


def test_sides_change_the_hash_and_round_trip():
    assert leak_token(_on()) == leak_token(_on(leak_sides=LEAK_SIDES_ALL))
    tokens = {leak_token(_on(leak_sides=m)) for m in LEAK_SIDES}
    assert len(tokens) == 3
    conf = _on(leak_sides=LEAK_SIDES_LEFT_RIGHT)
    assert WorkspaceConfig.from_flat_dict(conf.to_dict()).leak.leak_sides == LEAK_SIDES_LEFT_RIGHT
    assert LeakConfig(leak_sides="diagonal").leak_sides == LEAK_SIDES_ALL


def test_idle_points_are_the_middle_points_of_the_unused_sides():
    n = 4 * (LEAK_SIDE_POINTS + 1)
    tl, tr, br, bl = corner_indices(n)
    assert idle_points(n, LEAK_SIDES_ALL) == frozenset()
    assert idle_points(n, LEAK_SIDES_TOP_BOTTOM) == frozenset(range(tr + 1, br)) | frozenset(range(bl + 1, n))
    assert idle_points(n, LEAK_SIDES_LEFT_RIGHT) == frozenset(range(tl + 1, tr)) | frozenset(range(br + 1, bl))


def test_the_sides_choice_shows_the_saved_sides_and_sets_new_ones(qapp):
    from unittest.mock import MagicMock

    from negpy.desktop.session import AppState
    from negpy.desktop.view.sidebar.leak import LeakSidebar

    controller = MagicMock()
    controller.state = AppState()
    controller.state.config = _on(leak_sides=LEAK_SIDES_TOP_BOTTOM)
    card = LeakSidebar(controller)
    assert card.sides_btn.currentData() == LEAK_SIDES_TOP_BOTTOM
    assert card.sides_btn.isEnabled()
    card.sides_btn.setCurrentIndex(card.sides_btn.findData(LEAK_SIDES_LEFT_RIGHT))
    controller.set_leak.assert_called_with(leak_sides=LEAK_SIDES_LEFT_RIGHT)
