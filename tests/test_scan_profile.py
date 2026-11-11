from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.logic import (
    _paper_curve_params,
    apply_characteristic_curve,
    per_channel_curve_params,
    print_curve,
    scan_levels,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.papers import PAPER_PROFILES, effective_constants, scan_gamma
from negpy.features.process.models import ProcessMode
from negpy.kernel.image.logic import _WORKING_GAMMA
from negpy.services.rendering.engine import DarkroomEngine

SCAN = PAPER_PROFILES["scan"]
# A gray ramp prints the same through any row-normalized matrix; identity keeps the mirror exact.
_NEUTRAL_DYES = replace(SCAN, print_color=PAPER_PROFILES["neutral"].print_color)
_ARGS = dict(auto_normalize_contrast=True, strength=0.0, lum_range=1.3, textural_range=0.8, anchor=0.52)


def _params(paper, **over):
    kw = dict(_ARGS, paper=paper, shadow_point=0.93, highlight_point=0.04)
    kw.update(over)
    return per_channel_curve_params(115.0, 1.0, **kw)


def test_only_a_scan_profile_has_a_scan_gamma():
    assert scan_gamma(SCAN) == _WORKING_GAMMA
    assert all(scan_gamma(p) == 0.0 for key, p in PAPER_PROFILES.items() if key != "scan")
    assert scan_gamma(None) == 0.0


def test_the_levels_print_the_tails_at_the_scan_levels():
    slopes, pivots, _ = _params(SCAN, highlight_point=0.33)
    c = effective_constants(SCAN)
    d_max = float(c["d_max"])
    line = lambda u: slopes[1] * (u - pivots[1])  # noqa: E731
    assert line(0.93) == pytest.approx(d_max * (1.0 - c["scan_black_level"]))
    assert line(0.33) == pytest.approx(d_max * (1.0 - c["scan_white_level"]))


@pytest.mark.parametrize("highlight_point, span_key", [(0.04, "scan_levels_span_max"), (0.85, "scan_levels_span_min")])
def test_the_levels_hold_the_shadow_point_and_bound_the_span(highlight_point, span_key):
    slopes, pivots, _ = _params(SCAN, highlight_point=highlight_point)
    c = effective_constants(SCAN)
    d_max = float(c["d_max"])
    line = lambda u: slopes[1] * (u - pivots[1])  # noqa: E731
    assert line(0.93) == pytest.approx(d_max * (1.0 - c["scan_black_level"]))
    assert line(0.93 - c[span_key]) == pytest.approx(d_max * (1.0 - c["scan_white_level"]))


def test_a_paper_ignores_the_highlight_point():
    paper = PAPER_PROFILES["neutral"]
    assert _params(paper) == _params(paper, highlight_point=None)


@pytest.mark.parametrize("over", [{"auto_normalize_contrast": False}, {"anchor": None}, {"highlight_point": None}, {"shadow_point": None}])
def test_the_levels_need_both_autos_and_both_points(over):
    kw = dict(_ARGS, paper=SCAN, shadow_point=0.93, highlight_point=0.04)
    kw.update(over)
    points = {k: kw.pop(k) for k in ("highlight_point",)}
    assert per_channel_curve_params(115.0, 1.0, **kw, **points) == _paper_curve_params(115.0, 1.0, **kw)


def test_a_frame_with_no_range_keeps_the_paper_placement():
    params = ((3.0, 3.0, 3.0), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0))
    assert scan_levels(params, 3.0, 0.5, 0.5, 0.5, EXPOSURE_CONSTANTS) == params


def test_grade_and_density_act_about_the_levels():
    neutral = _params(SCAN)
    harder = per_channel_curve_params(90.0, 1.0, **dict(_ARGS, paper=SCAN, shadow_point=0.93, highlight_point=0.04))
    lighter = per_channel_curve_params(115.0, 0.5, **dict(_ARGS, paper=SCAN, shadow_point=0.93, highlight_point=0.04))
    assert harder[0][1] > neutral[0][1]
    assert lighter[0][1] == pytest.approx(neutral[0][1])
    assert lighter[1][1] > neutral[1][1]


def test_the_code_is_linear_in_the_density_a_paper_prints():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], 3, axis=2), 4, axis=0)
    line = ((0.2, 3.0),) * 3
    scan = np.asarray(apply_characteristic_curve(img, *line, paper=_NEUTRAL_DYES)).astype(np.float64)
    paper = np.asarray(apply_characteristic_curve(img, *line, paper=replace(_NEUTRAL_DYES, scan=False))).astype(np.float64)
    d_max = float(effective_constants(SCAN)["d_max"])
    lit = paper > 1e-2
    want = np.clip((d_max + np.log10(paper[lit])) / d_max, 0.0, 1.0)
    np.testing.assert_allclose(scan[lit] ** (1.0 / _WORKING_GAMMA), want, atol=2e-4)
    assert np.ptp(want) > 0.5


def test_the_chart_curve_matches_the_render():
    exposure = ExposureConfig(paper_profile="scan")
    ramp = np.linspace(0.05, 0.95, 32, dtype=np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], 3, axis=2), 2, axis=0)
    slope, pivot = 3.2, 0.3
    out = np.asarray(apply_characteristic_curve(img, (pivot, slope), (pivot, slope), (pivot, slope), paper=_NEUTRAL_DYES))[0, :, 1]
    chart = 10.0 ** -np.asarray(print_curve(exposure, slope, pivot, ProcessMode.C41, toe=0.0, shoulder=0.0)(ramp))
    np.testing.assert_allclose(out, chart, atol=2e-4)


def test_a_fresh_frame_prints_on_the_scan_stock():
    assert WorkspaceConfig().exposure.paper_profile == "scan"
    rng = np.random.default_rng(3)
    img = (rng.random((48, 48, 3), dtype=np.float32) * 0.6 + 0.2).astype(np.float32)
    cfg = WorkspaceConfig()
    scan = np.asarray(DarkroomEngine().process(img, cfg, source_hash="scan-default"))
    paper = replace(cfg, exposure=replace(cfg.exposure, paper_profile="neutral"))
    assert not np.allclose(scan, np.asarray(DarkroomEngine().process(img, paper, source_hash="scan-paper")))


def _green(params, u):
    slopes, pivots, curvs = params
    return slopes[1] * (u - pivots[1]) + curvs[1] * u * u


def test_whites_moves_the_white_end_and_holds_black():
    c = effective_constants(SCAN)
    d_max = float(c["d_max"])
    base, lifted = _params(SCAN), _params(SCAN, levels=(0.5, 0.0))
    black = next(u for u in np.linspace(0.0, 1.5, 3001) if _green(base, u) >= d_max)
    assert _green(lifted, black) == pytest.approx(_green(base, black), abs=2e-3)
    assert _green(lifted, 0.3) < _green(base, 0.3)


def test_blacks_moves_the_black_end_and_holds_white():
    base, lifted = _params(SCAN), _params(SCAN, levels=(0.0, 0.5))
    white = next(u for u in np.linspace(1.5, -0.5, 4001) if _green(base, u) <= 0.0)
    assert _green(lifted, white) == pytest.approx(_green(base, white), abs=2e-3)
    assert _green(lifted, 0.8) < _green(base, 0.8)


def test_neutral_levels_leave_the_curve_alone():
    assert _params(SCAN, levels=(0.0, 0.0)) == _params(SCAN)
    paper = PAPER_PROFILES["neutral"]
    assert _params(paper, levels=(0.4, -0.2)) != _params(paper)


def test_grade_turns_about_the_contrast_pivot():
    c = effective_constants(SCAN)
    target = float(c["d_max"]) * (1.0 - float(c["scan_contrast_pivot"]))
    kw = dict(_ARGS, paper=SCAN, shadow_point=0.93, highlight_point=0.04)
    base = per_channel_curve_params(115.0, 1.0, **kw)
    u = base[1][1] + target / base[0][1]
    for grade in (80.0, 150.0):
        graded = per_channel_curve_params(grade, 1.0, **kw)
        assert graded[0][1] != pytest.approx(base[0][1])
        assert _green(graded, u) == pytest.approx(target, abs=1e-6)


def test_brightness_scales_the_scan_gamma():
    assert scan_gamma(SCAN, 0.5) < scan_gamma(SCAN) < scan_gamma(SCAN, -0.5)
    assert scan_gamma(PAPER_PROFILES["neutral"], 0.5) == 0.0


def test_brightness_lightens_the_midtones_and_holds_the_ends():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    img = np.ascontiguousarray(np.repeat(ramp[None, :, None], 3, axis=2).repeat(4, axis=0))
    slopes, pivots, _ = _params(_NEUTRAL_DYES)
    kw = dict(params_r=(pivots[0], slopes[0]), params_g=(pivots[1], slopes[1]), params_b=(pivots[2], slopes[2]), paper=_NEUTRAL_DYES)
    base = np.asarray(apply_characteristic_curve(img, **kw))
    bright = np.asarray(apply_characteristic_curve(img, brightness=0.5, **kw))
    mid = (base[0, :, 1] > 0.1) & (base[0, :, 1] < 0.6)
    assert mid.any() and np.all(bright[0, mid, 1] > base[0, mid, 1])
    assert bright[0, :, 1].min() == pytest.approx(base[0, :, 1].min(), abs=1e-4)
    assert bright[0, :, 1].max() == pytest.approx(base[0, :, 1].max(), abs=1e-2)


def test_the_standard_tone_profile_is_the_scan_levels():
    from negpy.features.exposure.models import TONE_PROFILES

    assert TONE_PROFILES["standard"] == (EXPOSURE_CONSTANTS["scan_black_level"], EXPOSURE_CONSTANTS["scan_white_level"], 0.0, 1.0)
    assert _params(SCAN, tone_profile="standard") == _params(SCAN)
    assert _params(SCAN, tone_profile="no such profile") == _params(SCAN)


def test_a_linear_tone_profile_prints_the_tails_at_black_and_white():
    params = _params(SCAN, tone_profile="linear", highlight_point=0.33)
    d_max = float(effective_constants(SCAN)["d_max"])
    assert _green(params, 0.93) == pytest.approx(d_max)
    assert _green(params, 0.33) == pytest.approx(0.0, abs=1e-9)


def test_a_hard_tone_profile_is_steeper_than_a_soft_one():
    assert _params(SCAN, tone_profile="hard")[0][1] > _params(SCAN)[0][1] > _params(SCAN, tone_profile="soft")[0][1]


def test_a_paper_ignores_the_tone_profile():
    paper = PAPER_PROFILES["neutral"]
    assert _params(paper, tone_profile="hard") == _params(paper)
    assert scan_gamma(paper, 0.0, "deep") == 0.0
    assert scan_gamma(SCAN, 0.0, "deep") > scan_gamma(SCAN)
