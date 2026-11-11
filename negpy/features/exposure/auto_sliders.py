"""Auto Density and Auto Grade on their sliders.

The stored value is an offset on the meter; the slider shows meter + offset, and an edit
converts back. A toggle flips only the flag.
"""

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Tuple

from negpy.features.exposure.logic import (
    auto_base_slope,
    auto_highlight_from_metrics,
    auto_shadow_from_metrics,
    compute_pivot,
    curve_params_from_metrics,
    effective_grade_range,
    effective_midtone_gamma,
    grade_to_slope,
    slope_to_grade,
)
from negpy.features.exposure.models import EXPOSURE_CONSTANTS, ExposureConfig
from negpy.features.exposure.papers import effective_constants, effective_paper_profile, slope_gain

# The offsets at which the meter alone decides.
NEUTRAL: Dict[str, float] = {f: float(getattr(ExposureConfig(), f)) for f in ("density", "grade", "shadow_density", "highlight_density")}


def _paper(exposure: ExposureConfig, process_mode: Optional[str]) -> Tuple[Any, float]:
    profile = effective_paper_profile(exposure.paper_profile, process_mode)
    return profile, (profile.d_min if exposure.paper_dmin else 0.0)


def _density_shift(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[float]:
    """Print Density the metered anchor is worth: the gap between the metered and the
    reference pivot at green's slope, as compute_pivot is linear in density."""
    anchor = metrics.get("metered_anchor")
    if not exposure.auto_exposure or anchor is None:
        return None
    profile, d_min = _paper(exposure, process_mode)
    slope = curve_params_from_metrics(exposure, process_mode, metrics)[0][1]
    mg = effective_midtone_gamma(profile, exposure.midtone_gamma)
    gap = compute_pivot(slope, 1.0, d_min, float(anchor), profile, mg) - compute_pivot(slope, 1.0, d_min, paper=profile)
    return gap / float(effective_constants(profile)["density_multiplier"])


def _auto_slope(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[float]:
    lum_range, textural = metrics.get("norm_density_range"), metrics.get("textural_range")
    if not exposure.auto_normalize_contrast or lum_range is None or textural is None:
        return None
    profile, d_min = _paper(exposure, process_mode)
    anchor = metrics.get("metered_anchor") if exposure.auto_exposure else None
    mg = effective_midtone_gamma(profile, exposure.midtone_gamma)
    return auto_base_slope(exposure.grade, True, lum_range, textural, metrics.get("shadow_point"), anchor, d_min, profile, mg)


def _scan_levels(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    """Green's (slope, pivot) under a scan profile with both Autos on, None otherwise. Placed at
    the metered points, or unplaced while the shadow or highlight meter is missing."""
    profile, _ = _paper(exposure, process_mode)
    if not profile.scan or not exposure.auto_exposure or not exposure.auto_normalize_contrast:
        return None
    if any(metrics.get(k) is None for k in ("metered_anchor", "textural_range", "norm_density_range")):
        return None
    slopes, pivots, _ = curve_params_from_metrics(exposure, process_mode, metrics)
    return slopes[1], pivots[1]


def _scan_shown(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """The Density and Grade that print a scan profile's levels with both Autos off. With both
    off, the levels hold fixed points, so the placed slope scales with the manual slope and
    Density moves the placed pivot by density_multiplier per unit."""
    placed = _scan_levels(exposure, process_mode, metrics)
    if placed is None:
        return None
    slope, pivot = placed
    profile, _ = _paper(exposure, process_mode)
    lum_range = metrics["norm_density_range"]
    gain = slope_gain(profile)
    off = replace(exposure, auto_exposure=False, auto_normalize_contrast=False, grade=NEUTRAL["grade"], density=NEUTRAL["density"])
    neutral_slope = curve_params_from_metrics(off, process_mode, metrics)[0][1]
    grade = slope_to_grade(grade_to_slope(NEUTRAL["grade"], lum_range, gain) * slope / neutral_slope, lum_range, gain)
    graded_pivot = curve_params_from_metrics(replace(off, grade=grade), process_mode, metrics)[1][1]
    mult = float(effective_constants(profile)["density_multiplier"])
    return {"density": NEUTRAL["density"] - (pivot - graded_pivot) / mult, "grade": grade}


def print_shown_values(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Manual-equivalent value of every slider an auto drives, for the autos that are on
    and metered. Grade is the ISO R that prints the auto's slope on the frame's own range.
    Whites and Blacks act after either, so the values leave them out."""
    exposure = replace(exposure, whites=0.0, blacks=0.0)
    shown: Dict[str, float] = {}
    scan = _scan_shown(exposure, process_mode, metrics)
    shift = _density_shift(exposure, process_mode, metrics)
    if scan is not None:
        shown.update(scan)
    elif shift is not None:
        shown["density"] = exposure.density - shift
    slope = _auto_slope(exposure, process_mode, metrics)
    if scan is None and slope is not None:
        shown["grade"] = slope_to_grade(slope, metrics["norm_density_range"], slope_gain(_paper(exposure, process_mode)[0]))
    if exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        shown["highlight_density"] = exposure.highlight_density + auto_highlight_from_metrics(exposure, process_mode, metrics)
    if exposure.auto_normalize_contrast and metrics.get("shadow_point") is not None:
        shown["shadow_density"] = exposure.shadow_density + auto_shadow_from_metrics(exposure, process_mode, metrics)
    return shown


def print_stored_value(
    exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any], field: str, shown: float
) -> float:
    """Inverse of print_shown_values for one field. Where Shadow Reach floors the slope,
    a softer grade is stored but does not print."""
    exposure = replace(exposure, whites=0.0, blacks=0.0)
    scan = _scan_shown(exposure, process_mode, metrics)
    if field == "density":
        if scan is not None:
            # The levels shift the pivot by a constant, so shown and stored differ by one.
            return shown + exposure.density - scan["density"]
        shift = _density_shift(exposure, process_mode, metrics)
        return shown if shift is None else shown + shift
    if field == "grade":
        if _auto_slope(exposure, process_mode, metrics) is None:
            return shown
        lum_range = metrics["norm_density_range"]
        r_eff = effective_grade_range(True, lum_range, metrics["textural_range"])
        gain = slope_gain(_paper(exposure, process_mode)[0])
        target = grade_to_slope(shown, lum_range, gain)
        if scan is not None:
            # The levels scale the slope by one factor at every Grade.
            unplaced = curve_params_from_metrics(exposure, process_mode, {**metrics, "highlight_point": None})[0][1]
            target *= unplaced / grade_to_slope(scan["grade"], lum_range, gain)
        return slope_to_grade(target, r_eff, gain)
    if field == "highlight_density" and exposure.auto_normalize_contrast and metrics.get("highlight_point") is not None:
        return shown - auto_highlight_from_metrics(exposure, process_mode, metrics)
    if field == "shadow_density" and exposure.auto_normalize_contrast and metrics.get("shadow_point") is not None:
        return shown - auto_shadow_from_metrics(exposure, process_mode, metrics)
    return shown


def grade_at_softest(exposure: ExposureConfig, process_mode: Optional[str], metrics: Mapping[str, Any]) -> bool:
    """Whether the frame prints its softest slope while the Grade slider still shows travel
    toward the soft end: a flat negative reaches the slope floor before R180."""
    if metrics.get("norm_density_range") is None:
        return False
    softest = float(EXPOSURE_CONSTANTS["iso_r_max"])
    if print_shown_values(exposure, process_mode, metrics).get("grade", exposure.grade) >= softest - 0.5:
        return False
    slope = curve_params_from_metrics(exposure, process_mode, metrics)[0][1]
    return slope <= curve_params_from_metrics(replace(exposure, grade=softest), process_mode, metrics)[0][1] + 1e-6
