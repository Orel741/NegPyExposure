"""Characterization guard: pin the default + exposure-only full-engine output so the
look doesn't drift. Goldens are the scene-linear pipeline encoded with the Adobe RGB
working TRC."""

from dataclasses import replace

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.process.models import ProcessConfig
from negpy.services.rendering.engine import DarkroomEngine

_POINTS = [(0, 0), (8, 8), (20, 30), (32, 32), (50, 50), (60, 10)]

# Golden pixel values (Adobe-RGB-TRC-encoded engine output, scene-linear pipeline,
# default config with both Autos on — paper_dmin off, paper_black off, cast_removal_strength 0.5, sharpen 0.25,
# dye_separation 1.0).
_GOLDEN = {
    "default": [
        (0.041479, 0.966377, 0.953464),
        (0.041983, 0.958922, 0.964215),
        (0.243130, 0.233866, 0.234138),
        (0.166996, 0.170178, 0.170413),
        (0.041884, 0.040688, 0.953670),
        (0.950070, 0.961239, 0.041670),
    ],
    "expo_dark": [
        (0.301440, 0.991429, 0.990583),
        (0.303188, 0.990947, 0.991291),
        (0.638220, 0.625288, 0.625669),
        (0.535491, 0.539520, 0.539818),
        (0.302858, 0.298670, 0.990596),
        (0.990351, 0.991099, 0.302112),
    ],
    "expo_cmy": [
        (0.027331, 0.974900, 0.898536),
        (0.027632, 0.969662, 0.921983),
        (0.190812, 0.270169, 0.155072),
        (0.127739, 0.200311, 0.107596),
        (0.027570, 0.053988, 0.898974),
        (0.919879, 0.971295, 0.021725),
    ],
}


def _synthetic_image(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), dtype=np.float32)
    for y in range(64):
        for x in range(64):
            img[y, x] = 0.1 + 0.8 * ((x + y) / 126.0)
    img[0:16, 0:16] = [0.9, 0.1, 0.1]
    img[0:16, 48:64] = [0.1, 0.9, 0.1]
    img[48:64, 0:16] = [0.1, 0.1, 0.9]
    img[48:64, 48:64] = [0.9, 0.9, 0.1]
    img += rng.normal(0, 0.005, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _base_settings() -> WorkspaceConfig:
    geo = GeometryConfig(
        rotation=0,
        fine_rotation=0.0,
        flip_horizontal=False,
        flip_vertical=False,
        crop_rect=(0.0, 0.0, 1.0, 1.0),
        autocrop_offset=0,
    )
    return replace(
        WorkspaceConfig(),
        geometry=geo,
        process=replace(ProcessConfig(), white_point_offset=0.0, black_point_offset=0.0, color_range_clip=1.0),
        exposure=replace(WorkspaceConfig().exposure, paper_profile="neutral", auto_exposure=True, auto_normalize_contrast=True),
    )


def test_full_engine_output_preserved_after_relocation():
    base = _base_settings()
    configs = {
        "default": base,
        "expo_dark": replace(base, exposure=replace(base.exposure, density=-1.0, grade=2.0)),
        "expo_cmy": replace(base, exposure=replace(base.exposure, wb_cyan=0.3, wb_magenta=-0.2, wb_yellow=0.5)),
    }
    img = _synthetic_image()
    eng = DarkroomEngine()
    for name, cfg in configs.items():
        out = eng.process(img, cfg, f"relocation_{name}")
        got = np.array([out[y, x] for (y, x) in _POINTS], dtype=np.float32)
        want = np.array(_GOLDEN[name], dtype=np.float32)
        np.testing.assert_allclose(got, want, atol=1e-3, err_msg=f"config={name}")
