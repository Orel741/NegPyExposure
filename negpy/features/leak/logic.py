"""Light-leak removal: a source bake on the linear scan, measured on the film around the frame.

A leak adds light to the film. The clear film around the picture holds no image, so its density
above the clear base is the leak alone. Each side of the measuring frame is read as a profile, the
four profiles fill the frame as a Coons patch, and the leak comes out as exposure (`negpy_core.leak_correct`).
The frame is stored in the crop's space and mapped back through the geometry onto the source.
"""

import hashlib
from typing import Any, List, Optional, Sequence, Tuple

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.geometry.logic import map_coords_to_geometry
from negpy.features.leak.models import LEAK_SIDE_POINTS, LeakConfig

# Film contrast that turns a density step into exposure: a color negative's slope.
LEAK_GAMMA = 0.6
# Readings along each side, and the patch radius as a fraction of the short side: small enough to
# fit the strip between the perforations and the picture.
_SAMPLES_PER_SIDE = 48
_PATCH_FRACTION = 0.004
# Running median, then running mean, over the readings of one side: a median drops a reading on
# edge print or a perforation, the mean takes out grain.
_MEDIAN_SPAN = 7
_MEAN_SPAN = 5
# The leak map is smooth, so it is solved on a small grid and resized to the image.
_MAP_LONG_SIDE = 128
_MIN_T = 1e-6
_INVERSE_STEPS = 12

Point = Tuple[float, float]


def leak_active(config: LeakConfig) -> bool:
    return bool(config.leak_apply) and len(config.leak_frame) >= 4 and len(config.leak_frame) % 4 == 0 and config.leak_strength > 0.0


def _geometry_key(geometry: Any) -> tuple:
    return (
        geometry.crop_rect,
        int(geometry.rotation),
        float(geometry.fine_rotation),
        bool(geometry.flip_horizontal),
        bool(geometry.flip_vertical),
        float(geometry.distortion_k1),
        float(geometry.converge_v),
        float(geometry.converge_h),
    )


def leak_token(settings: Any) -> str:
    """Identity of the active correction on a WorkspaceConfig, folded into the render source hash.
    The geometry is in it, because the frame maps through the crop and the straightening. Empty when inactive."""
    leak = settings.leak
    if not leak_active(leak):
        return ""
    raw = repr((leak.leak_frame, leak.leak_anchor, round(leak.leak_strength, 6), _geometry_key(settings.geometry))).encode()
    return "|leak:" + hashlib.blake2b(raw, digest_size=8).hexdigest()


def anchored_frame(config: LeakConfig, crop_rect: Optional[Sequence[float]]) -> Tuple[Point, ...]:
    """The frame in the crop's space for a frame with *crop_rect*: moved and scaled from the rect it
    was placed against onto this crop. Without an anchor or a crop, the frame stays where it was placed."""
    if not config.leak_anchor or not crop_rect:
        return config.leak_frame
    ax1, ay1, ax2, ay2 = config.leak_anchor
    cx1, cy1, cx2, cy2 = crop_rect
    sx = (cx2 - cx1) / max(ax2 - ax1, 1e-9)
    sy = (cy2 - cy1) / max(ay2 - ay1, 1e-9)
    return tuple((cx1 + (x - ax1) * sx, cy1 + (y - ay1) * sy) for x, y in config.leak_frame)


def source_points(points: Sequence[Point], geometry: Any, shape: Tuple[int, int]) -> Tuple[Point, ...]:
    """*points* in the crop's space mapped back onto the source, normalized: the inverse of the
    geometry stage, solved by Newton steps on `map_coords_to_geometry`."""

    def forward(x: float, y: float) -> np.ndarray:
        return np.asarray(
            map_coords_to_geometry(
                x,
                y,
                shape,
                geometry.rotation,
                geometry.fine_rotation,
                geometry.flip_horizontal,
                geometry.flip_vertical,
                distortion_k1=geometry.distortion_k1,
                converge_v=geometry.converge_v,
                converge_h=geometry.converge_h,
            )
        )

    eps = 1e-4
    out: List[Point] = []
    for tx, ty in points:
        target = np.array([tx, ty])
        p = np.array([0.5, 0.5])
        for _ in range(_INVERSE_STEPS):
            f = forward(*p)
            err = target - f
            if np.max(np.abs(err)) < 1e-7:
                break
            jac = np.stack([(forward(p[0] + eps, p[1]) - f) / eps, (forward(p[0], p[1] + eps) - f) / eps], axis=1)
            try:
                p = p + np.linalg.solve(jac, err)
            except np.linalg.LinAlgError:
                break
        out.append((float(np.clip(p[0], 0.0, 1.0)), float(np.clip(p[1], 0.0, 1.0))))
    return tuple(out)


def default_points(rect: Tuple[float, float, float, float], per_side: int = LEAK_SIDE_POINTS) -> Tuple[Point, ...]:
    """A measuring frame on the rectangle (x1, y1, x2, y2), with *per_side* points between corners."""
    x1, y1, x2, y2 = rect
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    loop: List[Point] = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for k in range(per_side + 1):
            t = k / (per_side + 1)
            loop.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return tuple(loop)


def corner_indices(n_points: int) -> Tuple[int, int, int, int]:
    step = n_points // 4
    return 0, step, 2 * step, 3 * step


def sides(points: Sequence[Point]) -> Tuple[List[Point], List[Point], List[Point], List[Point]]:
    """The loop as (top TL->TR, bottom BL->BR, left TL->BL, right TR->BR)."""
    n = len(points)
    tl, tr, br, bl = corner_indices(n)
    pts = list(points)
    top = pts[tl : tr + 1]
    right = pts[tr : br + 1]
    bottom = pts[br : bl + 1][::-1]
    left = (pts[bl:] + [pts[0]])[::-1]
    return top, bottom, left, right


def _arc_positions(poly: Sequence[Point], n: int, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """*n* points evenly spaced by length along *poly* (pixel lengths), and their 0-1 positions."""
    w, h = size
    p = np.asarray(poly, dtype=np.float64) * (w, h)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    s = np.linspace(0.0, 1.0, n)
    at = s * total
    xs = np.interp(at, cum, p[:, 0])
    ys = np.interp(at, cum, p[:, 1])
    return np.stack([xs, ys], axis=1), s


def _running(values: np.ndarray, span: int, reduce) -> np.ndarray:
    half = span // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    return reduce(np.stack([padded[i : i + len(values)] for i in range(span)]), axis=0)


def side_densities(img: np.ndarray, poly: Sequence[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed per-channel densities along one side, and their 0-1 positions on it."""
    h, w = img.shape[:2]
    r = max(2, int(round(_PATCH_FRACTION * min(h, w))))
    pos, s = _arc_positions(poly, _SAMPLES_PER_SIDE, (w, h))
    values = np.empty((len(pos), 3), dtype=np.float64)
    for i, (x, y) in enumerate(pos):
        xi = int(np.clip(round(x), r, max(r, w - r - 1)))
        yi = int(np.clip(round(y), r, max(r, h - r - 1)))
        patch = img[max(0, yi - r) : yi + r + 1, max(0, xi - r) : xi + r + 1, :3]
        values[i] = np.median(patch.reshape(-1, 3), axis=0)
    density = -np.log10(np.maximum(values, _MIN_T))
    density = _running(density, _MEDIAN_SPAN, np.median)
    density = _running(density, _MEAN_SPAN, np.mean)
    return density, s


def _inverse_bilinear(x: np.ndarray, y: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(u, v) of each point in the quad TL, TR, BR, BL by Newton steps, clamped to the quad."""
    a, b, c, d = quad
    lo, hi = quad.min(axis=0), quad.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    u = np.clip((x - lo[0]) / span[0], 0.0, 1.0)
    v = np.clip((y - lo[1]) / span[1], 0.0, 1.0)
    for _ in range(8):
        px = (1 - u) * (1 - v) * a[0] + u * (1 - v) * b[0] + u * v * c[0] + (1 - u) * v * d[0]
        py = (1 - u) * (1 - v) * a[1] + u * (1 - v) * b[1] + u * v * c[1] + (1 - u) * v * d[1]
        dxu = (1 - v) * (b[0] - a[0]) + v * (c[0] - d[0])
        dyu = (1 - v) * (b[1] - a[1]) + v * (c[1] - d[1])
        dxv = (1 - u) * (d[0] - a[0]) + u * (c[0] - b[0])
        dyv = (1 - u) * (d[1] - a[1]) + u * (c[1] - b[1])
        det = dxu * dyv - dxv * dyu
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        ex, ey = x - px, y - py
        u = u + (ex * dyv - ey * dxv) / det
        v = v + (ey * dxu - ex * dyu) / det
        # Clamped each step: outside the quad the side's value carries straight on.
        u, v = np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    return u, v


def measure(img: np.ndarray, points: Sequence[Point], strength: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(base transmittance per channel, leak map on a small grid, side densities) for the frame
    *points* in source-normalized coords, or None for a frame that is not a loop."""
    if len(points) < 4 or len(points) % 4 or strength <= 0.0:
        return None
    top, bottom, left, right = sides(points)
    profiles = [side_densities(img, poly) for poly in (top, bottom, left, right)]
    # A leak only adds density, so the thinnest reading of each channel is the clear base.
    d_ref = np.min(np.concatenate([d for d, _ in profiles]), axis=0)
    leaks = [(np.power(10.0, (d - d_ref) / LEAK_GAMMA) - 1.0) * strength for d, _ in profiles]
    (lt, st), (lb, sb), (ll, sl), (lr, sr) = [(lk, s) for lk, (_, s) in zip(leaks, profiles)]

    h, w = img.shape[:2]
    scale = _MAP_LONG_SIDE / max(h, w)
    gw, gh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    gx, gy = np.meshgrid((np.arange(gw) + 0.5) / gw, (np.arange(gh) + 0.5) / gh)
    tl, tr, br, bl = corner_indices(len(points))
    quad = np.asarray([points[i] for i in (tl, tr, br, bl)], dtype=np.float64)
    u, v = _inverse_bilinear(gx, gy, quad)

    def along(s: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t, s, values[:, c]) for c in range(3)], axis=-1)

    top_u, bottom_u, left_v, right_v = along(st, lt, u), along(sb, lb, u), along(sl, ll, v), along(sr, lr, v)
    c00, c10 = (lt[0] + ll[0]) / 2, (lt[-1] + lr[0]) / 2
    c01, c11 = (lb[0] + ll[-1]) / 2, (lb[-1] + lr[-1]) / 2
    u3, v3 = u[..., None], v[..., None]
    patch = (
        (1 - v3) * top_u
        + v3 * bottom_u
        + (1 - u3) * left_v
        + u3 * right_v
        - ((1 - u3) * (1 - v3) * c00 + u3 * (1 - v3) * c10 + (1 - u3) * v3 * c01 + u3 * v3 * c11)
    )
    t_ref = np.power(10.0, -d_ref)
    return t_ref, np.maximum(patch, 0.0).astype(np.float32), np.stack([d for d, _ in profiles])


def apply_leak(image: ImageBuffer, settings: Any) -> ImageBuffer:
    """Take the measured leak out of the linear source, for a WorkspaceConfig. No-op when inactive."""
    leak = settings.leak
    if not leak_active(leak):
        return image
    frame = anchored_frame(leak, settings.geometry.crop_rect)
    found = measure(image, source_points(frame, settings.geometry, image.shape[:2]), leak.leak_strength)
    if found is None:
        return image
    t_ref, grid, _ = found
    h, w = image.shape[:2]
    leak_map = negpy_core.resize_linear(np.ascontiguousarray(grid), (w, h))
    t = tuple(float(max(v, _MIN_T)) for v in t_ref)
    return negpy_core.leak_correct(
        np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(leak_map, dtype=np.float32), t, LEAK_GAMMA
    )
