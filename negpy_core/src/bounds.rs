//! The density bounds sampling of negpy/features/exposure/normalization.py.

use crate::stats::np_percentiles_f64;

/// `np.percentile(grid[:, :, ch], qs)` for each channel of an HxWx3 float32 grid, with float64
/// interpolation (an `np.float64` or sequence q).
pub fn channel_percentiles(grid: &[f32], qs: &[f64]) -> [Vec<f64>; 3] {
    std::array::from_fn(|ch| {
        let plane: Vec<f32> = grid.iter().skip(ch).step_by(3).copied().collect();
        np_percentiles_f64(&plane, qs)
    })
}
