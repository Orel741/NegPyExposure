use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bounds;

fn grid_of<'a>(grid: &'a PyReadonlyArrayDyn<'_, f32>) -> PyResult<&'a [f32]> {
    if grid.ndim() != 3 || grid.shape()[2] != 3 {
        return Err(PyValueError::new_err("needs an HxWx3 array"));
    }
    grid.as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous array"))
}

/// `np.percentile(grid[:, :, ch], qs)` per channel of an HxWx3 float32 grid, float64 interpolation.
#[pyfunction]
fn channel_percentiles(
    py: Python<'_>,
    grid: PyReadonlyArrayDyn<'_, f32>,
    qs: Vec<f64>,
) -> PyResult<[Vec<f64>; 3]> {
    let g = grid_of(&grid)?;
    Ok(super::detach(py, g.len(), || {
        bounds::channel_percentiles(g, &qs)
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(channel_percentiles, m)?)
}
