use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::leak;

/// `apply_leak`'s per-pixel pass: `img` and `leak_map` are HxWx3 float32 of one shape.
#[pyfunction]
fn leak_correct<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, f32>,
    leak_map: PyReadonlyArrayDyn<'py, f32>,
    t_ref: (f32, f32, f32),
    gamma: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let [_, _, 3] = img.shape()[..] else {
        return Err(PyValueError::new_err("leak_correct needs an HxWx3 image"));
    };
    if img.shape() != leak_map.shape() {
        return Err(PyValueError::new_err(
            "leak_correct needs a leak map of the image's shape",
        ));
    }
    if gamma.is_nan() || gamma <= 0.0 || t_ref.0 <= 0.0 || t_ref.1 <= 0.0 || t_ref.2 <= 0.0 {
        return Err(PyValueError::new_err(
            "leak_correct needs a positive gamma and base",
        ));
    }
    let src = img
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous image"))?;
    let map = leak_map
        .as_slice()
        .map_err(|_| PyValueError::new_err("needs a contiguous leak map"))?;
    let out = PyArrayDyn::<f32>::zeros(py, img.shape(), false);
    {
        let mut rw = out.readwrite();
        let dst = rw
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let base = [t_ref.0, t_ref.1, t_ref.2];
        super::detach(py, dst.len(), || {
            leak::leak_correct(src, map, base, gamma, dst)
        });
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(leak_correct, m)?)?;
    Ok(())
}
