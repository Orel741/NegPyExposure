//! Light-leak removal on the linear negative scan: the leak is light added to the film, so it is
//! taken out as exposure, not as a density offset.

use crate::mathf::{exp_f64, ln_f64};

/// One channel value: `t` scan transmittance, `t_ref` the clear base, `l` the leak in units of
/// the base fog exposure and `gamma` the film contrast. With x = (t / t_ref)^(-1/gamma), the
/// exposure x - l is printed; below the leaked base the pixel takes the base's density shift.
#[inline(always)]
pub fn correct(t: f32, t_ref: f32, l: f32, gamma: f64) -> f32 {
    let l = l.max(0.0) as f64;
    let t64 = (t.max(0.0) / t_ref) as f64;
    // q = 1 / x = (t / t_ref)^(1/gamma); 0 for a black pixel.
    let q = if t64 > 0.0 {
        exp_f64(ln_f64(t64) / gamma)
    } else {
        0.0
    };
    let r = (1.0 - l * q).max(1.0 / (1.0 + l));
    (t as f64 * exp_f64(-gamma * ln_f64(r))) as f32
}

/// `correct` on every channel of an HxWx3 image, with an HxWx3 leak map and a per-channel base.
pub fn leak_correct(src: &[f32], leak: &[f32], t_ref: [f32; 3], gamma: f64, dst: &mut [f32]) {
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 14;
    dst.par_chunks_mut(3 * BLOCK)
        .zip(src.par_chunks(3 * BLOCK).zip(leak.par_chunks(3 * BLOCK)))
        .for_each(move |(d, (s, l))| {
            for ((d, s), l) in d
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(s.as_chunks::<3>().0)
                .zip(l.as_chunks::<3>().0)
            {
                for c in 0..3 {
                    d[c] = correct(s[c], t_ref[c], l[c], gamma);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_leak_keeps_the_pixel() {
        for t in [0.0f32, 0.01, 0.2, 0.7, 1.0] {
            assert_eq!(correct(t, 0.5, 0.0, 0.6), t);
        }
    }

    #[test]
    fn the_leaked_base_returns_to_the_clear_base() {
        let (t_ref, gamma, l) = (0.5f32, 0.6f64, 1.5f32);
        let leaked = t_ref * (1.0 + l).powf(-gamma as f32);
        assert!((correct(leaked, t_ref, l, gamma) - t_ref).abs() < 1e-5);
    }

    #[test]
    fn a_dense_pixel_barely_moves() {
        let out = correct(0.005, 0.5, 1.0, 0.6);
        assert!(out > 0.005 && out < 0.00502);
    }
}
