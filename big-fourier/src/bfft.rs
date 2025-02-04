//! Copyright (c) 2024, ETH Zurich


use bytemuck::Pod;
use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftDirection, FftNum};

use crate::{npy, Extents};

/// Determine how to split the data of length `n` into evenly sized chunks.
///
/// `n` must be a power of two. The number of chunks will be smaller than or
/// equal to the length of a chunk.
pub fn split_sizes(n: usize) -> Extents {
    // todo, possible extension: small prime factors suffice, no large difference between 2 and 3.

    assert_eq!(n & (n - 1), 0, "N must be a power of two.");
    let l2 = n.ilog2() as usize;

    let r = 1 << ((l2 + 1) / 2);
    let q = 1 << (l2 / 2);

    Extents { outer: q, inner: r }
}

/// Out-of-core FFT & IFFT
///
/// If the input is no longer needed then it's safe to pass the same path as `infile` and `scratchfile`.
/// This function therefore also assumes that `infile` is writable.
pub fn fft<T>(
    infile: &mut std::fs::File,
    outfile: &mut std::fs::File,
    scratchfile: &mut std::fs::File,
    direction: FftDirection,
) -> anyhow::Result<()>
where
    T: FftNum + Float + Pod,
    Complex<T>: std::ops::AddAssign<Complex<T>>,
    Complex<T>: std::ops::MulAssign<Complex<T>>,
{
    let header = npy::read_header(infile)?;
    let n = header.shape[0];
    let extents = split_sizes(n);

    let mut planner = rustfft::FftPlanner::<T>::new();
    let fft_r = planner.plan_fft(extents.inner, direction);
    let fft_q = planner.plan_fft(extents.outer, direction);

    let mut scratch_fft = {
        let len = fft_r
            .get_inplace_scratch_len()
            .max(fft_q.get_inplace_scratch_len());
        vec![Complex::new(T::zero(), T::zero()); len]
    };

    let mut buffer = {
        let len = extents.inner.max(extents.outer);
        vec![Complex::new(T::zero(), T::zero()); len]
    };

    // We have 3 files available. Input, Scratch and Output.
    // The process requiers 3 transpose operations which are out-of-place, and two fft operations which are in-place.
    // Strategy:
    // * Transpose Input -> Output
    // * FFT Output
    // * Scale Output
    // * Transpose Output -> Scratch
    // * FFT Scratch
    // * Transpose Scratch -> Output

    npy::transpose::<T>(infile, outfile, extents)?;

    let (ifft_factor, twiddle_sign) = match direction {
        FftDirection::Forward => (T::one(), 1),
        FftDirection::Inverse => (T::from_usize(n).unwrap().recip(), -1),
    };

    npy::modify_inplace(outfile, &mut buffer[..extents.outer], |idx, data| {
        fft_q.process_with_scratch(data, &mut scratch_fft);

        assert_eq!(data.len(), extents.outer);
        let i = idx / extents.outer;
        for j in 0..extents.outer {
            // ifft normalization happens here: divide by n
            data[j] *= twiddle((i * j) as i32 * twiddle_sign, n as i32) * ifft_factor;
        }
    })?;

    npy::transpose::<T>(outfile, scratchfile, extents.tranposed())?;

    npy::modify_inplace(scratchfile, &mut buffer[..extents.inner], |_, data| {
        assert_eq!(data.len(), extents.inner);
        fft_r.process_with_scratch(data, &mut scratch_fft);
    })?;

    npy::transpose::<T>(scratchfile, outfile, extents)?;

    Ok(())
}

/// Compute an FFT with the $\mathcal O(n^2)$ algorithm, useful for validating.
///
/// This uses the [NumPy convention](https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details):
/// $ y_k = \sum_{i=0}^{N-1} a_i \omega^{-i k}_N $
pub fn slow_fft<T>(a: &[Complex<T>]) -> Vec<Complex<T>>
where
    T: Float,
    Complex<T>: std::ops::AddAssign<Complex<T>>,
{
    let n = a.len();
    let mut out = vec![Complex::new(T::zero(), T::zero()); n];
    for k in 0..n {
        for i in 0..n {
            out[k] += a[i] * twiddle((k * i) as i32, n as i32);
        }
    }
    out
}

/// Compute the twiddle factor $\omega^{-k}_n$. Note the minus sign on the exponent.
pub fn twiddle<T: Float>(k: i32, n: i32) -> Complex<T> {
    let angle = -2. * std::f64::consts::PI * k as f64 / n as f64;
    Complex::from_polar(T::one(), T::from(angle).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split() {
        for i in 0..63 {
            let n = 1 << i;
            let extents = split_sizes(n);
            assert_eq!(extents.product(), n);
            assert_eq!(extents.inner & (extents.inner - 1), 0);
            assert_eq!(extents.outer & (extents.outer - 1), 0);
        }
    }

    /// Helper function to open a file in read/write mode
    fn open_rw(path: &str) -> std::io::Result<std::fs::File> {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
    }

    #[test]
    fn test_fft() {
        // Setup files in a temporary directory
        let dir = tempfile::tempdir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let mut infile = open_rw("in.npy").unwrap();
        let mut outfile = open_rw("out.npy").unwrap();
        let mut scratchfile = open_rw("scratch.npy").unwrap();

        // This is the vector to be FFT'd
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(0.0, 1.0),
        ];
        let mut buffer = vec![Complex::new(0.0, 0.0); input.len()];

        // Write the input array to "in.npy"
        npy::generate_zeros_array::<f64>(&mut infile, input.len()).unwrap();
        npy::modify_inplace::<f64>(&mut infile, &mut buffer, |_, data| {
            data[..input.len()].copy_from_slice(&input);
        })
        .unwrap();

        // Compute the FFT and store it in "out.npy"
        fft::<f64>(
            &mut infile,
            &mut outfile,
            &mut scratchfile,
            FftDirection::Forward,
        )
        .unwrap();

        // Calculate the FFT using the slow reference algorithm.
        let reference_output = slow_fft(&input);

        // Compare "out.npy" to the reference FFT
        npy::modify_inplace::<f64>(&mut outfile, &mut buffer, |_, data| {
            for (a, b) in data.iter().zip(reference_output.iter()) {
                assert!((a - b).norm() < 0.0001);
            }
        })
        .unwrap();

        // Compute the inverse FFT and store the result in "in.npy"
        fft::<f64>(
            &mut outfile,
            &mut infile,
            &mut scratchfile,
            FftDirection::Inverse,
        )
        .unwrap();

        // Compare the result of the inverse FFT to the original input vector
        npy::modify_inplace::<f64>(&mut infile, &mut buffer, |_, data| {
            for (a, b) in data.iter().zip(input.iter()) {
                assert!((a - b).norm() < 0.0001);
            }
        })
        .unwrap();
    }
}
