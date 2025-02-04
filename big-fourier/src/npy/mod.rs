//! Copyright (c) 2024, ETH Zurich

//! Working with NPY files
//!
//! This is the file format used by NumPy in `np.save('data.npy', data)`.
//! See the official [NumPy documentation][numpydoc] for more information about the format.
//!
//! The functions here are all optimized to work with arrays bigger than the available RAM.
//! Since such files can't easily be generated with Python, there's also a function to generate
//! arrays.
//!
//! [numpydoc]: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

use anyhow::{bail, Context};

use bytemuck::{cast_slice, cast_slice_mut, Pod};
use header::Value;
use num_complex::Complex;
use num_traits::{Num, Zero};
use rustix::fs::{fadvise, Advice};

use crate::Extents;

mod header;

#[derive(Debug, PartialEq, Eq)]
pub struct NpyHeader {
    /// Dtype specifier string as described in the [NumPy documentation](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    pub dtype: String,
    /// The list of array sizes
    pub shape: Vec<usize>,
    /// Number of bytes in the header, including all padding. The actual data will start at this offset.
    pub header_length: usize,
}

/// Parse header information given the beginning of a NPY file.
///
/// `data` doesn't need to contain the entire file, it suffices to cover just the header
/// of the file. Anything beyond the header is ignored.
fn parse_header(data: &[u8]) -> anyhow::Result<NpyHeader> {
    let nomresult: nom::IResult<&[u8], header::Value> = header::parse_header(data);
    let (header, len) = match nomresult {
        nom::IResult::Done(rest, val) => (val, data.len() - rest.len()),
        nom::IResult::Error(e) => Err(e).context("Cannot pare .npy header")?,
        nom::IResult::Incomplete(_) => {
            bail!("not enough data. TODO: fix this later if this issue occurs in practice")
        }
    };

    let mut dtype: Option<String> = None;
    let mut shape: Option<Vec<usize>> = None;

    if let Value::Map(fields) = header {
        for (name, field) in fields {
            if name == "fortran_order" && field == Value::Bool(true) {
                bail!("Fortran order is not supported");
            }
            if name == "descr" {
                if let Value::String(val) = &field {
                    dtype = Some(val.clone());
                }
            }
            if name == "shape" {
                if let Value::List(val) = field {
                    shape = Some(
                        val.into_iter()
                            .map(|val| {
                                if let Value::Integer(val) = val {
                                    Ok(val as usize)
                                } else {
                                    bail!("expected the shape list to only contain integers")
                                }
                            })
                            .collect::<anyhow::Result<Vec<_>>>()?,
                    );
                }
            }
        }
    }

    let info = NpyHeader {
        dtype: dtype.context("Header does not contain dtype")?,
        shape: shape.context("Header does not contain shape")?,
        header_length: len,
    };
    Ok(info)
}

/// Read the header information from an open NPY file.
pub fn read_header(file: &mut std::fs::File) -> anyhow::Result<NpyHeader> {
    file.rewind().context("rewinding to start")?;
    let mut header_buffer = vec![0u8; 1024];
    let bytes_read = file
        .read(&mut header_buffer)
        .context("reading into buffer")?;
    parse_header(&header_buffer[..bytes_read])
}

/// Generate a header for a NPY file of version 1.
///
/// See [numpy.org](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#format-version-1-0)
/// for the format documentation.
///
/// We always generate a header with a length of 128 bytes because the `data_length` is never so big as to require more space.
pub fn generate_header<T>(data_length: usize) -> Vec<u8> {
    let mut header = vec![0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    let complex_size = std::mem::size_of::<T>() * 2;
    let fields = format!(
        "{{'descr': '<c{complex_size}', 'fortran_order': False, 'shape': ({data_length},), }}"
    );
    let fields_bytes = fields.as_bytes();

    // header_len field. We subtract 2 to account for the 2 bytes of this header_len field.
    header.extend_from_slice(&(128 - 2 - header.len() as u16).to_le_bytes());
    header.extend_from_slice(fields_bytes);

    // pad with zeros up to 127 bytes, and then finish with a newline.
    header.extend_from_slice(&vec![0x20; 127 - header.len()]);
    header.push(0x0a);

    header
}

/// Generate a complex array containing all zeros and store it in a NPY file.
pub fn generate_zeros_array<T: Pod + Num>(
    mut output: impl Write,
    length: usize,
) -> anyhow::Result<()> {
    let mut writer = BufWriter::new(&mut output);
    writer.write_all(&generate_header::<T>(length))?;
    for _ in 0..length {
        let val = Complex::<T>::zero();
        writer.write_all(cast_slice(&[val]))?;
    }

    Ok(())
}

/// Transpose an array given in an NPY file. The array in the file is expected to be
/// 1d, but for the purposes of transposing it it will be interpreted as a matrix with
/// dimensions given in `input_extents`.
///
/// This algorithm is adapted from Godard et al. (2021).
pub fn transpose<T>(
    input: &mut std::fs::File,
    output: &mut std::fs::File,
    input_extents: Extents,
) -> anyhow::Result<()>
where
    T: Pod,
    Complex<T>: num_traits::Zero,
{
    let header = read_header(input).context("Could not read .npy header")?;
    assert_eq!(header.shape.len(), 1, "Only 1d arrays are supported");
    assert_eq!(header.shape[0], input_extents.product());

    let entry_size = std::mem::size_of::<Complex<T>>();
    assert_eq!(header.dtype, format!("<c{entry_size}"));

    // The chunk size is hardcoded for now. TODO: experiment with this
    // later for possible performance gains.
    let tile = Extents {
        outer: (4096 * 2).min(input_extents.outer),
        inner: (1024 * 2).min(input_extents.inner),
    };

    {
        // Copy the header to the output file.
        // Note that the header contains shape information about the python array,
        // but we don't change that since from the python perspective it is still
        // used as a 1d array after this transpose operation.
        input.rewind()?;
        output.rewind()?;
        let mut header = vec![0u8; header.header_length];
        input
            .read_exact(&mut header)
            .context("reading header from input file")?;
        output
            .write_all(&header)
            .context("writing header to output file")?;
    }

    let zero = <Complex<T> as num_traits::Zero>::zero();
    let mut in_buffer = vec![zero; tile.product()];
    let mut out_buffer = vec![zero; tile.product()];

    for bx in (0..input_extents.inner).step_by(tile.inner) {
        for by in (0..input_extents.outer).step_by(tile.outer) {
            for y in 0..tile.outer {
                let blocksize = 256; // todo: tune this parameter
                if y % blocksize == 0 {
                    for yp in y..tile.outer.min(y + blocksize) {
                        fadvise(
                            &input,
                            (((by + yp) * input_extents.inner + bx) * entry_size
                                + header.header_length) as u64,
                            (tile.inner * entry_size) as u64,
                            Advice::WillNeed,
                        )
                        .unwrap();
                    }
                }

                let start_index =
                    ((by + y) * input_extents.inner + bx) * entry_size + header.header_length;

                input.seek(std::io::SeekFrom::Start(start_index as u64))?;

                let nr_bytes = input.read(cast_slice_mut(
                    &mut in_buffer[y * tile.inner..(y + 1) * tile.inner],
                ))?;
                assert_eq!(
                    nr_bytes,
                    tile.inner * entry_size,
                    "read returned fewer bytes"
                );
            }

            transpose::transpose(&in_buffer, &mut out_buffer, tile.inner, tile.outer);

            for x in 0..tile.inner {
                let start_index =
                    ((x + bx) * input_extents.outer + by) * entry_size + header.header_length;
                output.seek(std::io::SeekFrom::Start(start_index as u64))?;
                output.write_all(cast_slice(
                    &out_buffer[x * tile.outer..(x + 1) * tile.outer],
                ))?;
            }
        }
    }

    Ok(())
}

/// Apply a function to conntiguous chunks of complex number in the given NPY file array and
/// save the transformed array back to the same file.
///
/// The mutator function receives an index and a mutable slice to the data in the current
/// chunk. The given index is the *offset of the first entry* in the current chunk, not the
/// chunk index.
///
/// If the total length of the array is not evenly divisible by the length of the given buffer,
/// then the last slice passed to the mutator function will be shorter. All other invocations are
/// guaranteed to receive the full buffer length.
pub fn modify_inplace<T: num_traits::Num + Clone + Copy + Pod>(
    file: &mut std::fs::File,
    buffer: &mut [Complex<T>],
    mut mutator: impl FnMut(usize, &mut [Complex<T>]),
) -> anyhow::Result<()> {
    // TODO: remove this function in favor of ChunkedEditor (or reimplement in terms of it)

    // TODO: prefetch the next chunk while processing the current one via fadvise.

    // Mark the whole file as processed sequentially. Todo: check if this is actually faster
    fadvise(&file, 0, 0, Advice::Sequential).unwrap();

    let header = read_header(file).context("Could not read .npy header")?;
    let entry_size: usize = std::mem::size_of::<Complex<T>>();
    debug_assert_eq!(entry_size, std::mem::size_of::<T>() * 2);
    assert_eq!(header.shape.len(), 1);

    for i in (0..header.shape[0]).step_by(buffer.len()) {
        let nr_elements_in_chunk = buffer.len().min(header.shape[0] - i);
        // Focus on the range of the buffer that we need in this iteration
        let buffer = &mut buffer[..nr_elements_in_chunk];

        file.seek(SeekFrom::Start(
            (i * entry_size + header.header_length) as u64,
        ))
        .context("seeking to next chunk start")?;
        file.read_exact(cast_slice_mut(buffer))
            .context("reading into buffer")?;

        mutator(i, buffer);
        file.seek(SeekFrom::Start(
            (i * entry_size + header.header_length) as u64,
        ))
        .context("seeking back to chunk start")?;
        file.write_all(cast_slice(buffer))
            .context("writing transformed buffer back to file")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER_BINARY: &[u8] = &[
        0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7b, 0x27, 0x64, 0x65, 0x73,
        0x63, 0x72, 0x27, 0x3a, 0x20, 0x27, 0x3c, 0x63, 0x31, 0x36, 0x27, 0x2c, 0x20, 0x27, 0x66,
        0x6f, 0x72, 0x74, 0x72, 0x61, 0x6e, 0x5f, 0x6f, 0x72, 0x64, 0x65, 0x72, 0x27, 0x3a, 0x20,
        0x46, 0x61, 0x6c, 0x73, 0x65, 0x2c, 0x20, 0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3a,
        0x20, 0x28, 0x35, 0x33, 0x36, 0x38, 0x37, 0x30, 0x39, 0x31, 0x32, 0x2c, 0x29, 0x2c, 0x20,
        0x7d, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x0a,
    ];

    #[test]
    fn test_generate_header() {
        let header = generate_header::<f64>(536870912);
        assert_eq!(header, HEADER_BINARY);
    }

    #[test]
    fn test_parse_header() {
        let header = parse_header(HEADER_BINARY).unwrap();
        assert_eq!(
            header,
            NpyHeader {
                dtype: "<c16".to_string(),
                shape: vec![536870912],
                header_length: 128,
            }
        );
    }
}
