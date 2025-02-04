//! Copyright (c) 2024, ETH Zurich

//! Chunkmat data format

use std::io::{Read, Seek, SeekFrom, Write};

use bincode::Options;
use bytemuck::{cast_slice, cast_slice_mut, Pod};
use num_complex::Complex;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub chunk_size: u32,
    pub dtype: String,
    pub state: TranposeState,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[repr(u8)]
pub enum TranposeState {
    RowMayor = 1,
    ColMayor = 2,
}

#[derive(Debug)]
struct ChunkInfo {
    /// Start index in the linearized array
    start: u64,
    /// One-past-end index in the linearized array
    end: u64,
    /// Width of this chunk
    width: u64,
    /// Height of this chunk
    height: u64,
}

impl Header {
    fn nr_chunks_x(&self) -> u32 {
        (self.width + self.chunk_size - 1) / self.chunk_size
    }

    fn nr_chunks_y(&self) -> u32 {
        (self.height + self.chunk_size - 1) / self.chunk_size
    }

    /// Start index and one-past-end index of the range representing this chunk
    fn get_chunk_info(&self, x: u32, y: u32) -> ChunkInfo {
        let x = x as u64;
        let y = y as u64;

        let w = self.nr_chunks_x() as u64;
        let h = self.nr_chunks_y() as u64;

        // The chunks at the upper x and y edges might not be the same size as the inner
        // chunks. Compute cw and ch, the width and height of the chunk located at x, y.
        let cs = self.chunk_size as u64;
        let cw = if x == w - 1 {
            self.width as u64 - x * cs
        } else {
            cs
        };
        let ch = if y == h - 1 {
            self.height as u64 - y * cs
        } else {
            cs
        };

        let start = y * cs * self.width as u64 + x * cs * ch;

        ChunkInfo {
            start,
            end: start + cw * ch,
            width: cw,
            height: ch,
        }
    }
}

/// Number of bytes used to store a value of this dtype
fn size_of_dtype(dtype: &str) -> usize {
    match dtype {
        "<c16" => 16,
        "<c8" => 8,
        _ => unimplemented!("we don't care about other dtypes"),
    }
}

pub struct ChunkmatHandle<F> {
    header: Header,
    header_size: u64,
    file: F,
}

fn bincode_options() -> impl bincode::Options {
    bincode::DefaultOptions::new().allow_trailing_bytes()
}
fn transpose<T>(mut file: impl Read + Write + Seek, header: &Header, header_size: u64)
where
    T: Zero + Pod,
{
    let cs = header.chunk_size as usize;
    let mut buffer_in = vec![T::zero(); cs * cs];
    let mut buffer_out = vec![T::zero(); cs * cs];
    let w = header.nr_chunks_x();
    let h = header.nr_chunks_y();

    let tsize = std::mem::size_of::<T>() as u64;

    for y in 0..h {
        for x in 0..w {
            let ci = header.get_chunk_info(x, y);

            let chunk_entries = (ci.width * ci.height) as usize;
            file.seek(SeekFrom::Start(ci.start * tsize + header_size))
                .unwrap();
            file.read_exact(cast_slice_mut(&mut buffer_in[..chunk_entries]))
                .unwrap();

            let decoded: &[f32] = cast_slice(&buffer_in[..chunk_entries]);

            let mut input_width = ci.width as usize;
            let mut input_height = ci.height as usize;
            if header.state == TranposeState::ColMayor {
                std::mem::swap(&mut input_width, &mut input_height);
            }
            transpose::transpose(
                &mut buffer_in[..chunk_entries],
                &mut buffer_out[..chunk_entries],
                input_width,
                input_height,
            );
            file.seek(SeekFrom::Start(ci.start * tsize + header_size))
                .unwrap();
            file.write_all(cast_slice(&buffer_out[..chunk_entries]))
                .unwrap();
        }
    }
}

fn extract_vector<F, T>(
    mut file: F,
    header: &Header,
    header_size: u64,
    vector_idx: u32,
    vec: &mut Vec<T>,
) where
    T: Zero + Pod,
    F: Read + Write + Seek,
{
    let tsize = std::mem::size_of::<T>() as u64;

    match header.state {
        TranposeState::RowMayor => {
            vec.resize(header.width as usize, T::zero());

            let w = header.nr_chunks_x();
            let y = vector_idx / header.chunk_size;
            for x in 0..w {
                let ci = header.get_chunk_info(x, y);
                let inner_y = (vector_idx - y * header.chunk_size) as u64;
                let inner_offset = inner_y * ci.width as u64;
                file.seek(SeekFrom::Start(
                    (ci.start + inner_offset) * tsize + header_size,
                ))
                .unwrap();
                let vs = header.chunk_size as usize * x as usize;
                file.read_exact(cast_slice_mut(&mut vec[vs..vs + ci.width as usize]))
                    .unwrap();
            }
        }
        TranposeState::ColMayor => {
            vec.resize(header.height as usize, T::zero());

            let h = header.nr_chunks_y();
            let x = vector_idx / header.chunk_size;
            for y in 0..h {
                let ci = header.get_chunk_info(x, y);
                let inner_x = (vector_idx - x * header.chunk_size) as u64;
                let inner_offset = inner_x * ci.height as u64;
                file.seek(SeekFrom::Start(
                    (ci.start + inner_offset) * tsize + header_size,
                ))
                .unwrap();
                let vs = header.chunk_size as usize * y as usize;
                file.read_exact(cast_slice_mut(&mut vec[vs..vs + ci.height as usize]))
                    .unwrap();
            }
        }
    }
}

fn save_vector<F, T>(mut file: F, header: &Header, header_size: u64, vector_idx: u32, vec: &[T])
where
    T: Pod,
    F: Read + Write + Seek,
{
    let tsize = std::mem::size_of::<T>() as u64;

    match header.state {
        TranposeState::RowMayor => {
            assert_eq!(vec.len(), header.width as usize);

            let w = header.nr_chunks_x();
            let y = vector_idx / header.chunk_size;
            for x in 0..w {
                let ci = header.get_chunk_info(x, y);
                let inner_y = (vector_idx - y * header.chunk_size) as u64;
                let inner_offset = inner_y * ci.width as u64;
                file.seek(SeekFrom::Start(
                    (ci.start + inner_offset) * tsize + header_size,
                ))
                .unwrap();
                let vs = header.chunk_size as usize * x as usize;
                file.write_all(cast_slice(&vec[vs..vs + ci.width as usize]))
                    .unwrap();
            }
        }
        TranposeState::ColMayor => {
            assert_eq!(vec.len(), header.height as usize);

            let h = header.nr_chunks_y();
            let x = vector_idx / header.chunk_size;
            for y in 0..h {
                let ci = header.get_chunk_info(x, y);
                let inner_x = (vector_idx - x * header.chunk_size) as u64;
                let inner_offset = inner_x * ci.height as u64;
                file.seek(SeekFrom::Start(
                    (ci.start + inner_offset) * tsize + header_size,
                ))
                .unwrap();
                let vs = header.chunk_size as usize * y as usize;
                file.write_all(cast_slice(&vec[vs..vs + ci.height as usize]))
                    .unwrap();
            }
        }
    }
}

impl<F: Read + Write + Seek> ChunkmatHandle<F> {
    /// Create a zeroed Chunkmat file
    ///
    /// Assumes that the given file is writeable and empty.
    pub fn new(mut file: F, header: Header) -> Self {
        file.seek(SeekFrom::Start(0)).unwrap();
        bincode_options()
            .serialize_into(&mut file, &header)
            .unwrap();
        let header_size = bincode_options().serialized_size(&header).unwrap();

        let data_size =
            header.width as usize * header.height as usize * size_of_dtype(&header.dtype);

        let mut zeros_written = 0;
        let zeros_buffer = vec![0u8; header.width as usize];
        while zeros_written < data_size {
            let write = zeros_buffer.len().min(data_size - zeros_written);
            zeros_written += file.write(&zeros_buffer[..write]).unwrap();
        }
        Self {
            header,
            header_size,
            file,
        }
    }

    pub fn open(mut file: F) -> Self {
        file.seek(SeekFrom::Start(0)).unwrap();
        let header: Header = bincode_options().deserialize_from(&mut file).unwrap();
        let header_size = bincode_options().serialized_size(&header).unwrap();

        Self {
            header,
            header_size,
            file,
        }
    }

    pub fn transpose(&mut self) {
        // todo: need an out-of-place transpose
        let old_header = self.header.clone();
        self.header.state = match self.header.state {
            TranposeState::RowMayor => TranposeState::ColMayor,
            TranposeState::ColMayor => TranposeState::RowMayor,
        };
        self.file.seek(SeekFrom::Start(0)).unwrap();
        bincode_options()
            .serialize_into(&mut self.file, &self.header)
            .unwrap();

        match &*self.header.dtype {
            "<c16" => transpose::<f64>(&mut self.file, &old_header, self.header_size),
            "<c8" => transpose::<f32>(&mut self.file, &old_header, self.header_size),
            _ => unimplemented!("we don't care about other dtypes"),
        }
    }

    pub fn extract_vector_c8(&mut self, vector_idx: u32, vec: &mut Vec<Complex<f32>>) {
        assert_eq!(self.header.dtype, "<c8");
        extract_vector(
            &mut self.file,
            &self.header,
            self.header_size,
            vector_idx,
            vec,
        )
    }

    pub fn save_vector_c8(&mut self, vector_idx: u32, vec: &[Complex<f32>]) {
        assert_eq!(self.header.dtype, "<c8");
        save_vector(
            &mut self.file,
            &self.header,
            self.header_size,
            vector_idx,
            vec,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_tranpose_regular() {
        let mut data = [99f32, 1., 2., 3., 4., 5., 6., 7., 8.];
        let expected = [99f32, 1., 3., 2., 4., 5., 7., 6., 8.];

        let arr = Cursor::<&mut [u8]>::new(cast_slice_mut(&mut data));
        transpose::<f32>(
            arr,
            &Header {
                width: 4,
                height: 2,
                chunk_size: 2,
                dtype: "asdf".to_string(),
                state: TranposeState::RowMayor,
            },
            std::mem::size_of::<f32>() as u64,
        );

        assert_eq!(data, expected);
    }

    #[test]
    fn test_tranpose_irregular() {
        // initial matrix
        // 01 02 03   11 12
        // 04 05 06   13 14
        // 07 08 09   15 16

        let data_orig = [
            99f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 11., 12., 13., 14., 15., 16.,
        ];
        let mut data = data_orig;
        let expected = [
            99f32, 1., 4., 7., 2., 5., 8., 3., 6., 9., 11., 13., 15., 12., 14., 16.,
        ];

        let arr = Cursor::<&mut [u8]>::new(cast_slice_mut(&mut data));
        transpose::<f32>(
            arr,
            &Header {
                width: 5,
                height: 3,
                chunk_size: 3,
                dtype: "asdf".to_string(),
                state: TranposeState::RowMayor,
            },
            std::mem::size_of::<f32>() as u64,
        );

        assert_eq!(data, expected);

        let arr = Cursor::<&mut [u8]>::new(cast_slice_mut(&mut data));
        transpose::<f32>(
            arr,
            &Header {
                width: 5,
                height: 3,
                chunk_size: 3,
                dtype: "asdf".to_string(),
                state: TranposeState::ColMayor,
            },
            std::mem::size_of::<f32>() as u64,
        );

        assert_eq!(data, data_orig);
    }

    #[test]
    fn test_extract_vector() {
        let mut data = [
            99f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 11., 12., 13., 14., 15., 16.,
        ];
        let mut arr = Cursor::<&mut [u8]>::new(cast_slice_mut(&mut data));
        let header_size = std::mem::size_of::<f32>() as u64;

        let mut out = Vec::new();
        extract_vector::<_, f32>(
            &mut arr,
            &Header {
                width: 5,
                height: 3,
                chunk_size: 3,
                dtype: "asdf".to_string(),
                state: TranposeState::RowMayor,
            },
            header_size,
            2,
            &mut out,
        );

        assert_eq!(out, [7., 8., 9., 15., 16.]);

        transpose::<f32>(
            &mut arr,
            &Header {
                width: 5,
                height: 3,
                chunk_size: 3,
                dtype: "asdf".to_string(),
                state: TranposeState::RowMayor,
            },
            header_size,
        );

        let header = Header {
            width: 5,
            height: 3,
            chunk_size: 3,
            dtype: "asdf".to_string(),
            state: TranposeState::ColMayor,
        };
        extract_vector::<_, f32>(&mut arr, &header, header_size, 1, &mut out);
        assert_eq!(out, [2., 5., 8.]);
        extract_vector::<_, f32>(&mut arr, &header, header_size, 3, &mut out);
        assert_eq!(out, [11., 13., 15.]);
    }
}
