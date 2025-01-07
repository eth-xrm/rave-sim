# Copyright (c) 2024, ETH Zurich
from pathlib import Path
import h5py
import numpy as np
import sys
from tqdm import tqdm

import bfpy


def npy_to_hdf5(input: Path, output: Path):
    ce = bfpy.ChunkedEditor(input)

    dtype = np.dtype(ce.dtype())
    N = len(ce)

    print(f"N = {N} = 2**{np.log2(N)}, dtype = {dtype}")

    size_factor = 16 if dtype == "c16" else 8

    with tqdm(total=N * size_factor, unit_scale=True, unit='B') as pbar:
        with h5py.File(output, "w") as outf:
            dset = outf.create_dataset("data", shape=(N,), dtype=dtype)

            buffer = np.array([0] * 2**20, dtype=dtype)
            while ce.position() < N:
                count = ce.read_chunk_c16(buffer) if dtype == "c16" else ce.read_chunk_c8(buffer)
                dset[ce.position():ce.position()+count] = buffer[:count]
                ce.advance(count)
                pbar.update(count * size_factor)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python3 npy-to-hdf5.py <input.npy> <output.h5>")
    input = Path(sys.argv[1])
    output = Path(sys.argv[2])    

    npy_to_hdf5(input, output)
