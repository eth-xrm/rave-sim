# Copyright (c) 2024, ETH Zurich


import numpy as np
from pathlib import Path

# This file contains python stubs definitions so that bfpy works with mypy.
# More info here: https://pyo3.rs/v0.16.4/python_typing_hints.html

PathOrStr = Path | str

def fft_c16(infile: PathOrStr, outfile: PathOrStr, scratchfile: PathOrStr) -> None: ...
def ifft_c16(infile: PathOrStr, outfile: PathOrStr, scratchfile: PathOrStr) -> None: ...
def generate_header_c16(filename: PathOrStr, len: int) -> None: ...
def fft_c8(infile: PathOrStr, outfile: PathOrStr, scratchfile: PathOrStr) -> None: ...
def ifft_c8(infile: PathOrStr, outfile: PathOrStr, scratchfile: PathOrStr) -> None: ...
def generate_header_c8(filename: PathOrStr, len: int) -> None: ...

class ChunkedEditor:
    """
    Walks over the given NPY file in mutable chunks. Can be used to edit a file in place.

    :param file: path to the .npy file
    """

    def __init__(self, file: PathOrStr) -> None: ...
    def position(self) -> int: ...
    def __len__(self) -> int: ...
    def advance(self, nr_elements: int) -> None: ...
    def read_chunk_c16(self, buffer: np.ndarray[np.c16]) -> int: ...
    def write_chunk_c16(self, buffer: np.ndarray[np.c16]) -> int: ...
    def read_chunk_c8(self, buffer: np.ndarray[np.c8]) -> int: ...
    def write_chunk_c8(self, buffer: np.ndarray[np.c8]) -> int: ...
    def dtype(self) -> str: ...
