# Copyright (c) 2024, ETH Zurich

import os
from pathlib import Path
from scipy import fft as spfft  # type: ignore # https://github.com/scipy/scipy/issues/17158
import shutil
from typing import Callable, Protocol, Self
import numpy as np

import bfpy


class Vector(Protocol):
    """
    A complex vector that supports the operations that
    we need to simulate the wave propagation.
    """

    # todo: the `Self` type of the destination might be too loose, check
    # if this allows for calling one type of vector with a different
    # output class, and if yes either fix it or document it.
    def fft(self: Self, destination: Self):
        """Perform a forward fourier transform"""
        ...

    def ifft(self: Self, destination: Self):
        """Perform an inverse fourier transform"""
        ...

    def __len__(self: Self) -> int:
        """Return the length of the vector"""
        ...

    def write_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        """
        Overwrite a vector in place with a function that can generate one chunk at a time.
        """

    def modify_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        """
        Modify a vector in place with a mutator function that can edit one chunk at a time.

        The mutator function receives the index of the first element in the current chunk, as
        well as the chunk as a numpy array. It should modify the given array in place, without
        returning anything.
        """
        ...

    def read_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        """
        Iterate over the vector read-only, one chunk at a time.

        The reader function receives the index of the first element in the current chunk, as
        well as the chunk as a numpy array. It should not modify or replace the given array.
        """

    def copy(self: Self) -> Self:
        """Return a copy of the vector"""
        ...

    def copy_to(self: Self, destination: Self) -> None:
        """Copy the vector, overwriting the destination"""
        ...

    def drop(self: Self) -> None:
        """
        Delete the data of the vector. The state of the vector object is undefined after this
        call, no further methods should be called on it.

        We use this instead of `__del__` to make sure we don't keep around unnecessary data.
        """
        ...

    @property
    def dtype(self: Self) -> np.dtype:
        ...


class DiskVector(Vector):
    """
    A vector backed by a .npy file on disk.

    The given file is not required to exist, but note that if the file doesn't contain a valid
    numpy array of the correct length, then the DiskVector object is in an invalid state until
    the `zero` method is called.

    Initialization does not modify existing files.

    The scratch file path is not required to exist, but it is required to point to a writeable
    location. Any existing data in the scratch file will be overwritten.

    If `copy()` is used then the directory of `file` has to be writeable.
    """

    file: Path
    scratchfile: Path
    len: int
    typ: np.dtype

    def __init__(self, file: Path, scratchfile: Path, len: int, dtype: np.dtype):
        if not isinstance(file, Path):
            file = Path(file)
        if not isinstance(scratchfile, Path):
            scratchfile = Path(scratchfile)

        self.file = file
        self.scratchfile = scratchfile
        self.len = len
        self.typ = np.dtype(dtype)

    def zero(self: Self, chunk_size: int) -> None:
        """
        Set the vector to all zeros, overwriting any existing data.
        """
        if self.typ == np.dtype(np.complex128):
            bfpy.generate_header_c16(self.file, self.len)
        else:
            bfpy.generate_header_c8(self.file, self.len)

        ce = bfpy.ChunkedEditor(self.file)
        buffer = np.zeros(chunk_size, dtype=self.typ)
        while ce.position() < self.len:
            if self.typ == np.dtype(np.complex128):
                ce.write_chunk_c16(buffer)
            else:
                ce.write_chunk_c8(buffer)
            ce.advance(chunk_size)

    def fft(self: Self, destination: Self):
        if self.typ == np.dtype(np.complex128):
            bfpy.fft_c16(self.file, destination.file, self.scratchfile)
        else:
            bfpy.fft_c8(self.file, destination.file, self.scratchfile)
        destination.len = self.len

    def ifft(self: Self, destination: Self):
        if self.typ == np.dtype(np.complex128):
            bfpy.ifft_c16(self.file, destination.file, self.scratchfile)
        else:
            bfpy.ifft_c8(self.file, destination.file, self.scratchfile)
        destination.len = self.len

    def write_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        ce = self._open_chunked_editor()

        buffer = np.zeros(chunk_size, dtype=self.typ)
        while ce.position() < self.len:
            nr_items = min(chunk_size, self.len - ce.position())
            fn(ce.position(), buffer[:nr_items])
            if self.typ == np.dtype(np.complex128):
                ce.write_chunk_c16(buffer[:nr_items])
            else:
                ce.write_chunk_c8(buffer[:nr_items])
            ce.advance(nr_items)

    def modify_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        ce = self._open_chunked_editor()

        buffer = np.zeros(chunk_size, dtype=self.typ)
        while ce.position() < self.len:
            nr_items: int
            if self.typ == np.dtype(np.complex128):
                nr_items = ce.read_chunk_c16(buffer)
            else:
                nr_items = ce.read_chunk_c8(buffer)
            fn(ce.position(), buffer[:nr_items])
            if self.typ == np.dtype(np.complex128):
                ce.write_chunk_c16(buffer[:nr_items])
            else:
                ce.write_chunk_c8(buffer[:nr_items])

            ce.advance(nr_items)

    def read_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        ce = self._open_chunked_editor()

        buffer = np.zeros(chunk_size, dtype=self.typ)
        while ce.position() < self.len:
            nr_items: int
            if self.typ == np.dtype(np.complex128):
                nr_items = ce.read_chunk_c16(buffer)
            else:
                nr_items = ce.read_chunk_c8(buffer)
            fn(ce.position(), buffer[:nr_items])

            ce.advance(nr_items)

    def __len__(self: Self) -> int:
        return self.len

    def copy(self: Self) -> Self:
        # generate a new unique filename (https://stackoverflow.com/a/57896232)
        newfile = self.file
        filename, extension = os.path.splitext(newfile)
        counter = 1
        while os.path.exists(newfile):
            newfile = Path(f"{filename}_{counter:02}{extension}")
            counter += 1

        shutil.copyfile(self.file, newfile)
        dv = DiskVector(newfile, self.scratchfile, self.len, self.typ)
        return dv  # type: ignore # see todo note in class Vector

    def copy_to(self: Self, destination: Self) -> None:
        shutil.copyfile(self.file, destination.file)
        destination.len = self.len

    def drop(self: Self) -> None:
        os.remove(self.file)

    @property
    def dtype(self: Self) -> np.dtype:
        return np.dtype(self.typ)

    def _open_chunked_editor(self: Self) -> bfpy.ChunkedEditor:
        ce = bfpy.ChunkedEditor(self.file)
        assert len(ce) == self.len
        assert np.dtype(ce.dtype()) == np.dtype(self.typ)
        return ce


class NumpyVector(Vector):
    """
    An implementation of the `Vector` protocol that assumes that everything fits in RAM as a NumPy array.

    This can be used as a reference to verify the out-of-core implementation.
    """

    vec: np.ndarray

    def __init__(self, vec: np.ndarray):
        assert vec.dtype == np.complex128 or vec.dtype == np.complex64
        self.vec = vec

    def __len__(self: Self) -> int:
        return len(self.vec)

    def fft(self: Self, destination: Self):
        destination.vec = spfft.fft(self.vec)

    def ifft(self: Self, destination: Self):
        destination.vec = spfft.ifft(self.vec)

    def write_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        # NumpyVector keeps the invariant that the length of the data remains constant,
        # which is why we can use the `modify_chunked` method here. For `DiskVector` we
        # would run into the issue that the data migth not exist yet, so there's nothing
        # to read.
        self.modify_chunked(chunk_size, fn)

    def modify_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        for i in range(0, len(self.vec), chunk_size):
            extract = self.vec[i : i + chunk_size]
            fn(i, extract)
            self.vec[i : i + chunk_size] = extract

    def read_chunked(
        self: Self, chunk_size: int, fn: Callable[[int, np.ndarray], None]
    ) -> None:
        for i in range(0, len(self.vec), chunk_size):
            extract = self.vec[i : i + chunk_size]
            fn(i, extract)

    def copy(self: Self) -> Self:
        nv = NumpyVector(self.vec.copy())
        return nv  # type: ignore # see todo note in class Vector

    def copy_to(self: Self, destination: Self) -> None:
        destination.vec = self.vec.copy()

    def drop(self: Self) -> None:
        self.vec = np.array([])

    @property
    def dtype(self: Self) -> np.dtype:
        return self.vec.dtype
