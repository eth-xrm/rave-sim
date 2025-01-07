# Copyright (c) 2024, ETH Zurich

from typing import Optional, Tuple
import numpy as np

from propagation import SimParams, square_and_downsample, propagate
from vector import Vector


class History:
    # a list of downscaled vectors in the order that they were added to the history
    entries: list[np.ndarray]

    # z values for every history entry. Length should be equal to `entries` length
    zs: list[float]

    def __init__(self) -> None:
        self.entries = []
        self.zs = []

    def push(self, downscaled: np.ndarray, z: float) -> None:
        """Add a new vector to the history"""
        self.entries.append(downscaled)
        self.zs.append(z)

    def get_history(self) -> np.ndarray:
        """Return the full history up to this point as a 2d NumPy array with shape (x, z)"""
        if len(self.entries) == 0:
            return np.empty((0, 0))
        else:
            return np.stack(self.entries, axis=-1)

    def get_z(self) -> list[float]:
        """Return the z coordinates for each entry in the history"""
        return self.zs

    def __len__(self) -> int:
        """The number of entries in the history up to this point"""
        assert len(self.entries) == len(self.zs)
        return len(self.entries)


def generate_analytic_history(
    history: History,
    zs: np.ndarray,
    x_source: float,
    params: SimParams,
    cutoff_freq: float,
):
    gradient = cutoff_freq * params.wl

    n = params.detector_size // params.detector_pixel_size_x
    x = (np.arange(n) - n // 2) * params.detector_pixel_size_x - x_source

    for z in zs:
        angle = np.arctan(x / z)
        r2 = x**2 + z**2
        intensity = (
            params.detector_pixel_size_x
            * params.detector_pixel_size_y
            * np.cos(angle)
            / r2
        )
        intensity[np.abs(x) > gradient * z] = 0
        history.push(intensity, z)


def propagate_with_history(
    u: Vector,
    U: Vector,
    sim_params: SimParams,
    dz: float,
    cutoff_freq: float,
    current_z: float,
    skip_fft: bool = False,
    history: Optional[Tuple[History, float]] = None,
):
    """
    Wrapper function aroung `propagate` that iterates in multiple
    small steps and adds every intermediary result to the history.
    """

    if dz == 0:
        return

    if history is not None:
        hist_obj = history[0]
        hist_dz = history[1]

        # list of all the z values we'll visit given as offsets from current_z
        zs = np.arange(0, dz, hist_dz)
        if dz - zs[-1] < hist_dz / 2.0 and len(zs) > 1:
            zs[-1] = dz
        else:
            zs = np.append(zs, [dz])

        if not skip_fft:
            u.fft(U)

        # Always jump to the current z value from the start (i.e. zs[0]), not
        # from the previous position (zs[i-1]) to prevent accumulation of
        # errors. For this we need to remember the start value.
        U_start = U.copy()

        for i in range(1, len(zs)):
            dz_step: float = zs[i] - zs[0]
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                dz_step,
                sim_params.chunk_size,
                cutoff_freq,
                True,
            )
            z = zs[i] + current_z
            hist_obj.push(square_and_downsample(u, sim_params, z), z)

            if i < len(zs) - 1:
                U_start.copy_to(U)

        U_start.drop()

    else:
        propagate(
            u,
            U,
            sim_params.dx,
            sim_params.wl,
            dz,
            sim_params.chunk_size,
            cutoff_freq,
            skip_fft,
        )
