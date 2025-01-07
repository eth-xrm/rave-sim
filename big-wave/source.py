# Copyright (c) 2024, ETH Zurich

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from history import History, generate_analytic_history, propagate_with_history
from propagation import (
    SimParams,
    apply_frequency_cutoff,
    grid_density_check,
    propagate_analytically,
)
from vector import Vector


class Source(Protocol):
    """
    A source of waves located at distance z
    """

    z: float

    def propagate_to(
        self,
        z_out: float,
        sim_params: SimParams,
        cutoff_freq: float,
        u: Vector,
        U: Vector,
        history: Optional[Tuple[History, float]],
    ) -> None:
        """
        Propagate waves to `z_out`. The output vector `u` is modified in place. `U` is
        not guaranteed to contain the fft of `u` afterwards.
        """
        ...


@dataclass
class PointSource(Source):
    x: float
    z: float

    def propagate_to(
        self,
        z_out: float,
        sim_params: SimParams,
        cutoff_freq: float,
        u: Vector,
        U: Vector,
        history: Optional[Tuple[History, float]],
    ) -> None:
        dz = z_out - self.z
        grid_density_check(dz, self.x, sim_params.N, sim_params.dx, sim_params.wl)
        propagate_analytically(u, dz, self.x, sim_params)

        # Discussion with Alex 2023-07-21: not sure if we want to apply frequency cutoff
        # after analytical propagation or not. Can still comment this out later.
        u.fft(U)
        apply_frequency_cutoff(U, cutoff_freq, sim_params.dx, sim_params.chunk_size)
        U.ifft(u)

        if history is not None:
            # don't generate a history entry at z=0 to avoid division by zero shennanigans
            zs = np.arange(0, z_out, history[1])[1:]
            generate_analytic_history(history[0], zs, self.x, sim_params, cutoff_freq)


@dataclass
class VectorSource(Source):
    input: Vector
    z: float

    def propagate_to(
        self,
        z_out: float,
        sim_params: SimParams,
        cutoff_freq: float,
        u: Vector,
        U: Vector,
        history: Optional[Tuple[History, float]],
    ) -> None:
        assert type(self.input) == type(u)

        dz = z_out - self.z
        if dz == 0:
            self.input.copy_to(u)
            return

        self.input.fft(U)
        propagate_with_history(
            u,
            U,
            sim_params,
            dz,
            cutoff_freq,
            self.z,
            skip_fft=True,
            history=history,
        )
