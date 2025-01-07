# Copyright (c) 2024, ETH Zurich

from dataclasses import dataclass
import logging
from typing import Optional, Protocol, Tuple, TypeVar
from nist_lookup.xraydb_plugin import xray_delta_beta  # type: ignore
import numpy as np

from history import History
from propagation import (
    SimParams,
    square_and_downsample,
    propagate,
)
from vector import Vector

logger = logging.getLogger("big-wave")


@dataclass
class Material:
    """Description of one material from which a grating or a sample is made"""

    mat: str
    density: float


# Each entry in the table is a tuple of the material and the corresponding
# deltabeta. Note that this table is only valid for the one wavelength for
# which it was computed.
DeltabetaTable = list[tuple[Material, np.complex128]]

ComplexOrNdarray = TypeVar("ComplexOrNdarray", np.complex128, np.ndarray)


def material_factor(
    deltabeta: ComplexOrNdarray, thickness: float, wl: float
) -> ComplexOrNdarray:
    return np.exp(2j * np.pi * thickness / wl * deltabeta)


def extract_from_deltabeta_table(
    material: Material, deltabeta_table: DeltabetaTable
) -> np.complex128:
    """
    Find the best match in the deltabeta_table for the given material. This table is precomputed and
    stored in `subconfig.yaml` when the project directory is generated. That way the `OpticalElement`
    classes don't have to call nist_lookup at runtime.
    """

    candidates = sorted(
        [
            (abs(tm[0].density - material.density), tm[1])
            for tm in deltabeta_table
            if tm[0].mat == material.mat
        ]
    )
    assert len(candidates) > 0, f"Material {material.mat} not found in deltabeta table"
    assert (
        candidates[0][0] < 1e-5
    ), f"Material {material.mat} found with density error of {candidates[0][0]}"

    return candidates[0][1]


class OpticalElement(Protocol):
    """An object that can be put between the source and the detector"""

    z_start: float
    x_positions: np.ndarray

    def get_thickness(self) -> float:
        """The z distance covered by the apply function"""
        ...

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        """
        Assumes that the vector is given at the start z of the
        object, and then iterates through the object from there on.

        The caller is responsible for keeping track of the new z position,
        i.e. z_start + thickness.

        This will modify the provided vectors, the output will be located in `u`.

        All implementations gurantee that `U` will contain the FFT of `u` afterwards.
        """
        ...

    def check_valid(self) -> None:
        """
        Assert that the given configuration makes sense. If not,
        the function should raise an exception.
        """
        ...

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        """
        The materials are specified as a name and a density, but for the calculations we
        need the delta and beta values.

        This method stores the delta and beta values for each material in `self`, using
        the table that we get from `subconfig.yaml` where all the required delta and beta
        values are stored.
        """


@dataclass
class Grating(OpticalElement):
    """Description of a grating. A higher duty cycle corresponds to using more of material A"""

    pitch: float
    dc: Tuple[float, float]  # duty cycle
    z_start: float
    thickness: float  # does not include the substrate
    nr_steps: int
    x_positions: np.ndarray
    substrate_thickness: float
    mat_a: Optional[Material]
    mat_b: Optional[Material]
    mat_substrate: Optional[Material]

    def get_thickness(self) -> float:
        return self.thickness + self.substrate_thickness

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        z_step = self.thickness / self.nr_steps

        # In every step we multiply by a vector containing only two different factors, one per
        # material. We therefore precompute these two factors and allocate the `factor` vector
        # which we will update for every step and every chunk and then multiply on to the chunks.
        factor_a = material_factor(self.db_a, z_step, sim_params.wl)
        factor_b = material_factor(self.db_b, z_step, sim_params.wl)

        factor = np.zeros(sim_params.chunk_size, dtype=u.dtype)

        for i in range(self.nr_steps):
            logger.debug(f"Grating step {i}/{self.nr_steps}")

            z_fraction = i / self.nr_steps

            def modifier(idx: int, chunk: np.ndarray) -> None:
                mask = generate_grating_chunk(
                    sim_params.dx,
                    idx - sim_params.N // 2,
                    len(chunk),
                    self,
                    z_fraction,
                    stepping_iteration,
                )
                np.place(factor[: len(chunk)], mask, factor_a)
                np.place(factor[: len(chunk)], 1.0 - mask, factor_b)
                chunk *= factor[: len(chunk)]

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                z_step,
                sim_params.chunk_size,
                cutoff_freq,
            )

            if history is not None:
                z = self.z_start + (i + 1) * z_step
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

        if self.substrate_thickness > 0:
            factor_substrate = material_factor(
                self.db_substrate, self.substrate_thickness, sim_params.wl
            )

            def modifier(idx: int, chunk: np.ndarray) -> None:
                chunk *= factor_substrate

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                self.substrate_thickness,
                sim_params.chunk_size,
                cutoff_freq,
            )

    def check_valid(self) -> None:
        assert self.substrate_thickness >= 0
        assert self.thickness > 0

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        self.db_a = np.complex128(0.0)
        if self.mat_a is not None:
            self.db_a = extract_from_deltabeta_table(self.mat_a, deltabeta_table)

        self.db_b = np.complex128(0.0)
        if self.mat_b is not None:
            self.db_b = extract_from_deltabeta_table(self.mat_b, deltabeta_table)

        self.db_substrate = np.complex128(0.0)
        if self.mat_substrate is not None:
            self.db_substrate = extract_from_deltabeta_table(
                self.mat_substrate, deltabeta_table
            )

@dataclass
class EnvGrating(OpticalElement):
    """Description of a grating. A higher duty cycle corresponds to using more of material A"""

    pitch0: float # small pitch
    pitch1: float # large pitch
    dc0: Tuple[float, float]  # duty cycle
    dc1: Tuple[float, float]  # duty cycle
    z_start: float
    thickness: float  # does not include the substrate
    nr_steps: int
    x_positions: np.ndarray
    substrate_thickness: float
    mat_a: Optional[Material]
    mat_b: Optional[Material]
    mat_substrate: Optional[Material]

    def get_thickness(self) -> float:
        return self.thickness + self.substrate_thickness

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        z_step = self.thickness / self.nr_steps

        # In every step we multiply by a vector containing only two different factors, one per
        # material. We therefore precompute these two factors and allocate the `factor` vector
        # which we will update for every step and every chunk and then multiply on to the chunks.
        factor_a = material_factor(self.db_a, z_step, sim_params.wl)
        factor_b = material_factor(self.db_b, z_step, sim_params.wl)

        factor = np.zeros(sim_params.chunk_size, dtype=u.dtype)

        for i in range(self.nr_steps):
            logger.debug(f"Grating step {i}/{self.nr_steps}")

            z_fraction = i / self.nr_steps

            def modifier(idx: int, chunk: np.ndarray) -> None:
                mask = generate_env_grating_chunk(
                    sim_params.dx,
                    idx - sim_params.N // 2,
                    len(chunk),
                    self,
                    z_fraction,
                    stepping_iteration,
                )
                np.place(factor[: len(chunk)], mask, factor_a)
                np.place(factor[: len(chunk)], 1.0 - mask, factor_b)
                chunk *= factor[: len(chunk)]

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                z_step,
                sim_params.chunk_size,
                cutoff_freq,
            )

            if history is not None:
                z = self.z_start + (i + 1) * z_step
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

        if self.substrate_thickness > 0:
            factor_substrate = material_factor(
                self.db_substrate, self.substrate_thickness, sim_params.wl
            )

            def modifier(idx: int, chunk: np.ndarray) -> None:
                chunk *= factor_substrate

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                self.substrate_thickness,
                sim_params.chunk_size,
                cutoff_freq,
            )

    def check_valid(self) -> None:
        assert self.substrate_thickness >= 0
        assert self.thickness > 0

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        self.db_a = np.complex128(0.0)
        if self.mat_a is not None:
            self.db_a = extract_from_deltabeta_table(self.mat_a, deltabeta_table)

        self.db_b = np.complex128(0.0)
        if self.mat_b is not None:
            self.db_b = extract_from_deltabeta_table(self.mat_b, deltabeta_table)

        self.db_substrate = np.complex128(0.0)
        if self.mat_substrate is not None:
            self.db_substrate = extract_from_deltabeta_table(
                self.mat_substrate, deltabeta_table
            )


def generate_grating_chunk(
    dx: float,
    start: int,
    chunk_size: int,
    grating: Grating,
    z_fraction: float,
    stepping_iteration: int,
) -> np.ndarray:
    """Generate a chunk of the grating as a boolean mask array"""

    dc = (1 - z_fraction) * grating.dc[0] + z_fraction * grating.dc[1]
    x_start = start * dx + grating.x_positions[stepping_iteration]
    x_stop = x_start + chunk_size * dx

    x = np.linspace(x_start, x_stop, chunk_size, endpoint=False)
    phase = np.mod(x / grating.pitch + dc / 2, 1)
    return phase < dc

def generate_env_grating_chunk(
    dx: float,
    start: int,
    chunk_size: int,
    grating: EnvGrating,
    z_fraction: float,
    stepping_iteration: int,
) -> np.ndarray:
    """Generate a chunk of the grating as a boolean mask array"""

    dc0 = (1 - z_fraction) * grating.dc0[0] + z_fraction * grating.dc0[1]
    dc1 = (1 - z_fraction) * grating.dc1[0] + z_fraction * grating.dc1[1]
    x_start = start * dx + grating.x_positions[stepping_iteration]
    x_stop = x_start + chunk_size * dx

    x = np.linspace(x_start, x_stop, chunk_size, endpoint=False)
    phase0 = np.mod(x / grating.pitch0 + dc0 / 2, 1) 
    phase1 = np.mod(x / grating.pitch1, 1)
    return (phase0 < dc0) * (phase1 < dc1)


@dataclass
class Sample(OpticalElement):
    """
    A sample to be analyzed, represented as a pixel grid of material indices.

    Axis 0 of the grid is the z position, axis 1 is the x position. For example, the
    following sample could represent a small section of a Si/Au grating with a Si substrate,
    where 0 represents holes in the grating.

    ```
    grid = [ [1, 2, 1, 0, 1, 2, 0],
             [1, 1, 1, 1, 1, 1, 1] ]
    materials = [Si, Au]
    ```

    Note that the indices into the materials list are 1-based because 0 represents an empty space.

    Currently does not support rotating the grid. Assumes rectangular pixels

    Samples are centered on the x-axis, i.e. the middle entry is at x=0
    """

    z_start: float
    pixel_size_x: float  # in metres
    pixel_size_z: float  # in metres
    grid: np.ndarray
    materials: list[Material]
    x_positions: np.ndarray

    def check_valid(self) -> None:
        shape = self.grid.shape
        assert len(shape) == 2
        assert shape[0] > 0
        assert shape[1] > 0
        assert self.grid.dtype == np.int8
        assert self.z_start > 0
        assert self.pixel_size_x > 0
        assert self.pixel_size_z > 0
        assert len(self.materials) >= np.max(self.grid)

    def get_thickness(self) -> float:
        return self.pixel_size_z * self.grid.shape[0]

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        self.db_list: np.ndarray = np.zeros(
            len(self.materials) + 1, dtype=np.complex128
        )
        for idx, material in enumerate(self.materials):
            self.db_list[idx + 1] = extract_from_deltabeta_table(
                material, deltabeta_table
            )

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        rowlen = self.grid.shape[1]
        row_x = (
            np.arange(rowlen) * self.pixel_size_x
            + self.x_positions[stepping_iteration]
            - rowlen * self.pixel_size_x * 0.5
        )
        for rowidx in range(self.grid.shape[0]):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

            # Calculate the vector of delta+ibeta values for this row without applying material_factor yet since
            # we want to interpolate the values in this space, instead of the space after taking the exp of this.
            row_deltabeta: np.ndarray = self.db_list[self.grid[rowidx, :]]

            def modifier(idx: int, chunk: np.ndarray) -> None:
                start = idx - sim_params.N / 2
                x_chunk = np.arange(start, start + len(chunk)) * sim_params.dx
                deltabeta = np.interp(
                    x_chunk, xp=row_x, fp=row_deltabeta, left=0.0, right=0.0
                )
                chunk *= material_factor(deltabeta, self.pixel_size_z, sim_params.wl)

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                self.pixel_size_z,
                sim_params.chunk_size,
                cutoff_freq,
            )


@dataclass
class SaveAndExit(OpticalElement):
    """
    This is not a real optical element, instead it's a marker object that tells the simulation
    to stop right there and save the current `u` vector as output. The saved `u` vector can be
    used as the input for a future simulation.
    """

    z_start: float
    x_positions: np.ndarray

    def get_thickness(self) -> float:
        return 0

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        raise ValueError(
            "Trying to apply a SaveAndExit element, this is not allowed since it's only a marker for the setup"
        )

    def check_valid(self) -> None:
        pass

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        pass


def collect_all_materials(
    optical_elements: list[OpticalElement],
) -> list[Material]:
    """
    Collect a deduplicated list of all materials present in the setup
    """

    materials = []

    def push(mat: Material):
        # insert into `materials` if it's not already present
        if mat not in materials:
            materials.append(mat)

    for el in optical_elements:
        if isinstance(el, Grating):
            if el.mat_a is not None:
                push(el.mat_a)
            if el.mat_b is not None:
                push(el.mat_b)
            if el.mat_substrate is not None:
                push(el.mat_substrate)
        if isinstance(el, EnvGrating):
            if el.mat_a is not None:
                push(el.mat_a)
            if el.mat_b is not None:
                push(el.mat_b)
            if el.mat_substrate is not None:
                push(el.mat_substrate)
        elif isinstance(el, Sample):
            for mat in el.materials:
                push(mat)

    return materials


def generate_deltabeta_table(
    materials: list[Material],
    energy: float,
) -> DeltabetaTable:
    """
    Precompute the delta and beta values for the current energy using the nist_lookup
    library and store them in a table.
    """

    table = []

    for m in materials:
        db = xray_delta_beta(m.mat, m.density, energy)
        table.append((m, (db[0] + 1j * db[1])))

    return table
