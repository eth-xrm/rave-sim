# Copyright (c) 2024, ETH Zurich

import logging
import math
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from history import History
from optical_element import (
    Grating,
    Material,
    Sample,
    collect_all_materials,
    generate_deltabeta_table,
)
from propagation import compute_cutoff_angles, convert_energy_wavelength
from source import PointSource
from util import setup_logger
from vector import DiskVector, NumpyVector, Vector
from wavesim import run_simulation, SimParams

logger = logging.getLogger("big-wave")


def circle(radius: float, scale_x: float, scale_z: float) -> np.ndarray:
    """
    Generate a circle sample with materials 1 inside and 0 outside the circle
    """

    len_x = int(math.ceil(radius * scale_x * 2))
    len_z = int(math.ceil(radius * scale_z * 2))

    arr = np.zeros((len_z, len_x), dtype=np.int8)

    for iz in range(len_z):
        z = (iz - len_z / 2) * scale_z
        for ix in range(len_x):
            x = (ix - len_x / 2) * scale_x

            if x * x + z * z <= radius * radius:
                arr[iz, ix] = 1

    return arr


def main() -> None:
    N = 8192 * 128

    tempdir = tempfile.TemporaryDirectory()
    path = Path(tempdir.name)
    u: Vector
    U: Vector
    use_disk_vector = False

    dtype = np.dtype(np.complex128)

    if use_disk_vector:
        scratchfile = path / "scratch.npy"
        u = DiskVector(path / "u.npy", scratchfile, N, dtype)
        U = DiskVector(path / "U.npy", scratchfile, N, dtype)
    else:
        del tempdir
        u = NumpyVector(np.zeros(N, dtype=dtype))
        U = NumpyVector(np.zeros(N, dtype=dtype))

    history = History()

    energy = 46000.0
    si = Material("Si", 2.34)
    au = Material("Au", 19.32)

    sim_params = SimParams(
        N,
        dx=8e-10,
        z_detector=1.77,
        detector_size=N / 6 * 8e-10,
        detector_pixel_size_x=5e-7,
        detector_pixel_size_y=5e-7,
        wl=convert_energy_wavelength(energy),
        chunk_size=1024 * 1024,
    )

    elements = [
        Grating(
            pitch=4.2e-6,
            dc=(0.5, 0.5),
            z_start=0.1,
            thickness=140 * 1e-6,
            nr_steps=10,
            x_positions=np.array([0.0]),
            substrate_thickness=(370 - 140) * 1e-6,
            mat_a=si,
            mat_b=au,
            mat_substrate=si,
        ),
        Grating(
            pitch=4.2e-6,
            dc=(0.5, 0.5),
            z_start=0.918,
            thickness=59 * 1e-6,
            nr_steps=10,
            x_positions=np.array([0.0]),
            substrate_thickness=(200 - 59) * 1e-6,
            mat_a=si,
            mat_b=None,
            mat_substrate=si,
        ),
        Sample(
            z_start=1.0,
            pixel_size_x=10 * 1e-6,
            pixel_size_z=10000 * 1e-6,
            grid=circle(3e-5, 10 * 1e-6, 10 * 1e-6),
            materials=[si],
            x_positions=np.array([0.0]),
        ),
        Grating(
            pitch=4.2e-6,
            dc=(0.5, 0.5),
            z_start=1.736,
            thickness=154 * 1e-6,
            nr_steps=10,
            x_positions=np.array([0.0]),  # np.linspace(0, 4.2e-6, 5, endpoint=False),
            substrate_thickness=(370 - 154) * 1e-6,
            mat_a=si,
            mat_b=au,
            mat_substrate=si,
        ),
    ]
    z_distances = [el.z_start for el in elements]
    element_heights = [el.get_thickness() for el in elements]
    angles, max_x_list = compute_cutoff_angles(
        detector_size=sim_params.detector_size,
        dx=sim_params.dx,
        energy_range=(46000, 46000),
        z_source=0.0,
        z_distances=z_distances,
        element_heights=element_heights,
        z_detector=sim_params.z_detector,
        max_x=0.0,
    )
    required_N = int(np.ceil(max_x_list[-1] / sim_params.dx) * 2)
    assert N >= required_N, f"N={N}, required_N={required_N}"

    deltabeta_table = generate_deltabeta_table(collect_all_materials(elements), energy)

    u_out = run_simulation(
        sim_params,
        PointSource(x=0.0, z=0.0),
        elements,
        angles,
        u,
        U,
        deltabeta_table,
        path,
        path,
        False,
        (history, 0.02),
    )

    plt.pcolormesh(  # type: ignore
        history.get_z(),
        compute_hist_x(sim_params),
        history.get_history(),
        cmap="Greys_r",
        vmin=-0.1,
        vmax=32,
        shading="nearest",
    )
    plt.xlabel("z (m)")
    plt.ylabel("x (m)")
    plt.show()


def compute_hist_x(sim_params: SimParams) -> np.ndarray:
    hist_n = np.floor(sim_params.detector_size / sim_params.detector_pixel_size_x)
    return (np.arange(hist_n) - hist_n / 2) * sim_params.detector_pixel_size_x


if __name__ == "__main__":
    setup_logger()
    main()
