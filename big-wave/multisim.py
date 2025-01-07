# Copyright (c) 2024, ETH Zurich

from datetime import datetime
import logging
import copy
import os
from pathlib import Path
import random
import shutil
import time
from typing import Optional, Tuple
import bfpy
import numpy as np
import h5py  # type: ignore

import config
from history import History
from optical_element import (
    DeltabetaTable,
    OpticalElement,
    SaveAndExit,
    collect_all_materials,
    generate_deltabeta_table,
)
from propagation import (
    SimParams,
    compute_cutoff_angles,
    convert_energy_wavelength,
    grid_density_check,
)
from util import detector_x_vector, get_sub_dir, setup_logger
from vector import DiskVector, NumpyVector, Vector
import wavesim

logger = logging.getLogger("big-wave")


def setup_sim_dir(save_dir: Path) -> Path:
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    full = now.strftime("%Y%m%d_%H%M%S%f")

    path = save_dir / year / month / full

    os.makedirs(path)
    return path


def run_single_simulation(
    sim_dir: Path,
    source_idx: int,
    scratch_dir: Path,
    save_keypoints_path: Optional[Path] = None,
    history_dz: Optional[float] = None,
):
    setup_logger()
    logger.info(
        f"Running single simulation for source {source_idx} in simulation {sim_dir}"
    )
    start = time.perf_counter()

    if not isinstance(sim_dir, Path):
        sim_dir = Path(sim_dir)
    if not isinstance(scratch_dir, Path):
        scratch_dir = Path(scratch_dir)
    if save_keypoints_path is not None and not isinstance(save_keypoints_path, Path):
        save_keypoints_path = Path(save_keypoints_path)

    dct = config.load(sim_dir / "config.yaml")
    computed = config.load(sim_dir / "computed.yaml")
    angles = list(map(float, computed["cutoff_angles"]))

    use_disk_vector = dct["use_disk_vector"]
    save_final_u_vectors = dct["save_final_u_vectors"]
    elements = [config.parse_optical_element(el, sim_dir) for el in dct["elements"]]
    sim_params = config.parse_sim_params(dct["sim_params"])
    dtype = config.parse_dtype(dct)

    sub_dir = get_sub_dir(sim_dir, source_idx)
    sub_dct = config.load(sub_dir / "subconfig.yaml")

    vectors_dir = scratch_dir / "vectors"
    scratchfile = vectors_dir / "scratch.npy"

    source = config.parse_source(
        sub_dct["source"], use_disk_vector, scratchfile, sim_params.N, dtype
    )

    energy = float(sub_dct["energy"])
    deltabeta_table: DeltabetaTable = [
        (config.parse_material(entry[0]), entry[1][0] + entry[1][1] * 1j)
        for entry in sub_dct["deltabeta_table"]
    ]

    u: Vector
    U: Vector
    if use_disk_vector:
        vectors_dir.mkdir()
        if dtype == np.complex128:
            bfpy.generate_header_c16(vectors_dir / "u.npy", sim_params.N)
        else:
            bfpy.generate_header_c8(vectors_dir / "u.npy", sim_params.N)

        u = DiskVector(vectors_dir / "u.npy", scratchfile, sim_params.N, dtype)
        U = DiskVector(vectors_dir / "U.npy", scratchfile, sim_params.N, dtype)

    else:
        u = NumpyVector(np.zeros(sim_params.N, dtype=dtype))
        U = NumpyVector(np.zeros(sim_params.N, dtype=dtype))

    sim_params.wl = convert_energy_wavelength(energy)

    history: Optional[Tuple[History, float]] = None
    if history_dz is not None:
        history = (History(), history_dz)

    detector_outputs = wavesim.run_simulation(
        params=sim_params,
        source=source,
        elements=elements,
        cutoff_angles=angles,
        u=u,
        U=U,
        deltabeta_table=deltabeta_table,
        sub_dir=sub_dir,
        vectors_dir=vectors_dir,
        save_final_u_vectors=save_final_u_vectors,
        history=history,
        save_keypoints_path=save_keypoints_path,
    )

    if history is not None:
        np.save(sub_dir / "history.npy", history[0].get_history())
        np.save(sub_dir / "history_z.npy", history[0].get_z())
        np.save(
            sub_dir / "history_x.npy",
            detector_x_vector(
                sim_params.detector_size, sim_params.detector_pixel_size_x
            ),
        )

    np.save(sub_dir / "detected.npy", detector_outputs)

    shutil.rmtree(vectors_dir, ignore_errors=True)

    end = time.perf_counter()
    logger.info(
        f"Finished running simulation for source {source_idx} in {end - start} seconds"
    )


def deltabeta_table_to_tuples(
    table: DeltabetaTable,
) -> list[tuple[tuple[str, float], tuple[float, float]]]:
    """
    We can't save the Material and complex types directly in yaml, so we first simplify them to tuples.
    """

    return [
        ((entry[0].mat, entry[0].density), (float(entry[1].real), float(entry[1].imag)))
        for entry in table
    ]


def reduce_simulation_setup_for_save_and_exit(
    dct: config.DictType, max_x_list: list[float]
) -> float:
    """
    If there is a save_and_exit element present, we remove all the further elements after it
    and move the z_detector to the z coordinate of the save_and_exit element.

    Returns the maximal absolute coordinate at which light will occur at the (potentially updated) z_detector.
    """

    reduced_elements: list[config.DictType] = []
    max_x = max_x_list[-1]
    for i, el in enumerate(dct["elements"]):
        if el["type"] == "save_and_exit":
            max_x = max_x_list[i]
            dct["sim_params"]["z_detector"] = el["z_start"]
            if dct["save_final_u_vectors"]:
                logger.warn(
                    "save_final_u_vectors was activated on a config that contains a save_and_exit element. Note that the vectors will be saved, but at the save_and_exit position instead of the originally specified z_detector."
                )
            dct["save_final_u_vectors"] = True
            dct["elements"] = reduced_elements
            break
        else:
            reduced_elements.append(el)

    return max_x


def check_simulation_inputs(
    params: SimParams,
    z_source: float,
    elements: list[OpticalElement],
    cutoff_angles: list[float],
):
    """Check if the inputs to the simulation are valid"""

    assert len(cutoff_angles) == len(elements) + 1

    # Accept small z-overlaps to prevent problems in cases where the overlap is just
    # due to float-decimal conversions.
    z_tolerance = 1e-8

    if len(elements) > 0:
        assert z_source <= elements[0].z_start, "The source must not be at a greater z-coordinate than the first optical element"

    nr_phase_steps: Optional[int] = None
    current_z: float = z_source
    for i, el in enumerate(elements):
        steps = len(el.x_positions)
        if steps > 1:
            assert (
                nr_phase_steps is None or nr_phase_steps == steps
            ), "If multiple elements have phase stepping enabled, then all elements must have the same number of steps."
            nr_phase_steps = steps

        el.check_valid()
        assert current_z <= el.z_start + z_tolerance, "Overlapping optical elements are not supported"
        current_z = el.z_start + el.get_thickness()

        if isinstance(el, SaveAndExit):
            if i == len(elements) - 1:
                assert (
                    params.z_detector == el.z_start
                ), "A save_and_exit element at the end must have the same z coordinate as the detector"
            else:
                assert (
                    elements[i + 1].z_start == el.z_start
                ), "A save_and_exit element must have the same z coordinate as the next optical element"

    assert current_z <= params.z_detector + z_tolerance, "Detector must be after the last optical element"

    for i in range(len(cutoff_angles) - 1):
        assert cutoff_angles[i] <= cutoff_angles[i + 1], "cutoff angles must be increasing"

    for a in cutoff_angles:
        assert 0 <= a <= np.pi / 2, "cutoff angles must be between 0 and pi/2"


def load_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        return np.array(f["energy"]), np.array(f["pdf"])


def generate_energies_from_spectrum(
    spectrum_energy: np.ndarray,
    spectrum_intensities: np.ndarray,
    rs: np.random.RandomState,
    energy_range: Tuple[float, float],
    nr_points: int,
) -> np.ndarray:
    """
    Use the given spectrum, cropped to energy_range, to generate a set of energies

    This assumes that the spectrum is given in equidistant energies and fills in the spaces between two energies as a uniform distribution
    """

    mask = np.logical_and(
        spectrum_energy >= energy_range[0], spectrum_energy < energy_range[1]
    )
    se = spectrum_energy[mask]
    si = spectrum_intensities[mask]

    if len(se) < 2:
        raise ValueError(
            "Not enough energy points of the spectrum lie within the given range"
        )

    bases = rs.choice(se, size=nr_points, replace=True, p=si / np.sum(si))

    spacing = se[1] - se[0]
    offsets = rs.rand(nr_points) * spacing  # type: ignore [attr-defined]

    return bases + offsets


def setup_simulation(dct: config.DictType, config_dir: Path, save_dir: Path) -> Path:
    """
    Create a directory for this simulation and set up all the files within it that
    are necessary so that the simulations for the individual sources can run afterwards.

    For an explanation of the directory structure see the `big-wave` Readme.

    Parameters
    ----------
    dct : dict
        The simulation configuration
    config_dir : Path
        Directory from where the relative lookup to the grid arrays and spectrum should happen.
        The grid array files will be copied over to the generated simulation directory.
    save_dir : Path
        Directory where simulation directories should be created.

    Returns the path to the created simulation directory.
    """

    setup_logger()
    logger.info("Setting up simulation")

    dct = copy.deepcopy(dct)

    if not isinstance(config_dir, Path):
        config_dir = Path(config_dir)
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    sim_params = config.parse_sim_params(dct["sim_params"])
    elements = [config.parse_optical_element(el, config_dir) for el in dct["elements"]]

    source_points: list[config.DictType]
    sub_dcts: list[config.DictType] = []
    energy_range: Tuple[float, float]
    max_x: float
    z_source: float

    assert (
        sim_params.N * sim_params.dx >= sim_params.detector_size
    ), f"Detector is bigger than simulation space ({sim_params.detector_size} > {sim_params.N * sim_params.dx})."

    materials = collect_all_materials(elements)

    multisource = dct["multisource"]
    if multisource["type"] == "points":
        nr_source_points = int(multisource["nr_source_points"])
        energy_range = (
            float(multisource["energy_range"][0]),
            float(multisource["energy_range"][1]),
        )
        x_range = (
            float(multisource["x_range"][0]),
            float(multisource["x_range"][1]),
        )
        z_source = float(multisource["z"])
        seed = int(multisource["seed"])
        if seed == -1:
            seed = random.randint(0, 1000000)

        first_z = sim_params.z_detector
        if len(elements) > 0:
            first_z = elements[0].z_start

        # Use the highest possible energy for the grid density check since that one
        # has the shortest wavelength and is thus more likely to run into Nyquist
        # issues.
        wl_e_max = convert_energy_wavelength(energy_range[1])
        grid_density_check(
            first_z - z_source, x_range[0], sim_params.N, sim_params.dx, wl_e_max
        )
        grid_density_check(
            first_z - z_source, x_range[1], sim_params.N, sim_params.dx, wl_e_max
        )

        rs = np.random.RandomState(seed)
        energies: np.ndarray
        if "spectrum" in multisource:
            spectrum_energy, spectrum_intensities = load_spectrum(
                config_dir / multisource["spectrum"]
            )
            energies = generate_energies_from_spectrum(
                spectrum_energy,
                spectrum_intensities,
                rs,
                energy_range,
                nr_source_points,
            )
        else:
            energies = (
                rs.rand(nr_source_points) * (energy_range[1] - energy_range[0])  # type: ignore [attr-defined]
                + energy_range[0]
            )
        sources = (
            rs.rand(nr_source_points) * (x_range[1] - x_range[0])  # type: ignore [attr-defined]
            + x_range[0]
        )
        del rs

        sub_dcts = [
            {
                "source": {"type": "point", "x": float(sources[i]), "z": z_source},
                "energy": float(energies[i]),
                "deltabeta_table": deltabeta_table_to_tuples(
                    generate_deltabeta_table(materials, energies[i])
                ),
            }
            for i in range(nr_source_points)
        ]

        source_points = [
            {
                "x": float(sources[i]),
                "z": z_source,
                "energy": float(energies[i]),
            }
            for i in range(nr_source_points)
        ]

        max_x = max(abs(x_range[0]), abs(x_range[1]))

    elif multisource["type"] == "vectors":
        base_sim_dir = Path(multisource["base_sim_dir"])
        input_u_index = int(multisource["input_u_index"])

        base_dct = config.load(base_sim_dir / "config.yaml")
        base_sim_params = config.parse_sim_params(base_dct["sim_params"])
        z_source = base_sim_params.z_detector

        assert (
            base_sim_params.N == sim_params.N and base_sim_params.dx == sim_params.dx
        ), "simulation N and dx must match those of the base simulation in {base_sim_dir}"

        assert base_dct[
            "save_final_u_vectors"
        ], "The base simulation must have save_final_u_vectors set to true or have a save_and_exit element"

        base_computed = config.load(base_sim_dir / "computed.yaml")
        energy_range = (
            float(base_computed["energy_range"][0]),
            float(base_computed["energy_range"][1]),
        )
        max_x = float(base_computed["max_x"])

        # get all subdirectories whose name consists of eight digits.
        # This assumes that the base simulation has already been configured, but doesn't
        # require that it has already been executed.
        nr_source_points = len(
            list(base_sim_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/"))
        )

        for i in range(nr_source_points):
            base_sub_dir = get_sub_dir(base_sim_dir, i)

            upath = base_sub_dir / f"u_{input_u_index:04}.npy"
            if not upath.is_file():
                logger.warning(
                    f"Cannot find u vector {upath} for source {i:08}. This probably means that the base simulation did not run yet "
                    + "for that source point. Make sure that the u vectors of the base simulation are available by the time this "
                    + "simulation is executed."
                )

            base_sub_dct = config.load(base_sub_dir / "subconfig.yaml")
            energy = float(base_sub_dct["energy"])

            sub_dcts.append(
                {
                    "source": {
                        "type": "vector",
                        "input_path": str(upath),
                        "z": z_source,
                    },
                    "energy": energy,
                    "deltabeta_table": deltabeta_table_to_tuples(
                        generate_deltabeta_table(materials, energy)
                    ),
                }
            )

        source_points = base_computed["source_points"]

    else:
        raise ValueError(f"Unknown multisource type {multisource['type']}.")

    z_distances = [el.z_start for el in elements]
    element_heights = [el.get_thickness() for el in elements]
    angles, max_x_list = compute_cutoff_angles(
        detector_size=sim_params.detector_size,
        dx=sim_params.dx,
        energy_range=energy_range,
        z_source=z_source,
        z_distances=z_distances,
        element_heights=element_heights,
        z_detector=sim_params.z_detector,
        max_x=max_x,
    )

    # Maximal absolute x coordinate at z_detector where rays should appear according to the cutoff angles
    max_x = max_x_list[-1]
    # size of the simulation such that no rays should reflect at all when using the computed cutoff angles.
    required_N = int(np.ceil(max_x / sim_params.dx) * 2)

    assert (
        sim_params.N >= required_N
    ), f"Reflections at the boundary might occur: N={sim_params.N}, required_N={required_N}"

    grid_files = []
    for el in dct["elements"]:
        if el["type"] == "sample":
            grid_files.append(config_dir / el["grid_path"])
            el["grid_path"] = os.path.basename(el["grid_path"])

    max_x = reduce_simulation_setup_for_save_and_exit(dct, max_x_list)

    check_simulation_inputs(sim_params, z_source, elements, angles)

    sim_dir = setup_sim_dir(save_dir)
    for gf in grid_files:
        shutil.copyfile(gf, sim_dir / os.path.basename(gf))

    config.save(sim_dir / "config.yaml", dct)

    config.save(
        sim_dir / "computed.yaml",
        {
            "cutoff_angles": angles,
            "max_x": max_x,
            "energy_range": energy_range,
            "source_points": source_points,
        },
    )

    for i in range(len(sub_dcts)):
        sub_dir = get_sub_dir(sim_dir, i)
        os.makedirs(sub_dir)
        config.save(sub_dir / "subconfig.yaml", sub_dcts[i])

    logger.info(f"Finished setting up simulation in {sim_dir}")
    return sim_dir
