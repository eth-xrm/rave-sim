# Copyright (c) 2024, ETH Zurich

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import numpy as np
from typing import Optional, Tuple

from history import History, propagate_with_history
from optical_element import DeltabetaTable, OpticalElement
from propagation import SimParams, square_and_downsample, propagate
from source import Source
from vector import DiskVector, NumpyVector, Vector

logger = logging.getLogger("big-wave")


@dataclass
class Snapshot:
    """
    A snapshot of the simulation that we can revert back to.

    This is used to store the state just before a grating or sample that is being phase
    stepped. Reverting back to the snapshot means we don't have to recalculate everything
    before that point for every phase step.
    """

    z: float
    element_idx: int
    u: Vector
    nr_phase_steps: int


def save_keypoint(
    save_keypoints_path: Optional[Path], element_idx: int, sub_idx: int, u: Vector
):
    """
    Save the u vector to a file if `save_keypoints_path` is set.

    Conceptually, every keypoint is associated with an optical element in the setup
    through its 0-based index. Additionally, we have a sub-index which for now is
    0 right before the element and 1 right after.

    For example, in tl3g without a sample, the keypoint after the grating and substrate
    of G2 is keypoint_2_1.npy, with an added sample it would be keypoint_3_1.npy because
    G2 now appears later in the list.
    """

    if save_keypoints_path is None:
        return

    path = save_keypoints_path / f"keypoint_{element_idx:02}_{sub_idx}.npy"
    if isinstance(u, NumpyVector):
        np.save(path, u.vec)
    elif isinstance(u, DiskVector):
        shutil.copy(u.file, path)


def save_u(
    u: Vector,
    phase_step: int,
    sub_dir: Path,
    vectors_dir: Path,
    save_final_u_vectors: bool,
):
    """
    Save the current u vector to a file.

    If the vector is a disk vector, the backing file is consumed in the process and thus the vector
    can no longer be used. The vector would first have to be the target of a copy_to call or similar
    before it can be used again.
    """

    if save_final_u_vectors:
        filename = f"u_{phase_step:04}.npy"

        if isinstance(u, NumpyVector):
            np.save(sub_dir / filename, u.vec)
        else:
            shutil.move(vectors_dir / "u.npy", sub_dir / filename)


def run_simulation(
    params: SimParams,
    source: Source,
    elements: list[OpticalElement],
    cutoff_angles: list[float],
    u: Vector,
    U: Vector,
    deltabeta_table: DeltabetaTable,
    sub_dir: Path,
    vectors_dir: Path,
    save_final_u_vectors: bool,
    history: Optional[Tuple[History, float]] = None,
    save_keypoints_path: Optional[Path] = None,
) -> list[np.ndarray]:
    """
    Run the simulation for one source.

    Parameters
    ----------
    params : SimParams
        Simulation parameters such as chunk size, wl, etc.

    source : Source
        Source object from where the simulation is started.

    elements : list[OpticalElement]
        Optical elements (gratings and samples) in the setup.

    cutoff_angles : list[float]
        List of angles according to which the frequency limiting is done.

        `len(cutoff_angles)` should be `len(elements) + 1`, so that the first entry is the cutoff angle between the
        source and the start of the first element, the `i-th` entry is the cutoff angle between the start of the
        `i-th` element and the start of the element after or the detector.

    u : Vector
        A vector with already the correct length. For disk vectors, only the header has to be initialized, it doesn't matter
        if the body is present or not. For ram vectors, a full numpy array of the correct length has to be provided.

    U : Vector
        A possibly uninitialized vector.

    deltabeta_table : DeltabetaTable
        A table with the deltabeta values for all materials, valid for the current wavelength.

    history : Optional[Tuple[History, float]]
        A history object and the dz with which the history should be recorded. If `None` is passed, then the simulation
        will be faster because no in-between steps will be computed in empty space.

    save_keypoints_path : Optional[Path]
        If set, keypoints (full u vectors) will be saved to this directory. See documentation of `save_keypoint()` for
        more details.
    """

    z_tolerance = 1e-8

    if save_keypoints_path is not None and not isinstance(save_keypoints_path, Path):
        save_keypoints_path = Path(save_keypoints_path)

    cutoff_frequencies = [np.sin(a) / params.wl for a in cutoff_angles]

    for el in elements:
        el.store_deltabetas(deltabeta_table)

    ######## Initialize u vector ########

    cutoff_freq = cutoff_frequencies[0]

    current_z = elements[0].z_start if len(elements) > 0 else params.z_detector
    source.propagate_to(current_z, params, cutoff_freq, u, U, history)

    ######## Apply optical elements (gratings and samples) ########

    snapshot: Optional[Snapshot] = None

    for el_idx, el in enumerate(elements):
        if current_z < el.z_start:
            logger.info(f"Propagating from z {current_z}m to {el.z_start}m")
            propagate_with_history(
                u,
                U,
                params,
                el.z_start - current_z,
                cutoff_freq,
                current_z,
                False,
                history,
            )

        current_z = el.z_start

        if len(el.x_positions) > 1 and snapshot is None:
            logger.info(f"Taking snapshot at z={current_z}")
            snapshot = Snapshot(
                z=current_z,
                element_idx=el_idx,
                u=u.copy(),
                nr_phase_steps=len(el.x_positions),
            )

        cutoff_freq = cutoff_frequencies[el_idx + 1]

        save_keypoint(save_keypoints_path, el_idx, 0, u)

        element_name = type(el).__name__
        logger.info(
            f"Applying optical element {el_idx + 1}/{len(elements)} ({element_name})"
        )
        el.apply(
            u, U, params, cutoff_freq, 0, history[0] if history is not None else None
        )

        save_keypoint(save_keypoints_path, el_idx, 1, u)

        current_z = el.z_start + el.get_thickness()

    ######## Cover remaining distance to the detector ########

    if current_z < params.z_detector - z_tolerance:
        logger.info(f"Propagating from z {current_z}m to {params.z_detector}m")

        # Source.propagate_to() doesn't guarantee that U will contain the
        # FFT of u afterwards, but OpticalElement.apply does. We can therefore
        # skip the FFT calculation only if the previous step wasn't a source
        # propagation step.
        skip_fft = len(elements) > 0
        propagate_with_history(
            u,
            U,
            params,
            params.z_detector - current_z,
            cutoff_freq,
            current_z,
            skip_fft=skip_fft,
            history=history,
        )

    ######## Result ########

    detector_outputs = [square_and_downsample(u, params, params.z_detector)]
    save_u(
        u,
        phase_step=0,
        sub_dir=sub_dir,
        vectors_dir=vectors_dir,
        save_final_u_vectors=save_final_u_vectors,
    )

    ######## Remaining stepping iterations ########

    # When phase stepping is activated, we first simulate the first run through all the elements normally
    # and take a snapshot just before the element that is being stepped. Then for the remaining steps
    # we only simulate the distance from that point on and don't keep the history for the remaining steps.

    if snapshot is not None:
        el_start_idx = snapshot.element_idx
        total_steps = snapshot.nr_phase_steps
        for step in range(1, total_steps):
            current_z = snapshot.z
            snapshot.u.copy_to(u)
            cutoff_freq = cutoff_frequencies[el_start_idx]

            logger.info(
                f"Running step {step + 1}/{total_steps} from snapshot at z {current_z}m, element {el_start_idx + 1}"
            )
            for el_idx, el in enumerate(elements[el_start_idx:], start=el_start_idx):
                if current_z < el.z_start:
                    logger.info(f"Propagating from z {current_z}m to {el.z_start}m")
                    propagate(
                        u,
                        U,
                        params.dx,
                        params.wl,
                        el.z_start - current_z,
                        params.chunk_size,
                        cutoff_freq,
                    )

                current_z = el.z_start

                cutoff_freq = cutoff_frequencies[el_idx + 1]

                element_name = type(el).__name__
                logger.info(
                    f"Applying optical element {el_idx + 1}/{len(elements)} ({element_name})"
                )
                el_step = step if len(el.x_positions) > 1 else 0
                el.apply(u, U, params, cutoff_freq, el_step, None)

                current_z = el.z_start + el.get_thickness()

            if current_z < params.z_detector - z_tolerance:
                logger.info(f"Propagating from z {current_z}m to {params.z_detector}m")
                propagate(
                    u,
                    U,
                    params.dx,
                    params.wl,
                    params.z_detector - current_z,
                    params.chunk_size,
                    cutoff_freq,
                )

            detector_outputs.append(square_and_downsample(u, params, params.z_detector))
            save_u(
                u,
                phase_step=step,
                sub_dir=sub_dir,
                vectors_dir=vectors_dir,
                save_final_u_vectors=save_final_u_vectors,
            )

        snapshot.u.drop()

    return detector_outputs
