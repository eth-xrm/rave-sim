# Copyright (c) 2024, ETH Zurich

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import config

logger = logging.getLogger("big-wave")


def setup_logger():
    if len(logger.handlers) > 0:
        # logging has already been set up
        return

    stderr_logger = logging.StreamHandler()
    stderr_logger.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(stderr_logger)


def load_wavefronts_filtered(
    sim_dir: Path,
    x_range: Optional[Tuple[float, float]] = None,
    energy_range: Optional[Tuple[float, float]] = None,
) -> list[Tuple[np.ndarray, float, float]]:
    """
    Load a list of wavefronts from the simulation for source points that fall in a given x and energy range.

    Returns:
        A list of tuples of the form (wavefront: np.ndarray, x: float, energy: float)
    """

    if not isinstance(sim_dir, Path):
        sim_dir = Path(sim_dir)

    source_points = config.load(sim_dir / "computed.yaml")["source_points"]

    idxs = [
        i
        for i, sp in enumerate(source_points)
        if (x_range is None or sp["x"] >= x_range[0])
        if (x_range is None or sp["x"] <= x_range[1])
        if (energy_range is None or sp["energy"] >= energy_range[0])
        if (energy_range is None or sp["energy"] <= energy_range[1])
    ]

    return [
        (
            np.load(get_sub_dir(sim_dir, i) / "detected.npy"),
            source_points[i]["x"],
            source_points[i]["energy"],
        )
        for i in idxs
    ]


def load_keypoints(keypoints_path: Path) -> list[np.ndarray]:
    if not isinstance(keypoints_path, Path):
        keypoints_path = Path(keypoints_path)

    l = []
    for el_idx in range(100):
        try:
            l.append(np.load(keypoints_path / f"keypoint_{el_idx:02}_0.npy"))
            l.append(np.load(keypoints_path / f"keypoint_{el_idx:02}_1.npy"))
        except FileNotFoundError:
            break

    return l


def detector_x_vector(detector_size: float, detector_pixel_size_x: float) -> np.ndarray:
    """Generate the vector of x values corresponding to the detector pixels."""
    n = detector_size // detector_pixel_size_x
    x = (np.arange(n) - n // 2) * detector_pixel_size_x
    return x


def full_x_vector(N: int, dx: float) -> np.ndarray:
    """
    Generate the vector of x coordinates for the whole simulation.

    Warning: this will fill up your ram if N is sufficiently large.
    """
    return (np.arange(N) - N / 2) * dx


def get_sub_dir(sim_dir: Path, source_idx: int) -> Path:
    if not isinstance(sim_dir, Path):
        sim_dir = Path(sim_dir)
    return sim_dir / f"{source_idx:08}"
