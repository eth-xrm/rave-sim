# Copyright (c) 2024, ETH Zurich

from dataclasses import dataclass
import math
from typing import Tuple
import numpy as np

from util import detector_x_vector
from vector import Vector


@dataclass
class SimParams:
    """Global parameters for the simulation"""

    N: int
    dx: float
    z_detector: float
    detector_size: float
    detector_pixel_size_x: float
    detector_pixel_size_y: float
    wl: float
    chunk_size: int


h = 6.62607004 * 10 ** (-34)  # planck constant in mˆ2 kg / s
c_0 = 299792458  # speed of light in m / s
eV_to_joule = 1.602176634 * 10 ** (-19)


def grid_density_check(dz: float, x_source: float, N: int, dx: float, wl: float):
    """
    Check if dx is sufficiently small so that the waves after the initial analytical
    propagation don't violate the Nyquist condition.
    """

    def dist_x(idx: int) -> float:
        return np.sqrt(((N / 2 - idx) * dx + np.abs(x_source)) ** 2 + dz**2)

    nyquist_satisfied = np.abs(dist_x(0) - dist_x(1)) <= wl / 2

    if not nyquist_satisfied:
        raise ValueError(
            "Nyquist not satisfied, dx too large. In `propagation.py` there is a util function to compute the maximal possible dx."
        )


def max_dx(dz: float, x_source: float, N: int, wl: float) -> float:
    """
    Calculate the maximal possible dx such that the waves after the initial analytical
    propagation don't violate the Nyquist condition.

    We use binary search here because inverting this analytically is a pain.
    """
    min: float = 0
    max: float = 0.1

    while (max - min) / max > 0.01:
        dx = (max + min) / 2
        try:
            grid_density_check(dz, x_source, N, dx, wl)
            min = dx
        except:
            max = dx

    return min


def convert_energy_wavelength(energy_or_wavelength: float) -> float:
    """Given an energy in eV, calculate the wavelength in metres or vice versa"""
    return h * c_0 / (energy_or_wavelength * eV_to_joule)


def apply_frequency_cutoff(
    vec: Vector, freq: float, dx: float, chunk_size: int
) -> None:
    """Remove waves that do not pass through the detector. Modify the vector in place."""

    # the idea here is that instead of generating fftfreq/dx and then
    # comparing it to `freq` pointwise, we note that fftfreq first goes
    # from zero to 0.5 and then -0.5 to 0. Therefore we just have to remove
    # a part centered on the middle, the outer sections remain as they are.
    frac = freq * dx
    n = len(vec)
    removal_start_idx = math.ceil(n * frac)
    if removal_start_idx > n / 2:
        # nothing to do
        return

    removal_end_idx = n - removal_start_idx + 1

    def modifier(idx: int, chunk: np.ndarray) -> None:
        start = max(removal_start_idx - idx, 0)
        end = max(removal_end_idx - idx, 0)
        chunk[start:end] = 0.0

    vec.modify_chunked(chunk_size, modifier)


def propagate(
    u: Vector,
    U: Vector,
    dx: float,
    wl: float,  # wavelength
    dz: float,  # z distance to propagate
    chunk_size: int,  # number of vector entries to load into memory at once
    cutoff_frequency: float,
    skip_fft: bool = False,
) -> None:
    """
    Modify the provided vector to propagate it through empty space in the z direction using the Fresnel
    approximation.

    The `U` argument doesn't have to be a valid vector at the beginning, it will be overwritten before
    anything is read from it. After this function returns it will contain the fourier transform of the
    propagated vector.

    However if `skip_fft` is True, then the `U` argument should be the fourier transform of `u` and then
    we can skip the fft operation.
    """

    n = len(u)

    def propagate_chunk(idx: int, chunk: np.ndarray) -> None:
        fx = fftfreq_chunk(n, idx, chunk_size) / dx
        chunk *= np.exp(-1j * 2 * np.pi / wl * dz) * np.exp(
            1j * np.pi * wl * dz * (fx**2)
        )

    if not skip_fft:
        u.fft(U)

    U.modify_chunked(
        chunk_size,
        propagate_chunk,
    )
    apply_frequency_cutoff(U, cutoff_frequency, dx, chunk_size)
    U.ifft(u)


def propagate_analytically(u: Vector, z: float, x_source: float, params: SimParams):
    def analytical(idx: int, chunk: np.ndarray):
        x = (
            idx + np.arange(len(chunk), dtype=np.float64) - params.N / 2
        ) * params.dx - x_source
        r = np.sqrt(x**2 + z**2)
        # we use the sqrt of r instead of r so that the probability, i.e. wave function squared, decreases linearly with
        # distance instead of quadratically. This is because we are effectively in 2d and not 3d.
        chunk[:] = np.exp(r * (-2j * np.pi / params.wl)) / np.sqrt(r)

    u.write_chunked(params.chunk_size, analytical)


def fftfreq_chunk(n: int, start: int, chunk_size: int) -> np.ndarray:
    """
    Generate a chunk of the fftfreq array as computed by np.fft.fftfreq.

    Adapted from https://github.com/numpy/numpy/blob/v1.24.0/numpy/fft/helper.py#L123-L169

    This leads to some ugly calculations, but we have full unit test coverage so it's fine :D
    """

    N = (n - 1) // 2 + 1

    # ph, nh = positive/negative half, the first half of the fftfreq array contains positive values
    start_ph = start
    end_ph = min(max(start_ph, N), start + chunk_size)
    start_nh = max(-(n // 2), start - n)
    end_nh = min(0, max(start + chunk_size - n, start_nh))

    ph = np.arange(start_ph, end_ph, dtype=int)
    nh = np.arange(start_nh, end_nh, dtype=int)

    fac = 1.0 / n
    return np.concatenate((ph, nh)) * fac


def compute_cutoff_angles(
    detector_size: float,
    dx: float,
    energy_range: Tuple[float, float],
    z_source: float,
    z_distances: list[float],
    element_heights: list[float],
    z_detector: float,
    max_x: float,
) -> Tuple[list[float], list[float]]:
    """
    Calculate the list of angles to which we limit the simulation that ensures that all interesting rays are kept
    but tries to remove as much of the excess rays as possible. In earlier z ranges we can keep a narrow angle,
    for later z values the angle will have to open up.

    Parameters
    ----------
    detector_size : float
        Width of the detector in metres
    dx : float
        Spacing between points in the simulation.
    energy_range : Tuple[float, float]
        A tuple of the minimum and maximum photon energy in eV
    z_distances : list[float]
        List of z distances from the source where gratings and samples are located. Note that
        this function doesn't consider the thickness of these optical elements.
    element_heights : list[float]
        List of the z-heights of all of the optical elements in the simulation
    z_detector : float
        Distance to the detector.
    max_x : float
        Maximal absolute source x position from which we want to consider rays

    Returns
    -------
    (cutoff angles, max x at each of the optical element starts and at z_detector) : Tuple[list[float], list[float]]
    """

    assert len(z_distances) == len(element_heights)
    assert energy_range[0] <= energy_range[1]

    wl_e_min = convert_energy_wavelength(energy_range[0])
    wl_e_max = convert_energy_wavelength(energy_range[1])

    direct_source_to_detector_angle = np.arctan(
        (detector_size / 2 + max_x) / z_detector
    )
    if 2 / (np.sin(direct_source_to_detector_angle) / wl_e_max) < dx:
        deg = direct_source_to_detector_angle * 180.0 / math.pi
        threshold = 2 / (np.sin(direct_source_to_detector_angle) / wl_e_max)
        raise ValueError(
            f"Grid sampled to coarsly, highest energy waves won't cover the full detector! Angle {deg} degrees requires dx <= {threshold}."
        )

    # When figuring out the largest angles we have to account for, it makes sense to first figure out how large the angles can even get
    # from a frequency standpoint. It doesn't make sense to allow for huge angles that will never be filled in the frequency spectrum
    # anyway. Low energies correspond to large angles, so that's where we are the most constrained.
    e_min_gradient: float
    if wl_e_min * 2 / dx < 1:
        e_min_angle = np.arcsin(wl_e_min * 2 / dx)
        e_min_gradient = np.tan(e_min_angle)
    else:
        # The angle is 90 degrees, no restriction gained
        e_min_gradient = float(np.inf)

    current_max_x = max_x
    current_z = z_source
    max_gradients: list[float] = []
    max_x_list: list[float] = []
    for next_z in z_distances + [z_detector]:
        if current_z == z_detector:
            raise ValueError(
                "An element in the list has the same z coordinate as the detector! This is not allowed."
            )

        max_gradients.append(
            min(
                (detector_size / 2 + current_max_x) / (z_detector - current_z),
                e_min_gradient,
            )
        )
        current_max_x += max_gradients[-1] * (next_z - current_z)
        max_x_list.append(float(current_max_x))
        current_z = next_z

    assert current_max_x >= detector_size / 2

    # When the detector is placed right after the last element without any gap in
    # between, we want to not increase the cutoff angle any further for this last
    # step, since we're not going to perform any further propagation there.
    skip_last_propagate = (
        len(z_distances) > 0
        and np.abs(z_detector - (z_distances[-1] + element_heights[-1])) < 1e-8
    )
    if skip_last_propagate:
        max_gradients[-1] = max_gradients[-2]
        max_x_list[-1] = max_x_list[-2] + max_gradients[-1] * element_heights[-1]

    return ([float(np.arctan(g)) for g in max_gradients], max_x_list)


def square_and_downsample(
    u: Vector, sim_params: SimParams, current_z: float
) -> np.ndarray:
    """
    Compute the detector output from `u` by cropping to the detector size, computing the intensity, and downscaling.
    """

    outsize = int(sim_params.detector_size // sim_params.detector_pixel_size_x)
    assert outsize <= sim_params.N

    out = np.zeros(outsize, dtype=np.float64)
    n = len(u)
    assert n >= outsize

    # Weird hack: np.histogram doesn't use half-open intervals for every bin, instead the upper range
    # end is actually inclusive, which just seems weird. We get around this by adding an additional
    # bin at the upper end, which we then discard. See the [:-1] below.
    detector_range = (
        -(outsize / 2) * sim_params.detector_pixel_size_x,
        (outsize / 2 + 1) * sim_params.detector_pixel_size_x,
    )
    nr_bins = outsize + 1

    def f(idx: int, chunk: np.ndarray) -> None:
        positions = (idx + np.arange(len(chunk)) - sim_params.N / 2) * sim_params.dx
        abs_chunk = chunk.real * chunk.real + chunk.imag * chunk.imag

        out[:] += np.histogram(
            positions, bins=nr_bins, range=detector_range, weights=abs_chunk
        )[0][:-1]

    u.read_chunked(sim_params.chunk_size, f)

    x = detector_x_vector(sim_params.detector_size, sim_params.detector_pixel_size_x)
    angles = np.arctan(x / current_z)
    return (
        out
        * np.cos(angles)
        * sim_params.dx
        * sim_params.detector_pixel_size_y
        / np.sqrt(x**2 + current_z**2)
    )
