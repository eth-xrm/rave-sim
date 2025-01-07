# Copyright (c) 2024, ETH Zurich

import functools
import logging
from pathlib import Path
import numpy as np
from typing import Any, Optional
from ruamel.yaml import comments as yc, YAML

import optical_element
import propagation
import source
import vector

logger = logging.getLogger("big-wave")


# The ruamel.yaml type here is useful for preserving the comments in yaml configurations. Other than
# that we use it in the same way as a normal python dict.
DictType = yc.CommentedMap | dict[str, Any]


def get_float(dct: DictType | list, path: list) -> float:
    pathname = ".".join(map(str, path))
    try:
        val = functools.reduce(lambda d, k: d[k], path, dct)
    except KeyError:
        raise KeyError(f"path '{pathname}' does not exist")

    assert isinstance(val, (float, int)), f"entry '{pathname}' must be a float"
    return float(val)


def get_int(dct: DictType | list, path: list) -> int:
    pathname = ".".join(map(str, path))
    try:
        val = functools.reduce(lambda d, k: d[k], path, dct)
    except KeyError:
        raise KeyError(f"path '{pathname}' does not exist")

    assert isinstance(val, int), f"entry '{pathname}' must be an int"
    return int(val)


def parse_sim_params(dct: DictType) -> propagation.SimParams:
    return propagation.SimParams(
        N=get_int(dct, ["N"]),
        dx=get_float(dct, ["dx"]),
        z_detector=get_float(dct, ["z_detector"]),
        detector_size=get_float(dct, ["detector_size"]),
        detector_pixel_size_x=get_float(dct, ["detector_pixel_size_x"]),
        detector_pixel_size_y=get_float(dct, ["detector_pixel_size_y"]),
        wl=0.0,  # the wavelength might differ for individual sources so we don't parse that here yet
        chunk_size=get_int(dct, ["chunk_size"]),
    )


def parse_material(input: list) -> optical_element.Material:
    assert isinstance(input[0], str)
    return optical_element.Material(input[0], get_float(input, [1]))


def parse_optional_material(
    dct: DictType, key: str
) -> Optional[optical_element.Material]:
    if not key in dct:
        return None

    input = dct[key]
    if input is None:
        return None
    else:
        return parse_material(input)


def parse_optical_element(
    dct: DictType, config_dir: Path
) -> optical_element.OpticalElement:
    """
    Parse a single OpticalElement from a config dict.

    Parameters
    ----------
    dct : DictType
        Only the part of the config dict corresponding to this element. Should not be called with the whole config dict.
    config_dir : Path
        Directory from where the relative lookup to the grid arrays should happen.
    """

    if dct["type"] == "grating":
        return optical_element.Grating(
            pitch=get_float(dct, ["pitch"]),
            dc=(get_float(dct, ["dc", 0]), get_float(dct, ["dc", 1])),
            z_start=get_float(dct, ["z_start"]),
            thickness=get_float(dct, ["thickness"]),
            nr_steps=get_int(dct, ["nr_steps"]),
            x_positions=np.array(dct["x_positions"]),
            substrate_thickness=get_float(dct, ["substrate_thickness"]),
            mat_a=parse_optional_material(dct, "mat_a"),
            mat_b=parse_optional_material(dct, "mat_b"),
            mat_substrate=parse_optional_material(dct, "mat_substrate"),
        )
    if dct["type"] == "env_grating":
        return optical_element.EnvGrating(
            pitch0=get_float(dct, ["pitch0"]),
            pitch1=get_float(dct, ["pitch1"]),
            dc0=(get_float(dct, ["dc0", 0]), get_float(dct, ["dc0", 1])),
            dc1=(get_float(dct, ["dc1", 0]), get_float(dct, ["dc1", 1])),
            z_start=get_float(dct, ["z_start"]),
            thickness=get_float(dct, ["thickness"]),
            nr_steps=get_int(dct, ["nr_steps"]),
            x_positions=np.array(dct["x_positions"]),
            substrate_thickness=get_float(dct, ["substrate_thickness"]),
            mat_a=parse_optional_material(dct, "mat_a"),
            mat_b=parse_optional_material(dct, "mat_b"),
            mat_substrate=parse_optional_material(dct, "mat_substrate"),
        )
    elif dct["type"] == "sample":
        sample_materials = [parse_material(m) for m in dct["materials"]]
        grid = np.load(config_dir / dct["grid_path"])
        assert grid.dtype == np.int8

        return optical_element.Sample(
            z_start=get_float(dct, ["z_start"]),
            pixel_size_x=get_float(dct, ["pixel_size_x"]),
            pixel_size_z=get_float(dct, ["pixel_size_z"]),
            grid=grid,
            materials=sample_materials,
            x_positions=np.array(dct["x_positions"]),
        )
    elif dct["type"] == "save_and_exit":
        return optical_element.SaveAndExit(
            z_start=get_float(dct, ["z_start"]), x_positions=np.array([0])
        )
    else:
        raise ValueError(f'Unknown optical element type: {dct["type"]}.')


def parse_source(
    dct: DictType, use_disk_vector: bool, scratchfile: Path, N: int, dtype: np.dtype
) -> source.Source:
    if dct["type"] == "point":
        return source.PointSource(
            x=get_float(dct, ["x"]),
            z=get_float(dct, ["z"]),
        )
    elif dct["type"] == "vector":
        path = Path(dct["input_path"])
        v: vector.Vector
        if use_disk_vector:
            v = vector.DiskVector(
                file=path, scratchfile=scratchfile, len=N, dtype=dtype
            )
        else:
            nparr = np.load(path)
            if nparr.dtype != dtype:
                logger.warn(
                    f"Dtype {nparr.dtype} of loaded source vector {path} does not match dtype {dtype}. Converting."
                )
                nparr = nparr.astype(dtype)
            v = vector.NumpyVector(nparr)

        return source.VectorSource(
            input=v,
            z=get_float(dct, ["z"]),
        )
    else:
        raise ValueError(f'Unknown source type: {dct["type"]}.')


def parse_dtype(dct: DictType) -> np.dtype:
    """
    Given the full config dict, parse the dtype. Defaults to c16 if not specified.
    """

    if "dtype" not in dct:
        return np.dtype(np.complex128)
    val = dct["dtype"]

    if not isinstance(val, str):
        raise ValueError(f"Expected dtype to be a string, got {val}")

    if val == "c8":
        return np.dtype(np.complex64)
    elif val == "c16":
        return np.dtype(np.complex128)
    else:
        raise ValueError(f"Unknown dtype: {val}. Has to be either c8 or c16")


def load(path: Path) -> DictType:
    yaml = YAML(typ="rt")
    return yaml.load(path)


def save(path: Path, dct: DictType) -> None:
    with open(path, "w") as f:
        yaml = YAML(typ="rt")
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(dct, f)
