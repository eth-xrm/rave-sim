# Copyright (c) 2024, ETH Zurich

from freezegun import freeze_time
from pathlib import Path
import bfpy
import numpy as np
import os
import tempfile
import unittest

import config
from history import History
import multisim
import optical_element
import propagation
import source
import util
import vector
import wavesim


class TempDirTest(unittest.TestCase):
    """Run unittests in a temporary directory"""

    tempdir: tempfile.TemporaryDirectory
    path: Path

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name)
        os.chdir(self.path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class TestBigFourier(TempDirTest):
    def test_chunked_editor(self) -> None:
        arr = np.array([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j])
        np.save("temp.npy", arr)

        ce = bfpy.ChunkedEditor(Path("temp.npy"))
        self.assertEqual(len(ce), len(arr))

        buffer = np.zeros(2, dtype="c16")
        for i in range(3):
            self.assertEqual(ce.position(), i * 2)
            nr_items = ce.read_chunk_c16(buffer)
            if i < 2:
                self.assertEqual(nr_items, 2)
            else:
                self.assertEqual(nr_items, 1)

            self.assertEqual(
                list(arr[i * 2 : i * 2 + nr_items]), list(buffer[:nr_items])
            )

            buffer[0] += 1j
            ce.write_chunk_c16(buffer[:nr_items])

            ce.advance(2)

        self.assertEqual(ce.position(), len(arr))

        modified = np.load("temp.npy")
        expected = arr + [1j, 0, 1j, 0, 1j]
        self.assertEqual(list(modified), list(expected))

    def test_fft_ifft(self) -> None:
        arr = np.array([1 + 0j, 2 + 5j, 3 + 1j, 4 + 0j, 5 - 6j, 6 - 1j, 7 + 1j, 8 + 0j])
        inpath = Path("in.npy")
        outpath = Path("out.npy")
        scratchpath = Path("scratch.npy")

        np.save(inpath, arr)

        bfpy.fft_c16(inpath, outpath, scratchpath)

        result = np.load(outpath)
        expected = np.fft.fft(arr)
        self.assertTrue(np.allclose(result, expected))

        bfpy.ifft_c16(outpath, inpath, scratchpath)

        result = np.load(inpath)
        self.assertTrue(np.allclose(result, arr))


class TestWaveSim(TempDirTest):
    def test_fftfreq(self) -> None:
        for n in [4, 5]:
            expected = np.fft.fftfreq(n)
            for start in range(0, n):
                for end in range(start, n):
                    chunk = propagation.fftfreq_chunk(n, start, end - start)
                    self.assertTrue(np.allclose(expected[start:end], chunk))
                    self.assertEqual(len(chunk), end - start)

    def test_history(self) -> None:
        history = History()

        gt = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=np.complex128,
        ).T

        self.assertEqual(len(history), 0)
        self.assertEqual(history.get_history().shape[1], 0)

        history.push(gt[:, 0], z=1.0)
        self.assertTrue(np.allclose(history.get_history(), gt[:, 0:1]))
        self.assertEqual(history.get_z(), [1.0])

        history.push(gt[:, 1], z=2.0)
        self.assertTrue(np.allclose(history.get_history(), gt))
        self.assertEqual(history.get_z(), [1.0, 2.0])

    def test_grating(self) -> None:
        n = 16
        pitch = 4
        dx = 0.5

        for offset in [0, 3]:
            # `offset` is counted in array indices, from this we calculate
            # the x positions.
            x_position = offset * dx + 0.01
            grating = optical_element.Grating(
                pitch=pitch,
                dc=(0.25, 0.5),
                z_start=0,
                thickness=1,
                nr_steps=1,
                x_positions=np.array([x_position]),
                substrate_thickness=0.0,
                mat_a=optical_element.Material("Si", 2.34),
                mat_b=None,
                mat_substrate=optical_element.Material("Si", 2.34),
            )

            for z_fraction, template in [
                (0, [1, 0, 0, 0, 0, 0, 0, 1]),  # 0.25 dc
                (1, [1, 1, 0, 0, 0, 0, 1, 1]),  # 0.5 dc
            ]:
                # we repeat the template multiple times and then later take a subrange of it
                expected = np.tile(template, 10)
                for start in range(n):
                    for end in range(start, n):
                        self.assertEqual(
                            list(expected[start + offset : end + offset]),
                            list(
                                optical_element.generate_grating_chunk(
                                    0.5, start, end - start, grating, z_fraction, 0
                                )
                            ),
                        )

    def test_grating_stepping(self) -> None:
        pitch = 4.2e-6

        grating = optical_element.Grating(
            pitch=pitch,
            dc=(0.25, 0.5),
            z_start=0,
            thickness=1,
            nr_steps=1,
            x_positions=np.array([0.0, pitch * 0.5, pitch, pitch * 2]),
            substrate_thickness=0.0,
            mat_a=optical_element.Material("Si", 2.34),
            mat_b=None,
            mat_substrate=optical_element.Material("Si", 2.34),
        )

        masks = [
            optical_element.generate_grating_chunk(
                1e-6, 123, 16, grating, 0, stepping_iteration
            ).tolist()
            for stepping_iteration in range(len(grating.x_positions))
        ]

        self.assertEqual(masks[0], masks[2])
        self.assertEqual(masks[0], masks[3])
        self.assertNotEqual(masks[0], masks[1])

    def test_frequency_cutoff(self) -> None:
        def reference(vec: np.ndarray, freq: float, dx: float):
            vec *= np.abs(np.fft.fftfreq(len(vec)) / dx) < freq

        for dx in [0.5, 1.0, 2.3]:
            for n in [3, 4, 5, 6]:
                for freq in [0.0, 0.1, 0.4, 1.2, 3.5, 100000.0]:
                    for chunk_size in [3, 4, 6]:
                        npvec = np.ones(n, dtype=np.complex128)
                        vec = vector.NumpyVector(npvec.copy())
                        propagation.apply_frequency_cutoff(vec, freq, dx, chunk_size)
                        reference(npvec, freq, dx)
                        self.assertEqual(list(vec.vec), list(npvec))

    def test_energy_conversion(self) -> None:
        e = 46000
        wl = 26.95e-12
        self.assertAlmostEqual(
            propagation.convert_energy_wavelength(e), wl, delta=0.01e-12
        )
        self.assertAlmostEqual(propagation.convert_energy_wavelength(wl), e, delta=10)

    def test_square_and_downsample(self) -> None:
        vec = vector.NumpyVector(np.arange(8, dtype=np.complex128))

        params = propagation.SimParams(
            N=8,
            dx=0.5,
            z_detector=1.0,
            detector_size=3.0,
            detector_pixel_size_x=1.0,
            detector_pixel_size_y=1.0,
            wl=1.0,
            chunk_size=3,
        )
        self.assertEqual(
            list(propagation.square_and_downsample(vec, params, 1.0)),
            [1.25, 12.5, 15.249999999999998],
        )

    def test_detector_x(self) -> None:
        vec = vector.NumpyVector(np.arange(118, dtype=np.complex128))

        for detector_pixel_size_x in [0.3, 0.5, 0.6, 1.0]:
            for detector_size in [1.0, 2.0, 10.0]:
                detector_output = propagation.square_and_downsample(
                    vec,
                    propagation.SimParams(
                        N=len(vec),
                        dx=0.2,
                        z_detector=1.0,
                        detector_size=detector_size,
                        detector_pixel_size_x=detector_pixel_size_x,
                        detector_pixel_size_y=1.0,
                        wl=1.0,
                        chunk_size=3,
                    ),
                    1.0,
                )
                x = util.detector_x_vector(detector_size, detector_pixel_size_x)

                self.assertEqual(len(detector_output), len(x))
                self.assertTrue(x[len(x) // 2] == 0)
                self.assertAlmostEqual(x[0], -x[-1], delta=detector_pixel_size_x * 1.01)

    def test_stepping(self) -> None:
        N = 6
        u = vector.NumpyVector(np.zeros(N, dtype=np.complex128))
        U = vector.NumpyVector(np.zeros(N, dtype=np.complex128))

        def grating(z_start, x_positions):
            return optical_element.Grating(
                pitch=4.2e-6,
                dc=(0.5, 0.5),
                z_start=z_start,
                thickness=100e-6,
                nr_steps=1,
                x_positions=np.array(x_positions),
                substrate_thickness=0.0,
                mat_a=None,
                mat_b=None,
                mat_substrate=None,
            )

        params = propagation.SimParams(
            N=N,
            dx=10e-9,
            z_detector=1.0,
            detector_size=40e-9,
            detector_pixel_size_x=20e-9,
            detector_pixel_size_y=1.0,
            wl=propagation.convert_energy_wavelength(46000),
            chunk_size=3,
        )

        hist = wavesim.History()

        u_out = wavesim.run_simulation(
            params,
            source.PointSource(x=0.0, z=0.0),
            [
                grating(0.1, [0.0]),
                grating(0.2, [0.0, 1.6e-6, 3.1e-6]),
            ],
            [0.1, 0.2, 0.3],
            u,
            U,
            deltabeta_table=[],
            sub_dir=self.path,
            vectors_dir=self.path,
            save_final_u_vectors=False,
            history=(hist, 0.1),
        )

        self.assertEqual(len(u_out), 3)
        self.assertEqual(u_out[0].shape, (2,))
        self.assertEqual(u_out[1].shape, (2,))
        self.assertEqual(u_out[2].shape, (2,))

        # none in propagate analytically, 9 steps in empty space, 1 step for each of the 2 gratings
        self.assertEqual(hist.get_history().shape, (2, 0 + 9 + 2))

        u_out = wavesim.run_simulation(
            params,
            source.PointSource(x=0.0, z=0.0),
            [
                grating(0.1, [0.0]),
                grating(0.2, [0.0, 1.6e-6, 3.1e-6]),
                grating(0.3, [0.0]),
            ],
            [0.1, 0.2, 0.3, 0.4],
            u,
            U,
            [],
            self.path,
            self.path,
            False,
            None,
        )

        self.assertEqual(len(u_out), 3)
        self.assertEqual(u_out[0].shape, (2,))
        self.assertEqual(u_out[1].shape, (2,))
        self.assertEqual(u_out[2].shape, (2,))

    def test_db_table_lookup(self) -> None:
        si1 = optical_element.Material("Si", 2.34)
        si2 = optical_element.Material("Si", 2.12)
        au = optical_element.Material("Au", 3.51)
        xy = optical_element.Material("Xy", 1.34)

        table: optical_element.DeltabetaTable = [
            (au, np.complex128(1)),
            (si1, np.complex128(2)),
            (si2, np.complex128(3)),
        ]

        self.assertEqual(optical_element.extract_from_deltabeta_table(si1, table), 2)

        self.assertEqual(optical_element.extract_from_deltabeta_table(au, table), 1)

        self.assertRaises(
            AssertionError,
            lambda: optical_element.extract_from_deltabeta_table(xy, table),
        )

    def test_max_dx(self) -> None:
        wl = 1.8e-11
        dz = 0.1
        N = 2**28
        x_source = 0.0
        dx_1 = propagation.max_dx(dz, x_source, N, wl)
        propagation.grid_density_check(dz, x_source, N, dx_1, wl)

        x_source = 0.01
        dx_2 = propagation.max_dx(dz, x_source, N, wl)
        propagation.grid_density_check(dz, x_source, N, dx_2, wl)

        # The larger the source the more constrained we'd expect dx to be
        self.assertLess(dx_2, dx_1)


class TestConfigParsing(unittest.TestCase):
    def test_material(self) -> None:
        dct = {"mat": ["Si", 2.34]}
        self.assertEqual(
            config.parse_optional_material(dct, "mat"),
            optical_element.Material("Si", 2.34),
        )

    def test_empty_material(self) -> None:
        self.assertEqual(config.parse_optional_material({"mat": None}, "mat"), None)


class TestNumpyVector(unittest.TestCase):
    def test_modify(self) -> None:
        v = vector.NumpyVector(np.zeros(5, dtype=np.complex128))

        def f(idx, chunk):
            chunk[0] += idx + 1

        v.modify_chunked(2, f)
        self.assertEqual(list(v.vec), [1, 0, 3, 0, 5])

    def test_len(self) -> None:
        v = vector.NumpyVector(np.zeros(7, dtype=np.complex128))
        self.assertEqual(len(v), 7)


def make_config(disk_vector: bool, save_final: bool) -> config.DictType:
    return {
        "sim_params": {
            "N": 4096 * 4,
            "dx": 8e-9,
            "z_detector": 1.77,
            "detector_size": 1e-5,
            "detector_pixel_size_x": 1e-6,
            "detector_pixel_size_y": 1e-6,
            "chunk_size": 10469376,
        },
        "use_disk_vector": disk_vector,
        "save_final_u_vectors": save_final,
        "dtype": "c16",
        "multisource": {
            "type": "points",
            "energy_range": [46000.0, 47000.0],
            "x_range": [0.0, 1e-6],
            "z": 0.0,
            "nr_source_points": 2,
            "seed": 4,
        },
        "elements": [
            {
                "type": "grating",
                "pitch": 4.2e-6,
                "dc": [0.5, 0.5],
                "z_start": 0.1,
                "thickness": 140e-6,
                "nr_steps": 2,
                "x_positions": [0.0],
                "substrate_thickness": 230e-6,
                "mat_a": ["Si", 2.34],
                "mat_b": ["Au", 19.32],
                "mat_substrate": ["Si", 2.34],
            },
            {
                "type": "sample",
                "z_start": 0.8,
                "pixel_size_x": 10e-6,
                "pixel_size_z": 10000e-6,
                "grid_path": "grid.npy",
                "materials": [["Si", 2.34], ["Au", 19.32]],
                "x_positions": [0.0],
            },
        ],
    }


class TestMultisim(TempDirTest):
    @freeze_time("2023-01-02 03:04:05")
    def test_setup_simple(self) -> None:
        grid = np.array([[1, 2, 0]], dtype=np.int8)
        np.save(self.path / "grid.npy", grid)

        dct = make_config(disk_vector=False, save_final=False)

        save_dir = self.path / "save"
        save_dir.mkdir()
        sim_dir = multisim.setup_simulation(dct, self.path, save_dir)

        expected_sim_dir = self.path / "save" / "2023" / "01" / "20230102_030405000000"
        self.assertTrue(expected_sim_dir.is_dir())
        self.assertEqual(sim_dir, expected_sim_dir)

        cfg = config.load(expected_sim_dir / "config.yaml")
        self.assertEqual(dct, cfg)

        computed = config.load(expected_sim_dir / "computed.yaml")
        self.assertEqual(len(computed["cutoff_angles"]), 3)
        self.assertTrue(computed["max_x"] > dct["multisource"]["x_range"][1])
        self.assertEqual(computed["energy_range"], dct["multisource"]["energy_range"])
        self.assertEqual(
            len(computed["source_points"]), dct["multisource"]["nr_source_points"]
        )

        for i in range(dct["multisource"]["nr_source_points"]):
            self.assertTrue((expected_sim_dir / f"{i:08}").is_dir())
            sub_dct = config.load(expected_sim_dir / f"{i:08}" / "subconfig.yaml")
            self.assertEqual(sub_dct["source"]["type"], "point")
            x_range = dct["multisource"]["x_range"]
            self.assertTrue(x_range[0] <= sub_dct["source"]["x"] < x_range[1])
            self.assertEqual(sub_dct["source"]["z"], dct["multisource"]["z"])
            energy_range = dct["multisource"]["energy_range"]
            self.assertTrue(energy_range[0] <= sub_dct["energy"] < energy_range[1])

            self.assertEqual(sub_dct["source"]["x"], computed["source_points"][i]["x"])
            self.assertEqual(sub_dct["energy"], computed["source_points"][i]["energy"])

        self.assertTrue((expected_sim_dir / "grid.npy").is_file())

        elements = [config.parse_optical_element(el, sim_dir) for el in cfg["elements"]]
        self.assertEqual(len(elements), 2)
        self.assertTrue(isinstance(elements[0], optical_element.Grating))
        self.assertTrue(isinstance(elements[1], optical_element.Sample))
        self.assertTrue(np.array_equal(elements[1].grid, grid))  # type: ignore [attr-defined]

    def test_run_individual(self) -> None:
        np.save(self.path / "grid.npy", np.array([[1, 2, 0]], dtype=np.int8))

        dct = make_config(disk_vector=True, save_final=False)
        sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        multisim.run_single_simulation(sim_dir, 0, sim_dir)
        upath = multisim.get_sub_dir(sim_dir, 0) / "u_0000.npy"
        self.assertFalse(upath.is_file())

        dct = make_config(disk_vector=True, save_final=True)
        sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        multisim.run_single_simulation(sim_dir, 1, sim_dir)
        upath = multisim.get_sub_dir(sim_dir, 1) / "u_0000.npy"
        self.assertTrue(upath.is_file())

        dct = make_config(disk_vector=False, save_final=True)
        sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        multisim.run_single_simulation(sim_dir, 0, sim_dir)
        upath = multisim.get_sub_dir(sim_dir, 0) / "u_0000.npy"
        self.assertTrue(upath.is_file())
        self.assertEqual(np.load(upath).shape, (dct["sim_params"]["N"],))

    def test_continuing_simulation(self) -> None:
        np.save(self.path / "grid.npy", np.array([[1, 2, 0]], dtype=np.int8))
        dct = make_config(disk_vector=True, save_final=True)
        base_sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        for i in range(2):
            multisim.run_single_simulation(base_sim_dir, i, base_sim_dir)

        # check if the output of the simulation is saved
        sp = dct["sim_params"]
        detector_output_len = sp["detector_size"] // sp["detector_pixel_size_x"]
        for i in range(2):
            sub_dir = multisim.get_sub_dir(base_sim_dir, i)
            file = sub_dir / "detected.npy"
            data = np.load(file)
            self.assertEqual(data.dtype, np.float64)
            self.assertEqual(data.shape, (1, detector_output_len))
            self.assertTrue(np.all(data >= 0))
            self.assertTrue((sub_dir / f"u_0000.npy").is_file())

        dct["sim_params"]["z_detector"] = 3.0
        dct["multisource"] = {
            "type": "vectors",
            "base_sim_dir": str(base_sim_dir),
            "input_u_index": 0,
        }
        dct["use_disk_vector"] = False
        dct["save_final_u_vectors"] = False
        dct["elements"] = [
            {
                "type": "grating",
                "pitch": 4.2e-6,
                "dc": [0.5, 0.5],
                "z_start": 1.8,
                "thickness": 140e-6,
                "nr_steps": 2,
                "x_positions": [0.0],
                "substrate_thickness": 230e-6,
                "mat_a": ["Si", 2.34],
                "mat_b": ["Au", 19.32],
                "mat_substrate": ["Si", 2.34],
            },
        ]

        sim_dir = multisim.setup_simulation(dct, self.path, self.path)

        computed = config.load(sim_dir / "computed.yaml")
        self.assertEqual(len(computed["source_points"]), 2)
        self.assertTrue("x" in computed["source_points"][0])
        self.assertTrue("energy" in computed["source_points"][0])

        for i in range(2):
            multisim.run_single_simulation(sim_dir, i, sim_dir)

        for i in range(2):
            file = multisim.get_sub_dir(sim_dir, i) / "detected.npy"
            data = np.load(file)
            self.assertEqual(data.dtype, np.float64)
            self.assertEqual(data.shape, (1, detector_output_len))
            self.assertTrue(np.all(data >= 0))

    def test_keypoints(self) -> None:
        np.save(self.path / "grid.npy", np.array([[1, 2, 0]], dtype=np.int8))
        dct = make_config(disk_vector=False, save_final=False)
        sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        kp_path = self.path / "keypoints"
        kp_path.mkdir()

        n = dct["sim_params"]["N"]

        multisim.run_single_simulation(sim_dir, 0, sim_dir, kp_path)

        kps = util.load_keypoints(kp_path)
        self.assertEqual(len(kps), 2 * 2)
        for kp in kps:
            self.assertEqual(kp.dtype, np.complex128)
            self.assertEqual(kp.shape, (n,))

    def test_load_wavefronts(self) -> None:
        np.save(self.path / "grid.npy", np.array([[1, 2, 0]], dtype=np.int8))
        dct = make_config(disk_vector=False, save_final=False)
        dct["sim_params"]["seed"] = 5
        sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        for i in range(2):
            multisim.run_single_simulation(sim_dir, i, sim_dir)

        # in the simulation params we have these ranges
        # energy: [46000.0, 47000.0]
        # x: [0.0, 1e-6]

        # using seed 5 we get these values:
        # point 0: 9.726e-07, 46967.029
        # point 1: 7.148e-07, 46547.232

        def count(x_range=None, energy_range=None) -> int:
            return len(util.load_wavefronts_filtered(sim_dir, x_range, energy_range))

        self.assertEqual(count(), 2)

        self.assertEqual(count(x_range=(5e-7, 6e-7)), 0)
        self.assertEqual(count(x_range=(6e-7, 8e-7)), 1)
        self.assertEqual(count(x_range=(8e-7, 9e-7)), 0)
        self.assertEqual(count(x_range=(9e-7, 10e-7)), 1)
        self.assertEqual(count(x_range=(10e-7, 11e-7)), 0)
        self.assertEqual(count(x_range=(5e-7, 11e-7)), 2)

        self.assertEqual(count(energy_range=(46400, 46500)), 0)
        self.assertEqual(count(energy_range=(46500, 46600)), 1)
        self.assertEqual(count(energy_range=(46600, 46700)), 0)
        self.assertEqual(count(energy_range=(46700, 46990)), 1)
        self.assertEqual(count(energy_range=(46990, 46990)), 0)
        self.assertEqual(count(energy_range=(46990, 46991)), 0)
        self.assertEqual(count(energy_range=(46400, 46991)), 2)

        self.assertEqual(count(x_range=(6e-7, 8e-7), energy_range=(46700, 46990)), 0)
        self.assertEqual(count(x_range=(9e-7, 10e-7), energy_range=(46700, 46990)), 1)

        sp = dct["sim_params"]
        detector_output_len = sp["detector_size"] // sp["detector_pixel_size_x"]

        self.assertEqual(len(util.load_wavefronts_filtered(sim_dir)[0]), 3)
        self.assertEqual(
            util.load_wavefronts_filtered(sim_dir)[0][0].shape, (1, detector_output_len)
        )
        self.assertTrue(
            util.load_wavefronts_filtered(sim_dir, x_range=(9e-7, 10e-7))[0][1] >= 9e-7
        )

    def test_generate_spectrum(self) -> None:
        spectrum_energies = np.arange(5) * 0.5
        spectrum_intensities = np.zeros_like(spectrum_energies)
        spectrum_intensities[3] = 2
        spectrum_intensities[4] = 10

        rs = np.random.RandomState(123)
        energies = multisim.generate_energies_from_spectrum(
            spectrum_energies, spectrum_intensities, rs, (0, 2), 100
        )

        self.assertEqual(len(energies), 100)
        self.assertTrue(np.all(energies >= 1.5))
        self.assertTrue(np.all(energies < 2))
        self.assertTrue(np.any(energies >= 1.75))

    def configure_from_ruamel_dict(self) -> None:
        """
        Ensure that the setup_simulation function works with a loaded yaml object,
        which might have ever so slightly different behaviour from a normal python
        dict.
        """

        odct = {
            "sim_params": {
                "N": 4096 * 4,
                "dx": 8e-9,
                "z_detector": 1.77,
                "detector_size": 1e-5,
                "detector_pixel_size_x": 1e-6,
                "detector_pixel_size_y": 1e-6,
                "chunk_size": 10469376,
            },
            "use_disk_vector": False,
            "save_final_u_vectors": False,
            "dtype": "c16",
            "multisource": {
                "type": "points",
                "energy_range": [46000.0, 47000.0],
                "x_range": [0.0, 1e-6],
                "z": 0.0,
                "nr_source_points": 1,
                "seed": 4,
            },
            "elements": [
                {
                    "type": "save_and_exit",
                    "z_start": 0.1,
                },
                {
                    "type": "grating",
                    "pitch": 4.2e-6,
                    "dc": [0.5, 0.5],
                    "z_start": 0.1,
                    "thickness": 140e-6,
                    "nr_steps": 1,
                    "x_positions": [0.0],
                    "substrate_thickness": 0,
                    "mat_a": ["Si", 2.34],
                    "mat_b": ["Au", 19.32],
                },
            ],
        }
        path = self.path / "cfg.yaml"
        config.save(path, odct)
        dct = config.load(path)
        multisim.setup_simulation(dct, self.path, self.path)

    def test_save_and_exit_element(self) -> None:
        dct = {
            "sim_params": {
                "N": 4096 * 4,
                "dx": 8e-9,
                "z_detector": 1.77,
                "detector_size": 1e-5,
                "detector_pixel_size_x": 1e-6,
                "detector_pixel_size_y": 1e-6,
                "chunk_size": 10469376,
            },
            "use_disk_vector": False,
            "save_final_u_vectors": False,
            "dtype": "c16",
            "multisource": {
                "type": "points",
                "energy_range": [46000.0, 47000.0],
                "x_range": [0.0, 1e-6],
                "z": 0.0,
                "nr_source_points": 1,
                "seed": 4,
            },
            "elements": [
                {
                    "type": "grating",
                    "pitch": 4.2e-6,
                    "dc": [0.5, 0.5],
                    "z_start": 0.1,
                    "thickness": 140e-6,
                    "nr_steps": 1,
                    "x_positions": [0.0],
                    "substrate_thickness": 0,
                    "mat_a": ["Si", 2.34],
                    "mat_b": ["Au", 19.32],
                },
                {
                    "type": "save_and_exit",
                    "z_start": 0.25,
                },
                {
                    "type": "grating",
                    "pitch": 4.2e-6,
                    "dc": [0.5, 0.5],
                    "z_start": 0.25,
                    "thickness": 140e-6,
                    "nr_steps": 1,
                    "x_positions": [0.0],
                    "substrate_thickness": 0,
                    "mat_a": ["Si", 2.34],
                    "mat_b": ["Au", 19.32],
                },
                {
                    "type": "grating",
                    "pitch": 4.2e-6,
                    "dc": [0.5, 0.5],
                    "z_start": 0.5,
                    "thickness": 140e-6,
                    "nr_steps": 1,
                    "x_positions": [0.0],
                    "substrate_thickness": 0,
                    "mat_a": ["Si", 2.34],
                    "mat_b": ["Au", 19.32],
                },
            ],
        }
        base_sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        dct_adjusted = config.load(base_sim_dir / "config.yaml")

        # save_final_u_vectors gets activated when a save_and_exit element is present
        self.assertTrue(dct_adjusted["save_final_u_vectors"])
        self.assertEqual(len(dct_adjusted["elements"]), 1)
        self.assertEqual(dct_adjusted["sim_params"]["z_detector"], 0.25)

        multisim.run_single_simulation(base_sim_dir, 0, base_sim_dir)
        sub_dir = multisim.get_sub_dir(base_sim_dir, 0)
        self.assertTrue((sub_dir / f"u_0000.npy").is_file())

        dct["multisource"] = {
            "type": "vectors",
            "base_sim_dir": str(base_sim_dir),
            "input_u_index": 0,
        }
        dct["elements"] = dct["elements"][2:]  # type: ignore [index]

        sim_dir = multisim.setup_simulation(dct, self.path, self.path)
        computed = config.load(sim_dir / "computed.yaml")
        base_computed = config.load(base_sim_dir / "computed.yaml")

        self.assertEqual(computed["cutoff_angles"], base_computed["cutoff_angles"][2:])

        multisim.run_single_simulation(sim_dir, 0, sim_dir)
        detected = np.load(multisim.get_sub_dir(sim_dir, 0) / "detected.npy")
        self.assertEqual(len(detected.shape), 2)
        self.assertEqual(detected.dtype, np.float64)


class TestDiskVector(TempDirTest):
    def test_modify(self) -> None:
        file = self.path / "data.npy"
        scratchfile = self.path / "scratch.npy"
        v = vector.DiskVector(file, scratchfile, 5, dtype=np.dtype(np.complex128))
        v.zero(123)

        def f(idx, chunk):
            chunk[0] += idx + 1

        v.modify_chunked(2, f)
        self.assertEqual(list(np.load(v.file)), [1, 0, 3, 0, 5])

    def test_zero(self) -> None:
        file = self.path / "data.npy"
        scratchfile = self.path / "scratch.npy"
        with open(file, "+a") as f:
            # add BS data to file
            f.write("asdf" * 64)

        v = vector.DiskVector(file, scratchfile, 5, dtype=np.dtype(np.complex128))
        v.zero(4)
        self.assertEqual(list(np.load(v.file)), [0, 0, 0, 0, 0])

    def test_dtype_check(self) -> None:
        file = self.path / "data.npy"
        scratchfile = self.path / "scratch.npy"
        a = np.array([1, 2])

        np.save(file, a.astype(np.complex64))

        v = vector.DiskVector(file, scratchfile, 2, dtype=np.dtype(np.complex128))
        with self.assertRaises(AssertionError):
            v.read_chunked(1, lambda a, b: None)

        np.save(file, a.astype(np.complex128))
        v.read_chunked(
            1, lambda idx, chunk: self.assertEqual(chunk.dtype, np.complex128)
        )

        np.save(file, a.astype(np.complex64))
        v = vector.DiskVector(file, scratchfile, 2, dtype=np.dtype(np.complex64))
        v.read_chunked(
            1, lambda idx, chunk: self.assertEqual(chunk.dtype, np.complex64)
        )


class TestSnapshot(TempDirTest):
    """
    Compare the output to the output of an earlier version of the code.

    This does not guarantee that the output is correct, it just guarantees
    that the output hasn't changed unexpectedly.
    """

    # This data was computed using commit 02ae86f
    known_data = np.array(
        [
            3.28369302e-09,
            2.60964897e-09,
            1.63323760e-08,
            5.46132025e-08,
            6.00826811e-09,
            2.67058826e-09,
            1.59698117e-08,
            2.21972147e-07,
            2.99597312e-08,
            8.22291746e-09,
            3.31645632e-09,
            2.41456161e-08,
            2.90209790e-08,
            3.38525845e-09,
            3.01619909e-09,
            4.75476553e-08,
            1.56174148e-07,
            1.68614602e-08,
            4.23078330e-09,
            6.64596968e-09,
            8.50022455e-08,
            1.21052107e-08,
            4.01329057e-09,
            6.95904457e-09,
            3.20000739e-08,
        ]
    )
    known_config = {
        "sim_params": {
            "N": 2**19,
            "dx": 3.1e-10,
            "z_detector": 1.77,
            "detector_size": 2.5e-5,
            "detector_pixel_size_x": 1e-6,
            "detector_pixel_size_y": 1.0,
            "chunk_size": 256 * 1024 * 1024 // 16,
        },
        "use_disk_vector": False,
        "save_final_u_vectors": False,
        "multisource": {
            "type": "points",
            "energy_range": [46000.0, 47000.0],
            "x_range": [-1e-6, 1e-6],
            "z": 0.0,
            "nr_source_points": 1,
            "seed": 1,
        },
        "elements": [
            {
                "type": "grating",
                "pitch": 4.2e-6,
                "dc": [0.5, 0.5],
                "z_start": 0.1,
                "thickness": 140e-6,
                "nr_steps": 4,
                "x_positions": [0.0],
                "substrate_thickness": (370 - 140) * 1e-6,
                "mat_a": ["Si", 2.34],
                "mat_b": ["Au", 19.32],
                "mat_substrate": ["Si", 2.34],
            },
            {
                "type": "grating",
                "pitch": 4.2e-6,
                "dc": [0.5, 0.5],
                "z_start": 0.918,
                "thickness": 59e-6,
                "nr_steps": 4,
                "x_positions": [0.0],
                "substrate_thickness": (200 - 59) * 1e-6,
                "mat_a": ["Si", 2.34],
                "mat_b": None,
                "mat_substrate": ["Si", 2.34],
            },
            {
                "type": "grating",
                "pitch": 4.2e-6,
                "dc": [0.5, 0.5],
                "z_start": 1.736,
                "thickness": 154e-6,
                "nr_steps": 4,
                "x_positions": [4.2e-6 * 0.25],
                "substrate_thickness": (370 - 154) * 1e-6,
                "mat_a": ["Si", 2.34],
                "mat_b": ["Au", 19.32],
                "mat_substrate": ["Si", 2.34],
            },
        ],
    }

    def test(self) -> None:
        sim_path = multisim.setup_simulation(self.known_config, self.path, self.path)
        multisim.run_single_simulation(sim_path, 0, sim_path)
        wavefronts = util.load_wavefronts_filtered(sim_path)
        wf = wavefronts[0][0][0]

        self.assertTrue(np.allclose(wf, self.known_data, rtol=0, atol=1e-14))


if __name__ == "__main__":
    unittest.main()
