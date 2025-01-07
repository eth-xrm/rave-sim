<!-- Copyright (c) 2024, ETH Zurich -->

# Big Wave

The simulation framework for out-of-core wave simulations.

Make sure to first build the library in the `../big-fourier` directory.

```bash
# unittests
python test.py

# type checking
mypy *.py
```

## Code overview

* *vector.py* contains a `Vector` abstraction that allows simulations to either be run in RAM or with the vector stored on disk. The rest of the code doesn't have to care about this distinction, the abstraction takes care of it.
* *propagation.py* provides low-level utilities for the optics computations.
* *optical_element.py* represents anything that can be placed in between the source and the detector.
* *source.py* contains different kinds of ray sources.
* *wavesim.py* deals with running the simulation for one source with one frequency.
* *multisim.py* adds functionality for multiple source points and frequencies. It also generates the output directory structure.
* *history.py* is a container for recording the wavefront at various points throughout the simulation such that a 2d z/x image can be generated where interference patterns can be observed.
* *test.py* contains unit tests for the `big-wave` framework as well as some for the `big-fourier` library.

## Simulation directory structure

Simulations are stored according to this layout:

```
simulations
├── 2023                            # year
│   └── 07                          # month
│       └── 20230724_123033483230   # time of `setup_simulation`
│           ├── 00000000            # first source point
│           │   ├── detected.npy
│           │   ├── subconfig.yaml
│           │   └── u_0000.npy
│           ├── 00000001            # second source point
│           │   ├── detected.npy
│           │   ├── subconfig.yaml
│           │   └── u_0000.npy
│           ├── 00000002
│           │   ├── detected.npy
│           │   ├── subconfig.yaml
│           │   └── u_0000.npy
│           ├── ...
│           ├── computed.yaml
│           ├── config.yaml
│           └── my_sample.npy
...
```

* *config.yaml* is the input configuration that was passed to `multisim.setup_simulation`. If `save_and_exit` elements were used, the configuration is modified to simulate only up to that point, anything beyond that will be cut off. Otherwise the configuration is equivalent, except for some file path adjustments.
* *computed.yaml* contains the `cutoff_angles`, `max_x`, `energy_range`, and `source_points`. This is used for other simulations that continue based on the output of this simulation, as well as for loading detector outputs.
    * `cutoff_angles` is a list of length `nr_elements+1`, where the `0`-th entry is valid before the first optical element, and the `i`-th entry is valid from the start of the `i-1`-th element up to the start of the `i`-th element or the detector.
    * On a continued simulation, `source_points` will also contain source x coordinates, even though the sources in this simulation are full vectors instead of point sources. The coordinates there are taken from the base simulation from where the simulation is continued.
    * `max_x` is the maximal absolute x coordinate on `z_detector` that is reachable by following exactly the cutoff angles. This is used for computing subsequent `cutoff_angles` lists in case a simulation is continued from this one.
* *my_sample.npy* is a 2d numpy array containing a sample that is inserted into the simulation. Optional.
* *subconfig.yaml* is generated for every source point. The call to `multisim.run_single_simulation` uses this to figure out the parameters for the individual source point. It also stores the precomputed table of delta and beta values for all materials used in the simulation, using the energy of this source point.
* *u_0000.npy* appears if the `save_final_u_vectors` option was activated and contains the full `u` vector at the detector plane. Alternatively, if a `save_and_exit` element was added to the setup, the `u` vector from that z coordinate will be saved. Saving an `u` vector is needed if we want other simulations to continue from there on. Otherwise optional. If phase stepping is active for the simulation, multiple `u` vectors will be saved, numbered sequentially.
* *detected.npy* is the actual detector outputs as a 2d numpy array. The first dimension corresponds to phase stepping. If no phase stepping was used, this has length 1. The second dimensions corresponds to detector pixels.

## Running on Euler

To run this code on Euler, we need to activate a more recent Python version, and set up the appropriate environment. Apparently, `venv` is preferred over `conda` on Euler.

Setting up the environment:

```bash
ssh euler

# install a rust compiler (has to be done only once)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install 1.72 # or whatever newer version you want to use. rust is fully forward compatible. You might have to open a new shell after installing rust.

# activate a recent Python (has to be done every time after activating a new shell)
env2lmod && module load gcc/8.2.0 python/3.11.2

# setup the venv and install Python dependencies
cd rave-sim
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

# Compile the big-fourier library. Note that venv has to be active while running this, because this step installs
# the python wrapper library bfpy into the currently active python environment. This compilation has to be done
# only once, and afterwards only after changes in the rust code.
cd big-fourier
maturin develop --release
```
