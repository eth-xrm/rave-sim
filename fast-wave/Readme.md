<!-- Copyright (c) 2024, ETH Zurich -->

# Fast Wave

Sacrifice the ability to compute arbitrarily large simulations for perforance gains.

This program is separated into various CMake targets:

* `fwcuda`: located in the `fwcuda/` directory, this is the part that directly calls CUDA kernels. This is the only part of the main that is compiled with nvcc. We separate this out so that the rest of the code can use more C++ features and libraries without being limited by nvcc bugs.
* `fastwave_lib`: The library that implements the main application logic of running the simulation according to the given configuration
* `fastwave`: The executable that the end user calls from the command line to use `fastwave_lib`.
* `fwtest`: The unit testing executable. This part is also compiled with nvcc so that we can directly call kernels from the tests.

This code is sparsely documented. Much of the functionality is a one-to-one translation from the Python version
in `big-wave`, so check there if anything is unclear.

## Usage

Once the `fastwave` program has been compiled (see build instructions below), you can run simulations with a similar workflow to what is used for `big-wave`.

Use `sim_dir = multisim.setup_simulation(...)` from `big-wave` to create a simulation directory. Then run the simulation like that:

```bash
./fastwave /path/to/sim/dir -s 123
```

where 123 is the source index (0-based). To run multiple source points just use a for loop.

To run on a specific GPU, pass the 0-based gpu index as an env variable:

```bash
# run on the second gpu
CUDA_VISIBLE_DEVICES=1 ./fastwave /path/to/sim/dir -s 123
```

The history can be saved by using the optional argument `--history_dz 0.025`.

Remember to recompile if the c++ or cuda code has changed.

## Build Instructions

The project is built in the normal CMake way. On Euler you'll have to load a recent cuda version first.

```bash
# this first line is only needed on euler
env2lmod && module load gcc/9.3.0 cuda/12.1.1

cd fast-wave

mkdir build-Release
cd build-Release

# On Hari add the -DCUDAToolkit_INCLUDE_DIR option. This might not be
# required on other machines.
cmake -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_INCLUDE_DIR=/usr/include ..

# The above step doesn't compile the program yet, it's just the CMake configure step. This next line
# is the build step. If the code has changed, only this next line has to be re-run.
cmake --build . --target fastwave

# Run the tests. This requires c++20 and therefore does not work
# on euler unless you want to compile your own non-ancient gcc.
cmake --build . --target fwtest
ctest
```

The cuda toolkit has to be installed for this to work. Other dependencies are fetched from the internet during configure time (during the first of the cmake commands above).

If you run into git submodule errors while running CMake, it might be that you don't have an ssh key for github set up on that machine. One of the dependencies uses git submodules with an ssh address instead of an https address, which breaks in that case. You can fix this issue by adding these lines to your `~/.gitconfig`:

```
[url "https://github.com/"]
    insteadOf = git@github.com:
```

This project does not depend on big-fourier or big-wave, you don't have to build them to use this.

## todo:

does it make sense to keep the data in a chunked format? Govindaraju et al.: Our hierarchical FFT minimizes the number of memory accesses by combining transpose operations with the FFT computation.
