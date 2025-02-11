<!-- Copyright (c) 2024, ETH Zurich -->

# RAVE SIM: Really big/fast wAVE SIMulation

Framework as published in Optics Express DOI: https://doi.org/10.1364/OE.543500

Framework contains code for Pascal Sommer's master thesis as well as Alexandre Vieira Pereira's and Simon Spindler's PhD theses.


The Python environment is shared across the entire codebase. The `requirements.txt` file in the root directory therefore contains the requirements for all the subprojects.

```bash
conda activate myenv
pip install -r requirements.txt
```

Example notebooks such as the notebooks to generate the results from the publication are given in the folder "notebooks".

## Directory Structure

* `big-fourier` contains a rust implementation for out-of-core fourier transforms, along with a python wrapper.
* `big-wave` implements an out-of-core wave simulation using the big fourier library.
* `fast-wave` uses the GPU to run simulations as fast as possible.
* `nist_lookup` is a third party library to look up material properties.
