# ADIOS4DOLFINx - A framework for checkpointing in DOLFINx

![MIT](https://img.shields.io/github/license/jorgensd/adios4dolfinx)
[![status](https://joss.theoj.org/papers/7866cb142db8a803e32d79a109573d25/status.svg)](https://joss.theoj.org/papers/7866cb142db8a803e32d79a109573d25)


ADIOS4DOLFINx is an extension for [DOLFINx](https://github.com/FEniCS/dolfinx/) to checkpoint meshes, meshtags and functions using [ADIOS 2](https://adios2.readthedocs.io/en/latest/).

The code uses the ADIOS2 Python-wrappers to write DOLFINx objects to file, supporting N-to-M (_recoverable_) and N-to-N (_snapshot_) checkpointing.
See: [Checkpointing in DOLFINx - FEniCS 23](https://jsdokken.com/checkpointing-presentation/#/) or the examples in the [Documentation](https://jsdokken.com/adios4dolfinx/) for more information.

For scalability, the code uses [MPI Neighbourhood collectives](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node200.htm) for communication across processes.

## Installation
Compatibility with DOLFINx:
- ADIOS4DOLFINx v0.7.3 is compatible with DOLFINx v0.7.x
- ADIOS4DOLFINx v0.8.x is compatible with the main branch of DOLFINx

### Docker

ADIOS2 is installed in the official DOLFINx containers, and thus there are no additional dependencies required to install `adios4dolfinx`
on top of DOLFINx in these images.

Create a Docker container, named for instance `dolfinx-checkpoint`.
Use the `nightly` tag to get the main branch of DOLFINx, or `stable` to get the lastest stable release
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --name=dolfinx-checkpoint ghcr.io/fenics/dolfinx/dolfinx:nightly
```
For the latest version compatible with nightly (with the ability to run the test suite), use
```bash
python3 -m pip install adios4dolfinx[test]@git+https://github.com/jorgensd/adios4dolfinx@main
```
If you are using the `stable` image, you can install `adios4dolfinx` from [PYPI](https://pypi.org/project/adios4dolfinx/) with
```bash
python3 -m pip install adios4dolfinx[test]
```

This docker container can be opened with
```bash
docker container start -i dolfinx-checkpoint
```
at a later instance

### Conda

> [!NOTE]  
> Conda supports the stable release of DOLFINx, and thus the appropriate version should be installed, see the section above for more details.

Following is a minimal recipe of how to install adios4dolfinx, given that you have conda installed on your system.
```bash
conda create -n dolfinx-checkpoint python=3.10
conda activate dolfinx-checkpoint
conda install -c conda-forge fenics-dolfinx pip adios2
python3 -m pip install adios4dolfinx[test]@git+https://github.com/jorgensd/adios4dolfinx@v0.7.3
```

> [!NOTE]
> To run the tests or demos associated with the code, install `ipyparallel` in your environment, for instance by calling
> ```bash
> python3 -m pip install adios4dolfinx[test]@git+https://github.com/jorgensd/adios4dolfinx@v0.7.3
> ```
## Functionality

### DOLFINx

- Reading and writing meshes, using `adios4dolfinx.read/write_mesh`
- Reading and writing meshtags associated to meshes `adios4dolfinx.read/write_meshtags`
- Reading checkpoints for any element (serial and parallel, arbitrary number of functions and timesteps per file). Use `adios4dolfinx.read/write_function`.
- Writing standalone function checkpoints relating to "original meshes", i.e. meshes read from `XDMFFile`. Use `adios4dolfinx.write_function_on_input_mesh` for this.
- Store mesh partitioning and re-read the mesh with this information, avoiding calling SCOTCH, Kahip or Parmetis.

> [!IMPORTANT]  
> For checkpoints written with `write_function` to be valid, you first have to store the mesh with `write_mesh` to the checkpoint file.

> [!IMPORTANT]  
> A checkpoint file supports multiple functions and multiple time steps, as long as the functions are associated with the same mesh

> [!IMPORTANT]  
> Only one mesh per file is allowed

## Example Usage
The repository contains many documented examples of usage, in the `docs`-folder, including
- [Reading and writing mesh checkpoints](./docs/writing_mesh_checkpoint.py)
- [Storing mesh partitioning data](./docs/partitioned_mesh.py)
- [Writing mesh-tags to a checkpoint](./docs/meshtags.py)
- [Reading and writing function checkpoints](./docs/writing_functions_checkpoint.py)
- [Checkpoint on input mesh](./docs/original_checkpoint.py)
Further examples can be found at [ADIOS4DOLFINx examples](https://jsdokken.com/adios4dolfinx/)

### Backwards compatibility

> [!WARNING]
> If you are using v0.7.2, you are adviced to upgrade to v0.7.3, as it contains som crucial fixes for openmpi.

### Legacy DOLFIN

Only checkpoints for `Lagrange` or `DG` functions are supported from legacy DOLFIN

- Reading meshes from the DOLFIN HDF5File-format
- Reading checkpoints from the DOLFIN HDF5File-format (one checkpoint per file only)
- Reading checkpoints from the DOLFIN XDMFFile-format (one checkpoint per file only, and only uses the `.h5` file)

See the [API](./docs/api) for more information.

## Long term plan

The long term plan is to get this library merged into DOLFINx (rewritten in C++ with appropriate Python-bindings).
