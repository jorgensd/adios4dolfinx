# ADIOS2Wrappers for DOLFINx

[![MIT](https://img.shields.io/github/license/jorgensd/adios4dolfinx)](LICENSE)
[Read Latest Documentation](https://jsdokken.com/adios4dolfinx/)

This is an extension for [DOLFINx](https://github.com/FEniCS/dolfinx/) to checkpoint meshes, meshtags and functions using [ADIOS2](https://adios2.readthedocs.io/en/latest/).

The code uses the adios2 Python-wrappers to write DOLFINx objects to file, supporting N-to-M (*recoverable*) and N-to-N (*snapshot*) checkpointing.
See: [Checkpointing in DOLFINx - FEniCS 23](https://jsdokken.com/checkpointing-presentation/#/) for more information.

For scalability, the code uses [MPI Neighbourhood collectives](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node200.htm) for communication across processes.

## Installation

### Docker
ADIOS2 is installed in the official DOLFINx containers.
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --name=dolfinx-checkpoint ghcr.io/fenics/dolfinx/dolfinx:nightly
```

### Conda
To use with conda (DOLFINx release v0.7.0 works with v0.7.2 of ADIOS4DOLFINx)
```bash
conda create -n dolfinx-checkpoint python=3.10
conda activate dolfinx-checkpoint
conda install -c conda-forge fenics-dolfinx pip adios2
python3 -m pip install git+https://github.com/jorgensd/adios4dolfinx@v0.7.2
```

## Functionality 



### DOLFINx
- Reading and writing meshes, using `adios4dolfinx.read/write_mesh`
- Reading and writing meshtags associated to meshes `adios4dolfinx.read/write_meshtags`
- Reading checkpoints for any element (serial and parallel, arbitrary number of functions and timesteps per file). Use `adios4dolfinx.read/write_function`.

> [!IMPORTANT]  
> For a checkpoint to be valid, you first have to store the mesh with `write_mesh`, then use `write_function` to append to the checkpoint file.

> [!IMPORTANT]  
> A checkpoint file supports multiple functions and multiple time steps, as long as the functions are associated with the same mesh

> [!IMPORTANT]  
> Only one mesh per file is allowed


### Backwards compatibility
> [!WARNING]
> If you are using checkpoints written with `adios4dolfinx<0.7.2` please use the `legacy=True` flag for reading in the checkpoint with
> with any newer version


### Legacy DOLFIN
Only checkpoints for `Lagrange` or `DG` functions are supported from legacy DOLFIN
- Reading meshes from the DOLFIN HDF5File-format
- Reading checkpoints from the DOLFIN HDF5File-format (one checkpoint per file only)
- Reading checkpoints from the DOLFIN XDMFFile-format (one checkpoint per file only, and only uses the `.h5` file)

See the [API](./docs/api) for more information.

## Long term plan
The long term plan is to get this library merged into DOLFINx (rewritten in C++ with appropriate Python-bindings).
