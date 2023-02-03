# ADIOS2Wrappers for DOLFINx

[![MIT](https://img.shields.io/github/license/jorgensd/adios4dolfinx)](LICENSE)
[Read Latest Documentation](https://jorgensd.github.io/adios4dolfinx/)

This is an experimental library for checkpoint with [DOLFINx](https://github.com/FEniCS/dolfinx/) using [ADIOS2](https://adios2.readthedocs.io/en/latest/).
ADIOS2 is installed in the official DOLFINx containers.
To get access to the Python interface of ADIOS2, you might have to extend your Python-path:
```bash
export PYTHONPATH=/usr/local/lib/python3/dist-packages:$PYTHONPATH
```

# Long term plan
The long term plan is to get this library merged into DOLFINx (rewritten in C++ with appropriate Python-bindings).
_________________

# Functionality 

## Reading Legacy mesh files

This library supports:

### Legacy DOLFIN
Only checkpoints for `Lagrange` or `DG` functions are supported from legacy DOLFIN
- Reading meshes from the DOLFIN HDF5File-format
- Reading checkpoints from the DOLFIN HDF5File-format (one checkpoint per file only)
- Reading checkpoints from the DOLFIN XDMFFile-format (one checkpoint per file only, and only uses the `.h5` file)

### DOLFINx
- Reading and writing meshes
- Reading checkpoints for Lagrange/DG elements (serial and parallel, one checkpoint per file)
- Reading checkpoints for edge based elements (serial only, one checkpoint per file)
_________________
