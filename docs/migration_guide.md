---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Migration Guide: Transitions to `io4dolfinx`


This guide outlines the necessary steps to transition your code from {py:mod}`adios4dolfinx` new API introduced in `io4dolfinx`.

## Major Changes

The library has undergone a major refactor to support multiple IO backends.
**{py:mod}`adios4dolfinx` now supports both [ADIOS2](https://adios2.readthedocs.io/en/latest/) and [h5py](https://docs.h5py.org/en/stable/) backends.**

This allows users to choose between the high-performance, adaptable ADIOS2 framework and the standard HDF5 format via {py:class}`h5py.File`, all using the same high-level API.
It also opens the door for new backends in the future.

### Key API Updates

1.  **Backend Agnosticism**: You can now switch between {py:class}`adios2.ADIOS` (default) and {py:class}`h5py.File` by passing a `backend` argument.
2.  **Engine Configuration**: The explicit `engine` argument (e.g., "BP4", "HDF5", see ADIOS2 [Engine Types](https://adios2.readthedocs.io/en/latest/engines/engines.html)) has been removed from function signatures. It is now passed via a dictionary `backend_args`. This is because different backends may have different engine options. For example {py:class}`adios2.ADIOS` supports "BP4" and "HDF5", while `h5py` does not use engines.


### Example Transition

#### Writing a Mesh

`````{tab-set}
````{tab-item} Old API
```python
import adios4dolfinx
adios4dolfinx.write_mesh("mesh.bp", mesh, engine="BP4")
```
````

````{tab-item} New API (ADIOS2 Backend)
```python
import io4dolfinx
io4dolfinx.write_mesh("mesh.bp", mesh, backend="adios2", backend_args={"engine": "BP4"})
```
````

````{tab-item} New API (h5py Backend)
```python
import io4dolfinx
io4dolfinx.write_mesh("mesh.bp", mesh, backend="h5py")
```
````

`````


#### Writing a Function

`````{tab-set}
````{tab-item} Old API
```python
import adios4dolfinx
adios4dolfinx.write_function("solution.bp", u, time=0.0, engine="BP4")
```
````
````{tab-item} New API (ADIOS2 Backend)
```python
import io4dolfinx
io4dolfinx.write_function("solution.bp", u, time=0.0, backend="adios2", backend_args={"engine": "BP4"})
```
````
````{tab-item} New API (h5py Backend)
```python
import io4dolfinx
io4dolfinx.write_function("solution.bp", u, time=0.0, backend="h5py")
```
````
`````


