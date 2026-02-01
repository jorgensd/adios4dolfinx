# Adding a custom backend

{py:mod}`adios4dolfinx` is designed to be backend-agnostic, meaning you can implement custom readers and writers for different file formats by adhering to a specific protocol.

## The IOBackend Protocol

Any backend must implement the {py:class}`IOBackend<adios4dolfinx.backends.IOBackend>` protocol defined in {py:mod}`adios4dolfinx.backends`. This protocol ensures that the backend provides all necessary methods for reading and writing meshes, functions, and attributes.

To use a custom backend, you simply pass the python module name (as a string) to the `backend` argument of any {py:mod}`adios4dolfinx` function.
The library will attempt to import the module and use it as the backend.

## Required Data Structures

Your backend will interact with several data classes defined in {py:mod}`adios4dolfinx.structures`.
You should import these to type-hint your implementation correctly:

* {py:class}`MeshData<adios4dolfinx.structures.MeshData>`: Contains local geometry, topology, and partitioning information for writing {py:class}`meshes<dolfinx.mesh.Mesh>`.
* {py:class}`ReadMeshData<adios4dolfinx.structures.ReadMeshData>`: A container for returning mesh data (cells, geometry, etc.) when reading.
* {py:class}`FunctionData<adios4dolfinx.structures.FunctionData>`: Contains function values, dofmaps, and permutation info for writing functions.
* {py:class}`MeshTagsData<adios4dolfinx.structures.MeshTagsData>`: Contains indices, values, and metadata for mesh tags.

## Implementation Checklist

Your backend module must implement the functions listed below. Note that `comm` is always an {py:class}`MPI.Intracomm<mpi4py.MPI.Intracomm>` and `filename` is a
{py:class}`pathlib.Path` or {py:class}`str`.

### General Configuration

* {py:func}`~adios4dolfinx.backends.IOBackend.get_default_backend_args`
    * Returns a dictionary of default arguments (e.g., engine type) for your backend.

### Attribute IO

* {py:func}`~adios4dolfinx.backends.IOBackend.write_attributes`
    * Writes a dictionary of attributes (key-value pairs) to the file under the specified `name` (group).
* {py:func}`~adios4dolfinx.backends.IOBackend.read_attributes`
    * Reads and returns attributes associated with `name`.

### Mesh IO

* {py:func}`~adios4dolfinx.backends.IOBackend.write_mesh`
    * Writes mesh geometry, topology, and optionally partitioning data.
    * Must handle {py:class}`FileMode.write` (new file) and {py:class}`FileMode.append`.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_mesh_data`
    * Reads mesh geometry and topology at a specific `time`.
    * If `read_from_partition` is True, it should read pre-calculated partitioning data to avoid re-partitioning.

### MeshTags IO

* {py:func}`~adios4dolfinx.backends.IOBackend.write_meshtags`
    * Writes mesh tag indices and values.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_meshtags_data`
    * Reads mesh tags identified by `name`.

### Function IO

* {py:func}`~adios4dolfinx.backends.IOBackend.write_function`
    * Writes function values, global dofmaps, and cell permutations.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_dofmap`
    * Reads the dofmap (connectivity) for the function `name`.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_dofs`
    * Reads the local chunk of function values for a specific `time`.
    * Returns the array of values and the global starting index of that chunk.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_cell_perms`
    * Reads cell permutation data used to map input cells to the current mesh.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_timestamps`
    * Returns all available time-steps for a given function.

### Legacy Support (Optional but defined in protocol)

* {py:func}`~adios4dolfinx.backends.IOBackend.read_legacy_mesh`
    * Reads mesh data from legacy DOLFIN HDF5/XDMF formats.
* {py:func}`~adios4dolfinx.backends.IOBackend.read_hdf5_array`
    * Reads a raw array from an HDF5-like structure (used for legacy vector reading).

### Snapshots

* {py:func}`~adios4dolfinx.backends.IOBackend.snapshot_checkpoint`
    * Handles lightweight N-to-N checkpointing where data is saved exactly as distributed in memory without global reordering.

## Example Skeleton

```python
from typing import Any
from pathlib import Path
from mpi4py import MPI
import numpy as np
import dolfinx
from adios4dolfinx.structures import MeshData, FunctionData, MeshTagsData, ReadMeshData
from adios4dolfinx.backends import FileMode

def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    return arguments or {}

def write_mesh(filename: Path | str, comm: MPI.Intracomm, mesh: MeshData,
               backend_args: dict[str, Any] | None, mode: FileMode, time: float):
    # Implementation here
    pass

# ... Implement all other methods defined in IOBackend ...
```