# Adding a custom backend

`adios4dolfinx` is designed to be backend-agnostic, meaning you can implement custom readers and writers for different file formats by adhering to a specific protocol.

## The IOBackend Protocol

Any backend must implement the `IOBackend` protocol defined in `src/adios4dolfinx/backends/__init__.py`. This protocol ensures that the backend provides all necessary methods for reading and writing meshes, functions, and attributes.

To use a custom backend, you simply pass the python module name (as a string) to the `backend` argument of any `adios4dolfinx` function. The library will attempt to import the module and use it as the backend.

## Required Data Structures

Your backend will interact with several data classes defined in `src/adios4dolfinx/structures.py`. You should import these to type-hint your implementation correctly:

* **`MeshData`**: Contains local geometry, topology, and partitioning information for writing meshes.
* **`ReadMeshData`**: A container for returning mesh data (cells, geometry, etc.) when reading.
* **`FunctionData`**: Contains function values, dofmaps, and permutation info for writing functions.
* **`MeshTagsData`**: Contains indices, values, and metadata for mesh tags.

## Implementation Checklist

Your backend module must implement the functions listed below. Note that `comm` is always an `MPI.Intracomm` and `filename` is a `pathlib.Path` or `str`.

### General Configuration

* `get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]`
    * Returns a dictionary of default arguments (e.g., engine type) for your backend.

### Attribute IO

* `write_attributes(filename, comm, name, attributes, backend_args)`
    * Writes a dictionary of attributes (key-value pairs) to the file under the specified `name` (group).
* `read_attributes(filename, comm, name, backend_args) -> dict`
    * Reads and returns attributes associated with `name`.

### Mesh IO

* `write_mesh(filename, comm, mesh: MeshData, backend_args, mode, time)`
    * Writes mesh geometry, topology, and optionally partitioning data.
    * Must handle `FileMode.write` (new file) and `FileMode.append`.
* `read_mesh_data(filename, comm, time, read_from_partition, backend_args) -> ReadMeshData`
    * Reads mesh geometry and topology at a specific `time`.
    * If `read_from_partition` is True, it should read pre-calculated partitioning data to avoid re-partitioning.

### MeshTags IO

* `write_meshtags(filename, comm, data: MeshTagsData, backend_args)`
    * Writes mesh tag indices and values.
* `read_meshtags_data(filename, comm, name, backend_args) -> MeshTagsData`
    * Reads mesh tags identified by `name`.

### Function IO

* `write_function(filename, comm, u: FunctionData, time, mode, backend_args)`
    * Writes function values, global dofmaps, and cell permutations.
* `read_dofmap(filename, comm, name, backend_args) -> dolfinx.graph.AdjacencyList`
    * Reads the dofmap (connectivity) for the function `name`.
* `read_dofs(filename, comm, name, time, backend_args) -> tuple[np.ndarray, int]`
    * Reads the local chunk of function values for a specific `time`.
    * Returns the array of values and the global starting index of that chunk.
* `read_cell_perms(comm, filename, backend_args) -> np.ndarray`
    * Reads cell permutation data used to map input cells to the current mesh.
* `read_timestamps(filename, comm, function_name, backend_args) -> np.ndarray`
    * Returns all available time-steps for a given function.

### Legacy Support (Optional but defined in protocol)

* `read_legacy_mesh(filename, comm, group) -> tuple`
    * Reads mesh data from legacy DOLFIN HDF5/XDMF formats.
* `read_hdf5_array(comm, filename, group, backend_args) -> tuple`
    * Reads a raw array from an HDF5-like structure (used for legacy vector reading).

### Snapshots

* `snapshot_checkpoint(filename, mode, u: dolfinx.fem.Function, backend_args)`
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