from mpi4py import MPI
import pathlib
import numpy as np
import numpy.typing as npt
import adios2
from .utils import compute_local_range
from typing import Tuple


"""
Helpers reading/writing data with ADIOS2
"""

__all__ = ["read_dofmap", "read_array"]


def read_dofmap(comm: MPI.Comm, filename: pathlib.Path,
                dofmap: str, dofmap_offsets: str, num_cells_global: int,
                engine: str,
                cells: npt.NDArray[np.int64],
                dof_pos: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
    """
    Read dofmap with given communicator, split in continuous chunks based on number of
    cells in the mesh (global).

    NOTE: No MPI communication is done during this call
    """
    # Open ADIOS engine
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("DofmapReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    infile.BeginStep()

    local_cell_range = compute_local_range(comm, num_cells_global)

    # Get dofmap offsets from file
    if dofmap_offsets not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at '{dofmap_offsets}'")
    d_offsets = io.InquireVariable(dofmap_offsets)
    shape = d_offsets.Shape()
    assert len(shape) == 1
    # As the offsets are one longer than the number of cells, we need to read in with an overlap
    d_offsets.SetSelection([[local_cell_range[0]], [local_cell_range[1]+1-local_cell_range[0]]])
    dofmap_offsets = np.empty(local_cell_range[1]+1-local_cell_range[0], dtype=np.dtype(d_offsets.Type().strip("_t")))
    infile.Get(d_offsets, dofmap_offsets, adios2.Mode.Sync)
    # Get the relevant part of the dofmap
    if dofmap not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at {dofmap}")
    cell_dofs = io.InquireVariable(dofmap)
    cell_dofs.SetSelection([[dofmap_offsets[0]], [dofmap_offsets[-1]-dofmap_offsets[0]]])
    in_dofmap = np.empty(dofmap_offsets[-1]-dofmap_offsets[0], dtype=np.dtype(cell_dofs.Type().strip("_t")))
    infile.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)

    in_dofmap = in_dofmap.astype(np.int64)

    # Extract dofmap data
    global_dofs = np.zeros_like(cells, dtype=np.int64)
    for i, (cell, pos) in enumerate(zip(cells, dof_pos.astype(np.uint64))):
        input_cell_pos = cell-local_cell_range[0]
        read_pos = dofmap_offsets[input_cell_pos] + pos - dofmap_offsets[0]
        #print(input_cell_pos, i, read_pos)
        global_dofs[i] = in_dofmap[read_pos]
    #print("Second", global_dofs, in_dofmap)

    infile.EndStep()
    adios.RemoveIO("DofmapReader")
    return global_dofs


def read_array(filename: pathlib.Path, path: str, engine: str,
               comm: MPI.Comm) -> Tuple[npt.NDArray[np.float64], int]:

    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("ArrayReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)
    infile.BeginStep()

    if path not in io.AvailableVariables().keys():
        raise KeyError(f"No array found at {path}")
    arr = io.InquireVariable(path)
    arr_shape = arr.Shape()
    assert len(arr_shape) == 1
    arr_range = compute_local_range(comm, arr_shape[0])
    assert (arr_range[0] == arr_range[0])
    arr.SetSelection([[arr_range[0]], [arr_range[1]-arr_range[0]]])
    vals = np.empty(arr_range[1]-arr_range[0], dtype=np.dtype(arr.Type().strip("_t")))
    infile.Get(arr, vals, adios2.Mode.Sync)
    infile.EndStep()
    adios.RemoveIO("ArrayReader")
    return vals, arr_range[0]
