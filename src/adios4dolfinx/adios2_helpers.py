from mpi4py import MPI
import pathlib
import numpy as np
import numpy.typing as npt
import adios2
from .utils import compute_local_range
from typing import Tuple
import dolfinx.graph
import dolfinx.cpp.graph

"""
Helpers reading/writing data with ADIOS2
"""

__all__ = ["read_dofmap", "read_array", "read_dofmap_new", "read_cell_perms"]


def read_cell_perms(comm: MPI.Intracomm, filename: pathlib.Path,
                    variable: str, num_cells_global: np.int64,
                    engine: str) -> npt.NDArray[np.uint32]:
    """
    Read dofmap with given communicator,
    split in continuous chunks based on number of cells in the mesh (global).

    .. note::
        No MPI communication is done during this call
    """
    local_cell_range = compute_local_range(comm, num_cells_global)

    # Open ADIOS engine
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("CellPermReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    # Get dofmap offsets from file
    for i in range(infile.Steps()):
        infile.BeginStep()
        if variable in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if variable not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at '{variable}'")
    perm_var = io.InquireVariable(variable)
    shape = perm_var.Shape()
    assert len(shape) == 1
    # As the offsets are one longer than the number of cells, we need to read in with an overlap
    perm_var.SetSelection([[local_cell_range[0]], [local_cell_range[1]-local_cell_range[0]]])
    in_perm = np.empty(local_cell_range[1]-local_cell_range[0], dtype=perm_var.Type().strip("_t"))
    infile.Get(perm_var, in_perm, adios2.Mode.Sync)
    infile.EndStep()
    adios.RemoveIO("CellPermReader")
    return in_perm


def read_dofmap_new(comm: MPI.Intracomm, filename: pathlib.Path,
                    dofmap: str, dofmap_offsets: str, num_cells_global: np.int64,
                    engine: str) -> dolfinx.cpp.graph.AdjacencyList_int64:
    """
    Read dofmap with given communicator,
    split in continuous chunks based on number of cells in the mesh (global).

    .. note::
        No MPI communication is done during this call
    """
    local_cell_range = compute_local_range(comm, num_cells_global)

    # Open ADIOS engine
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("DofmapReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    # Get dofmap offsets from file
    for i in range(infile.Steps()):
        infile.BeginStep()
        if dofmap_offsets in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if dofmap_offsets not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at '{dofmap_offsets}'")
    d_offsets = io.InquireVariable(dofmap_offsets)
    shape = d_offsets.Shape()
    assert len(shape) == 1
    # As the offsets are one longer than the number of cells, we need to read in with an overlap
    d_offsets.SetSelection([[local_cell_range[0]], [local_cell_range[1]+1-local_cell_range[0]]])
    in_offsets = np.empty(local_cell_range[1]+1-local_cell_range[0], dtype=d_offsets.Type().strip("_t"))
    infile.Get(d_offsets, in_offsets, adios2.Mode.Sync)
    # Get the relevant part of the dofmap
    if dofmap not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at {dofmap}")
    cell_dofs = io.InquireVariable(dofmap)
    cell_dofs.SetSelection([[in_offsets[0]], [in_offsets[-1]-in_offsets[0]]])
    in_dofmap = np.empty(in_offsets[-1]-in_offsets[0], dtype=cell_dofs.Type().strip("_t"))
    infile.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)

    in_dofmap = in_dofmap.astype(np.int64)
    in_offsets -= in_offsets[0]

    infile.EndStep()
    adios.RemoveIO("DofmapReader")
    return dolfinx.graph.create_adjacencylist(in_dofmap, in_offsets.astype(np.int32))


def read_dofmap(comm: MPI.Intracomm, filename: pathlib.Path,
                dofmap: str, dofmap_offsets: str, num_cells_global: np.int64,
                engine: str,
                cells: npt.NDArray[np.int64],
                dof_pos: npt.NDArray[np.int32]) -> npt.NDArray[np.int64]:
    """
    Read dofmap with given communicator, split in continuous chunks based on number of
    cells in the mesh (global).

    .. note::
        No MPI communication is done during this call
    """
    local_cell_range = compute_local_range(comm, num_cells_global)

    # Open ADIOS engine
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("DofmapReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    for i in range(infile.Steps()):
        infile.BeginStep()
        if dofmap_offsets in io.AvailableVariables().keys():
            break
        infile.EndStep()

    d_offsets = io.InquireVariable(dofmap_offsets)
    shape = d_offsets.Shape()
    assert len(shape) == 1
    # As the offsets are one longer than the number of cells, we need to read in with an overlap
    d_offsets.SetSelection([[local_cell_range[0]], [local_cell_range[1]+1-local_cell_range[0]]])
    in_offsets = np.empty(local_cell_range[1]+1-local_cell_range[0], dtype=d_offsets.Type().strip("_t"))
    infile.Get(d_offsets, in_offsets, adios2.Mode.Sync)
    # Get the relevant part of the dofmap
    if dofmap not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at {dofmap}")
    cell_dofs = io.InquireVariable(dofmap)
    cell_dofs.SetSelection([[in_offsets[0]], [in_offsets[-1]-in_offsets[0]]])
    in_dofmap = np.empty(in_offsets[-1]-in_offsets[0], dtype=cell_dofs.Type().strip("_t"))
    infile.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)

    in_dofmap = in_dofmap.astype(np.int64)

    # Extract dofmap data
    global_dofs = np.zeros_like(cells, dtype=np.int64)
    for i, (cell, pos) in enumerate(zip(cells, dof_pos.astype(np.int64))):
        input_cell_pos = cell-local_cell_range[0]
        read_pos = np.int32(in_offsets[input_cell_pos] + pos - in_offsets[0])
        global_dofs[i] = in_dofmap[read_pos]
        del input_cell_pos, read_pos
    infile.EndStep()
    adios.RemoveIO("DofmapReader")
    return global_dofs


def read_array(filename: pathlib.Path, path: str, engine: str,
               comm: MPI.Intracomm) -> Tuple[npt.NDArray[np.float64], int]:

    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("ArrayReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    for i in range(infile.Steps()):
        infile.BeginStep()
        if path in io.AvailableVariables().keys():
            break
        infile.EndStep()
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