import pathlib
from typing import Tuple

import adios2
import dolfinx.cpp.graph
import dolfinx.graph
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .utils import compute_local_range, valid_function_types

"""
Helpers reading/writing data with ADIOS2
"""

__all__ = ["read_array", "read_dofmap", "read_cell_perms", "adios_to_numpy_dtype"]

adios_to_numpy_dtype = {"float": np.float32, "double": np.float64,
                        "float complex": np.complex64, "double complex": np.complex128,
                        "uint32_t": np.uint32}


def read_cell_perms(
    adios: adios2.adios2.ADIOS,
    comm: MPI.Intracomm,
    filename: pathlib.Path,
    variable: str,
    num_cells_global: np.int64,
    engine: str,
) -> npt.NDArray[np.uint32]:
    """
    Read cell permutation from file with given communicator,
    Split in continuous chunks based on number of cells in the mesh (global).

    Args:
        adios: The ADIOS instance
        comm: The MPI communicator used to read the data
        filename: Path to input file
        variable: Name of cell-permutation variable
        num_cells_global: Number of cells in the mesh (global)
        engine: Type of ADIOS engine to use for reading data

    Returns:
        Cell-permutations local to the process

    .. note::
        No MPI communication is done during this call
    """

    # Open ADIOS engine
    io_name = f"{variable=}_reader"
    io = adios.DeclareIO(io_name)
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    # Find step that has cell permutation
    for i in range(infile.Steps()):
        infile.BeginStep()
        if variable in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if variable not in io.AvailableVariables().keys():
        raise KeyError(f"Variable {variable} not found in '{filename}'")

    # Get variable and get global shape
    perm_var = io.InquireVariable(variable)
    shape = perm_var.Shape()
    assert len(shape) == 1

    # Get local selection
    local_cell_range = compute_local_range(comm, num_cells_global)
    perm_var.SetSelection(
        [[local_cell_range[0]], [local_cell_range[1] - local_cell_range[0]]]
    )
    in_perm = np.empty(
        local_cell_range[1] - local_cell_range[0], dtype=adios_to_numpy_dtype[perm_var.Type()]
    )
    infile.Get(perm_var, in_perm, adios2.Mode.Sync)
    infile.EndStep()

    # Close IO and remove io
    adios.RemoveIO(io_name)
    return in_perm


def read_dofmap(
    adios: adios2.adios2.ADIOS,
    comm: MPI.Intracomm,
    filename: pathlib.Path,
    dofmap: str,
    dofmap_offsets: str,
    num_cells_global: np.int64,
    engine: str,
) -> dolfinx.cpp.graph.AdjacencyList_int64:
    """
    Read dofmap with given communicator,
    split in continuous chunks based on number of cells in the mesh (global).

    Args:
        adios: The ADIOS instance
        comm: The MPI communicator used to read the data
        filename: Path to input file
        dofmap: Name of variable containing dofmap
        dofmap_offsets: Name of variable containing dofmap_offsets
        num_cells_global: Number of cells in the mesh (global)
        engine: Type of ADIOS engine to use for reading data

    Returns:
        The local part of dofmap from input dofs

    .. note::
        No MPI communication is done during this call
    """
    local_cell_range = compute_local_range(comm, num_cells_global)

    # Open ADIOS engine
    io_name = f"{dofmap=}_reader"
    io = adios.DeclareIO(io_name)
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    # First find step with dofmap offsets, to be able to read in a full row of the dofmap
    for i in range(infile.Steps()):
        infile.BeginStep()
        if dofmap_offsets in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if dofmap_offsets not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at '{dofmap_offsets}'")

    # Get global shape of dofmap-offset, and read in data with an overlap
    d_offsets = io.InquireVariable(dofmap_offsets)
    shape = d_offsets.Shape()
    assert len(shape) == 1
    # As the offsets are one longer than the number of cells, we need to read in with an overlap
    d_offsets.SetSelection(
        [[local_cell_range[0]], [local_cell_range[1] + 1 - local_cell_range[0]]]
    )
    in_offsets = np.empty(
        local_cell_range[1] + 1 - local_cell_range[0],
        dtype=d_offsets.Type().strip("_t"),
    )
    infile.Get(d_offsets, in_offsets, adios2.Mode.Sync)

    # Assuming dofmap is saved in stame step
    # Get the relevant part of the dofmap
    if dofmap not in io.AvailableVariables().keys():
        raise KeyError(f"Dof offsets not found at {dofmap}")
    cell_dofs = io.InquireVariable(dofmap)
    cell_dofs.SetSelection([[in_offsets[0]], [in_offsets[-1] - in_offsets[0]]])
    in_dofmap = np.empty(
        in_offsets[-1] - in_offsets[0], dtype=cell_dofs.Type().strip("_t")
    )
    infile.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)

    in_dofmap = in_dofmap.astype(np.int64)
    in_offsets -= in_offsets[0]

    infile.EndStep()
    infile.Close()
    adios.RemoveIO(io_name)
    # Return local dofmap
    return dolfinx.graph.adjacencylist(in_dofmap, in_offsets.astype(np.int32))


def read_array(
            adios: adios2.adios2.ADIOS,
            filename: pathlib.Path, array_name: str, engine: str, comm: MPI.Intracomm
) -> Tuple[npt.NDArray[valid_function_types], int]:
    """
    Read an array from file, return the global starting position of the local array

    Args:
        adios: The ADIOS instance
        filename: Path to file to read array from
        array_name: Name of array in file
        engine: Name of engine to use to read file
        comm: MPI communicator used for reading the data
    Returns:
        Local part of array and its global starting position
    """
    io = adios.DeclareIO("ArrayReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    for i in range(infile.Steps()):
        infile.BeginStep()
        if array_name in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if array_name not in io.AvailableVariables().keys():
        raise KeyError(f"No array found at {array_name}")

    arr = io.InquireVariable(array_name)
    arr_shape = arr.Shape()
    assert len(arr_shape) >= 1  # TODO: Should we always pick the first element?
    arr_range = compute_local_range(comm, arr_shape[0])

    if len(arr_shape) == 1:
        arr.SetSelection([[arr_range[0]], [arr_range[1] - arr_range[0]]])
        vals = np.empty(arr_range[1] - arr_range[0], dtype=adios_to_numpy_dtype[arr.Type()])
    else:
        arr.SetSelection([[arr_range[0], 0], [arr_range[1] - arr_range[0], arr_shape[1]]])
        vals = np.empty((arr_range[1] - arr_range[0], arr_shape[1]), dtype=adios_to_numpy_dtype[arr.Type()])

    infile.Get(arr, vals, adios2.Mode.Sync)
    infile.EndStep()
    infile.Close()
    adios.RemoveIO("ArrayReader")
    return vals, arr_range[0]
