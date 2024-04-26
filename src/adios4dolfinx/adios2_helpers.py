from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Union

from mpi4py import MPI

import adios2
import dolfinx.cpp.graph
import dolfinx.graph
import numpy as np
import numpy.typing as npt

from .utils import compute_local_range, valid_function_types


def resolve_adios_scope(adios2):
    return adios2.bindings if hasattr(adios2, "bindings") else adios2


adios2 = resolve_adios_scope(adios2)

"""
Helpers reading/writing data with ADIOS2
"""

__all__ = ["read_array", "read_adjacency_list", "read_cell_perms", "adios_to_numpy_dtype"]

adios_to_numpy_dtype = {
    "float": np.float32,
    "double": np.float64,
    "float complex": np.complex64,
    "double complex": np.complex128,
    "uint32_t": np.uint32,
}


class AdiosFile(NamedTuple):
    io: adios2.IO
    file: adios2.Engine


@contextmanager
def ADIOSFile(
    adios: adios2.ADIOS,
    filename: Union[Path, str],
    engine: str,
    mode: adios2.Mode,
    io_name: str,
):
    io = adios.DeclareIO(io_name)
    io.SetEngine(engine)
    file = io.Open(str(filename), mode)
    try:
        yield AdiosFile(io=io, file=file)
    finally:
        file.Close()
        adios.RemoveIO(io_name)


def read_cell_perms(
    adios: adios2.ADIOS,
    comm: MPI.Intracomm,
    filename: Union[Path, str],
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

    with ADIOSFile(
        adios=adios,
        engine=engine,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name=io_name,
    ) as adios_file:
        # Find step that has cell permutation
        for i in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if variable in adios_file.io.AvailableVariables().keys():
                break
            adios_file.file.EndStep()
        if variable not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Variable {variable} not found in '{filename}'")

        # Get variable and get global shape
        perm_var = adios_file.io.InquireVariable(variable)
        shape = perm_var.Shape()
        assert len(shape) == 1

        # Get local selection
        local_cell_range = compute_local_range(comm, num_cells_global)
        perm_var.SetSelection([[local_cell_range[0]], [local_cell_range[1] - local_cell_range[0]]])
        in_perm = np.empty(
            local_cell_range[1] - local_cell_range[0],
            dtype=adios_to_numpy_dtype[perm_var.Type()],
        )
        adios_file.file.Get(perm_var, in_perm, adios2.Mode.Sync)
        adios_file.file.EndStep()

    return in_perm


def read_adjacency_list(
    adios: adios2.ADIOS,
    comm: MPI.Intracomm,
    filename: Union[Path, str],
    dofmap: str,
    dofmap_offsets: str,
    num_cells_global: np.int64,
    engine: str,
) -> Union[dolfinx.cpp.graph.AdjacencyList_int64, dolfinx.cpp.graph.AdjacencyList_int32]:
    """
    Read an adjacency-list from an ADIOS file with given communicator.
    The adjancency list is split in to a flat array (data) and its corresponding offset.

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

    with ADIOSFile(
        adios=adios,
        engine=engine,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name=io_name,
    ) as adios_file:
        # First find step with dofmap offsets, to be able to read
        # in a full row of the dofmap
        for i in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if dofmap_offsets in adios_file.io.AvailableVariables().keys():
                break
            adios_file.file.EndStep()
        if dofmap_offsets not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Dof offsets not found at '{dofmap_offsets}' in {filename}")

        # Get global shape of dofmap-offset, and read in data with an overlap
        d_offsets = adios_file.io.InquireVariable(dofmap_offsets)
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
        adios_file.file.Get(d_offsets, in_offsets, adios2.Mode.Sync)

        # Assuming dofmap is saved in stame step
        # Get the relevant part of the dofmap
        if dofmap not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Dof offsets not found at {dofmap} in {filename}")
        cell_dofs = adios_file.io.InquireVariable(dofmap)
        cell_dofs.SetSelection([[in_offsets[0]], [in_offsets[-1] - in_offsets[0]]])
        in_dofmap = np.empty(in_offsets[-1] - in_offsets[0], dtype=cell_dofs.Type().strip("_t"))
        adios_file.file.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)
        in_offsets -= in_offsets[0]

        adios_file.file.EndStep()

    # Return local dofmap
    return dolfinx.graph.adjacencylist(in_dofmap, in_offsets.astype(np.int32))


def read_array(
    adios: adios2.ADIOS,
    filename: Union[Path, str],
    array_name: str,
    engine: str,
    comm: MPI.Intracomm,
    time: float = 0.0,
    time_name: str = "",
    legacy: bool = False,
) -> tuple[npt.NDArray[valid_function_types], int]:
    """
    Read an array from file, return the global starting position of the local array

    Args:
        adios: The ADIOS instance
        filename: Path to file to read array from
        array_name: Name of array in file
        engine: Name of engine to use to read file
        comm: MPI communicator used for reading the data
        time_name: Name of time variable for modern checkpoints
        legacy: If True ignore time_name and read the first available step
    Returns:
        Local part of array and its global starting position
    """

    with ADIOSFile(
        adios=adios,
        engine=engine,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name="ArrayReader",
    ) as adios_file:
        # Get time-stamp from first available step
        if legacy:
            for i in range(adios_file.file.Steps()):
                adios_file.file.BeginStep()
                if array_name in adios_file.io.AvailableVariables().keys():
                    break
                adios_file.file.EndStep()
            if array_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No array found at {array_name}")
        else:
            for i in range(adios_file.file.Steps()):
                adios_file.file.BeginStep()
                if time_name in adios_file.io.AvailableVariables().keys():
                    arr = adios_file.io.InquireVariable(time_name)
                    time_shape = arr.Shape()
                    arr.SetSelection([[0], [time_shape[0]]])
                    times = np.empty(time_shape[0], dtype=adios_to_numpy_dtype[arr.Type()])
                    adios_file.file.Get(arr, times, adios2.Mode.Sync)
                    if times[0] == time:
                        break
                if i == adios_file.file.Steps() - 1:
                    raise KeyError(
                        f"No data associated with {time_name}={time} found in {filename}"
                    )

                adios_file.file.EndStep()

            if time_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No data associated with {time_name}={time} found in {filename}")

            if array_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No array found at {time=} for {array_name}")

        arr = adios_file.io.InquireVariable(array_name)
        arr_shape = arr.Shape()
        assert len(arr_shape) >= 1  # TODO: Should we always pick the first element?
        arr_range = compute_local_range(comm, arr_shape[0])

        if len(arr_shape) == 1:
            arr.SetSelection([[arr_range[0]], [arr_range[1] - arr_range[0]]])
            vals = np.empty(arr_range[1] - arr_range[0], dtype=adios_to_numpy_dtype[arr.Type()])
        else:
            arr.SetSelection([[arr_range[0], 0], [arr_range[1] - arr_range[0], arr_shape[1]]])
            vals = np.empty(
                (arr_range[1] - arr_range[0], arr_shape[1]),
                dtype=adios_to_numpy_dtype[arr.Type()],
            )
            assert arr_shape[1] == 1

        adios_file.file.Get(arr, vals, adios2.Mode.Sync)
        adios_file.file.EndStep()

    return vals.reshape(-1), arr_range[0]
