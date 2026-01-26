import warnings
from pathlib import Path
from typing import Any, Union

from mpi4py import MPI

import adios2
import numpy as np
import numpy.typing as npt

from ...structures import MeshData
from ...utils import check_file_exists
from .. import FileMode
from .helpers import ADIOSFile, adios_to_numpy_dtype, resolve_adios_scope

adios2 = resolve_adios_scope(adios2)


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    if "engine" not in args.keys():
        args["engine"] = "BP4"
    return args


def convert_file_mode(mode: FileMode) -> adios2.Mode:  # type: ignore[override]
    match mode:
        case FileMode.append:
            return adios2.Mode.Append
        case FileMode.write:
            return adios2.Mode.Write
        case FileMode.read:
            return adios2.Mode.Read
        case _:
            raise NotImplementedError(f"FileMode {mode} not implemented.")


def write_attributes(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None = None,
):
    """Write attributes to file using ADIOS2.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attributes
        attributes: Dictionary of attributes to write to file
        engine: ADIOS2 engine to use
    """

    adios = adios2.ADIOS(comm)
    backend_args = get_default_backend_args(backend_args)

    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Append,
        io_name="AttributeWriter",
        **backend_args,
    ) as adios_file:
        adios_file.file.BeginStep()

        for k, v in attributes.items():
            adios_file.io.DefineAttribute(f"{name}_{k}", v)

        adios_file.file.PerformPuts()
        adios_file.file.EndStep()


def read_attributes(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Read attributes from file using ADIOS2.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attributes
        engine: ADIOS2 engine to use
    Returns:
        The attributes
    """
    check_file_exists(filename)
    adios = adios2.ADIOS(comm)
    backend_args = get_default_backend_args(backend_args)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        **backend_args,
        io_name="AttributesReader",
    ) as adios_file:
        adios_file.file.BeginStep()
        attributes = {}
        for k in adios_file.io.AvailableAttributes().keys():
            if k.startswith(f"{name}_"):
                a = adios_file.io.InquireAttribute(k)
                attributes[k[len(name) + 1 :]] = a.Data()
        adios_file.file.EndStep()
    return attributes


def read_timestamps(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Read time-stamps from a checkpoint file.

    Args:
        comm: MPI communicator
        filename: Path to file
        function_name: Name of the function to read time-stamps for
        backend_args: Arguments for backend, for instance file type.
        backend: What backend to use for writing.
    Returns:
        The time-stamps
    """
    check_file_exists(filename)

    adios = adios2.ADIOS(comm)
    backend_args = get_default_backend_args(backend_args)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        **backend_args,
        io_name="TimestepReader",
    ) as adios_file:
        time_name = f"{function_name}_time"
        time_stamps = []
        for _ in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if time_name in adios_file.io.AvailableVariables().keys():
                arr = adios_file.io.InquireVariable(time_name)
                time_shape = arr.Shape()
                arr.SetSelection([[0], [time_shape[0]]])
                times = np.empty(
                    time_shape[0],
                    dtype=adios_to_numpy_dtype[arr.Type()],
                )
                adios_file.file.Get(arr, times, adios2.Mode.Sync)
                time_stamps.append(times[0])
            adios_file.file.EndStep()

    return np.array(time_stamps)


def write_mesh(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """
    Write a mesh to file using ADIOS2

    Parameters:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        backend_args: File mode and potentially the io-name.
        mode: Mode to use (write or append)
        time: Time stamp
    """
    backend_args = get_default_backend_args(backend_args)
    if "io_name" not in backend_args.keys():
        backend_args["io_name"] = "MeshWriter"

    mode = convert_file_mode(mode)
    gdim = mesh.local_geometry.shape[1]
    adios = adios2.ADIOS(comm)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=mode,
        comm=comm,
        **backend_args,
    ) as adios_file:
        adios_file.file.BeginStep()
        # Write geometry
        pointvar = adios_file.io.DefineVariable(
            "Points",
            mesh.local_geometry,
            shape=[mesh.num_nodes_global, gdim],
            start=[mesh.local_geometry_pos[0], 0],
            count=[mesh.local_geometry_pos[1] - mesh.local_geometry_pos[0], gdim],
        )
        adios_file.file.Put(pointvar, mesh.local_geometry, adios2.Mode.Sync)

        if mode == adios2.Mode.Write:
            adios_file.io.DefineAttribute("CellType", mesh.cell_type)
            adios_file.io.DefineAttribute("Degree", np.array([mesh.degree], dtype=np.int32))
            adios_file.io.DefineAttribute(
                "LagrangeVariant", np.array([mesh.lagrange_variant], dtype=np.int32)
            )
            # Write topology (on;y on first write as topology is constant)
            num_dofs_per_cell = mesh.local_topology.shape[1]
            dvar = adios_file.io.DefineVariable(
                "Topology",
                mesh.local_topology,
                shape=[mesh.num_cells_global, num_dofs_per_cell],
                start=[mesh.local_topology_pos[0], 0],
                count=[
                    mesh.local_topology_pos[1] - mesh.local_topology_pos[0],
                    num_dofs_per_cell,
                ],
            )
            adios_file.file.Put(dvar, mesh.local_topology)

            # Add partitioning data
            if mesh.store_partition:
                assert mesh.partition_range is not None
                par_data = adios_file.io.DefineVariable(
                    "PartitioningData",
                    mesh.ownership_array,
                    shape=[mesh.partition_global],
                    start=[mesh.partition_range[0]],
                    count=[
                        mesh.partition_range[1] - mesh.partition_range[0],
                    ],
                )
                adios_file.file.Put(par_data, mesh.ownership_array)
                assert mesh.ownership_offset is not None
                par_offset = adios_file.io.DefineVariable(
                    "PartitioningOffset",
                    mesh.ownership_offset,
                    shape=[mesh.num_cells_global + 1],
                    start=[mesh.local_topology_pos[0]],
                    count=[mesh.local_topology_pos[1] - mesh.local_topology_pos[0] + 1],
                )
                adios_file.file.Put(par_offset, mesh.ownership_offset)
                assert mesh.partition_processes is not None
                adios_file.io.DefineAttribute(
                    "PartitionProcesses", np.array([mesh.partition_processes], dtype=np.int32)
                )
        if mode == adios2.Mode.Append and mesh.store_partition:
            warnings.warn("Partitioning data is not written in append mode")

        # Add time step to file
        t_arr = np.array([time], dtype=np.float64)
        time_var = adios_file.io.DefineVariable(
            "MeshTime",
            t_arr,
            shape=[1],
            start=[0],
            count=[1 if comm.rank == 0 else 0],
        )
        adios_file.file.Put(time_var, t_arr)

        adios_file.file.PerformPuts()
        adios_file.file.EndStep()
