import warnings
from pathlib import Path
from typing import Any, Union

from mpi4py import MPI

import adios2
import numpy as np
import numpy.typing as npt

from ...structures import MeshData, ReadMeshData
from ...utils import check_file_exists, compute_local_range
from .. import FileMode
from .helpers import ADIOSFile, adios_to_numpy_dtype, read_adjacency_list, resolve_adios_scope

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


def read_mesh_data(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    time: float = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """
    Read an ADIOS2 mesh data for use with DOLFINx.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        engine: ADIOS engine to use for reading (BP4, BP5 or HDF5)
        time: Time stamp associated with mesh
        legacy: If checkpoint was made prior to time-dependent mesh-writer set to True
        read_from_partition: Read mesh with partition from file
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """

    adios = adios2.ADIOS(comm)
    backend_args = backend_args if backend_args is not None else {}
    legacy = backend_args.pop("legacy", False)
    io_name = backend_args.pop("io_name", "MeshReader")
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        **backend_args,
        io_name=io_name,
    ) as adios_file:
        # Get time independent mesh variables (mesh topology and cell type info) first
        adios_file.file.BeginStep()
        # Get mesh topology (distributed)
        if "Topology" not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Mesh topology not found at Topology in {filename}")
        topology = adios_file.io.InquireVariable("Topology")
        shape = topology.Shape()
        local_range = compute_local_range(comm, shape[0])
        topology.SetSelection([[local_range[0], 0], [local_range[1] - local_range[0], shape[1]]])
        mesh_topology = np.empty((local_range[1] - local_range[0], shape[1]), dtype=np.int64)
        adios_file.file.Get(topology, mesh_topology, adios2.Mode.Deferred)

        # Check validity of partitioning information
        if read_from_partition:
            if "PartitionProcesses" not in adios_file.io.AvailableAttributes().keys():
                raise KeyError(f"Partitioning information not found in {filename}")
            par_num_procs = adios_file.io.InquireAttribute("PartitionProcesses")
            num_procs = par_num_procs.Data()[0]
            if num_procs != comm.size:
                raise ValueError(f"Number of processes in file ({num_procs})!=({comm.size=})")

        # Get mesh cell type
        if "CellType" not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"Mesh cell type not found at CellType in {filename}")
        celltype = adios_file.io.InquireAttribute("CellType")
        cell_type = celltype.DataString()[0]

        # Get basix info
        if "LagrangeVariant" not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"Mesh LagrangeVariant not found in {filename}")
        lvar = adios_file.io.InquireAttribute("LagrangeVariant").Data()[0]
        if "Degree" not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"Mesh degree not found in {filename}")
        degree = adios_file.io.InquireAttribute("Degree").Data()[0]

        if not legacy:
            time_name = "MeshTime"
            for i in range(adios_file.file.Steps()):
                if i > 0:
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

        # Get mesh geometry
        if "Points" not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Mesh coordinates not found at Points in {filename}")
        geometry = adios_file.io.InquireVariable("Points")
        x_shape = geometry.Shape()
        geometry_range = compute_local_range(comm, x_shape[0])
        geometry.SetSelection(
            [
                [geometry_range[0], 0],
                [geometry_range[1] - geometry_range[0], x_shape[1]],
            ]
        )
        mesh_geometry = np.empty(
            (geometry_range[1] - geometry_range[0], x_shape[1]),
            dtype=adios_to_numpy_dtype[geometry.Type()],
        )
        adios_file.file.Get(geometry, mesh_geometry, adios2.Mode.Deferred)
        adios_file.file.PerformGets()
        adios_file.file.EndStep()

    if read_from_partition:
        partition_graph = read_adjacency_list(
            adios,
            comm,
            filename,
            "PartitioningData",
            "PartitioningOffset",
            shape[0],
            backend_args["engine"],
        )
    else:
        partition_graph = None

    return ReadMeshData(
        cells=mesh_topology,
        cell_type=cell_type,
        x=mesh_geometry,
        degree=degree,
        lvar=lvar,
        partition_graph=partition_graph,
    )
