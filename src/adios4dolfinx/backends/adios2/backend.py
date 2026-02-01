import warnings
from pathlib import Path
from typing import Any

from mpi4py import MPI

import adios2
import dolfinx
import numpy as np
import numpy.typing as npt

from ...structures import FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, compute_local_range
from .. import FileMode, ReadMode
from .helpers import (
    ADIOSFile,
    adios_to_numpy_dtype,
    check_variable_exists,
    read_adjacency_list,
    read_array,
    resolve_adios_scope,
)

adios2 = resolve_adios_scope(adios2)

read_mode = ReadMode.parallel


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
    filename: Path | str,
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
    filename: Path | str,
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
    filename: Path | str,
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
    filename: Path | str,
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
    filename: Path | str,
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
    backend_args = get_default_backend_args(backend_args)
    legacy = backend_args.get("legacy", False)
    io_name = backend_args.get("io_name", "MeshReader")
    engine = backend_args["engine"]
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        engine=engine,
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


def write_meshtags(
    filename: str | Path,
    comm: MPI.Intracomm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None = None,
):
    backend_args = {} if backend_args is None else backend_args
    io_name = backend_args.get("io_name", "MeshTagWriter")
    engine = backend_args.get("engine", "BP4")
    adios = adios2.ADIOS(comm)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Append,
        engine=engine,
        io_name=io_name,
    ) as adios_file:
        adios_file.file.BeginStep()

        # Write meshtag topology
        topology_var = adios_file.io.DefineVariable(
            data.name + "_topology",
            data.indices,
            shape=[data.num_entities_global, data.num_dofs_per_entity],
            start=[data.local_start, 0],
            count=[len(data.indices), data.num_dofs_per_entity],
        )
        adios_file.file.Put(topology_var, data.indices, adios2.Mode.Sync)

        # Write meshtag values
        values_var = adios_file.io.DefineVariable(
            data.name + "_values",
            data.values,
            shape=[data.num_entities_global],
            start=[data.local_start],
            count=[len(data.indices)],
        )
        adios_file.file.Put(values_var, data.values, adios2.Mode.Sync)

        # Write meshtag dim
        adios_file.io.DefineAttribute(data.name + "_dim", np.array([data.dim], dtype=np.uint8))
        adios_file.file.PerformPuts()
        adios_file.file.EndStep()


def read_meshtags_data(
    filename: str | Path, comm: MPI.Intracomm, name: str, backend_args: dict[str, Any] | None = None
) -> MeshTagsData:
    adios = adios2.ADIOS(comm)
    backend_args = {} if backend_args is None else backend_args
    io_name = backend_args.get("io_name", "MeshTagsReader")
    engine = backend_args.get("engine", "BP4")
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        engine=engine,
        io_name=io_name,
    ) as adios_file:
        # Get mesh cell type
        dim_attr_name = f"{name}_dim"
        step = 0
        for i in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if dim_attr_name in adios_file.io.AvailableAttributes().keys():
                step = i
                break
            adios_file.file.EndStep()
        if dim_attr_name not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"{dim_attr_name} not found in {filename}")

        m_dim = adios_file.io.InquireAttribute(dim_attr_name)
        dim = int(m_dim.Data()[0])

        # Get mesh tags entites
        topology_name = f"{name}_topology"
        for i in range(step, adios_file.file.Steps()):
            if i > step:
                adios_file.file.BeginStep()
            if topology_name in adios_file.io.AvailableVariables().keys():
                break
            adios_file.file.EndStep()
        if topology_name not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"{topology_name} not found in {filename}")

        topology = adios_file.io.InquireVariable(topology_name)
        top_shape = topology.Shape()
        topology_range = compute_local_range(comm, top_shape[0])

        topology.SetSelection(
            [
                [topology_range[0], 0],
                [topology_range[1] - topology_range[0], top_shape[1]],
            ]
        )
        mesh_entities = np.empty(
            (topology_range[1] - topology_range[0], top_shape[1]), dtype=np.int64
        )
        adios_file.file.Get(topology, mesh_entities, adios2.Mode.Deferred)

        # Get mesh tags values
        values_name = f"{name}_values"
        if values_name not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"{values_name} not found")

        values = adios_file.io.InquireVariable(values_name)
        val_shape = values.Shape()
        assert val_shape[0] == top_shape[0]
        values.SetSelection([[topology_range[0]], [topology_range[1] - topology_range[0]]])
        tag_values = np.empty((topology_range[1] - topology_range[0]), dtype=np.int32)
        adios_file.file.Get(values, tag_values, adios2.Mode.Deferred)

        adios_file.file.PerformGets()
        adios_file.file.EndStep()

        return MeshTagsData(name=name, values=tag_values, indices=mesh_entities, dim=dim)


def read_dofmap(
    filename: str | Path, comm: MPI.Intracomm, name: str, backend_args: dict[str, Any] | None = None
) -> dolfinx.graph.AdjacencyList:
    backend_args = {} if backend_args is None else backend_args

    # Handles legacy adios4dolfinx files, modern files, and custom location of dofmap.
    legacy = backend_args.get("legacy", False)
    xdofmap_path: str | None
    dofmap_path: str | None
    if (dofmap_path := backend_args.get("dofmap", None)) is None:
        if legacy:
            dofmap_path = "Dofmap"
        else:
            dofmap_path = f"{name}_dofmap"

    if (xdofmap_path := backend_args.get("offsets", None)) is None:
        if legacy:
            xdofmap_path = "XDofmap"
        else:
            xdofmap_path = f"{name}_XDofmap"

    engine = backend_args.get("engine", "BP4")

    adios = adios2.ADIOS(comm)
    check_file_exists(filename)
    assert isinstance(xdofmap_path, str)
    return read_adjacency_list(adios, comm, filename, dofmap_path, xdofmap_path, engine=engine)


def read_dofs(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    time: float,
    backend_args: dict[str, Any] | None = None,
) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
    backend_args = {} if backend_args is None else backend_args
    legacy = backend_args.get("legacy", False)
    engine = backend_args.get("engine", "BP4")
    io_name = backend_args.get("io_name", f"{name}_FunctionReader")
    # Check that file contains the function to read
    adios = adios2.ADIOS(comm)
    check_file_exists(filename)

    if not legacy:
        with ADIOSFile(
            adios=adios,
            filename=filename,
            mode=adios2.Mode.Read,
            engine=engine,
            io_name=io_name,
        ) as adios_file:
            variables = set(
                sorted(
                    map(
                        lambda x: x.split("_time")[0],
                        filter(lambda x: x.endswith("_time"), adios_file.io.AvailableVariables()),
                    )
                )
            )
            if name not in variables:
                raise KeyError(f"{name} not found in {filename}. Did you mean one of {variables}?")

    if legacy:
        array_path = "Values"
    else:
        array_path = f"{name}_values"

    time_name = f"{name}_time"
    return read_array(adios, filename, array_path, engine, comm, time, time_name, legacy=legacy)


def read_cell_perms(
    comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None = None
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
    adios = adios2.ADIOS(comm)
    check_file_exists(filename)

    # Open ADIOS engine
    backend_args = {} if backend_args is None else backend_args
    engine = backend_args.get("engine", "BP4")

    cell_perms, _ = read_array(
        adios, filename, "CellPermutations", engine=engine, comm=comm, legacy=True
    )

    return cell_perms.astype(np.uint32)


def read_hdf5_array(
    comm: MPI.Intracomm,
    filename: Path | str,
    group: str,
    backend_args: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    adios = adios2.ADIOS(comm)
    return read_array(adios, filename, group, engine="HDF5", comm=comm, legacy=True)


def write_function(
    filename: Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    time: float = 0.0,
    mode: FileMode = FileMode.append,
    backend_args: dict[str, Any] | None = None,
):
    """
    Write a function to file using ADIOS2

    Parameters:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        engine: ADIOS2 engine to use
        mode: ADIOS2 mode to use (write or append)
        time: Time stamp associated with function
        io_name: Internal name used for the ADIOS IO object
    """
    adios_mode = convert_file_mode(mode)
    backend_args = get_default_backend_args(backend_args)
    engine = backend_args["engine"]
    io_name = backend_args.get("io_name", "{name}_writer")

    adios = adios2.ADIOS(comm)
    cell_permutations_exists = False
    dofmap_exists = False
    XDofmap_exists = False
    if mode == adios2.Mode.Append:
        cell_permutations_exists = check_variable_exists(
            adios, filename, "CellPermutations", engine=engine
        )
        dofmap_exists = check_variable_exists(adios, filename, f"{u.name}_dofmap", engine=engine)
        XDofmap_exists = check_variable_exists(adios, filename, f"{u.name}_XDofmap", engine=engine)

    with ADIOSFile(
        adios=adios, filename=filename, mode=adios_mode, engine=engine, io_name=io_name, comm=comm
    ) as adios_file:
        adios_file.file.BeginStep()

        if not cell_permutations_exists:
            # Add mesh permutations
            pvar = adios_file.io.DefineVariable(
                "CellPermutations",
                u.cell_permutations,
                shape=[u.num_cells_global],
                start=[u.local_cell_range[0]],
                count=[u.local_cell_range[1] - u.local_cell_range[0]],
            )
            adios_file.file.Put(pvar, u.cell_permutations)

        if not dofmap_exists:
            # Add dofmap
            dofmap_var = adios_file.io.DefineVariable(
                f"{u.name}_dofmap",
                u.dofmap_array,
                shape=[u.global_dofs_in_dofmap],
                start=[u.dofmap_range[0]],
                count=[u.dofmap_range[1] - u.dofmap_range[0]],
            )
            adios_file.file.Put(dofmap_var, u.dofmap_array)

        if not XDofmap_exists:
            # Add XDofmap
            xdofmap_var = adios_file.io.DefineVariable(
                f"{u.name}_XDofmap",
                u.dofmap_offsets,
                shape=[u.num_cells_global + 1],
                start=[u.local_cell_range[0]],
                count=[u.local_cell_range[1] - u.local_cell_range[0] + 1],
            )
            adios_file.file.Put(xdofmap_var, u.dofmap_offsets)

        val_var = adios_file.io.DefineVariable(
            f"{u.name}_values",
            u.values,
            shape=[u.num_dofs_global],
            start=[u.dof_range[0]],
            count=[u.dof_range[1] - u.dof_range[0]],
        )
        adios_file.file.Put(val_var, u.values)

        # Add time step to file
        t_arr = np.array([time], dtype=np.float64)
        time_var = adios_file.io.DefineVariable(
            f"{u.name}_time",
            t_arr,
            shape=[1],
            start=[0],
            count=[1 if comm.rank == 0 else 0],
        )
        adios_file.file.Put(time_var, t_arr)
        adios_file.file.PerformPuts()
        adios_file.file.EndStep()


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Intracomm, group: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], str | None]:
    # Create ADIOS2 reader
    adios = adios2.ADIOS(comm)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name="Mesh reader",
        engine="HDF5",
    ) as adios_file:
        # Get mesh topology (distributed)
        if f"{group}/topology" not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Mesh topology not found at '{group}/topology'")
        topology = adios_file.io.InquireVariable(f"{group}/topology")
        shape = topology.Shape()
        local_range = compute_local_range(comm, shape[0])
        topology.SetSelection([[local_range[0], 0], [local_range[1] - local_range[0], shape[1]]])

        mesh_topology = np.empty(
            (local_range[1] - local_range[0], shape[1]),
            dtype=topology.Type().strip("_t"),
        )
        adios_file.file.Get(topology, mesh_topology, adios2.Mode.Sync)

        # Get mesh cell type
        cell_type = None
        if f"{group}/topology/celltype" in adios_file.io.AvailableAttributes().keys():
            celltype = adios_file.io.InquireAttribute(f"{group}/topology/celltype")
            cell_type = celltype.DataString()[0]

        # Get mesh geometry

        for geometry_key in [f"{group}/geometry", f"{group}/coordinates"]:
            if geometry_key in adios_file.io.AvailableVariables().keys():
                break
        else:
            raise KeyError(
                f"Mesh coordinates not found at '{group}/coordinates' or '{group}/geometry'"
            )

        geometry = adios_file.io.InquireVariable(geometry_key)
        shape = geometry.Shape()
        local_range = compute_local_range(comm, shape[0])
        geometry.SetSelection([[local_range[0], 0], [local_range[1] - local_range[0], shape[1]]])
        mesh_geometry = np.empty(
            (local_range[1] - local_range[0], shape[1]),
            dtype=adios_to_numpy_dtype[geometry.Type()],
        )
        adios_file.file.Get(geometry, mesh_geometry, adios2.Mode.Sync)

    return mesh_topology, mesh_geometry, cell_type


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    adios_mode = convert_file_mode(mode)
    adios = adios2.ADIOS(u.function_space.mesh.comm)
    backend_args = {} if backend_args is None else backend_args
    io_name = backend_args.get("io_name", "SnapshotCheckPoint")
    engine = backend_args.get("engine", "BP4")
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios_mode,
        io_name=io_name,
        engine=engine,
    ) as adios_file:
        if adios_mode == adios2.Mode.Write:
            dofmap = u.function_space.dofmap
            num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
            local_dofs = u.x.array[:num_dofs_local].copy()

            # Write to file
            adios_file.file.BeginStep()
            dofs = adios_file.io.DefineVariable("dofs", local_dofs, count=[num_dofs_local])
            adios_file.file.Put(dofs, local_dofs, adios2.Mode.Sync)
            adios_file.file.EndStep()
        elif adios_mode == adios2.Mode.Read:
            adios_file.file.BeginStep()
            in_variable = adios_file.io.InquireVariable("dofs")
            in_variable.SetBlockSelection(u.function_space.mesh.comm.rank)
            adios_file.file.Get(in_variable, u.x.array, adios2.Mode.Sync)
            adios_file.file.EndStep()
            u.x.scatter_forward()
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for snapshot checkpoint")


def read_point_data(
    filename: Path | str, name: str, mesh: dolfinx.mesh.Mesh
) -> dolfinx.fem.Function:
    """Read data from te nodes of a mesh.

    Parameters:
        filename: Path to file
        name: Name of point data
        mesh: The corresponding :py:class:`dolfinx.mesh.Mesh`.

    Returns:
        A function in the space equivalent to the mesh
        coordinate element (up to shape).
    """
    raise NotImplementedError("The ADIOS2 backend cannot read point data.")
