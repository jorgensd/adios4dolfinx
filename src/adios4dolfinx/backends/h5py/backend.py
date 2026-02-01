"""
H5py interface to ADIOS4DOLFINx

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Henrik N.T. Finsberg, Simula Research Laboratory
"""

import contextlib
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
from dolfinx.graph import adjacencylist

from ...structures import FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, compute_local_range
from .. import FileMode, ReadMode

read_mode = ReadMode.parallel


@contextlib.contextmanager
def h5pyfile(h5name, filemode="r", force_serial: bool = False, comm=None):
    """Context manager for opening an HDF5 file with h5py.

    Args:
        h5name: The name of the HDF5 file.
        filemode: The file mode.
        force_serial: Force serial access to the file.
        comm: The MPI communicator

    """
    import h5py

    if comm is None:
        comm = MPI.COMM_WORLD

    if h5py.h5.get_config().mpi and not force_serial:
        h5file = h5py.File(h5name, filemode, driver="mpio", comm=comm)
    else:
        if comm.size > 1 and not force_serial:
            raise ValueError(
                f"h5py is not installed with MPI support, while using {comm.size} processes.",
                "If you really want to do this, turn on the `force_serial` flag.",
            )
        h5file = h5py.File(h5name, filemode)

    try:
        yield h5file
    finally:
        if h5file.id:
            h5file.close()


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}

    if arguments:
        # Currently no default arguments for h5py backend
        # TODO: Pehaps we would like to make this into a warning instead?
        raise RuntimeError("Unexpected backend arguments to h5py backend")

    return args


def convert_file_mode(mode: FileMode) -> str:
    match mode:
        case FileMode.append:
            return "a"
        case FileMode.read:
            return "r"
        case FileMode.write:
            return "w"
        case _:
            raise NotImplementedError(f"File mode {mode} not implemented")


def write_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None = None,
):
    """Write attributes to file using H5PY.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attributes
        attributes: Dictionary of attributes to write to file
        engine: ADIOS2 engine to use
    """
    filemode = "a" if Path(filename).exists() else "w"
    with h5pyfile(filename, filemode=filemode, comm=comm, force_serial=False) as h5file:
        if name not in h5file.keys():
            h5file.create_group(name, track_order=True)
        group = h5file[name]

        for key, val in attributes.items():
            group.attrs[key] = val


def read_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read attributes from file using H5PY.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attributes
    Returns:
        The attributes
    """
    check_file_exists(filename)
    output_attrs = {}
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        for key, val in h5file[name].attrs.items():
            output_attrs[key] = val
    return output_attrs


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
    mesh_name = "mesh"
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        mesh_directory = h5file[mesh_name]
        functions = mesh_directory["functions"]
        u = functions[function_name]
        timestamps = u.attrs["timestamps"]
    return timestamps


def write_mesh(
    filename: Path | str,
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """
    Write a mesh to file using H5PY

    Parameters:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to.
        mode: Mode to use (write or append)
        time: Time stamp
    """
    backend_args = get_default_backend_args(backend_args)
    h5_mode = convert_file_mode(mode)
    mesh_name = "mesh"
    with h5pyfile(filename, filemode=h5_mode, comm=comm, force_serial=False) as h5file:
        if mesh_name in h5file.keys() and h5_mode == "a":
            mesh_directory = h5file[mesh_name]
            timestamps = mesh_directory.attrs["timestamps"]
            if np.isclose(time, timestamps).any():
                raise ValueError("Mesh has already been stored at time={time_stamp}.")
            else:
                mesh_directory.attrs["timestamps"] = np.append(
                    mesh_directory.attrs["timestamps"], time
                )
                idx = len(mesh_directory.attrs["timestamps"]) - 1
                write_topology = False
        else:
            mesh_directory = h5file.create_group(mesh_name)
            mesh_directory.attrs["timestamps"] = np.array([time], dtype=np.float64)
            idx = 0
            write_topology = True

        geometry_group = mesh_directory.create_group(f"{idx}")

        # Write geometry data
        gdim = mesh.local_geometry.shape[1]
        geometry_dataset = geometry_group.create_dataset(
            "Points", [mesh.num_nodes_global, gdim], dtype=mesh.local_geometry.dtype
        )
        geometry_dataset[slice(*mesh.local_geometry_pos), :] = mesh.local_geometry

        # Write static partitioning data
        if "PartitioningData" not in mesh_directory.keys() and mesh.store_partition:
            assert mesh.partition_range is not None
            assert mesh.ownership_array is not None
            par_dataset = mesh_directory.create_dataset(
                "PartitioningData", [mesh.partition_global], dtype=mesh.ownership_array.dtype
            )
            par_dataset[slice(*mesh.partition_range)] = mesh.ownership_array

        if "PartitioningOffset" not in mesh_directory.keys() and mesh.store_partition:
            assert mesh.ownership_offset is not None
            par_dataset = mesh_directory.create_dataset(
                "PartitioningOffset", [mesh.num_cells_global + 1], dtype=np.int64
            )
            par_dataset[mesh.local_topology_pos[0] : mesh.local_topology_pos[1] + 1] = (
                mesh.ownership_offset
            )

        if "PartitionProcesses" not in mesh_directory.attrs.keys() and mesh.store_partition:
            mesh_directory.attrs["PartitionProcesses"] = mesh.partition_processes

        # Write static data
        if write_topology:
            mesh_directory.attrs["CellType"] = mesh.cell_type
            mesh_directory.attrs["Degree"] = mesh.degree
            mesh_directory.attrs["LagrangeVariant"] = mesh.lagrange_variant
            num_dofs_per_cell = mesh.local_topology.shape[1]
            topology_dataset = mesh_directory.create_dataset(
                "Topology", [mesh.num_cells_global, num_dofs_per_cell], dtype=np.int64
            )
            topology_dataset[slice(*mesh.local_topology_pos), :] = mesh.local_topology


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: float = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """
    Read mesh data from h5py based checkpoint files.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        time: Time stamp associated with mesh
        read_from_partition: Read mesh with partition from file
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """

    backend_args = get_default_backend_args(backend_args)

    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        if "mesh" not in h5file.keys():
            raise KeyError("Could not find mesh in file")
        mesh_group = h5file["mesh"]
        timestamps = mesh_group.attrs["timestamps"]
        parent_group = np.flatnonzero(np.isclose(timestamps, time))
        if len(parent_group) != 1:
            raise RuntimeError(
                f"Time step {time} not found in file, available steps is {timestamps}"
            )
        time_group = f"{parent_group[0]:d}"

        # Get mesh topology (distributed)
        topology = mesh_group["Topology"]
        local_range = compute_local_range(comm, topology.shape[0])
        mesh_topology = topology[slice(*local_range), :]

        cell_type = mesh_group.attrs["CellType"]
        lvar = mesh_group.attrs["LagrangeVariant"]
        degree = mesh_group.attrs["Degree"]

        geometry_group = mesh_group[time_group]
        geometry_dataset = geometry_group["Points"]
        x_shape = geometry_dataset.shape

        geometry_range = compute_local_range(comm, x_shape[0])
        mesh_geometry = geometry_dataset[slice(*geometry_range), :]

        # Check validity of partitioning information
        if read_from_partition:
            if "PartitionProcesses" not in mesh_group.attrs.keys():
                raise KeyError(f"Partitioning information not found in {filename}")
            par_keys = ("PartitioningData", "PartitioningOffset")
            for key in par_keys:
                if key not in mesh_group.keys():
                    raise KeyError(f"Partitioning information not found in {filename}")

            par_num_procs = mesh_group.attrs["PartitionProcesses"]
            if par_num_procs != comm.size:
                raise ValueError(f"Number of processes in file ({par_num_procs})!=({comm.size=})")

            # First read in offsets based on the number of cells [0, num_cells_local]
            par_offsets = mesh_group["PartitioningOffset"][local_range[0] : local_range[1] + 1]
            # Then read the data based of offsets
            par_data = mesh_group["PartitioningData"][par_offsets[0] : par_offsets[-1]]
            # Then make offsets local
            par_offsets[:] -= par_offsets[0]
            partition_graph = adjacencylist(par_data, par_offsets.astype(np.int32))
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
    backend_args = get_default_backend_args(backend_args)

    with h5pyfile(filename, filemode="a", comm=comm, force_serial=False) as h5file:
        if "mesh" not in h5file.keys():
            raise KeyError("Could not find mesh in file")
        mesh_group = h5file["mesh"]
        if "tags" not in mesh_group.keys():
            tags = mesh_group.create_group("tags")
        else:
            tags = mesh_group["tags"]
        if data.name in tags.keys():
            raise KeyError(f"MeshTags with {data.name=} already exists in this file")
        tag = tags.create_group(data.name)

        # Add topology
        topology = tag.create_dataset(
            "Topology", shape=[data.num_entities_global, data.num_dofs_per_entity], dtype=np.int64
        )
        assert data.local_start is not None
        topology[data.local_start : data.local_start + len(data.indices), :] = data.indices

        # Add cell_type attribute
        tag.attrs["CellType"] = data.cell_type

        # Add values
        values = tag.create_dataset(
            "Values", shape=[data.num_entities_global], dtype=data.values.dtype
        )
        values[data.local_start : data.local_start + len(data.indices)] = data.values

        # Add dimension
        tag.attrs["dim"] = data.dim


def read_meshtags_data(
    filename: str | Path, comm: MPI.Intracomm, name: str, backend_args: dict[str, Any] | None = None
) -> MeshTagsData:
    backend_args = get_default_backend_args(backend_args)

    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        if "mesh" not in h5file.keys():
            raise KeyError("No mesh found")
        mesh = h5file["mesh"]
        if "tags" not in mesh.keys():
            raise KeyError("Could not find 'tags' in file, are you sure this is a checkpoint?")
        tags = mesh["tags"]
        if name not in tags.keys():
            raise KeyError(f"Could not find {name} in '/mesh/tags/' in {filename}")
        tag = tags[name]

        dim = tag.attrs["dim"]
        topology = tag["Topology"]
        num_entities_global = topology.shape[0]
        topology_range = compute_local_range(comm, num_entities_global)
        indices = topology[slice(*topology_range), :]
        values = tag["Values"]
        vals = values[slice(*topology_range)]
        return MeshTagsData(name=name, values=vals, indices=indices, dim=dim)


def read_dofmap(
    filename: str | Path, comm: MPI.Intracomm, name: str, backend_args: dict[str, Any] | None
) -> dolfinx.graph.AdjacencyList:
    backend_args = {} if backend_args is None else backend_args
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        # If dofmap is read with full path, it is passed through backend_args
        dofmap_key = backend_args.get("dofmap", None)
        if dofmap_key is None:
            mesh_name = "mesh"  # Prepare for multiple meshes
            if mesh_name not in h5file.keys():
                raise KeyError(f"No mesh '{mesh_name}' found in {filename}")
            mesh = h5file[mesh_name]
            if "functions" not in mesh.keys():
                raise KeyError(f"No functions stored in '{mesh_name}' in {filename}")
            functions = mesh["functions"]
            if name not in functions.keys():
                raise KeyError(
                    f"No function with name '{name}' on '{mesh_name}' stored in {filename}"
                )
            function = functions[name]
            offset_key = "dofmap_offsets"
            dofmap_key = "dofmap"
            offsets = function[offset_key]
            dofmap = function[dofmap_key]
        else:
            offset_key = backend_args["offsets"]
            dofmap = h5file[dofmap_key]
            offsets = h5file[offset_key]

        num_cells = offsets.shape[0] - 1
        local_range = compute_local_range(comm, num_cells)

        # First read in offsets based on the number of cells [0, num_cells_local]
        glob_offsets = offsets[local_range[0] : local_range[1] + 1].flatten().astype(np.int64)

        # Then read the data based of offsets
        dofmap_data = dofmap[glob_offsets[0] : glob_offsets[-1]].flatten()

    # Then make offsets local
    loc_offsets = (glob_offsets - glob_offsets[0]).astype(np.int32)
    return adjacencylist(dofmap_data, loc_offsets)


def read_dofs(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    time: float,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        mesh_name = "mesh"  # Prepare for multiple meshes
        if mesh_name not in h5file.keys():
            raise RuntimeError(f"No mesh '{mesh_name}' found in {filename}")
        mesh = h5file[mesh_name]
        if "functions" not in mesh.keys():
            raise RuntimeError(f"No functions stored in '{mesh_name}' in {filename}")
        functions = mesh["functions"]
        if name not in functions.keys():
            raise RuntimeError(
                f"No function with name '{name}' on '{mesh_name}' stored in {filename}"
            )
        function = functions[name]
        timestamps = function.attrs["timestamps"]
        idx = np.flatnonzero(np.isclose(timestamps, time))
        if len(idx) != 1:
            raise RuntimeError("Could not find {name}(t={time}) on grid {mesh_name} in {filename}.")
        u_t = function[f"{idx[0]:d}"]
        data_group = u_t["array"]
        num_dofs_global = data_group.shape[0]
        local_range = compute_local_range(comm, num_dofs_global)
        local_array = data_group[slice(*local_range)]
        return local_array, local_range[0]


def read_cell_perms(
    comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None
) -> npt.NDArray[np.uint32]:
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        mesh_name = "mesh"  # Prepare for multiple meshes
        if mesh_name not in h5file.keys():
            raise RuntimeError(f"No mesh '{mesh_name}' found in {filename}")
        mesh = h5file[mesh_name]
        data_group = mesh["CellPermutations"]
        num_cells_global = data_group.shape[0]
        local_range = compute_local_range(comm, num_cells_global)
        local_array = data_group[slice(*local_range)]
        return local_array


def write_function(
    filename: str | Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None = None,
):
    mesh_name = "mesh"  # Prepare for multiple meshes
    backend_args = get_default_backend_args(backend_args)
    h5_mode = convert_file_mode(mode)
    with h5pyfile(filename, filemode=h5_mode, comm=comm, force_serial=False) as h5file:
        cell_permutations_exist = False
        dofmap_exists = False
        dofmap_offsets_exists = False
        if h5_mode == "a":
            if mesh_name not in h5file.keys():
                mesh = h5file.create_group(mesh_name)
            else:
                mesh = h5file[mesh_name]

            cell_permutations_exist = "CellPermutations" in mesh.keys()

            if "functions" not in mesh.keys():
                functions = mesh.create_group("functions")
            else:
                functions = mesh["functions"]

            if u.name not in functions.keys():
                function = functions.create_group(u.name)
            else:
                function = functions[u.name]

            dofmap_exists = "dofmap" in function.keys()
            dofmap_offsets_exists = "dofmap_offsets" in function.keys()

        if not cell_permutations_exist:
            cell_perms = mesh.create_dataset(
                "CellPermutations", shape=[u.num_cells_global], dtype=np.uint32
            )
            cell_perms[slice(*u.local_cell_range)] = u.cell_permutations

        if not dofmap_exists:
            dofmap = function.create_dataset(
                "dofmap", shape=[u.global_dofs_in_dofmap], dtype=np.int64
            )
            dofmap[slice(*u.dofmap_range)] = u.dofmap_array

        if not dofmap_offsets_exists:
            dofmap_offsets = function.create_dataset(
                "dofmap_offsets", shape=[u.num_cells_global + 1], dtype=np.int64
            )
            dofmap_offsets[u.local_cell_range[0] : u.local_cell_range[1] + 1] = u.dofmap_offsets

        # Write timestamp
        if "timestamps" in function.attrs.keys():
            timestamps = function.attrs["timestamps"]
            if np.isclose(time, timestamps).any():
                raise RuntimeError("FUnction has already been stored at time={time_stamp}.")
            else:
                function.attrs["timestamps"] = np.append(function.attrs["timestamps"], time)
        else:
            function.attrs["timestamps"] = np.array([time])
        idx = len(function.attrs["timestamps"]) - 1

        data_group = function.create_group(f"{idx:d}")
        array = data_group.create_dataset("array", shape=[u.num_dofs_global], dtype=u.values.dtype)
        array[slice(*u.dof_range)] = u.values


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Intracomm, group: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], str | None]:
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        if group not in h5file.keys():
            raise KeyError(f"Could not find {group}  in {filename}.")
        mesh = h5file[group]
        if "topology" not in mesh.keys():
            raise KeyError(f"Could not find '{group}/topology'  in {filename}.")

        topology = mesh["topology"]
        shape = topology.shape
        local_range = compute_local_range(comm, shape[0])
        mesh_topology = topology[slice(*local_range)].astype(np.int64)

        # Get mesh cell type
        cell_type = None
        if "celltype" in topology.attrs.keys():
            cell_type = topology.attrs["celltype"]
            if isinstance(cell_type, np.bytes_):
                cell_type = cell_type.decode("utf-8")

        for geometry_key in ["geometry", "coordinates"]:
            if geometry_key in mesh.keys():
                break
        else:
            raise KeyError(
                "Could not find geometry in '/mesh/geometry' or '/mesh/coordinates'"
                + f" in {filename}."
            )
        geometry = mesh[geometry_key]
        g_shape = geometry.shape
        local_g_range = compute_local_range(comm, g_shape[0])
        mesh_geometry = geometry[slice(*local_g_range)]

    return mesh_topology, mesh_geometry, cell_type


def read_hdf5_array(
    comm: MPI.Intracomm,
    filename: Path | str,
    group: str,
    backend_args: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        data = h5file[group]
        shape = data.shape[0]
        local_range = compute_local_range(comm, shape)
        out_data = data[slice(*local_range)].flatten()
    return out_data, local_range[0]


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    comm = u.function_space.mesh.comm
    dofmap = u.function_space.dofmap
    local_range = np.array(dofmap.index_map.local_range) * dofmap.index_map_bs
    num_dofs_local = local_range[1] - local_range[0]
    num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
    h5mode = convert_file_mode(mode)
    if h5mode == "w":
        with h5pyfile(filename, filemode=h5mode, comm=comm, force_serial=False) as h5file:
            local_dofs = u.x.array[:num_dofs_local].copy()
            data = h5file.create_group("snapshot")
            dataset = data.create_dataset("dofs", shape=num_dofs_global, dtype=local_dofs.dtype)
            dataset[slice(*local_range)] = local_dofs
    elif h5mode == "r":
        with h5pyfile(filename, filemode=h5mode, comm=comm, force_serial=False) as h5file:
            data = h5file["snapshot"]["dofs"]
            assert data.shape[0] == num_dofs_global
            u.x.array[:num_dofs_local] = data[slice(*local_range)]
            u.x.scatter_forward()


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
    raise NotImplementedError("The h5py backend cannot read point data.")
