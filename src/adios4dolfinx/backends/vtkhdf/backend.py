"""
Module that can read the VTKHDF format using h5py.
"""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import h5py
import numpy as np
import numpy.typing as npt

from adios4dolfinx.structures import FunctionData, MeshData, MeshTagsData, ReadMeshData
from adios4dolfinx.utils import check_file_exists, compute_local_range

from .. import FileMode, ReadMode
from ..h5py.backend import convert_file_mode, h5pyfile
from ..pyvista.backend import _arbitrary_lagrange_vtk, _cell_degree, _first_order_vtk

read_mode = ReadMode.parallel


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {"name": "mesh"}
    return args


def find_all_unique_cell_types(comm, cell_types, num_nodes):
    """
    Given a set of cell types and number of nodes per cell, find all unique cell types
    across all ranks.

    Args:
        comm: MPI communicator
        cell_types: Local cell types
        num_nodes: Number of nodes per cell


    Returns:
        A 2D array where each row corresponds to a cell type (vtk int)
        and the number of nodes.
    """
    # Combine cell_types, num_nodes as tuple
    c_hash = np.zeros((2, len(cell_types)), dtype=np.int32)
    c_hash[0] = cell_types
    c_hash[1] = num_nodes
    indexes = np.unique(c_hash.T, axis=0, return_index=True)[1]
    local_unique_cells = c_hash.T[indexes]

    all_cell_types = np.vstack(comm.allgather(local_unique_cells))
    indexes = np.unique(all_cell_types, axis=0, return_index=True)[1]
    all_unique_cell_types = all_cell_types[indexes]
    return all_unique_cell_types


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: str | float | None,
    read_from_partition: bool,
    backend_args: dict[str, Any] | None,
) -> ReadMeshData:
    """Read mesh data from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh data read from file
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    if read_from_partition:
        raise RuntimeError("Cannot read partition data with VTKHDF")
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        # Check for type of VTKHDF file
        if (file_type := hdf.attrs["Type"]) in (b"MultiBlockDataSet", "MultiBlockDataSet"):
            # Recursive search of subgroups to find the unstructured grid
            name = backend_args["name"]
            ass = hdf["Assembly"]
            mesh_node = []

            def visitor(path):
                if path.split("/")[-1] == name:
                    # Retrieve the link object to check its type
                    obj = ass.get(path)
                    if "Type" in obj.attrs.keys() and obj.attrs["Type"] == "UnstructuredGrid":
                        mesh_node.append(path)
                        return 1
                # Return None to continue searching, or return a value to stop
                return None

            ass.visit_links(visitor)
            assert len(mesh_node) == 1
            hdf = ass[mesh_node[0]]
        elif file_type in (b"UnstructuredGrid", "UnstructuredGrid"):
            pass
        else:
            raise RuntimeError(f"Not supported file type {file_type}")
        if time is None:
            num_cells_global = hdf["Types"].size
            local_cell_range = compute_local_range(comm, num_cells_global)
            cell_types_local = hdf["Types"][slice(*local_cell_range)]

            num_points_global = hdf["NumberOfPoints"][0]
            local_point_range = compute_local_range(comm, num_points_global)
            points_local = hdf["Points"][slice(*local_point_range)]

            # Connectivity read
            offsets = hdf["Offsets"]
            local_connectivity_offset = offsets[local_cell_range[0] : local_cell_range[1] + 1]
            topology = hdf["Connectivity"][
                local_connectivity_offset[0] : local_connectivity_offset[-1]
            ]
            offset = local_connectivity_offset - local_connectivity_offset[0]
        else:
            if "Steps" not in hdf.keys():
                raise RuntimeError(f"No timestepping information found in {filename}.")
            stamps = hdf["Steps"]["Values"][:]
            pos = np.flatnonzero(np.isclose(stamps, time))
            if len(pos) == 0:
                raise RuntimeError(f"Could not find mesh at t={time} in {filename}.")
            elif len(pos) > 1:
                raise RuntimeError(f"Multiple time steps for mesh at t={time} in {filename}")

            # Get number of points
            point_node = hdf["Points"]
            step_start = hdf["Steps"]["PointOffsets"][pos[0]]

            # NOTE: currently, it doesn't seem like we follow:
            # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-unstructuredgrid-and-polydata
            # As only one num_points is stored irregardless of time steps added.
            if hdf["NumberOfPoints"].shape[0] != len(stamps):
                num_pts = hdf["NumberOfPoints"][0]
            else:
                num_pts = hdf["NumberOfPoints"][pos[0]]
            lr = compute_local_range(comm, num_pts)
            points_local = point_node[step_start + lr[0] : step_start + lr[1]]

            # Get cell-types in step
            cell_start = hdf["Steps"]["CellOffsets"][pos[0]]
            if hdf["NumberOfCells"].shape[0] != len(stamps):
                num_cells = hdf["NumberOfCells"][0]
            else:
                num_cells = hdf["NumberOfCells"][pos[0]]
            local_cell_range = compute_local_range(comm, num_cells)
            cell_types_local = hdf["Types"][
                cell_start + local_cell_range[0] : cell_start + local_cell_range[1]
            ]

            # Get connectivity in step
            connectivity_start = hdf["Steps"]["ConnectivityIdOffsets"][pos[0]]
            # Connectivity read
            offsets = hdf["Offsets"]
            local_connectivity_offset = offsets[
                connectivity_start + local_cell_range[0] : connectivity_start
                + local_cell_range[1]
                + 1
            ]
            topology = hdf["Connectivity"][
                local_connectivity_offset[0] : local_connectivity_offset[-1]
            ]
            offset = local_connectivity_offset - local_connectivity_offset[0]

    # NOTE: Currently we limit ourselfs to a single celltype, as it makes life easier,
    # other things have to change in `MeshReadData` to support this.
    num_nodes_per_cell = offset[1:] - offset[:-1]
    unique_cells = find_all_unique_cell_types(MPI.COMM_WORLD, cell_types_local, num_nodes_per_cell)
    if unique_cells.shape[0] > 1:
        raise NotImplementedError("adios4dolfinx does not support mixed celltype grids")
    topology = topology.reshape(-1, num_nodes_per_cell[0])
    cell_type, number_of_nodes = unique_cells[0]
    gtype = backend_args.get("dtype", points_local.dtype)
    if cell_type in _first_order_vtk.keys():
        ct = _first_order_vtk[cell_type]
        degree = 1
    elif cell_type in _arbitrary_lagrange_vtk.keys():
        ct = _arbitrary_lagrange_vtk[cell_type]
        degree = _cell_degree(ct, num_nodes=number_of_nodes)
    else:
        raise ValueError(f"Unknown VTK cell type {cell_type} in {filename}")
    perm = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(ct), number_of_nodes)
    topology = topology[:, perm]
    lvar = int(basix.LagrangeVariant.equispaced)
    return ReadMeshData(
        cells=topology, cell_type=ct, x=points_local.astype(gtype), lvar=lvar, degree=degree
    )


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: float | str | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read data from the nodes of a mesh.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments

    Returns:
       Data local to process (contiguous, no mpi comm) and local start range
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        if "PointData" not in hdf.keys():
            raise ValueError(f"No point data found in {filename}.")
        point_data = hdf["PointData"]
        assert point_data is not None
        if name not in point_data.keys():
            raise ValueError(f"No point data named {name} in {filename}.")
        func_node = point_data[name]

        if time is None:
            data_shape = func_node.shape[0]
            lr = compute_local_range(comm, data_shape)
            data = func_node[slice(*lr)]
            return data, lr[0]
        else:
            if "Steps" not in hdf.keys():
                raise RuntimeError(f"No timestepping information found in {filename}.")
            stamps = hdf["Steps"]["Values"][:]
            pos = np.flatnonzero(np.isclose(stamps, time))
            if len(pos) == 0:
                raise RuntimeError(f"Could not find {name}(t={time}) in {filename}.")
            elif len(pos) > 1:
                raise RuntimeError(f"Multiple time steps for {name}(t={time}) in {filename}")
            step_start = hdf["Steps"]["PointDataOffsets"][name][pos[0]]
            # NOTE: currently, it doesn't seem like we follow:
            # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-unstructuredgrid-and-polydata
            # As only one num_points is stored irregardless of time steps added.
            if hdf["NumberOfPoints"].shape[0] != len(stamps):
                num_pts = hdf["NumberOfPoints"][0]
            else:
                num_pts = hdf["NumberOfPoints"][pos[0]]
            lr = compute_local_range(comm, num_pts)
            return func_node[step_start + lr[0] : step_start + lr[1]], lr[0]


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        if "CellData" not in hdf.keys():
            raise RuntimeError(f"No cell data found in {filename}.")
        cell_data = hdf["CellData"]
        assert cell_data is not None
        if name not in cell_data.keys():
            raise ValueError(f"No cell data with name {name} in {filename}")
        cell_data_node = cell_data[name]
        assert cell_data_node is not None
        if time is None:
            cell_data_shape = cell_data_node.shape
            num_cells_global = hdf["Types"].size
            assert num_cells_global == cell_data_shape[0]
            local_cell_range = compute_local_range(comm, num_cells_global)
            data = cell_data_node[slice(*local_cell_range)]
        else:
            if "Steps" not in hdf.keys():
                raise RuntimeError(f"No timestepping information found in {filename}.")
            stamps = hdf["Steps"]["Values"][:]
            pos = np.flatnonzero(np.isclose(stamps, time))
            if len(pos) == 0:
                raise RuntimeError(f"Could not find {name}(t={time}) in {filename}.")
            elif len(pos) > 1:
                raise RuntimeError(f"Multiple time steps for {name}(t={time}) in {filename}")
            cd_start = hdf["Steps"]["CellDataOffsets"][name][pos[0]]

            # NOTE: currently, it doesn't seem like we follow:
            # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-unstructuredgrid-and-polydata
            # As only one num_points is stored irregardless of time steps added.
            if hdf["NumberOfCells"].shape[0] != len(stamps):
                number_of_cells = hdf["NumberOfCells"][0]
            else:
                number_of_cells = hdf["NumberOfCells"][pos[0]]
            lr = compute_local_range(comm, number_of_cells)
            data = cell_data_node[cd_start + lr[0] : cd_start + lr[1]]

    # NOTE: THis could be optimized by hand-coding some communication in
    # `read_cell_data` on the frontend side
    md = read_mesh_data(filename, comm, time=time, read_from_partition=False, backend_args=None)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    return md.cells, data


def write_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None,
):
    """Write attributes to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attribute group
        attributes: Dictionary of attributes to write
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot write attributes.")


def read_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read attributes from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attribute group
        backend_args: Arguments to backend

    Returns:
        Dictionary of attributes read from file
    """
    raise NotImplementedError("The Pyvista backend cannot read attributes.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read timestamps from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        function_name: Name of the function to read timestamps for
        backend_args: Arguments to backend

    Returns:
        Numpy array of timestamps read from file
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    # Temporal data storage as described in
    # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-data
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        if "Steps" in hdf.keys():
            timestamps = hdf["Steps"]["Values"][:]
            # For either point data or cell data, check if function exists,
            # and check if offsets in time are changing between steps.
            if "CellData" in hdf.keys() and function_name in hdf["CellData"].keys():
                offsets = hdf["Steps"]["CellDataOffsets"][function_name]
                step_offsets = offsets[:]
                step_diff = np.flatnonzero(step_offsets[1:] - step_offsets[:-1])
                return timestamps[step_diff]
            elif "PointData" in hdf.keys() and function_name in hdf["PointData"].keys():
                offsets = hdf["Steps"]["PointDataOffsets"][function_name]
                step_offsets = offsets[:]
                step_diff = np.flatnonzero(step_offsets[1:] - step_offsets[:-1])
                # This only finds when the offset changes, does not capture first step
                return np.hstack([[timestamps[0]], timestamps[step_diff]])
            else:
                raise RuntimeError(f"Function {function_name} is not assoicated with a time-stamp.")
        else:
            raise RuntimeError(f"{filename} does not contain time-stepping information.")


def read_function_names(
    filename: Path | str, comm: MPI.Intracomm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read all function names from a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        A list of function names.
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        function_names = set()
        if "CellData" in hdf.keys():
            for item in hdf["CellData"].keys():
                function_names.add(item)
        if "PointData" in hdf.keys():
            for item in hdf["PointData"].keys():
                function_names.add(item)
    return list(function_names)


def _create_dataset(
    root: h5py.File | h5py.Group,
    name: str,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    chunks: bool,
    maxshape: tuple[int, ...],
    mode: str,
    resize: bool = True,
) -> h5py.Dataset:
    if mode == "w" or (mode == "a" and name not in root.keys()):
        dataset = root.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, maxshape=maxshape
        )
    elif mode == "a":
        dataset = root[name]
        old_shape = dataset.shape
        # Only resize for dimension
        if resize:
            if len(old_shape) == 1:
                new_shape = (old_shape[0] + shape[0],)
            else:
                new_shape = (old_shape[0] + shape[0], *old_shape[1:])
            dataset.resize(new_shape)
    else:
        raise ValueError(f"Unknown file mode '{mode}' when creating dataset {name} in {root}")
    return dataset


def _create_group(root: h5py.File | h5py.Group, name: str, mode: str) -> h5py.Group:
    if mode == "w" or (mode == "a" and name not in root.keys()):
        # Track order has to be on to make multiblock work:
        # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#partitioneddatasetcollection-and-multiblockdataset
        group = root.create_group(name, track_order=True)
    elif mode == "a":
        group = root[name]
    else:
        raise ValueError("Unknown file mode '{h5_mode}'")
    return group


def _compute_append_slice(
    dataset: h5py.Dataset, input_size: int, original_slice: tuple[int, int] | np.ndarray, mode: str
) -> slice:
    append_offset = dataset.shape[0] - input_size if mode == "a" else 0
    return slice(*(np.asarray(original_slice) + append_offset).astype(np.int64))


def write_mesh(
    filename: Path | str,
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None,
    mode: FileMode,
    time: float,
):
    """
    Write a mesh to file.

    Args:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        backend_args: Arguments to backend
        mode: File-mode to store the mesh
        time: Time stamp associated with the mesh
    """
    h5_mode = convert_file_mode(mode)
    backend_args = get_default_backend_args(backend_args)
    name = backend_args["name"]
    with h5pyfile(filename, h5_mode, comm=comm) as h5file:
        hdf = _create_group(h5file, "/VTKHDF", h5_mode)
        hdf.attrs.create("Type", "MultiBlockDataSet")
        hdf.attrs["Version"] = np.array([2, 1], dtype=np.int32)

        mesh_group = _create_group(hdf, name, h5_mode)
        mesh_group.attrs.create("Type", "UnstructuredGrid")

        mesh_group.attrs["Version"] = np.array([2, 1], dtype=np.int32)

        assembly = _create_group(hdf, "Assembly", h5_mode)

        mesh_assembly = _create_group(assembly, name, h5_mode)
        if name not in mesh_assembly.keys():
            mesh_assembly[name] = h5py.SoftLink(f"/VTKHDF/{name}")

        # Partition split. We use no partitioning
        num_cells = _create_dataset(
            mesh_group,
            "NumberOfCells",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Resize should really be True, see issue below
        )  # VTKHDFReader issue: https://gitlab.kitware.com/vtk/vtk/-/issues/19257
        num_cells[-1] = mesh.num_cells_global

        number_of_points = _create_dataset(
            mesh_group,
            "NumberOfPoints",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        number_of_points[-1] = mesh.num_nodes_global

        # Single celltype assumption
        num_dofs_per_cell = mesh.local_topology.shape[1]
        number_of_connectivities = _create_dataset(
            mesh_group,
            "NumberOfConnectivityIds",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Resize should really be True, see issue below
        )  # VTKHDFReader issue: https://gitlab.kitware.com/vtk/vtk/-/issues/19257
        number_of_connectivities[-1] = mesh.num_cells_global * num_dofs_per_cell

        # Store nodes
        points = _create_dataset(
            mesh_group,
            "Points",
            shape=(mesh.num_nodes_global, mesh.local_geometry.shape[1]),
            dtype=mesh.local_geometry.dtype,
            chunks=True,
            maxshape=(None, 3),
            mode=h5_mode,
        )
        insert_slice = _compute_append_slice(
            points, mesh.num_nodes_global, mesh.local_geometry_pos, h5_mode
        )
        points[insert_slice] = mesh.local_geometry

        # Store topology offsets (single celltype assumption)
        offsets = _create_dataset(
            mesh_group,
            "Offsets",
            shape=(mesh.num_cells_global + 1,),
            dtype=np.int64,
            chunks=True,
            mode=h5_mode,
            maxshape=(None,),
            resize=False,  # Resize should really be True, see issue below
        )  # VTKHDFReader issue: https://gitlab.kitware.com/vtk/vtk/-/issues/19257
        offset_data = np.arange(0, mesh.local_topology.size + 1, mesh.local_topology.shape[1])
        offset_data += num_dofs_per_cell * mesh.local_topology_pos[0]
        insert_slice = _compute_append_slice(
            offsets,
            mesh.num_cells_global + 1,
            (mesh.local_topology_pos[0], mesh.local_topology_pos[1] + 1),
            mode=h5_mode,
        )
        offsets[insert_slice] = offset_data
        del offset_data

        # Permute and store topology data
        dx_ct = dolfinx.mesh.to_type(mesh.cell_type)
        top_perm = np.argsort(dolfinx.cpp.io.perm_vtk(dx_ct, num_dofs_per_cell))
        topology_data = mesh.local_topology[:, top_perm].flatten()
        topology = _create_dataset(
            mesh_group,
            "Connectivity",
            shape=(mesh.num_cells_global * num_dofs_per_cell,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Resize should really be True, see issue below
        )  # VTKHDFReader issue: https://gitlab.kitware.com/vtk/vtk/-/issues/19257
        insert_slice = _compute_append_slice(
            topology,
            mesh.num_cells_global * num_dofs_per_cell,
            np.array(mesh.local_topology_pos) * num_dofs_per_cell,
            mode=h5_mode,
        )
        topology[insert_slice] = topology_data
        del topology_data

        # Store celltypes
        cell_types = np.full(
            mesh.local_topology.shape[0],
            dolfinx.cpp.io.get_vtk_cell_type(dx_ct, dolfinx.mesh.cell_dim(dx_ct)),
        )
        types = _create_dataset(
            mesh_group,
            "Types",
            shape=(mesh.num_cells_global,),
            dtype=np.uint8,
            maxshape=(None,),
            chunks=True,
            mode=h5_mode,
            resize=False,  # Resize should really be True, see issue below
        )  # VTKHDFReader issue: https://gitlab.kitware.com/vtk/vtk/-/issues/19257
        insert_slice = _compute_append_slice(
            types, mesh.num_cells_global, mesh.local_topology_pos, h5_mode
        )
        types[insert_slice] = cell_types
        del cell_types

        steps = _create_group(mesh_group, "Steps", mode=h5_mode)
        # First fetch time-steps to see if we have stored this timestep already
        if h5_mode == "w":
            values = _create_dataset(
                steps,
                "Values",
                shape=(1,),
                dtype=np.float64,
                chunks=True,
                maxshape=(None,),
                mode="w",
                resize=False,
            )
            values[0] = time
        else:
            values = _create_dataset(
                steps,
                "Values",
                shape=(1,),
                dtype=np.float64,
                chunks=True,
                maxshape=(None,),
                mode="a",
                resize=False,
            )
            existing_steps = values[:]
            if len(np.flatnonzero(np.isclose(existing_steps, time))) > 0:
                raise RuntimeError(f"Mesh already exists at time {time} in {filename}.")
            values = _create_dataset(
                steps,
                "Values",
                shape=(1,),
                dtype=np.float64,
                chunks=True,
                maxshape=(None,),
                mode="a",
                resize=True,
            )
            values[-1] = time
        steps.attrs.create("NSteps", np.int64(len(values)), dtype=np.int64)

        # Write single partition data
        num_parts = _create_dataset(
            steps,
            "NumberOfParts",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        num_parts[-1] = 1
        part_offset = _create_dataset(
            steps,
            "PartOffsets",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        part_offset[-1] = 0

        # Create offsets for data
        point_offset = _create_dataset(
            steps,
            "PointOffsets",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        point_offset[-1] = points.shape[0] - mesh.num_nodes_global
        cell_offset = _create_dataset(
            steps,
            "CellOffsets",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        cell_offset[-1] = types.shape[0] - mesh.num_cells_global

        connectivity_offsets = _create_dataset(
            steps,
            "ConnectivityIdOffsets",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        connectivity_offsets[-1] = offsets.shape[0] - (mesh.num_cells_global + 1)


def write_meshtags(
    filename: str | Path,
    comm: MPI.Intracomm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None,
):
    """Write mesh tags to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        data: Internal data structure for the mesh tags to save to file
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot write meshtags.")


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> MeshTagsData:
    """Read mesh tags from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the mesh tags to read
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh tags read from file
    """
    raise NotImplementedError("The Pyvista backend cannot read meshtags.")


def read_dofmap(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dolfinx.graph.AdjacencyList:
    """Read the dofmap of a function with a given name.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofmap for
        backend_args: Arguments to backend

    Returns:
        Dofmap as an AdjacencyList
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_dofs(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    time: float,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
    """Read the dofs (values) of a function with a given name from a given timestep.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofs for
        time: Time stamp associated with the function to read
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of degrees of freedom (with respect to input data)
        and the global starting point on the process.
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_cell_perms(
    comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None
) -> npt.NDArray[np.uint32]:
    """
    Read cell permutation from file with given communicator,
    Split in continuous chunks based on number of cells in the input data.

    Args:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of permutations (with respect to input data)
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def write_function(
    filename: Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """
    Write a function to file.

    Args:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        time: Time stamp associated with function
        mode: File-mode to store the function
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Intracomm, group: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], str | None]:
    """Read in the mesh topology, geometry and (optionally) cell type from a
    legacy DOLFIN HDF5-file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        group: Group in HDF5 file where mesh is stored

    Returns:
        Tuple containing:
            - Topology as a (num_cells, num_vertices_per_cell) array of global vertex indices
            - Geometry as a (num_vertices, geometric_dimension) array of vertex coordinates
            - Cell type as a string (e.g. "tetrahedron") or None if not found
    """
    raise NotImplementedError("The Pyvista backend cannot read legacy DOLFIN meshes.")


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    """Create a snapshot checkpoint of a dolfinx function.

    Args:
        filename: Path to file to read from
        mode: File-mode to store the function
        u: dolfinx function to create a snapshot checkpoint for
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_hdf5_array(
    comm: MPI.Intracomm,
    filename: Path | str,
    group: str,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read an array from an HDF5 file.

    Args:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        group: Group in HDF5 file where array is stored
        backend_args: Arguments to backend

    Returns:
        Tuple containing:
            - Numpy array read from file
            - Global starting point on the process.
                Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The Pyvista backend cannot read HDF5 arrays")
