"""
Module that uses pyvista to import grids.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import pyvista
except ImportError:
    raise ModuleNotFoundError("This module requires pyvista")
from pathlib import Path

from mpi4py import MPI

import basix
import dolfinx

from adios4dolfinx.structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from adios4dolfinx.utils import check_file_exists

from .. import FileMode, ReadMode

# Cell types can be found at
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
_first_order_vtk = {
    1: "point",
    3: "interval",
    5: "triangle",
    9: "quadrilateral",
    10: "tetrahedron",
    12: "hexahedron",
}

_arbitrary_lagrange_vtk = {
    68: "interval",
    69: "triangle",
    70: "quadrilateral",
    71: "tetrahedron",
    72: "hexahedron",
    73: "prism",
    74: "pyramid",
}


read_mode = ReadMode.serial


def _cell_degree(ct: str, num_nodes: int) -> int:
    if ct == "point":
        return 1
    elif ct == "interval":
        return int(num_nodes - 1)
    elif ct == "triangle":
        n = (np.sqrt(1 + 8 * num_nodes) - 1) / 2
        if 2 * num_nodes != n * (n + 1):
            raise ValueError(f"Unknown triangle layout. Number of nodes: {num_nodes}")
        return int(n - 1)
    elif ct == "tetrahedron":
        n = 0
        while n * (n + 1) * (n + 2) < 6 * num_nodes:
            n += 1
        if n * (n + 1) * (n + 2) != 6 * num_nodes:
            raise ValueError(f"Unknown tetrahedron layout. Number of nodes: {num_nodes}")
        return int(n - 1)

    elif ct == "quadrilateral":
        n = np.sqrt(num_nodes)
        if num_nodes != n * n:
            raise ValueError(f"Unknown quadrilateral layout. Number of nodes: {num_nodes}")
        return int(n - 1)
    elif ct == "hexahedron":
        n = np.cbrt(num_nodes)
        if num_nodes != n * n * n:
            raise ValueError(f"Unknown hexahedron layout. Number of nodes: {num_nodes}")
        return int(n - 1)
    elif ct == "prism":
        if num_nodes == 6:
            return 1
        elif num_nodes == 15:
            return 2
        else:
            raise ValueError(f"Unknown prism layout. Number of nodes: {num_nodes}")
    elif ct == "pyramid":
        if num_nodes == 5:
            return 1
        elif num_nodes == 13:
            return 2
        else:
            raise ValueError(f"Unknown pyramid layout. Number of nodes: {num_nodes}")
    else:
        raise ValueError(f"Unknown cell type {ct} with {num_nodes=}.")


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {}
    return args


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: str | float | None = None,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
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
        raise RuntimeError("Cannot read partition data with Pyvista")
    cells: npt.NDArray[np.int64]
    geom: npt.NDArray[np.float64 | np.float32]
    if comm.rank == 0:
        in_data = pyvista.read(filename)
        if isinstance(in_data, pyvista.UnstructuredGrid):
            grid = in_data
        elif isinstance(in_data, pyvista.core.composite.MultiBlock):
            # To handle multiblock like pvd
            pyvista._VTK_SNAKE_CASE_STATE = "allow"
            number_of_blocks = in_data.number_of_blocks
            assert number_of_blocks == 1
            b0 = in_data.get_block(0)
            assert isinstance(b0, pyvista.UnstructuredGrid)
            grid = b0
        else:
            raise RuntimeError(f"Unknown data type {type(in_data)}")
        geom = grid.points
        num_cells_global = grid.number_of_cells
        cells = grid.cells.reshape(num_cells_global, -1).astype(np.int64)
        nodes_per_cell_type = cells[:, 0]
        assert np.allclose(nodes_per_cell_type, nodes_per_cell_type[0]), "Single celltype support"
        cells = cells[:, 1:].astype(np.int64)
        cell_types = grid.celltypes
        assert len(np.unique(cell_types)) == 1
        if (cell_type := cell_types[0]) in _first_order_vtk.keys():
            cell_type = _first_order_vtk[cell_type]
            order = 1
        elif cell_type in _arbitrary_lagrange_vtk.keys():
            cell_type = _arbitrary_lagrange_vtk[cell_type]
            order = _cell_degree(cell_type, cells.shape[1])
        perm = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(cell_type), cells.shape[1])
        cells = cells[:, perm]
        lvar = int(basix.LagrangeVariant.equispaced)
        gtype = backend_args.get("dtype", geom.dtype)
        order, lvar, nodes_per_cell, cell_type, gtype, gdim = comm.bcast(
            (order, lvar, cells.shape[1], cell_type, gtype, geom.shape[1]), root=0
        )
    else:
        order, lvar, nodes_per_cell, cell_type, gtype, gdim = comm.bcast(None, root=0)
        geom = np.zeros((0, gdim), dtype=gtype)
        cells = np.zeros((0, nodes_per_cell), dtype=np.int64)

    return ReadMeshData(
        cells=cells, cell_type=cell_type, x=geom.astype(gtype), lvar=lvar, degree=order
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
    dataset: np.ndarray
    if MPI.COMM_WORLD.rank == 0:
        in_data = pyvista.read(filename)
        if isinstance(in_data, pyvista.UnstructuredGrid):
            grid = in_data
        elif isinstance(in_data, pyvista.core.composite.MultiBlock):
            # To handle multiblock like pvd
            pyvista._VTK_SNAKE_CASE_STATE = "allow"
            number_of_blocks = in_data.number_of_blocks
            assert number_of_blocks == 1
            b0 = in_data.get_block(0)
            assert isinstance(b0, pyvista.UnstructuredGrid)
            grid = b0

        dataset = grid.point_data[name]
        if len(dataset.shape) == 1:
            num_components = 1
            dataset = dataset.reshape(-1, num_components)
        else:
            num_components = dataset.shape[1]
        if np.issubdtype(dataset.dtype, np.integer):
            gtype = in_data.points.dtype
            dataset = dataset.astype(gtype)
        else:
            gtype = in_data.dtype
        num_components, gtype = comm.bcast((num_components, gtype), root=0)
        local_range_start = 0
    else:
        num_components, gtype = comm.bcast(None, root=0)
        dataset = np.zeros((0, num_components), dtype=gtype)
        local_range_start = 0

    return dataset, int(local_range_start)


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    dataset: np.ndarray
    topology: np.ndarray
    if MPI.COMM_WORLD.rank == 0:
        in_data = pyvista.read(filename)
        if isinstance(in_data, pyvista.UnstructuredGrid):
            grid = in_data
        elif isinstance(in_data, pyvista.core.composite.MultiBlock):
            # To handle multiblock like pvd
            pyvista._VTK_SNAKE_CASE_STATE = "allow"
            number_of_blocks = in_data.number_of_blocks
            assert number_of_blocks == 1
            b0 = in_data.get_block(0)
            assert isinstance(b0, pyvista.UnstructuredGrid)
            grid = b0

        dataset = grid.cell_data[name]
        if len(dataset.shape) == 1:
            num_components = 1
            dataset = dataset.reshape(-1, num_components)
        else:
            num_components = dataset.shape[1]

        if np.issubdtype(dataset.dtype, np.integer):
            gtype = in_data.points.dtype
            dataset = dataset.astype(gtype)
        else:
            gtype = dataset.dtype
        num_components, gtype = comm.bcast((num_components, gtype), root=0)
    else:
        num_components, gtype = comm.bcast(None, root=0)
        dataset = np.zeros((0, num_components), dtype=gtype)
    _time = float(time) if time is not None else None
    topology = read_mesh_data(filename, comm, _time, False, backend_args=None).cells
    return topology, dataset


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
    raise NotImplementedError("The Pyvista backend cannot read timestamps.")


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
    in_data = pyvista.read(filename)
    if isinstance(in_data, pyvista.UnstructuredGrid):
        grid = in_data
    elif isinstance(in_data, pyvista.core.composite.MultiBlock):
        # To handle multiblock like pvd
        pyvista._VTK_SNAKE_CASE_STATE = "allow"
        number_of_blocks = in_data.number_of_blocks
        assert number_of_blocks == 1
        b0 = in_data.get_block(0)
        assert isinstance(b0, pyvista.UnstructuredGrid)
        grid = b0

    point_data = list(grid.point_data.keys())
    cell_data = list(grid.cell_data.keys())
    return point_data + cell_data


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
    raise NotImplementedError("The Pyvista backend cannot write meshes.")


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


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Intracomm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a 2D-array to file (distributed across proceses with MPI).

    Args:
        filename: Path to file
        array_data: Data to write to file
        comm: MPI communicator to open the file with
        time: Time stamp
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("The pyvista backend does not support writing point data")
