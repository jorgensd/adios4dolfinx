"""
Module that uses pyvista to import grids.
"""

from typing import Any

import numpy as np

try:
    import pyvista
except ImportError:
    raise ModuleNotFoundError("This module requires pyvista")
from pathlib import Path

from mpi4py import MPI

import basix
import dolfinx

from adios4dolfinx.comm_helpers import send_dofs_and_recv_values
from adios4dolfinx.structures import ReadMeshData
from adios4dolfinx.utils import check_file_exists, index_owner

from .. import ReadMode

# Cell types can be found at
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
_first_order_vtk = {
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


def _cell_degree(ct: str, num_nodes: int):
    if ct == "point":
        return 1
    elif ct == "interval":
        return num_nodes - 1
    elif ct == "triangle":
        n = (np.sqrt(1 + 8 * num_nodes) - 1) / 2
        if 2 * num_nodes != n * (n + 1):
            raise ValueError(f"Unknown triangle layout. Number of nodes: {num_nodes}")
        return n - 1
    elif ct == "tetrahedron":
        n = 0
        while n * (n + 1) * (n + 2) < 6 * num_nodes:
            n += 1
        if n * (n + 1) * (n + 2) != 6 * num_nodes:
            raise ValueError(f"Unknown tetrahedron layout. Number of nodes: {num_nodes}")
        return n - 1

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

    Parameters:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {}
    return args


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: float,
    read_from_partition: bool,
    backend_args: dict[str, Any] | None,
) -> ReadMeshData:
    """Read mesh data from file.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh data read from file
    """
    check_file_exists(filename)
    if read_from_partition:
        raise RuntimeError("Cannot read partition data with Pyvista")
    if comm.rank == 0:
        grid = pyvista.read(filename)
        if isinstance(grid, pyvista.UnstructuredGrid):
            pass
        elif isinstance(grid, pyvista.core.composite.MultiBlock):
            # To handle multiblock like pvd
            pyvista._VTK_SNAKE_CASE_STATE = "allow"
            number_of_blocks = grid.number_of_blocks
            assert number_of_blocks == 1
            grid = grid.get_block(0)
        else:
            raise RuntimeError(f"Unknown data type {type(grid)}")
        geom = grid.points
        num_cells_global = grid.number_of_cells
        cells = grid.cells.reshape(num_cells_global, -1)
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
        order, lvar, nodes_per_cell, cell_type, gtype, gdim = comm.bcast(
            (order, lvar, cells.shape[1], cell_type, geom.dtype, geom.shape[1]), root=0
        )
    else:
        order, lvar, nodes_per_cell, cell_type, gtype, gdim = comm.bcast(None, root=0)
        geom = np.zeros((0, gdim), dtype=gtype)
        cells = np.zeros((0, nodes_per_cell), dtype=np.int64)

    return ReadMeshData(cells=cells, cell_type=cell_type, x=geom, lvar=lvar, degree=order)


def read_point_data(
    filename: Path | str, name: str, mesh: dolfinx.mesh.Mesh
) -> dolfinx.fem.Function:
    if MPI.COMM_WORLD.rank == 0:
        grid = pyvista.read(filename)
        if isinstance(grid, pyvista.UnstructuredGrid):
            pass
        elif isinstance(grid, pyvista.core.composite.MultiBlock):
            # To handle multiblock like pvd
            pyvista._VTK_SNAKE_CASE_STATE = "allow"
            number_of_blocks = grid.number_of_blocks
            assert number_of_blocks == 1
            grid = grid.get_block(0)
        dataset = grid.point_data[name]
        if len(dataset.shape) == 1:
            num_components = 1
            dataset = dataset.reshape(-1, num_components)
        else:
            num_components = dataset.shape[1]
        num_components, gtype = mesh.comm.bcast((num_components, dataset.dtype), root=0)
        local_range_start = 0
    else:
        num_components, gtype = mesh.comm.bcast(None, root=0)
        dataset = np.zeros((0, num_components), dtype=gtype)
        local_range_start = 0

    # NOTE: THe below should be moved out of backend.

    # Create appropriate function space (based on coordinate map)
    if num_components == 1:
        shape = ()
    else:
        shape = (num_components,)
    element = basix.ufl.element(
        basix.ElementFamily.P,
        mesh.topology.cell_name(),
        mesh.geometry.cmap.degree,
        mesh.geometry.cmap.variant,
        shape=shape,
        dtype=mesh.geometry.x.dtype,
    )

    # Assumption: Same doflayout for geometry and function space, cannot test in python
    V = dolfinx.fem.functionspace(mesh, element)
    uh = dolfinx.fem.Function(V, name=name, dtype=dataset.dtype)
    # Assume that mesh is first order for now
    x_dofmap = mesh.geometry.dofmap
    igi = np.array(mesh.geometry.input_global_indices, dtype=np.int64)

    # This is dependent on how the data is read in. If distributed equally this is correct
    global_geom_input = igi[x_dofmap]
    from adios4dolfinx.backends import get_backend

    backend_cls = get_backend("pyvista")
    if backend_cls.read_mode == ReadMode.parallel:
        num_nodes_global = mesh.geometry.index_map().size_global
        global_geom_owner = index_owner(mesh.comm, global_geom_input.reshape(-1), num_nodes_global)
    elif backend_cls.read_mode == ReadMode.serial:
        # This is correct if everything is read in on rank 0
        global_geom_owner = np.zeros(len(global_geom_input.flatten()), dtype=np.int32)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

    for i in range(num_components):
        arr_i = send_dofs_and_recv_values(
            global_geom_input.reshape(-1),
            global_geom_owner,
            mesh.comm,
            dataset[:, i],
            local_range_start,
        )
        dof_pos = x_dofmap.reshape(-1) * num_components + i
        uh.x.array[dof_pos] = arr_i

    return uh
