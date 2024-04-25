import itertools
from typing import Tuple

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt
import pytest

from adios4dolfinx.utils import compute_dofmap_pos, unroll_dofmap

write_comm = [MPI.COMM_SELF, MPI.COMM_WORLD]  # Communicators for creating mesh
ghost_mode = [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
two_dimensional_cell_types = [
    dolfinx.mesh.CellType.triangle,
    dolfinx.mesh.CellType.quadrilateral,
]
three_dimensional_cell_types = [dolfinx.mesh.CellType.hexahedron]

two_dim_combinations = itertools.product(two_dimensional_cell_types, write_comm, ghost_mode)
three_dim_combinations = itertools.product(three_dimensional_cell_types, write_comm, ghost_mode)


@pytest.fixture(params=two_dim_combinations, scope="module")
def mesh_2D(request):
    cell_type, write_comm, ghost_mode = request.param
    mesh = dolfinx.mesh.create_unit_square(
        write_comm, 10, 10, cell_type=cell_type, ghost_mode=ghost_mode
    )
    return mesh


@pytest.fixture(params=three_dim_combinations, scope="module")
def mesh_3D(request):
    cell_type, write_comm, ghost_mode = request.param
    M = 5
    mesh = dolfinx.mesh.create_unit_cube(
        write_comm, M, M, M, cell_type=cell_type, ghost_mode=ghost_mode
    )
    return mesh


def compute_positions(
    dofs: npt.NDArray[np.int32],
    dofmap_bs: int,
    num_owned_dofs: int,
    num_owned_cells: int,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Support function for test.
    Given a dofmap, compute the local cell and position in the dofmap for each owned dof.
    The last cell (wrt) local index will be the one in the output map
    """
    dof_to_cell_map = np.zeros(num_owned_dofs, dtype=np.int32)
    dof_to_pos_map = np.zeros(num_owned_dofs, dtype=np.int32)
    for c in range(num_owned_cells):
        for i, dof in enumerate(dofs[c]):
            for b in range(dofmap_bs):
                local_dof = dof * dofmap_bs + b
                if local_dof < num_owned_dofs:
                    dof_to_cell_map[local_dof] = c
                    dof_to_pos_map[local_dof] = i * dofmap_bs + b
    return dof_to_cell_map, dof_to_pos_map


@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
def test_unroll_P(family, degree, mesh_2D):
    V = dolfinx.fem.functionspace(mesh_2D, (family, degree))
    dofmap = V.dofmap

    unrolled_map = unroll_dofmap(dofmap.list, dofmap.bs)

    normal_unroll = np.zeros(
        (dofmap.list.shape[0], dofmap.list.shape[1] * dofmap.bs), dtype=np.int32
    )
    for i, dofs in enumerate(dofmap.list):
        for j, dof in enumerate(dofs):
            for k in range(dofmap.bs):
                normal_unroll[i, j * dofmap.bs + k] = dof * dofmap.bs + k

    np.testing.assert_allclose(unrolled_map, normal_unroll)


@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_unroll_RTCF(family, degree, mesh_3D):
    el = basix.ufl.element(family, mesh_3D.ufl_cell().cellname(), degree)

    V = dolfinx.fem.functionspace(mesh_3D, el)
    dofmap = V.dofmap

    unrolled_map = unroll_dofmap(dofmap.list, dofmap.bs)

    normal_unroll = np.zeros(
        (dofmap.list.shape[0], dofmap.list.shape[1] * dofmap.bs), dtype=np.int32
    )
    for i, dofs in enumerate(dofmap.list):
        for j, dof in enumerate(dofs):
            for k in range(dofmap.bs):
                normal_unroll[i, j * dofmap.bs + k] = dof * dofmap.bs + k

    np.testing.assert_allclose(unrolled_map, normal_unroll)


@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_compute_dofmap_pos_RTCF(family, degree, mesh_3D):
    el = basix.ufl.element(family, mesh_3D.ufl_cell().cellname(), degree)

    V = dolfinx.fem.functionspace(mesh_3D, el)
    local_cells, local_pos = compute_dofmap_pos(V)

    num_cells_local = mesh_3D.topology.index_map(mesh_3D.topology.dim).size_local
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    reference_cells, reference_pos = compute_positions(
        V.dofmap.list, V.dofmap.bs, num_dofs_local, num_cells_local
    )
    np.testing.assert_allclose(reference_cells, local_cells)
    np.testing.assert_allclose(reference_pos, local_pos)


@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
def test_compute_dofmap_pos_P(family, degree, mesh_2D):
    el = basix.ufl.element(family, mesh_2D.ufl_cell().cellname(), degree)

    V = dolfinx.fem.functionspace(mesh_2D, el)
    local_cells, local_pos = compute_dofmap_pos(V)

    num_cells_local = mesh_2D.topology.index_map(mesh_2D.topology.dim).size_local
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    reference_cells, reference_pos = compute_positions(
        V.dofmap.list, V.dofmap.bs, num_dofs_local, num_cells_local
    )
    np.testing.assert_allclose(reference_cells, local_cells)
    np.testing.assert_allclose(reference_pos, local_pos)


def test_compute_send_sizes():
    np.random.seed(42)
    N = 0
    M = 10
    num_data = 100

    # Set of ranks to recieve data
    dest_ranks = np.arange(N, M, dtype=np.int32)

    # Random data owners
    data_owners = np.random.randint(N, M, num_data).astype(np.int32)

    # Compute the number of data to send to each rank with loops
    out_size = np.zeros(len(dest_ranks), dtype=np.int32)
    for owner in data_owners:
        for j, rank in enumerate(dest_ranks):
            if owner == rank:
                out_size[j] += 1
                break

    process_pos_indicator = data_owners.reshape(-1, 1) == dest_ranks
    vectorized_out_size = np.count_nonzero(process_pos_indicator, axis=0)
    np.testing.assert_allclose(vectorized_out_size, out_size)
