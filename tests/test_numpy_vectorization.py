from mpi4py import MPI
import basix.ufl
import dolfinx
import numpy as np
import itertools
import pytest
from adios4dolfinx.utils import unroll_dofmap

write_comm = [MPI.COMM_SELF, MPI.COMM_WORLD]  # Communicators for creating mesh
two_dimensional_cell_types = [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
three_dimensional_cell_types = [dolfinx.mesh.CellType.hexahedron]

two_dim_combinations = itertools.product(two_dimensional_cell_types, write_comm)
three_dim_combinations = itertools.product(three_dimensional_cell_types, write_comm)


@pytest.fixture(params=two_dim_combinations, scope="module")
def mesh_2D(request):
    cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type)
    return mesh


@pytest.fixture(params=three_dim_combinations, scope="module")
def mesh_3D(request):
    cell_type, write_comm = request.param
    M = 5
    mesh = dolfinx.mesh.create_unit_cube(write_comm, M, M, M, cell_type=cell_type)
    return mesh


@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
def test_unroll_P(family, degree, mesh_2D):

    V = dolfinx.fem.functionspace(mesh_2D, (family, degree))
    dofmap = V.dofmap

    unrolled_map = unroll_dofmap(dofmap.list, dofmap.bs)

    normal_unroll = np.zeros((dofmap.list.shape[0], dofmap.list.shape[1] * dofmap.bs),
                             dtype=np.int32)
    for i, dofs in enumerate(dofmap.list):
        for j, dof in enumerate(dofs):
            for k in range(dofmap.bs):
                normal_unroll[i, j * dofmap.bs+k] = dof * dofmap.bs + k

    np.testing.assert_allclose(unrolled_map, normal_unroll)


@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_unroll_RTCF(family, degree, mesh_3D):
    el = basix.ufl.element(family,
                           mesh_3D.ufl_cell().cellname(),
                           degree,
                           gdim=mesh_3D.geometry.dim)

    V = dolfinx.fem.functionspace(mesh_3D, el)
    dofmap = V.dofmap

    unrolled_map = unroll_dofmap(dofmap.list, dofmap.bs)

    normal_unroll = np.zeros((dofmap.list.shape[0], dofmap.list.shape[1] * dofmap.bs),
                             dtype=np.int32)
    for i, dofs in enumerate(dofmap.list):
        for j, dof in enumerate(dofs):
            for k in range(dofmap.bs):
                normal_unroll[i, j * dofmap.bs+k] = dof * dofmap.bs + k

    np.testing.assert_allclose(unrolled_map, normal_unroll)
