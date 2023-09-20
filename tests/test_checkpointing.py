import itertools

import basix
import basix.ufl
import dolfinx
import numpy as np
import pytest
from mpi4py import MPI

from .test_utils import read_function, write_function, get_dtype

dtypes = [np.float64, np.float32]  # Mesh geometry dtypes
write_comm = [MPI.COMM_SELF, MPI.COMM_WORLD]  # Communicators for creating mesh

two_dimensional_cell_types = [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
three_dimensional_cell_types = [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]

two_dim_combinations = itertools.product(dtypes, two_dimensional_cell_types, write_comm)
three_dim_combinations = itertools.product(dtypes, three_dimensional_cell_types, write_comm)


@pytest.fixture(params=two_dim_combinations, scope="module")
def mesh_2D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.fixture(params=three_dim_combinations, scope="module")
def mesh_3D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 5, 5, 5, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_2D(read_comm, family, degree, complex, mesh_2D):
    mesh = mesh_2D
    f_dtype = get_dtype(mesh.geometry.x.dtype, complex)

    el = basix.ufl.element(family,
                           mesh.ufl_cell().cellname(),
                           degree,
                           basix.LagrangeVariant.gll_warped,
                           gdim=mesh.geometry.dim,
                           shape=(mesh.geometry.dim, ))

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0] + x[1] * 1j
        values[1] = x[0] + 3j * x[1]
        return values

    hash = write_function(mesh, el, f, f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, f_dtype)


@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_3D(read_comm, family, degree, complex, mesh_3D):
    mesh = mesh_3D
    f_dtype = get_dtype(mesh.geometry.x.dtype, complex)
    el = basix.ufl.element(family,
                           mesh.ufl_cell().cellname(),
                           degree,
                           basix.LagrangeVariant.gll_warped,
                           gdim=mesh.geometry.dim,
                           shape=(mesh.geometry.dim, ))

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.pi + x[0] + 2j*x[2]
        values[1] = x[1] + 2 * x[0]
        values[2] = 1j*x[1] + np.cos(x[2])
        return values

    hash = write_function(mesh, el, f, f_dtype)

    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, f_dtype)
