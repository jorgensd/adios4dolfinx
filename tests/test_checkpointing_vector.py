import itertools

from mpi4py import MPI

import basix
import basix.ufl
import dolfinx
import numpy as np
import pytest

dtypes = [np.float64, np.float32]  # Mesh geometry dtypes
write_comm = [MPI.COMM_SELF, MPI.COMM_WORLD]  # Communicators for creating mesh


simplex_two_dim = itertools.product(dtypes, [dolfinx.mesh.CellType.triangle], write_comm)
simplex_three_dim = itertools.product(dtypes, [dolfinx.mesh.CellType.tetrahedron], write_comm)

non_simplex_two_dim = itertools.product(dtypes, [dolfinx.mesh.CellType.quadrilateral], write_comm)
non_simplex_three_dim = itertools.product(dtypes, [dolfinx.mesh.CellType.hexahedron], write_comm)


@pytest.fixture(params=simplex_two_dim, scope="module")
def simplex_mesh_2D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.fixture(params=simplex_three_dim, scope="module")
def simplex_mesh_3D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 5, 5, 5, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.fixture(params=non_simplex_two_dim, scope="module")
def non_simplex_mesh_2D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.fixture(params=non_simplex_three_dim, scope="module")
def non_simplex_mesh_3D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 5, 5, 5, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_2D(
    read_comm, family, degree, is_complex, simplex_mesh_2D, get_dtype, write_function, read_function
):
    mesh = simplex_mesh_2D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = x[1]
        if is_complex:
            values[0] += 2j * x[1]
            values[1] += 2j * x[0]
        return values

    fname = write_function(mesh, el, f, f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, fname, f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_3D(
    read_comm, family, degree, is_complex, simplex_mesh_3D, get_dtype, write_function, read_function
):
    mesh = simplex_mesh_3D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi)
        values[1] = x[1] + 2 * x[0]
        values[2] = np.cos(x[2])
        if is_complex:
            values[0] += 2j * x[2]
            values[1] += 2j * np.cos(x[2])
        return values

    fname = write_function(mesh, el, f, dtype=f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, fname, dtype=f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_2D_quad(
    read_comm,
    family,
    degree,
    is_complex,
    non_simplex_mesh_2D,
    get_dtype,
    write_function,
    read_function,
):
    mesh = non_simplex_mesh_2D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi)
        values[1] = x[1] + 2 * x[0]
        if is_complex:
            values[0] += 2j * x[2]
            values[1] += 2j * np.cos(x[2])
        return values

    hash = write_function(mesh, el, f, f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["NCF"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_hex(
    read_comm,
    family,
    degree,
    is_complex,
    non_simplex_mesh_3D,
    get_dtype,
    write_function,
    read_function,
):
    mesh = non_simplex_mesh_3D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = np.cos(x[2])
        values[2] = x[0]
        if is_complex:
            values[0] += 2j * x[2]
            values[2] -= 1j * x[1]
        return values

    hash = write_function(mesh, el, f, dtype=f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtype=f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_multiple(
    read_comm,
    family,
    degree,
    is_complex,
    non_simplex_mesh_2D,
    get_dtype,
    write_function,
    read_function,
):
    mesh = non_simplex_mesh_2D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi)
        values[1] = x[1] + 2 * x[0]
        if is_complex:
            values[0] -= 2j * x[2]
            values[1] += 2j * np.cos(x[2])
        return values

    def g(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = 2 * x[1]
        values[1] = 3 * x[0]
        if is_complex:
            values[0] += 3j * x[0]
            values[1] += 2j * x[0] * x[1]
        return values

    hash_f = write_function(mesh, el, f, dtype=f_dtype, name="f", append=False)
    hash_g = write_function(mesh, el, g, dtype=f_dtype, name="g", append=True)
    assert hash_f == hash_g

    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash_f, dtype=f_dtype, name="f")
    read_function(read_comm, el, g, hash_g, dtype=f_dtype, name="g")
