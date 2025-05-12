import itertools

from mpi4py import MPI

import basix
import basix.ufl
import dolfinx
import numpy as np
import pytest

import adios4dolfinx

dtypes = [np.float64, np.float32]  # Mesh geometry dtypes
write_comm = [MPI.COMM_SELF, MPI.COMM_WORLD]  # Communicators for creating mesh

two_dimensional_cell_types = [
    dolfinx.mesh.CellType.triangle,
    dolfinx.mesh.CellType.quadrilateral,
]
three_dimensional_cell_types = [
    dolfinx.mesh.CellType.tetrahedron,
    dolfinx.mesh.CellType.hexahedron,
]

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
    M = 5
    mesh = dolfinx.mesh.create_unit_cube(write_comm, M, M, M, cell_type=cell_type, dtype=dtype)
    return mesh


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_2D(
    read_comm, family, degree, is_complex, mesh_2D, get_dtype, write_function, read_function
):
    mesh = mesh_2D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)

    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = x[0]
        if is_complex:
            values[0] += 1j * x[1]
            values[1] -= 3j * x[1]
        return values

    hash = write_function(mesh, el, f, f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_3D(
    read_comm, family, degree, is_complex, mesh_3D, get_dtype, write_function, read_function
):
    mesh = mesh_3D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.pi + x[0]
        values[1] = x[1] + 2 * x[0]
        values[2] = np.cos(x[2])
        if is_complex:
            values[0] -= 2j * x[2]
            values[2] += 1j * x[1]
        return values

    hash = write_function(mesh, el, f, f_dtype)

    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_2D_time(
    read_comm,
    family,
    degree,
    is_complex,
    mesh_2D,
    get_dtype,
    write_function_time_dep,
    read_function_time_dep,
):
    mesh = mesh_2D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)

    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )

    def f0(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = x[0]
        if is_complex:
            values[0] += x[1] * 1j
            values[1] -= 3j * x[1]
        return values

    def f1(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = 2 * np.full(x.shape[1], np.pi) + x[0]
        values[1] = -x[0] + 2 * x[1]
        if is_complex:
            values[0] += x[1] * 1j
            values[1] += 3j * x[1]
        return values

    t0 = 0.8
    t1 = 0.6
    hash = write_function_time_dep(mesh, el, f0, f1, t0, t1, f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function_time_dep(read_comm, el, f0, f1, t0, t1, hash, f_dtype)


@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_3D_time(
    read_comm,
    family,
    degree,
    is_complex,
    mesh_3D,
    get_dtype,
    write_function_time_dep,
    read_function_time_dep,
):
    mesh = mesh_3D
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.pi + x[0]
        values[1] = x[1] + 2 * x[0]
        values[2] = np.cos(x[2])
        if is_complex:
            values[0] += 2j * x[2]
            values[2] += 5j * x[1]
        return values

    def g(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = x[0]
        values[1] = 2 * x[0]
        values[2] = x[0]
        if is_complex:
            values[0] += np.pi * 2j * x[2]
            values[1] += 1j * x[2]
            values[2] += 1j * np.cos(x[1])
        return values

    t0 = 0.1
    t1 = 1.3
    hash = write_function_time_dep(mesh, el, g, f, t0, t1, f_dtype)
    MPI.COMM_WORLD.Barrier()
    read_function_time_dep(read_comm, el, g, f, t0, t1, hash, f_dtype)


@pytest.mark.parametrize(
    "func, args",
    [
        (adios4dolfinx.read_attributes, ("nonexisting_file.bp", MPI.COMM_WORLD, "")),
        (adios4dolfinx.read_timestamps, ("nonexisting_file.bp", MPI.COMM_WORLD, "")),
        (adios4dolfinx.read_meshtags, ("nonexisting_file.bp", MPI.COMM_WORLD, None, "")),
        (adios4dolfinx.read_function, ("nonexisting_file.bp", None)),
        (adios4dolfinx.read_mesh, ("nonexisting_file.bp", MPI.COMM_WORLD)),
    ],
)
def test_read_nonexisting_file_raises_FileNotFoundError(func, args):
    with pytest.raises(FileNotFoundError):
        func(*args)


def test_read_function_with_invalid_name_raises_KeyError(tmp_path):
    comm = MPI.COMM_WORLD
    f_path = comm.bcast(tmp_path, root=0)
    filename = f_path / "func.bp"
    mesh = dolfinx.mesh.create_unit_square(comm, 10, 10, cell_type=dolfinx.mesh.CellType.triangle)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = dolfinx.fem.Function(V)
    adios4dolfinx.write_function(filename, u, time=0, name="some_name")
    adios4dolfinx.write_function(filename, u, time=0, name="some_other_name")
    variables = set(sorted(["some_name", "some_other_name"]))
    with pytest.raises(KeyError) as e:
        adios4dolfinx.read_function(filename, u, time=0, name="nonexisting_name")

    assert e.value.args[0] == (
        f"nonexisting_name not found in {filename}. Did you mean one of {variables}?"
    )


def test_read_timestamps(get_dtype, mesh_2D, tmp_path):
    mesh = mesh_2D
    dtype = get_dtype(mesh.geometry.x.dtype, False)

    el = basix.ufl.element(
        "Lagrange",
        mesh.ufl_cell().cellname(),
        1,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )
    V = dolfinx.fem.functionspace(mesh, el)

    u = dolfinx.fem.Function(V, dtype=dtype, name="u")
    v = dolfinx.fem.Function(V, dtype=dtype, name="v")

    f_path = mesh.comm.bcast(tmp_path, root=0)
    filename = f_path / "read_time_stamps.bp"

    t_u = [0.1, 1.4]
    t_v = [0.45, 1.2]

    adios4dolfinx.write_mesh(filename, mesh)
    adios4dolfinx.write_function(filename, u, time=t_u[0])
    adios4dolfinx.write_function(filename, v, time=t_v[0])
    adios4dolfinx.write_function(filename, u, time=t_u[1])
    adios4dolfinx.write_function(filename, v, time=t_v[1])

    timestamps_u = adios4dolfinx.read_timestamps(
        comm=mesh.comm, filename=filename, function_name="u"
    )
    timestamps_v = adios4dolfinx.read_timestamps(
        comm=mesh.comm, filename=filename, function_name="v"
    )

    assert np.allclose(timestamps_u, t_u)
    assert np.allclose(timestamps_v, t_v)
