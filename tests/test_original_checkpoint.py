from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from pathlib import Path

from mpi4py import MPI

import basix
import basix.ufl
import dolfinx
import numpy as np
import pytest

import adios4dolfinx

dtypes = [np.float64, np.float32]  # Mesh geometry dtypes

two_dimensional_cell_types = [
    dolfinx.mesh.CellType.triangle,
    dolfinx.mesh.CellType.quadrilateral,
]
three_dimensional_cell_types = [
    dolfinx.mesh.CellType.tetrahedron,
    dolfinx.mesh.CellType.hexahedron,
]

two_dim_combinations = itertools.product(dtypes, two_dimensional_cell_types)
three_dim_combinations = itertools.product(dtypes, three_dimensional_cell_types)


@pytest.fixture(scope="module")
def create_simplex_mesh_2D(tmp_path_factory):
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        10,
        10,
        cell_type=dolfinx.mesh.CellType.triangle,
        dtype=np.float64,
    )
    fname = tmp_path_factory.mktemp("output") / "original_mesh_2D_simplex.xdmf"
    fname = MPI.COMM_WORLD.bcast(fname, root=0)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def create_simplex_mesh_3D(tmp_path_factory):
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        5,
        5,
        5,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        dtype=np.float64,
    )
    fname = tmp_path_factory.mktemp("output") / "original_mesh_3D_simplex.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def create_non_simplex_mesh_2D(tmp_path_factory):
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        10,
        10,
        cell_type=dolfinx.mesh.CellType.quadrilateral,
        dtype=np.float64,
    )
    fname = tmp_path_factory.mktemp("output") / "original_mesh_2D_non_simplex.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def create_non_simplex_mesh_3D(tmp_path_factory):
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        5,
        5,
        5,
        cell_type=dolfinx.mesh.CellType.hexahedron,
        dtype=np.float64,
    )
    fname = tmp_path_factory.mktemp("output") / "original_mesh_3D_non_simplex.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(params=two_dim_combinations, scope="module")
def create_2D_mesh(request, tmpdir_factory):
    dtype, cell_type = request.param
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 7, cell_type=cell_type, dtype=dtype)
    fname = Path(tmpdir_factory.mktemp("output")) / f"original_mesh_2D_{dtype}_{cell_type}.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(params=three_dim_combinations, scope="module")
def create_3D_mesh(request, tmpdir_factory):
    dtype, cell_type = request.param
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 3, cell_type=cell_type, dtype=dtype)
    fname = Path(tmpdir_factory.mktemp("output")) / f"original_mesh_3D_{dtype}_{cell_type}.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


def write_function_original(
    write_mesh: bool,
    mesh: dolfinx.mesh.Mesh,
    el: basix.ufl._ElementBase,
    f: Callable[[np.ndarray], np.ndarray],
    dtype: np.dtype,
    name: str,
    path: Path,
) -> Path:
    """Convenience function for writing function to file on the original input mesh"""
    V = dolfinx.fem.functionspace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=dtype)
    uh.interpolate(f)
    uh.name = name
    el_hash = (
        V.element.signature()
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )

    file_hash = f"{el_hash}_{np.dtype(dtype).name}"
    filename = path / f"mesh_{file_hash}.bp"
    if write_mesh:
        adios4dolfinx.write_mesh_input_order(filename, mesh)
    adios4dolfinx.write_function_on_input_mesh(filename, uh, time=0.0)
    return filename


def read_function_original(
    mesh_fname: Path,
    u_fname: Path,
    u_name: str,
    family: str,
    degree: int,
    f: Callable[[np.ndarray], np.ndarray],
    u_dtype: np.dtype,
):
    """
    Convenience function for reading mesh with IPython-parallel and compare to exact solution
    """
    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    assert MPI.COMM_WORLD.size > 1
    if mesh_fname.suffix == ".xdmf":
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_fname, "r") as xdmf:
            mesh = xdmf.read_mesh()
    elif mesh_fname.suffix == ".bp":
        mesh = adios4dolfinx.read_mesh(
            mesh_fname, MPI.COMM_WORLD, "BP4", dolfinx.mesh.GhostMode.shared_facet
        )
    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V, name=u_name, dtype=u_dtype)
    adios4dolfinx.read_function(u_fname, u, time=0.0)
    MPI.COMM_WORLD.Barrier()

    u_ex = dolfinx.fem.Function(V, name="exact", dtype=u_dtype)
    u_ex.interpolate(f)
    u_ex.x.scatter_forward()
    atol = 10 * np.finfo(u_dtype).resolution
    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=atol)  # type: ignore


def write_function_vector(
    write_mesh: bool,
    fname: Path,
    family: str,
    degree: int,
    f: Callable[[np.ndarray], np.ndarray],
    dtype: np.dtype,
    name: str,
    dir: Path,
) -> Path:
    """Convenience function for writing function to file on the original input mesh"""
    from mpi4py import MPI

    import basix.ufl
    import dolfinx

    import adios4dolfinx

    assert MPI.COMM_WORLD.size > 1
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as xdmf:
        mesh = xdmf.read_mesh()
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype)
    V = dolfinx.fem.functionspace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=dtype)
    uh.interpolate(f)
    uh.name = name
    el_hash = (
        V.element.signature()
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )

    file_hash = f"{el_hash}_{np.dtype(dtype).name}"
    filename = dir / f"mesh_{file_hash}.bp"

    if write_mesh:
        adios4dolfinx.write_mesh_input_order(filename, mesh)
    adios4dolfinx.write_function_on_input_mesh(filename, uh, time=0.0)
    return filename


def read_function_vector(
    mesh_fname: Path,
    u_fname: Path,
    u_name: str,
    family: str,
    degree: int,
    f: Callable[[np.ndarray], np.ndarray],
    u_dtype: np.dtype,
):
    """
    Convenience function for reading mesh with IPython-parallel and compare to exact solution
    """

    if mesh_fname.suffix == ".xdmf":
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_fname, "r") as xdmf:
            mesh = xdmf.read_mesh()
    elif mesh_fname.suffix == ".bp":
        mesh = adios4dolfinx.read_mesh(
            mesh_fname, MPI.COMM_WORLD, "BP4", dolfinx.mesh.GhostMode.shared_facet
        )
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree)

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V, name=u_name, dtype=u_dtype)
    adios4dolfinx.read_function(u_fname, u, time=0.0)
    MPI.COMM_WORLD.Barrier()

    u_ex = dolfinx.fem.Function(V, name="exact", dtype=u_dtype)
    u_ex.interpolate(f)
    u_ex.x.scatter_forward()
    atol = 10 * np.finfo(u_dtype).resolution
    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=atol)  # type: ignore


@pytest.mark.skipif(
    os.cpu_count() == 1, reason="Test requires that the system has more than one process"
)
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("write_mesh", [True, False])
def test_read_write_P_2D(
    write_mesh, family, degree, is_complex, create_2D_mesh, cluster, get_dtype, tmp_path
):
    fname = create_2D_mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as xdmf:
        mesh = xdmf.read_mesh()
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
            values[0] -= 3j * x[1]
            values[1] += 2j * x[0]
        return values

    hash = write_function_original(write_mesh, mesh, el, f, f_dtype, "u_original", tmp_path)

    if write_mesh:
        mesh_fname = hash
    else:
        mesh_fname = fname
    query = cluster[:].apply_async(
        read_function_original, mesh_fname, hash, "u_original", family, degree, f, f_dtype
    )
    query.wait()
    assert query.successful(), query.error


@pytest.mark.skipif(
    os.cpu_count() == 1, reason="Test requires that the system has more than one process"
)
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("write_mesh", [True, False])
def test_read_write_P_3D(
    write_mesh, family, degree, is_complex, create_3D_mesh, cluster, get_dtype, tmp_path
):
    fname = create_3D_mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as xdmf:
        mesh = xdmf.read_mesh()
    f_dtype = get_dtype(mesh.geometry.x.dtype, is_complex)
    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
    )

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.pi + x[0]
        values[1] = x[1] + 2 * x[0]
        values[2] = np.cos(x[2])
        if is_complex:
            values[0] -= np.pi * x[1]
            values[1] += 3j * x[2]
            values[2] += 2j
        return values

    hash = write_function_original(write_mesh, mesh, el, f, f_dtype, "u_original", tmp_path)
    MPI.COMM_WORLD.Barrier()

    if write_mesh:
        mesh_fname = hash
    else:
        mesh_fname = fname

    query = cluster[:].apply_async(
        read_function_original, mesh_fname, hash, "u_original", family, degree, f, f_dtype
    )
    query.wait()
    assert query.successful(), query.error


@pytest.mark.skipif(
    os.cpu_count() == 1, reason="Test requires that the system has more than one process"
)
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_2D_vector_simplex(
    write_mesh, family, degree, is_complex, create_simplex_mesh_2D, cluster, get_dtype, tmp_path
):
    fname = create_simplex_mesh_2D

    f_dtype = get_dtype(np.float64, is_complex)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = x[1]
        if is_complex:
            values[0] -= np.sin(x[1]) * 2j
            values[1] += 3j
        return values

    query = cluster[:].apply_async(
        write_function_vector, write_mesh, fname, family, degree, f, f_dtype, "u_original", tmp_path
    )
    query.wait()
    assert query.successful(), query.error
    paths = query.result()
    file_path = paths[0]
    assert all([file_path == path for path in paths])
    if write_mesh:
        mesh_fname = file_path
    else:
        mesh_fname = fname

    read_function_vector(mesh_fname, file_path, "u_original", family, degree, f, f_dtype)


@pytest.mark.skipif(
    os.cpu_count() == 1, reason="Test requires that the system has more than one process"
)
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_3D_vector_simplex(
    write_mesh, family, degree, is_complex, create_simplex_mesh_3D, cluster, get_dtype, tmp_path
):
    fname = create_simplex_mesh_3D

    f_dtype = get_dtype(np.float64, is_complex)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi)
        values[1] = x[1] + 2 * x[0]
        values[2] = np.cos(x[2])
        if is_complex:
            values[0] += 2j * x[2]
            values[1] += 2j * np.cos(x[2])
        return values

    query = cluster[:].apply_async(
        write_function_vector, write_mesh, fname, family, degree, f, f_dtype, "u_original", tmp_path
    )
    query.wait()
    assert query.successful(), query.error
    paths = query.result()
    file_path = paths[0]
    assert all([file_path == path for path in paths])
    if write_mesh:
        mesh_fname = file_path
    else:
        mesh_fname = fname

    read_function_vector(mesh_fname, file_path, "u_original", family, degree, f, f_dtype)


@pytest.mark.skipif(
    os.cpu_count() == 1, reason="Test requires that the system has more than one process"
)
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_read_write_2D_vector_non_simplex(
    write_mesh, family, degree, is_complex, create_non_simplex_mesh_2D, cluster, get_dtype, tmp_path
):
    fname = create_non_simplex_mesh_2D

    f_dtype = get_dtype(np.float64, is_complex)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi)
        values[1] = x[1] + 2 * x[0]
        if is_complex:
            values[0] += 2j * x[1]
            values[1] -= np.sin(x[0]) * 9j
        return values

    query = cluster[:].apply_async(
        write_function_vector, write_mesh, fname, family, degree, f, f_dtype, "u_original", tmp_path
    )
    query.wait()
    assert query.successful(), query.error
    paths = query.result()
    file_path = paths[0]
    assert all([file_path == path for path in paths])
    if write_mesh:
        mesh_fname = file_path
    else:
        mesh_fname = fname

    read_function_vector(mesh_fname, file_path, "u_original", family, degree, f, f_dtype)


@pytest.mark.skipif(
    os.cpu_count() == 1, reason="Test requires that the system has more than one process"
)
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("family", ["NCF"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_3D_vector_non_simplex(
    write_mesh, family, degree, is_complex, create_non_simplex_mesh_3D, cluster, get_dtype, tmp_path
):
    fname = create_non_simplex_mesh_3D

    f_dtype = get_dtype(np.float64, is_complex)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = np.cos(x[2])
        values[2] = x[0]
        if is_complex:
            values[2] += x[0] * x[1] * 3j
        return values

    query = cluster[:].apply_async(
        write_function_vector, write_mesh, fname, family, degree, f, f_dtype, "u_original", tmp_path
    )
    query.wait()
    assert query.successful(), query.error
    paths = query.result()
    file_path = paths[0]
    assert all([file_path == path for path in paths])
    if write_mesh:
        mesh_fname = file_path
    else:
        mesh_fname = fname

    read_function_vector(mesh_fname, file_path, "u_original", family, degree, f, f_dtype)
