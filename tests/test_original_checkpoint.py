import itertools
from pathlib import Path
from typing import Callable

from mpi4py import MPI

import basix
import basix.ufl
import dolfinx
import ipyparallel as ipp
import numpy as np
import pytest

import adios4dolfinx

from .test_utils import get_dtype

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
def create_simplex_mesh_2D():
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        10,
        10,
        cell_type=dolfinx.mesh.CellType.triangle,
        dtype=np.float64,
    )
    fname = Path("output/original_mesh_2D_simplex.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def create_simplex_mesh_3D():
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        5,
        5,
        5,
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        dtype=np.float64,
    )
    fname = Path("output/original_mesh_3D_simplex.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def create_non_simplex_mesh_2D():
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        10,
        10,
        cell_type=dolfinx.mesh.CellType.quadrilateral,
        dtype=np.float64,
    )
    fname = Path("output/original_mesh_2D_non_simplex.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def create_non_simplex_mesh_3D():
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        5,
        5,
        5,
        cell_type=dolfinx.mesh.CellType.hexahedron,
        dtype=np.float64,
    )
    fname = Path("output/original_mesh_3D_non_simplex.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(params=two_dim_combinations, scope="module")
def create_2D_mesh(request):
    dtype, cell_type = request.param
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 5, 7, cell_type=cell_type, dtype=dtype
    )
    fname = Path("output/original_mesh_2D_{dtype}_{cell_type}.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(params=three_dim_combinations, scope="module")
def create_3D_mesh(request):
    dtype, cell_type = request.param
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 5, 7, 3, cell_type=cell_type, dtype=dtype
    )
    fname = Path("output/original_mesh_3D_{dtype}_{cell_type}.xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return fname


@pytest.fixture(scope="module")
def cluster():
    cluster = ipp.Cluster(engines="mpi", n=2)
    rc = cluster.start_and_connect_sync()
    yield rc
    cluster.stop_cluster_sync()


def write_function(
    write_mesh: bool,
    mesh: dolfinx.mesh.Mesh,
    el: basix.ufl._ElementBase,
    f: Callable[[np.ndarray], np.ndarray],
    dtype: np.dtype,
    name: str,
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
    filename = Path(f"output/mesh_{file_hash}.bp")

    if write_mesh:
        adios4dolfinx.write_mesh_input_order(mesh, filename)
    adios4dolfinx.write_function_on_input_mesh(uh, filename, time=0.0)
    return filename


def read_function(
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
            MPI.COMM_WORLD, mesh_fname, "BP4", dolfinx.mesh.GhostMode.shared_facet
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
    adios4dolfinx.read_function(u, u_fname, time=0.0)
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
) -> Path:
    """Convenience function for writing function to file on the original input mesh"""
    from mpi4py import MPI
    import basix.ufl
    import dolfinx
    import adios4dolfinx

    assert MPI.COMM_WORLD.size > 1
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as xdmf:
        mesh = xdmf.read_mesh()
    el = basix.ufl.element(
        family, mesh.ufl_cell().cellname(), degree, dtype=mesh.geometry.x.dtype
    )
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
    filename = Path(f"output/mesh_{file_hash}.bp")

    if write_mesh:
        adios4dolfinx.write_mesh_input_order(mesh, filename)
    adios4dolfinx.write_function_on_input_mesh(uh, filename, time=0.0)
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
            MPI.COMM_WORLD, mesh_fname, "BP4", dolfinx.mesh.GhostMode.shared_facet
        )
    el = basix.ufl.element(family, mesh.ufl_cell().cellname(), degree)

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V, name=u_name, dtype=u_dtype)
    adios4dolfinx.read_function(u, u_fname, time=0.0)
    MPI.COMM_WORLD.Barrier()

    u_ex = dolfinx.fem.Function(V, name="exact", dtype=u_dtype)
    u_ex.interpolate(f)
    u_ex.x.scatter_forward()
    atol = 10 * np.finfo(u_dtype).resolution
    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=atol)  # type: ignore


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("write_mesh", [True, False])
def test_read_write_P_2D(write_mesh, family, degree, complex, create_2D_mesh, cluster):
    fname = create_2D_mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as xdmf:
        mesh = xdmf.read_mesh()
    f_dtype = get_dtype(mesh.geometry.x.dtype, complex)

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
        values[0] = np.full(x.shape[1], np.pi) + x[0] + x[1] * 1j
        values[1] = x[0] + 3j * x[1]
        return values

    hash = write_function(write_mesh, mesh, el, f, f_dtype, "u_original")

    if write_mesh:
        mesh_fname = fname
    else:
        mesh_fname = hash
    query = cluster[:].apply_async(
        read_function, mesh_fname, hash, "u_original", family, degree, f, f_dtype
    )
    query.wait()
    assert query.successful(), query.error


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("write_mesh", [True, False])
def test_read_write_P_3D(write_mesh, family, degree, complex, create_3D_mesh, cluster):
    fname = create_3D_mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, fname, "r") as xdmf:
        mesh = xdmf.read_mesh()
    f_dtype = get_dtype(mesh.geometry.x.dtype, complex)
    el = basix.ufl.element(
        family,
        mesh.ufl_cell().cellname(),
        degree,
        basix.LagrangeVariant.gll_warped,
        shape=(mesh.geometry.dim,),
    )

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.pi + x[0] + 2j * x[2]
        values[1] = x[1] + 2 * x[0]
        values[2] = 1j * x[1] + np.cos(x[2])
        return values

    hash = write_function(write_mesh, mesh, el, f, f_dtype, "u_original")
    MPI.COMM_WORLD.Barrier()

    if write_mesh:
        mesh_fname = fname
    else:
        mesh_fname = hash

    query = cluster[:].apply_async(
        read_function, mesh_fname, hash, "u_original", family, degree, f, f_dtype
    )
    query.wait()
    assert query.successful(), query.error


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_2D_vector_simplex(
    write_mesh, family, degree, complex, create_simplex_mesh_2D, cluster
):
    fname = create_simplex_mesh_2D

    f_dtype = get_dtype(np.float64, complex)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0] + 2j * x[1]
        values[1] = x[1] + 2j * x[0]
        return values

    query = cluster[:].apply_async(
        write_function_vector,
        write_mesh,
        fname,
        family,
        degree,
        f,
        f_dtype,
        "u_original",
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

    read_function_vector(
        mesh_fname, file_path, "u_original", family, degree, f, f_dtype
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_3D_vector_simplex(
    write_mesh, family, degree, complex, create_simplex_mesh_3D, cluster
):
    fname = create_simplex_mesh_3D

    f_dtype = get_dtype(np.float64, complex)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + 2j * x[2]
        values[1] = x[1] + 2 * x[0] + 2j * np.cos(x[2])
        values[2] = np.cos(x[2])
        return values

    query = cluster[:].apply_async(
        write_function_vector,
        write_mesh,
        fname,
        family,
        degree,
        f,
        f_dtype,
        "u_original",
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

    read_function_vector(
        mesh_fname, file_path, "u_original", family, degree, f, f_dtype
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_read_write_2D_vector_non_simplex(
    write_mesh, family, degree, complex, create_non_simplex_mesh_2D, cluster
):
    fname = create_non_simplex_mesh_2D

    f_dtype = get_dtype(np.float64, complex)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + 2j * x[2]
        values[1] = x[1] + 2 * x[0] + 2j * np.cos(x[2])
        return values

    query = cluster[:].apply_async(
        write_function_vector,
        write_mesh,
        fname,
        family,
        degree,
        f,
        f_dtype,
        "u_original",
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

    read_function_vector(
        mesh_fname, file_path, "u_original", family, degree, f, f_dtype
    )


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Test uses ipythonparallel for MPI")
@pytest.mark.parametrize("write_mesh", [True, False])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("family", ["NCF"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_3D_vector_non_simplex(
    write_mesh, family, degree, complex, create_non_simplex_mesh_3D, cluster
):
    fname = create_non_simplex_mesh_3D

    f_dtype = get_dtype(np.float64, complex)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=f_dtype)
        values[0] = np.full(x.shape[1], np.pi) + x[0]
        values[1] = np.cos(x[2])
        values[2] = 1j * x[1] + x[0]
        return values

    query = cluster[:].apply_async(
        write_function_vector,
        write_mesh,
        fname,
        family,
        degree,
        f,
        f_dtype,
        "u_original",
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

    read_function_vector(
        mesh_fname, file_path, "u_original", family, degree, f, f_dtype
    )
