from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest

from adios4dolfinx import FileMode, snapshot_checkpoint as _stapshot_checkpoint
from adios4dolfinx.utils import skip_if_not_implemented, suffix


def snapshot_checkpoint(*args, **kwargs):
    with skip_if_not_implemented():
        _stapshot_checkpoint(*args, **kwargs)


triangle = dolfinx.mesh.CellType.triangle
quad = dolfinx.mesh.CellType.quadrilateral
tetra = dolfinx.mesh.CellType.tetrahedron
hex = dolfinx.mesh.CellType.hexahedron


@pytest.mark.parametrize(
    "cell_type, family", [(triangle, "N1curl"), (triangle, "RT"), (quad, "RTCF")]
)
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_2D(family, degree, cell_type, tmp_path, backend):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=cell_type)
    el = basix.ufl.element(family, mesh.basix_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi) + x[0], x[1])

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / Path("snapshot_2D_vs").with_suffix(suffix(backend))
    snapshot_checkpoint(u, file, FileMode.write, backend=backend)

    v = dolfinx.fem.Function(V)
    snapshot_checkpoint(v, file, FileMode.read, backend=backend)
    assert np.allclose(u.x.array, v.x.array)


@pytest.mark.parametrize("cell_type, family", [(tetra, "N1curl"), (tetra, "RT"), (hex, "NCF")])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_3D(family, degree, cell_type, tmp_path, backend):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, cell_type=cell_type)
    el = basix.ufl.element(family, mesh.basix_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi) + x[0], x[1], x[1] * x[2])

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / Path("snapshot_3D_vs").with_suffix(suffix(backend))
    snapshot_checkpoint(u, file, FileMode.write, backend=backend)

    v = dolfinx.fem.Function(V)
    snapshot_checkpoint(v, file, FileMode.read, backend=backend)
    assert np.allclose(u.x.array, v.x.array)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_P_2D(family, degree, cell_type, tmp_path, backend):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5, cell_type=cell_type)
    el = basix.ufl.element(family, mesh.basix_cell(), degree, shape=(mesh.geometry.dim,))

    def f(x):
        return (np.full(x.shape[1], np.pi) + x[0], x[1])

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / Path("snapshot_2D_p").with_suffix(suffix(backend))
    snapshot_checkpoint(u, file, FileMode.write, backend=backend)

    v = dolfinx.fem.Function(V)
    snapshot_checkpoint(v, file, FileMode.read, backend=backend)
    assert np.allclose(u.x.array, v.x.array)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
def test_read_write_P_3D(family, degree, cell_type, tmp_path, backend):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type=cell_type)
    el = basix.ufl.element(family, mesh.basix_cell(), degree, shape=(mesh.geometry.dim,))

    def f(x):
        return (np.full(x.shape[1], np.pi) + x[0], x[1] + 2 * x[0], np.cos(x[2]))

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / Path("snapshot_3D_p").with_suffix(suffix(backend))

    snapshot_checkpoint(u, file, FileMode.write, backend=backend)

    v = dolfinx.fem.Function(V)
    snapshot_checkpoint(v, file, FileMode.read, backend=backend)
    assert np.allclose(u.x.array, v.x.array)
