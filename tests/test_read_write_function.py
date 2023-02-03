import pytest
import dolfinx
import adios4dolfinx
import numpy as np
from mpi4py import MPI
import ufl
import pathlib


def write_function(mesh, el, f) -> str:
    V = dolfinx.fem.FunctionSpace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)
    el_hash = V.element.signature().replace(' ', '').replace(',', '').replace("(", "").replace(')', "")
    if mesh.comm.size != 1:
        adios4dolfinx.write_function(u, pathlib.Path(f"output/u{el_hash}.bp"))
        adios4dolfinx.write_mesh(mesh, pathlib.Path(f"output/mesh{el_hash}.bp"))
    else:
        if MPI.COMM_WORLD.rank == 0:
            adios4dolfinx.write_function(u, pathlib.Path(f"output/u{el_hash}.bp"))
            adios4dolfinx.write_mesh(mesh, pathlib.Path(f"output/mesh{el_hash}.bp"))
    return el_hash


def read_function(comm, el, f, hash):
    mesh = adios4dolfinx.read_mesh(comm, f"output/mesh{hash}.bp", "BP4", dolfinx.mesh.GhostMode.shared_facet)
    V = dolfinx.fem.FunctionSpace(mesh, el)
    v = dolfinx.fem.Function(V)
    adios4dolfinx.read_function(v, f"output/u{hash}.bp")
    v_ex = dolfinx.fem.Function(V)
    v_ex.interpolate(f)
    assert np.allclose(v.x.array, v_ex.x.array)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.quadrilateral])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_2D(read_comm, write_comm, family, degree, cell_type):
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type)
    el = ufl.VectorElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1])

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.tetrahedron,
                                       dolfinx.mesh.CellType.hexahedron])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_3D(read_comm, write_comm, family, degree, cell_type):
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 5, 5, 5, cell_type=cell_type)
    el = ufl.VectorElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1]+2*x[0], np.cos(x[2]))

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_2D(read_comm, write_comm, family, degree, cell_type):
    if read_comm.size > 1 or write_comm.size > 1:
        pytest.xfail("Edge based elements not supported in parallel")
    mesh = dolfinx.mesh.create_unit_square(write_comm, 1, 1, cell_type=cell_type)
    el = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1])

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)


@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_3D(read_comm, write_comm, family, degree):
    if read_comm.size > 1 or write_comm.size > 1:
        pytest.xfail("Edge based elements not supported in parallel")
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 1, 1, 1)
    el = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1]+2*x[0], np.cos(x[2]))

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.quadrilateral])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_2D_quad(read_comm, write_comm, family, degree, cell_type):
    if read_comm.size > 1 or write_comm.size > 1:
        pytest.xfail("Edge based elements not supported in parallel")
    mesh = dolfinx.mesh.create_unit_square(write_comm, 3, 3, cell_type=cell_type)
    el = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1])

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)


@pytest.mark.parametrize("family", ["NCF"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_hex(read_comm, write_comm, family, degree):
    if read_comm.size > 1 or write_comm.size > 1:
        pytest.xfail("Edge based elements not supported in parallel")
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 1, 1, 1, cell_type=dolfinx.mesh.CellType.hexahedron)
    el = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1]+2*x[0], np.cos(x[2]))

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)
