import pathlib

import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI
import basix.ufl
import basix
import adios4dolfinx


def write_function(mesh, el, f, dtype) -> str:
    V = dolfinx.fem.FunctionSpace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=dtype)
    uh.interpolate(f)
    el_hash = (
        V.element.signature()
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )
    filename = pathlib.Path(f"output/mesh{el_hash}_{dtype}.bp")
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh(mesh, filename)
        adios4dolfinx.write_function(uh, filename)

    else:
        if MPI.COMM_WORLD.rank == 0:
            adios4dolfinx.write_mesh(mesh, filename)
            adios4dolfinx.write_function(uh, filename)
    return f"{el_hash}_{dtype}"


def read_function(comm, el, f, hash, dtype):
    filename = f"output/mesh{hash}.bp"
    engine = "BP4"
    mesh = adios4dolfinx.read_mesh(
        comm, filename, engine, dolfinx.mesh.GhostMode.shared_facet
    )
    V = dolfinx.fem.functionspace(mesh, el)
    v = dolfinx.fem.Function(V, dtype=dtype)
    adios4dolfinx.read_function(v, filename, engine)
    v_ex = dolfinx.fem.Function(V, dtype=dtype)
    v_ex.interpolate(f)
    assert np.allclose(v.x.array, v_ex.x.array)


@pytest.mark.parametrize("dtypes", [(np.float64, np.float64), (np.float32, np.float32),  (np.float32, np.complex64),
                                    (np.float64, np.complex128)])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_2D(read_comm, write_comm, family, degree, cell_type, dtypes):
    mesh = dolfinx.mesh.create_unit_square(write_comm, 1, 1, cell_type=cell_type, dtype=dtypes[0])
    el = basix.ufl.element(family, 
                           mesh.ufl_cell().cellname(),
                           degree,
                           basix.LagrangeVariant.gll_warped,
                           gdim=mesh.geometry.dim,
                           shape=(mesh.geometry.dim, ))
    def f(x):
        values = np.empty((2, x.shape[1]), dtype=dtypes[1])
        values[0] = np.full(x.shape[1], np.pi) + x[0]+x[1]*1j 
        values[1] = x[0]+3j*x[1]
        return values

    hash = write_function(mesh, el, f, dtypes[1])
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtypes[1])


@pytest.mark.parametrize("dtypes", [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.complex64),
                                    (np.float64, np.complex128)])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_P_3D(read_comm, write_comm, family, degree, cell_type, dtypes):
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 5, 5, 5, cell_type=cell_type, dtype=dtypes[0])
    el = basix.ufl.element(family, 
                           mesh.ufl_cell().cellname(),
                           degree,
                           basix.LagrangeVariant.gll_warped,
                           gdim=mesh.geometry.dim,
                           shape=(mesh.geometry.dim, ))
    def f(x):
        values = np.empty((3, x.shape[1]), dtype=dtypes[1])
        values[0] = np.pi + x[0] + 2j*x[2]
        values[1] = x[1] + 2 * x[0]
        values[2] = 1j*x[1] + np.cos(x[2])
        return values

    hash = write_function(mesh, el, f, dtypes[1])

    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtypes[1])


@pytest.mark.parametrize("dtypes", [(np.float64, np.float64), (np.float32, np.float32)])
#,(np.float32, np.complex64),
#                                    (np.float64, np.complex128)])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_2D(read_comm, write_comm, family, degree, cell_type, dtypes):
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type, dtype=dtypes[0])
    el = basix.ufl.element(family, 
                           mesh.ufl_cell().cellname(),
                           degree,
                           gdim=mesh.geometry.dim)
    def f(x):
        values = np.empty((2, x.shape[1]), dtype=dtypes[1])
        values[0] = np.full(x.shape[1], np.pi) + x[0] + 2j*x[1]
        values[1] = x[1] + 2j*x[0]
        return values

    hash = write_function(mesh, el, f, dtypes[1])
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtypes[1])


@pytest.mark.parametrize("dtypes", [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.complex64),
                                    (np.float64, np.complex128)])
@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_3D(read_comm, write_comm, family, degree, dtypes):
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 3, 5, 7, dtype=dtypes[0])
    el = basix.ufl.element(family, 
                           mesh.ufl_cell().cellname(),
                           degree,
                           gdim=mesh.geometry.dim)

    def f(x):
        values = np.empty((3, x.shape[1]), dtype=dtypes[1])
        values[0] = np.full(x.shape[1], np.pi) + 2j*x[2]
        values[1] = x[1] + 2 * x[0] + 2j*np.cos(x[2])
        values[2] = np.cos(x[2])
        return values
    hash = write_function(mesh, el, f, dtype=dtypes[1])
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtype=dtypes[1])


@pytest.mark.parametrize("dtypes", [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.complex64),
                                    (np.float64, np.complex128)])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.quadrilateral])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_2D_quad(read_comm, write_comm, family, degree, cell_type, dtypes):
    mesh = dolfinx.mesh.create_unit_square(write_comm, 5, 7, cell_type=cell_type, dtype=dtypes[0])
    el = basix.ufl.element(family, 
                           mesh.ufl_cell().cellname(),
                           degree,
                           gdim=mesh.geometry.dim)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=dtypes[1])
        values[0] = np.full(x.shape[1], np.pi) + 2j*x[2]
        values[1] = x[1] + 2 * x[0] + 2j*np.cos(x[2])
        return values

    hash = write_function(mesh, el, f, dtypes[1])
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtypes[1])


@pytest.mark.parametrize("dtypes", [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.complex64),
                                    (np.float64, np.complex128)])
@pytest.mark.parametrize("family", ["NCF"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_hex(read_comm, write_comm, family, degree, dtypes):
    mesh = dolfinx.mesh.create_unit_cube(
        write_comm, 3, 7, 5, cell_type=dolfinx.mesh.CellType.hexahedron, dtype=dtypes[0]
    )
    el = basix.ufl.element(family, 
                           mesh.ufl_cell().cellname(),
                           degree,
                           gdim=mesh.geometry.dim)

    def f(x):
        values = np.empty((2, x.shape[1]), dtype=dtypes[1])
        values[0] = np.full(x.shape[1], np.pi) + x[0], x[1] + 2 * x[0]
        values[1] = np.cos(x[2])
        return values

    hash = write_function(mesh, el, f, dtype=dtypes[1])
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash, dtype=dtypes[1])
