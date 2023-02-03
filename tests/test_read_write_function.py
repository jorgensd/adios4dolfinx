import pytest
import dolfinx
import adios4dolfinx
import numpy as np
from mpi4py import MPI
import ufl


def write_function(mesh, el, f):
    V = dolfinx.fem.FunctionSpace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)
    print(mesh.comm.rank, "pre write", u.x.array)
    adios4dolfinx.write_function(u, "u.bp")
    adios4dolfinx.write_mesh(mesh, "test_mesh.bp")


def read_function(comm, el, f):
    mesh2 = adios4dolfinx.read_mesh(MPI.COMM_WORLD, "test_mesh.bp", "BP4", dolfinx.mesh.GhostMode.shared_facet)
    V2 = dolfinx.fem.FunctionSpace(mesh2, el)
    v = dolfinx.fem.Function(V2)
    adios4dolfinx.read_function(v, "u.bp")

    v_ex = dolfinx.fem.Function(V2)
    v_ex.interpolate(f)

    x = V2.tabulate_dof_coordinates()
    assert np.allclose(v.x.array, v_ex.x.array)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle,
                                       dolfinx.mesh.CellType.quadrilateral])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
@pytest.mark.parametrize("write_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_function(read_comm, write_comm, family, degree, cell_type):
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10)
    el = ufl.VectorElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1])

    write_function(mesh, el, f)
    read_function(read_comm, el, f)
