import pathlib

import dolfinx
import numpy as np
import pytest
import ufl
from mpi4py import MPI

import adios4dolfinx
from adios4dolfinx import read_function_perm, read_mesh, write_mesh_perm


def write_function(mesh, el, f) -> str:
    V = dolfinx.fem.FunctionSpace(mesh, el)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)
    el_hash = V.element.signature().replace(' ', '').replace(',', '').replace("(", "").replace(')', "")
    filename = pathlib.Path(f"output/mesh{el_hash}.bp")
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh_perm(mesh, filename)
        adios4dolfinx.write_function(u, filename)

    else:
        if MPI.COMM_WORLD.rank == 0:
            adios4dolfinx.write_mesh_perm(mesh, filename)
            adios4dolfinx.write_function(u, filename)
    return el_hash


def read_function(comm, el, f, hash):
    filename = f"output/mesh{hash}.bp"
    engine = "BP4"
    mesh = adios4dolfinx.read_mesh(comm, filename, engine, dolfinx.mesh.GhostMode.shared_facet)
    V = dolfinx.fem.FunctionSpace(mesh, el)
    v = dolfinx.fem.Function(V)
    adios4dolfinx.read_function_perm(v, filename, engine)
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
    mesh = dolfinx.mesh.create_unit_square(write_comm, 5, 5, cell_type=cell_type)
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
    mesh = dolfinx.mesh.create_unit_square(write_comm, 10, 10, cell_type=cell_type)
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
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 3, 3, 3)
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
    mesh = dolfinx.mesh.create_unit_square(write_comm, 5, 7, cell_type=cell_type)
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
    mesh = dolfinx.mesh.create_unit_cube(write_comm, 3, 7, 5, cell_type=dolfinx.mesh.CellType.hexahedron)
    el = ufl.FiniteElement(family, mesh.ufl_cell(), degree)

    def f(x):
        return (np.full(x.shape[1], np.pi)+x[0], x[1]+2*x[0], np.cos(x[2]))

    hash = write_function(mesh, el, f)
    MPI.COMM_WORLD.Barrier()
    read_function(read_comm, el, f, hash)


# @pytest.mark.parametrize("encoder, suffix", [("BP4", ".bp"), ("HDF5", ".h5"), ("BP5", ".bp")])
# @pytest.mark.parametrize("ghost_mode", [dolfinx.mesh.GhostMode.shared_facet])
# def test_mesh_read_writer(encoder, suffix, ghost_mode):
#     encoder = "BP4"
#     suffix = ".bp"
#     N = 25
#     ghost_mode = dolfinx.mesh.GhostMode.shared_facet
#     file = pathlib.Path(f"output/adios_mesh_{encoder}")
#     if MPI.COMM_WORLD.rank == 0:
#         mesh_loc = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, N, N, ghost_mode=ghost_mode)
#         write_mesh_perm(mesh_loc, file.with_suffix(suffix), encoder)
#         V = dolfinx.fem.FunctionSpace(mesh_loc, ("N1curl", 1))
#         u = dolfinx.fem.Function(V)
#         u.interpolate(lambda x: (x[0], 100*x[1]+3))
#         write_function(u, file.with_suffix(suffix), encoder)
#     MPI.COMM_WORLD.Barrier()

#     mesh = read_mesh(MPI.COMM_WORLD, file.with_suffix(suffix), encoder, dolfinx.mesh.GhostMode.shared_facet)
#     V = dolfinx.fem.FunctionSpace(mesh, ("N1curl", 1))
#     u = dolfinx.fem.Function(V)

#     read_function_perm(u, file.with_suffix(suffix), encoder)
#     w = dolfinx.fem.Function(V)
#     w.interpolate(lambda x: (x[0], 100*x[1]+3))
#     assert np.allclose(w.x.array, u.x.array)
