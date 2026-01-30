from mpi4py import MPI

import dolfinx
import numpy as np
import ufl

import adios4dolfinx


def test_xdmf_mesh(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    tmp_file = tmp_path / "mesh.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)

    MPI.COMM_WORLD.barrier()
    in_grid = adios4dolfinx.read_mesh(tmp_file, MPI.COMM_WORLD, backend="xdmf")

    assert mesh.topology.dim == in_grid.topology.dim
    assert mesh.geometry.dim == in_grid.geometry.dim
    for i in range(mesh.topology.dim):
        mesh.topology.create_entities(i)
        o_map = mesh.topology.index_map(i)
        in_grid.topology.create_entities(i)
        map = in_grid.topology.index_map(i)
        assert o_map.size_global == map.size_global

    assert mesh.geometry.index_map().size_global == in_grid.geometry.index_map().size_global
    org_area = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh))), op=MPI.SUM
    )
    area = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=in_grid))), op=MPI.SUM
    )
    assert np.isclose(area, org_area)


def test_xdmf_function(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 10)

    def f(x):
        return (x[0], x[1], -2 * x[1], 3 * x[0])

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (4,)))
    u = dolfinx.fem.Function(V, name="u")
    u.interpolate(f)

    tmp_file = tmp_path / "function.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    MPI.COMM_WORLD.barrier()

    import adios4dolfinx.backends.xdmf

    in_grid = adios4dolfinx.read_mesh(tmp_file, MPI.COMM_WORLD, backend="xdmf")
    u = adios4dolfinx.backends.xdmf.backend.read_point_data(tmp_file, "u", in_grid)

    u_ref = dolfinx.fem.Function(u.function_space)
    u_ref.interpolate(f)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(u.x.array, u_ref.x.array, atol=eps)
