from mpi4py import MPI
import dolfinx
import numpy as np
import adios4dolfinx
import ufl


def test_write_submesh_codim0(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(
        comm, 50, 50, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    def locate_cells(x):
        return (x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2 < 0.2

    dim = mesh.topology.dim
    cells = dolfinx.mesh.locate_entities(mesh, dim, locate_cells)

    submesh, _, _, node_map = dolfinx.mesh.create_submesh(mesh, dim, cells)

    f_path = mesh.comm.bcast(tmp_path, root=0)
    outfile = f_path / "submesh_codim0.bp"
    adios4dolfinx.write_mesh(outfile, mesh, name="mesh")

    adios4dolfinx.checkpointing.write_submesh(outfile, mesh, submesh, "submesh", node_map)
    V_sub = dolfinx.fem.functionspace(submesh, ("N1curl", 2))
    u_sub = dolfinx.fem.Function(V_sub, name="u_sub")

    def f(x):
        return np.sin(1.57 * x[0] + 0.2), x[1]

    u_sub.interpolate(f)
    adios4dolfinx.write_function(outfile, u_sub, time=0, name="u_sub")

    del submesh, u_sub, mesh

    comm.Barrier()
    new_mesh = adios4dolfinx.read_mesh(outfile, MPI.COMM_WORLD, name="mesh")
    new_submesh, _, _, _, input_indices = adios4dolfinx.checkpointing.read_submesh(
        outfile, new_mesh, "submesh"
    )

    V_new = dolfinx.fem.functionspace(new_submesh, ("N1curl", 2))
    u_sub_new = dolfinx.fem.Function(V_new, name="u_sub_new")
    adios4dolfinx.read_function(
        outfile, u_sub_new, time=0, name="u_sub", original_cell_index=input_indices
    )

    u_ref = dolfinx.fem.Function(V_new, name="u_ref")
    u_ref.interpolate(f)

    tol = 100 * np.finfo(dolfinx.default_scalar_type).eps
    np.testing.assert_allclose(u_sub_new.x.array, u_ref.x.array, atol=tol)

    L2_squared = dolfinx.fem.form(ufl.inner(u_sub_new - u_ref, u_sub_new - u_ref) * ufl.dx)
    L2_local = dolfinx.fem.assemble_scalar(L2_squared)
    L2_global = np.sqrt(new_submesh.comm.allreduce(L2_local, op=MPI.SUM))

    np.testing.assert_allclose(L2_global, 0.0, atol=tol)




def test_write_submesh_codim1(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(
        comm, 10,11,13, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    def locate_cells(x):
        return (x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2 < 0.2

    dim = mesh.topology.dim-1
    cells = dolfinx.mesh.locate_entities(mesh, dim, locate_cells)

    submesh, _, _, node_map = dolfinx.mesh.create_submesh(mesh, dim, cells)

    f_path = mesh.comm.bcast(tmp_path, root=0)
    outfile = f_path / "submesh_codim1.bp"
    adios4dolfinx.write_mesh(outfile, mesh, name="mesh")

    adios4dolfinx.checkpointing.write_submesh(outfile, mesh, submesh, "submesh", node_map)
    el = ("Lagrange", 1, (3, ))
    V_sub = dolfinx.fem.functionspace(submesh, el)
    u_sub = dolfinx.fem.Function(V_sub, name="u_sub")

    def f(x):
        return np.sin(1.57 * x[0] + 0.2), x[1], x[2] -np.cos(x[0])

    u_sub.interpolate(f)
    adios4dolfinx.write_function(outfile, u_sub, time=0, name="u_sub")
    with dolfinx.io.VTXWriter(submesh.comm, "test_codim_pre.bp", [u_sub]) as writer:
        writer.write(0.0)

    del submesh, u_sub, mesh

    comm.Barrier()
    new_mesh = adios4dolfinx.read_mesh(outfile, MPI.COMM_WORLD, name="mesh")
    new_submesh, _, _, _, input_indices = adios4dolfinx.checkpointing.read_submesh(
        outfile, new_mesh, "submesh"
    )

    V_new = dolfinx.fem.functionspace(new_submesh, el)
    u_sub_new = dolfinx.fem.Function(V_new, name="u_sub_new")
    adios4dolfinx.read_function(
        outfile, u_sub_new, time=0, name="u_sub", original_cell_index=input_indices
    )

    u_ref = dolfinx.fem.Function(V_new, name="u_ref")
    u_ref.interpolate(f)

    with dolfinx.io.VTXWriter(new_submesh.comm, "test_codim.bp", [u_sub_new]) as writer:
        writer.write(0.0)

    tol = 100 * np.finfo(dolfinx.default_scalar_type).eps
    np.testing.assert_allclose(u_sub_new.x.array - u_ref.x.array, 0, atol=tol)

    L2_squared = dolfinx.fem.form(ufl.inner(u_sub_new-u_ref, u_sub_new - u_ref) * ufl.dx)
    L2_local = dolfinx.fem.assemble_scalar(L2_squared)
    L2_global = np.sqrt(new_submesh.comm.allreduce(L2_local, op=MPI.SUM))

    np.testing.assert_allclose(L2_global, 0.0, atol=tol)
