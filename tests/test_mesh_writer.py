
from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from adios4dolfinx import read_mesh, write_mesh
from adios4dolfinx.adios2_helpers import adios2


@pytest.mark.parametrize("encoder, suffix", [("BP4", ".bp"), ("HDF5", ".h5")])
# , ("BP5", ".bp")]) # Deactivated, see: https://github.com/jorgensd/adios4dolfinx/issues/7
@pytest.mark.parametrize("ghost_mode", [dolfinx.mesh.GhostMode.shared_facet])
def test_mesh_read_writer(encoder, suffix, ghost_mode, tmp_path):
    N = 25
    # Consistent tmp dir across processes
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / f"adios_mesh_{encoder}"
    xdmf_file = fname / "xdmf_mesh"
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=ghost_mode)

    write_mesh(file.with_suffix(suffix), mesh, encoder)
    mesh.comm.Barrier()
    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
    mesh.comm.Barrier()

    mesh_adios = read_mesh(file.with_suffix(suffix), MPI.COMM_WORLD, encoder, ghost_mode)
    mesh_adios.comm.Barrier()

    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "r") as xdmf:
        mesh_xdmf = xdmf.read_mesh(ghost_mode=ghost_mode)

    for i in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(i)
        mesh_xdmf.topology.create_entities(i)
        mesh_adios.topology.create_entities(i)
        assert (
            mesh_xdmf.topology.index_map(i).size_global
            == mesh_adios.topology.index_map(i).size_global
        )

    # Check that integration over different entities are consistent
    for measure in [ufl.ds, ufl.dS, ufl.dx]:
        c_adios = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_adios)))
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        c_xdmf = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_xdmf)))
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_xdmf, MPI.SUM),
        )
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )


@pytest.mark.parametrize("encoder, suffix", [("BP4", ".bp"), ("BP5", ".bp")])
@pytest.mark.parametrize(
    "ghost_mode", [dolfinx.mesh.GhostMode.shared_facet, dolfinx.mesh.GhostMode.none]
)
def test_timedep_mesh(encoder, suffix, ghost_mode, tmp_path):
    # Currently unsupported, unclear why ("HDF5", ".h5"),
    N = 25
    # Consistent tmp dir across processes
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / f"adios_time_dep_mesh_{encoder}"
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=ghost_mode)

    def u(x):
        return np.asarray([x[0] + 0.1 * np.sin(x[1]), 0.2 * np.cos(x[1]), x[2]])

    write_mesh(file.with_suffix(suffix), mesh, encoder)
    delta_x = u(mesh.geometry.x.T).T
    mesh.geometry.x[:] += delta_x
    write_mesh(file.with_suffix(suffix), mesh, encoder, mode=adios2.Mode.Append, time=3.0)
    mesh.geometry.x[:] -= delta_x

    mesh_first = read_mesh(file.with_suffix(suffix), MPI.COMM_WORLD, encoder, ghost_mode, time=0.0)
    mesh_first.comm.Barrier()

    # Check that integration over different entities are consistent
    measures = [ufl.ds, ufl.dx]
    if ghost_mode == dolfinx.mesh.GhostMode.shared_facet:
        measures.append(ufl.dx)
    for measure in measures:
        c_adios = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_first)))
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        assert np.isclose(
            mesh_first.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )

    mesh.geometry.x[:] += delta_x
    mesh_second = read_mesh(file.with_suffix(suffix), MPI.COMM_WORLD, encoder, ghost_mode, time=3.0)
    mesh_second.comm.Barrier()
    measures = [ufl.ds, ufl.dx]
    if ghost_mode == dolfinx.mesh.GhostMode.shared_facet:
        measures.append(ufl.dx)
    for measure in measures:
        c_adios = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_second)))
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        assert np.isclose(
            mesh_second.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )
