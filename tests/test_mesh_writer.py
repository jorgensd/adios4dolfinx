import time

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from adios4dolfinx import read_mesh, write_mesh


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

    start = time.perf_counter()
    write_mesh(file.with_suffix(suffix), mesh, encoder)
    end = time.perf_counter()
    print(f"Write ADIOS2 mesh: {end-start}")

    mesh.comm.Barrier()
    start = time.perf_counter()
    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
    end = time.perf_counter()
    print(f"Write XDMF mesh: {end-start}")
    mesh.comm.Barrier()

    start = time.perf_counter()
    mesh_adios = read_mesh(file.with_suffix(suffix), MPI.COMM_WORLD, encoder, ghost_mode)
    end = time.perf_counter()
    print(f"Read ADIOS2 mesh: {end-start}")
    mesh.comm.Barrier()

    start = time.perf_counter()
    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "r") as xdmf:
        mesh_xdmf = xdmf.read_mesh(ghost_mode=ghost_mode)
    end = time.perf_counter()
    print(f"Read XDMF mesh: {end-start}")

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
