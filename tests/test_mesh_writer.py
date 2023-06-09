from adios4dolfinx import write_mesh, read_mesh
import dolfinx
from mpi4py import MPI
import pathlib
import time
import pytest
import ufl
import numpy as np


@pytest.mark.parametrize("encoder, suffix", [("BP4", ".bp"), ("HDF5", ".h5")])
# , ("BP5", ".bp")]) # Deactivated, see: https://github.com/jorgensd/adios4dolfinx/issues/7
@pytest.mark.parametrize("ghost_mode", [dolfinx.mesh.GhostMode.shared_facet])
def test_mesh_read_writer(encoder, suffix, ghost_mode):
    N = 25
    file = pathlib.Path(f"output/adios_mesh_{encoder}")
    xdmf_file = pathlib.Path("output/xdmf_mesh")
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=ghost_mode)

    start = time.perf_counter()
    write_mesh(mesh, file.with_suffix(suffix), encoder)
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
    mesh_adios = read_mesh(
        MPI.COMM_WORLD, file.with_suffix(suffix), encoder, ghost_mode
    )
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
        c_adios = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(1 * measure(domain=mesh_adios))
        )
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        c_xdmf = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(1 * measure(domain=mesh_xdmf))
        )
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_xdmf, MPI.SUM),
        )
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )
