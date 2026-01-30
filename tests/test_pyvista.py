from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import adios4dolfinx

pyvista = pytest.importorskip("pyvista")


def test_read_mesh(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "grid.vtu"
    if MPI.COMM_WORLD.rank == 0:
        grid = pyvista.examples.load_hexbeam()
        grid.save(filename)
    MPI.COMM_WORLD.barrier()

    mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="pyvista")

    vol = dolfinx.fem.form(1 * ufl.dx(domain=mesh))
    surf = dolfinx.fem.form(1 * ufl.ds(domain=mesh))

    vol_ref = 5 * 1 * 1
    surf_ref = 5 * 4 + 2

    vol_glob = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(vol), op=MPI.SUM)
    surf_glob = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(surf), op=MPI.SUM)
    assert np.isclose(vol_glob, vol_ref)
    assert np.isclose(surf_glob, surf_ref)
