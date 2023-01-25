# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


from adios4dolfinx import read_mesh_from_legacy_h5
from mpi4py import MPI
import dolfinx
import pathlib
from ufl import dx, ds
import numpy as np


def test_legacy_readers():
    comm = MPI.COMM_WORLD
    path = pathlib.Path("legacy").joinpath("mesh.h5").absolute()
    print(str(path))
    mesh = read_mesh_from_legacy_h5(comm, path, "/mesh")
    assert (mesh.topology.dim == 3)
    volume = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1*dx(domain=mesh))), op=MPI.SUM)
    surface = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(1*ds(domain=mesh))), op=MPI.SUM)
    assert np.isclose(volume, 1)
    assert np.isclose(surface, 6)

    mesh.topology.create_entities(mesh.topology.dim-1)
    num_facets = mesh.topology.index_map(mesh.topology.dim-1).size_global
    assert num_facets == 18
