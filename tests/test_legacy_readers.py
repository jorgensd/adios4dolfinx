# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


from adios4dolfinx import read_mesh_from_legacy_h5, read_function_from_legacy_h5
from mpi4py import MPI
import dolfinx
import pathlib
import ufl
import numpy as np


def test_legacy_mesh():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh.h5").absolute()
    mesh = read_mesh_from_legacy_h5(comm, path, "/mesh")
    assert mesh.topology.dim == 3
    volume = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh))),
        op=MPI.SUM,
    )
    surface = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.ds(domain=mesh))),
        op=MPI.SUM,
    )
    assert np.isclose(volume, 1)
    assert np.isclose(surface, 6)

    mesh.topology.create_entities(mesh.topology.dim - 1)
    num_facets = mesh.topology.index_map(mesh.topology.dim - 1).size_global
    assert num_facets == 18


def test_legacy_function():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh.h5").absolute()
    mesh = read_mesh_from_legacy_h5(comm, path, "/mesh")
    V = dolfinx.fem.FunctionSpace(mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    L = ufl.inner(f, v) * ufl.dx

    uh = dolfinx.fem.Function(V)
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, [], uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(mesh.comm, path, u_in)
    assert np.allclose(uh.x.array, u_in.x.array)
