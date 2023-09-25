# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


import pathlib

import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from adios4dolfinx import (read_function_from_legacy_h5,
                           read_mesh_from_legacy_h5)


def test_legacy_mesh():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh.h5").absolute()
    mesh = read_mesh_from_legacy_h5(comm=comm, filename=path, group="/mesh")
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


def test_read_legacy_mesh_from_checkpoint():
    comm = MPI.COMM_WORLD
    filename = (pathlib.Path("legacy") / "mesh_checkpoint.h5").absolute()
    mesh = read_mesh_from_legacy_h5(comm=comm, filename=filename, group="/Mesh/mesh")
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
    problem = LinearProblem(
        a, L, [], uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(mesh.comm, path, u_in, group="v")
    assert np.allclose(uh.x.array, u_in.x.array)

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim, )))
    wh = dolfinx.fem.Function(W)
    wh.interpolate(lambda x: (x[0], 3*x[2], 7*x[1]))
    w_in = dolfinx.fem.Function(W)

    read_function_from_legacy_h5(mesh.comm, path, w_in, group="w")

    assert np.allclose(wh.x.array, w_in.x.array)


def test_read_legacy_function_from_checkpoint():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh_checkpoint.h5").absolute()
    mesh = read_mesh_from_legacy_h5(comm, path, "/Mesh/mesh")

    V = dolfinx.fem.FunctionSpace(mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    L = ufl.inner(f, v) * ufl.dx

    uh = dolfinx.fem.Function(V)
    problem = LinearProblem(
        a, L, [], uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(mesh.comm, path, u_in, group="v",  step=0)
    assert np.allclose(uh.x.array, u_in.x.array)

    # Check second step
    uh.interpolate(lambda x: x[0])
    read_function_from_legacy_h5(mesh.comm, path, u_in, group="v",  step=1)
    assert np.allclose(uh.x.array, u_in.x.array)

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim, )))
    wh = dolfinx.fem.Function(W)
    wh.interpolate(lambda x: (x[0], 3*x[2], 7*x[1]))
    w_in = dolfinx.fem.Function(W)

    read_function_from_legacy_h5(mesh.comm, path, w_in, group="w", step=0)
    assert np.allclose(wh.x.array, w_in.x.array)

    wh.interpolate(lambda x: np.vstack((x[0], 0*x[0], x[1])))
    read_function_from_legacy_h5(mesh.comm, path, w_in, group="w", step=1)
    assert np.allclose(wh.x.array, w_in.x.array)
