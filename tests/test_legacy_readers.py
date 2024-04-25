# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


import pathlib

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx.fem.petsc import LinearProblem

from adios4dolfinx import (
    read_function,
    read_function_from_legacy_h5,
    read_mesh,
    read_mesh_from_legacy_h5,
)


def test_legacy_mesh():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh.h5").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    mesh = read_mesh_from_legacy_h5(filename=path, comm=comm, group="/mesh")
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
    if not filename.exists():
        pytest.skip(f"{filename} does not exist")
    mesh = read_mesh_from_legacy_h5(filename=filename, comm=comm, group="/Mesh/mesh")
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
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    mesh = read_mesh_from_legacy_h5(path, comm, "/mesh")
    V = dolfinx.fem.functionspace(mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    L = ufl.inner(f, v) * ufl.dx

    uh = dolfinx.fem.Function(V)
    problem = LinearProblem(a, L, [], uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(path, mesh.comm, u_in, group="v")
    np.testing.assert_allclose(uh.x.array, u_in.x.array, atol=1e-14)

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim,)))
    wh = dolfinx.fem.Function(W)
    wh.interpolate(lambda x: (x[0], 3 * x[2], 7 * x[1]))
    w_in = dolfinx.fem.Function(W)

    read_function_from_legacy_h5(path, mesh.comm, w_in, group="w")

    np.testing.assert_allclose(wh.x.array, w_in.x.array, atol=1e-14)


def test_read_legacy_function_from_checkpoint():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh_checkpoint.h5").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    mesh = read_mesh_from_legacy_h5(path, comm, "/Mesh/mesh")

    V = dolfinx.fem.functionspace(mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    L = ufl.inner(f, v) * ufl.dx

    uh = dolfinx.fem.Function(V)
    problem = LinearProblem(a, L, [], uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(path, mesh.comm, u_in, group="v", step=0)
    assert np.allclose(uh.x.array, u_in.x.array)

    # Check second step
    uh.interpolate(lambda x: x[0])
    read_function_from_legacy_h5(path, mesh.comm, u_in, group="v", step=1)
    assert np.allclose(uh.x.array, u_in.x.array)

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim,)))
    wh = dolfinx.fem.Function(W)
    wh.interpolate(lambda x: (x[0], 3 * x[2], 7 * x[1]))
    w_in = dolfinx.fem.Function(W)

    read_function_from_legacy_h5(path, mesh.comm, w_in, group="w", step=0)
    np.testing.assert_allclose(wh.x.array, w_in.x.array, atol=1e-14)

    wh.interpolate(lambda x: np.vstack((x[0], 0 * x[0], x[1])))
    read_function_from_legacy_h5(path, mesh.comm, w_in, group="w", step=1)
    np.testing.assert_allclose(wh.x.array, w_in.x.array, atol=1e-14)


def test_adios4dolfinx_legacy():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy_checkpoint") / "adios4dolfinx_checkpoint.bp").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")

    el = ("N1curl", 3)
    mesh = read_mesh(path, comm, "BP4", dolfinx.mesh.GhostMode.shared_facet, legacy=True)

    def f(x):
        values = np.zeros((2, x.shape[1]), dtype=np.float64)
        values[0] = x[0]
        values[1] = -x[1]
        return values

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    read_function(path, u, engine="BP4", legacy=True)

    u_ex = dolfinx.fem.Function(V)
    u_ex.interpolate(f)

    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=1e-14)
