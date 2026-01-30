# Copyright (C) 2023-2026 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


import inspect
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


def test_legacy_mesh(backend):
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh.h5").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    mesh = read_mesh_from_legacy_h5(filename=path, comm=comm, group="/mesh", backend=backend)
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


def test_read_legacy_mesh_from_checkpoint(backend):
    comm = MPI.COMM_WORLD
    filename = (pathlib.Path("legacy") / "mesh_checkpoint.h5").absolute()
    if not filename.exists():
        pytest.skip(f"{filename} does not exist")
    mesh = read_mesh_from_legacy_h5(
        filename=filename, comm=comm, group="/Mesh/mesh", backend=backend
    )
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


def test_legacy_function(backend):
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh.h5").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    mesh = read_mesh_from_legacy_h5(path, comm, "/mesh", backend=backend)
    V = dolfinx.fem.functionspace(mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    L = ufl.inner(f, v) * ufl.dx

    uh = dolfinx.fem.Function(V)
    if "petsc_options_prefix" in inspect.signature(LinearProblem.__init__).parameters.keys():
        extra_options = {"petsc_options_prefix": "legacy_test"}
    else:
        extra_options = {}
    problem = LinearProblem(
        a, L, bcs=[], u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, **extra_options
    )
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(path, mesh.comm, u_in, group="v", backend=backend)
    np.testing.assert_allclose(uh.x.array, u_in.x.array, atol=1e-14)

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim,)))
    wh = dolfinx.fem.Function(W)
    wh.interpolate(lambda x: (x[0], 3 * x[2], 7 * x[1]))
    w_in = dolfinx.fem.Function(W)

    read_function_from_legacy_h5(path, mesh.comm, w_in, group="w", backend=backend)

    np.testing.assert_allclose(wh.x.array, w_in.x.array, atol=1e-14)


def test_read_legacy_function_from_checkpoint(backend):
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh_checkpoint.h5").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")
    mesh = read_mesh_from_legacy_h5(path, comm, "/Mesh/mesh", backend=backend)

    V = dolfinx.fem.functionspace(mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    L = ufl.inner(f, v) * ufl.dx

    uh = dolfinx.fem.Function(V)
    if "petsc_options_prefix" in inspect.signature(LinearProblem.__init__).parameters.keys():
        extra_options = {"petsc_options_prefix": "legacy_checkpoint_test"}
    else:
        extra_options = {}
    problem = LinearProblem(
        a, L, bcs=[], u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, **extra_options
    )
    problem.solve()

    u_in = dolfinx.fem.Function(V)
    read_function_from_legacy_h5(path, mesh.comm, u_in, group="v", step=0, backend=backend)
    assert np.allclose(uh.x.array, u_in.x.array)

    # Check second step
    uh.interpolate(lambda x: x[0])
    read_function_from_legacy_h5(path, mesh.comm, u_in, group="v", step=1, backend=backend)
    assert np.allclose(uh.x.array, u_in.x.array)

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim,)))
    wh = dolfinx.fem.Function(W)
    wh.interpolate(lambda x: (x[0], 3 * x[2], 7 * x[1]))
    w_in = dolfinx.fem.Function(W)

    read_function_from_legacy_h5(path, mesh.comm, w_in, group="w", step=0, backend=backend)
    np.testing.assert_allclose(wh.x.array, w_in.x.array, atol=1e-14)

    wh.interpolate(lambda x: np.vstack((x[0], 0 * x[0], x[1])))
    read_function_from_legacy_h5(path, mesh.comm, w_in, group="w", step=1, backend=backend)
    np.testing.assert_allclose(wh.x.array, w_in.x.array, atol=1e-14)


def test_adios4dolfinx_legacy():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy_checkpoint") / "adios4dolfinx_checkpoint.bp").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")

    el = ("N1curl", 3)
    backend_args = {"engine": "BP4", "legacy": True}
    mesh = read_mesh(
        path, comm, dolfinx.mesh.GhostMode.shared_facet, backend_args=backend_args, backend="adios2"
    )

    def f(x):
        values = np.zeros((2, x.shape[1]), dtype=np.float64)
        values[0] = x[0]
        values[1] = -x[1]
        return values

    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    read_function(path, u, backend_args=backend_args, backend="adios2")

    u_ex = dolfinx.fem.Function(V)
    u_ex.interpolate(f)

    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=1e-14)


def test_legacy_vtu_mesh():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh_P1000000.vtu").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")

    mesh = read_mesh(path, comm, backend="pyvista")

    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    assert num_cells_global == 12 * 13 * 2

    area = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh))), op=MPI.SUM
    )
    assert np.isclose(area, 1.9 * 2.8)


def test_legacy_pvd():
    comm = MPI.COMM_WORLD
    path = (pathlib.Path("legacy") / "mesh_P1_func.pvd").absolute()
    if not path.exists():
        pytest.skip(f"{path} does not exist")

    mesh = read_mesh(path, comm, backend="pyvista")

    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    assert num_cells_global == 12 * 13 * 2

    area = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh))), op=MPI.SUM
    )
    assert np.isclose(area, 1.9 * 2.8)

    from adios4dolfinx.backends.pyvista.backend import read_point_data

    u = read_point_data(path, "w", mesh)
    u_ref = dolfinx.fem.Function(u.function_space)
    u_ref.interpolate(lambda x: (x[0], x[1] - x[0], np.zeros_like(x[0])))

    np.testing.assert_allclose(u.x.array, u_ref.x.array)
