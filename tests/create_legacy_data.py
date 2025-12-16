# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


"""
Functions to create checkpoints with Legacy dolfin
"""

import argparse
import pathlib

import dolfin
import numpy as np
import ufl_legacy as ufl


def create_reference_data(
    h5_file: pathlib.Path,
    xdmf_file: pathlib.Path,
    mesh_name: str,
    function_name: str,
    family: str,
    degree: int,
    function_name_vec: str,
    cellfunction_name: str = "cell_function",
    facetfunction_name: str = "facet_function",
) -> dolfin.Function:
    mesh = dolfin.UnitCubeMesh(2, 2, 2)
    V = dolfin.FunctionSpace(mesh, family, degree)
    W = dolfin.VectorFunctionSpace(mesh, family, degree)
    x = dolfin.SpatialCoordinate(mesh)

    class LeftCell(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] <= 0.5 + dolfin.DOLFIN_EPS

    left_cell = LeftCell()
    cfun = dolfin.MeshFunction("size_t", mesh, 3, mesh.domains())
    cfun.set_all(0)
    left_cell.mark(cfun, 1)

    class RightFacet(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] >= 0.5 - dolfin.DOLFIN_EPS and on_boundary

    ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())
    ffun.set_all(0)
    right_facet = RightFacet()
    right_facet.mark(ffun, 2)

    f0 = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    v0 = dolfin.project(f0, V)
    w0 = dolfin.interpolate(dolfin.Expression(("x[0]", "3*x[2]", "7*x[1]"), degree=1), W)

    v1 = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), V)
    w1 = dolfin.interpolate(dolfin.Expression(("x[0]", "0", "x[1]"), degree=1), W)

    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_file), "w") as hdf:
        hdf.write(mesh, mesh_name)
        hdf.write(v0, function_name)
        hdf.write(w0, function_name_vec)
        hdf.write(cfun, cellfunction_name)
        hdf.write(ffun, facetfunction_name)

    with dolfin.XDMFFile(mesh.mpi_comm(), str(xdmf_file)) as xdmf:
        xdmf.write(mesh)
        xdmf.write_checkpoint(v0, function_name, 0, dolfin.XDMFFile.Encoding.HDF5, append=True)
        xdmf.write_checkpoint(w0, function_name_vec, 0, dolfin.XDMFFile.Encoding.HDF5, append=True)
        xdmf.write_checkpoint(v1, function_name, 1, dolfin.XDMFFile.Encoding.HDF5, append=True)
        xdmf.write_checkpoint(w1, function_name_vec, 1, dolfin.XDMFFile.Encoding.HDF5, append=True)

    with dolfin.XDMFFile(mesh.mpi_comm(), "test.xdmf") as xdmf:
        xdmf.write(mesh)
    return v0, w0, v1, w1


def verify_hdf5(
    v_ref: dolfin.Function,
    w_ref: dolfin.Function,
    h5_file: pathlib.Path,
    mesh_name: str,
    function_name: str,
    family: str,
    degree: int,
    function_name_vec: str,
):
    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_file), "r") as hdf:
        hdf.read(mesh, mesh_name, False)
        V = dolfin.FunctionSpace(mesh, family, degree)
        v = dolfin.Function(V)
        hdf.read(v, function_name)

        W = dolfin.VectorFunctionSpace(mesh, family, degree)
        w = dolfin.Function(W)
        hdf.read(w, function_name_vec)

    assert np.allclose(v.vector().get_local(), v_ref.vector().get_local())
    assert np.allclose(w.vector().get_local(), w_ref.vector().get_local())


def verify_xdmf(
    v0_ref: dolfin.Function,
    w0_ref: dolfin.Function,
    v1_ref: dolfin.Function,
    w1_ref: dolfin.Function,
    xdmf_file: pathlib.Path,
    function_name: str,
    family: str,
    degree: int,
    function_name_vec: str,
):
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(mesh.mpi_comm(), str(xdmf_file)) as xdmf:
        xdmf.read(mesh)
        V = dolfin.FunctionSpace(mesh, family, degree)
        v0 = dolfin.Function(V)
        xdmf.read_checkpoint(v0, function_name, 0)
        v1 = dolfin.Function(V)
        xdmf.read_checkpoint(v1, function_name, 1)

        W = dolfin.VectorFunctionSpace(mesh, family, degree)
        w0 = dolfin.Function(W)
        xdmf.read_checkpoint(w0, function_name_vec, 0)
        w1 = dolfin.Function(W)
        xdmf.read_checkpoint(w1, function_name_vec, 1)

    assert np.allclose(v0.vector().get_local(), v0_ref.vector().get_local())
    assert np.allclose(w0.vector().get_local(), w0_ref.vector().get_local())

    assert np.allclose(v1.vector().get_local(), v1_ref.vector().get_local())
    assert np.allclose(w1.vector().get_local(), w1_ref.vector().get_local())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--family", type=str, default="DG")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="legacy", dest="dir")
    parser.add_argument("--mesh-name", type=str, default="mesh", dest="name")
    parser.add_argument("--function-name", type=str, default="v", dest="f_name")
    parser.add_argument("--function-name-vec", type=str, default="w", dest="f_name_vec")
    parser.add_argument(
        "--cellfunction-name", type=str, default="cell_function", dest="cellfunction_name"
    )
    parser.add_argument(
        "--facetfunction-name", type=str, default="facet_function", dest="facetfunction_name"
    )

    inputs = parser.parse_args()
    path = pathlib.Path(inputs.dir)
    path.mkdir(exist_ok=True, parents=True)
    h5_filename = path / f"{inputs.name}.h5"
    xdmf_filename = path / f"{inputs.name}_checkpoint.xdmf"

    v0_ref, w0_ref, v1_ref, w1_ref = create_reference_data(
        h5_filename,
        xdmf_filename,
        inputs.name,
        inputs.f_name,
        inputs.family,
        inputs.degree,
        inputs.f_name_vec,
        inputs.cellfunction_name,
        inputs.facetfunction_name,
    )

    verify_hdf5(
        v0_ref,
        w0_ref,
        h5_filename,
        inputs.name,
        inputs.f_name,
        inputs.family,
        inputs.degree,
        inputs.f_name_vec,
    )

    verify_xdmf(
        v0_ref,
        w0_ref,
        v1_ref,
        w1_ref,
        xdmf_filename,
        inputs.f_name,
        inputs.family,
        inputs.degree,
        inputs.f_name_vec,
    )
