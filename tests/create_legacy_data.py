# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


"""
Functions to create checkpoints with Legacy dolfin
"""

import dolfin
import numpy as np
import os
import pathlib
import argparse
import ufl


def create_reference_data(
    h5_file: pathlib.Path,
    xdmf_file: pathlib.Path,
    mesh_name: str,
    function_name: str,
    family: str,
    degree: int,
    function_name_vec: str
) -> dolfin.Function:
    mesh = dolfin.UnitCubeMesh(1, 1, 1)
    V = dolfin.FunctionSpace(mesh, family, degree)
    W = dolfin.VectorFunctionSpace(mesh, family, degree)
    x = dolfin.SpatialCoordinate(mesh)
    f = ufl.conditional(ufl.gt(x[0], 0.5), x[1], 2 * x[0])
    v = dolfin.project(f, V)
    w = dolfin.interpolate(dolfin.Expression(("1", "0", "0"), degree=1), W)
    
        
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_file), "w") as hdf:
        hdf.write(mesh, mesh_name)
        hdf.write(v, function_name)
        hdf.write(w, function_name_vec)

    
    with dolfin.XDMFFile(mesh.mpi_comm(), str(xdmf_file)) as xdmf:
        xdmf.write(mesh)
        xdmf.write_checkpoint(
            v, function_name, 0, dolfin.XDMFFile.Encoding.HDF5, append=True
        )
        xdmf.write_checkpoint(
            w, function_name_vec, 0, dolfin.XDMFFile.Encoding.HDF5, append=True
        )

    with dolfin.XDMFFile(mesh.mpi_comm(), "test.xdmf") as xdmf:
        xdmf.write(mesh)
    return v, w


def verify_hdf5(
    v_ref: dolfin.Function,
    w_ref: dolfin.Function,
    h5_file: pathlib.Path,
    mesh_name: str,
    function_name: str,
    family: str,
    degree: int,
    function_name_vec: str
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
    v_ref: dolfin.Function,
    w_ref: dolfin.Function,
    xdmf_file: pathlib.Path,
    function_name: str,
    family: str,
    degree: int,
    function_name_vec: str
):
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(mesh.mpi_comm(), str(xdmf_file)) as xdmf:
        xdmf.read(mesh)
        V = dolfin.FunctionSpace(mesh, family, degree)
        v = dolfin.Function(V)
        xdmf.read_checkpoint(v, function_name, 0)

        W = dolfin.VectorFunctionSpace(mesh, family, degree)
        w = dolfin.Function(W)
        xdmf.read_checkpoint(w, function_name_vec, 0)

    assert np.allclose(v.vector().get_local(), v_ref.vector().get_local())
    assert np.allclose(w.vector().get_local(), w_ref.vector().get_local())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--family", type=str, default="DG")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="legacy", dest="dir")
    parser.add_argument("--mesh-name", type=str, default="mesh", dest="name")
    parser.add_argument("--function-name", type=str, default="v", dest="f_name")
    parser.add_argument("--function-name-vec", type=str, default="w", dest="f_name_vec")

    inputs = parser.parse_args()
    path = pathlib.Path(inputs.dir)
    if not os.path.exists(path):
        os.mkdir(path)
    h5_filename = path / f"{inputs.name}.h5"
    xdmf_filename = path / f"{inputs.name}_checkpoint.xdmf"

    v_ref, w_ref = create_reference_data(
        h5_filename,
        xdmf_filename,
        inputs.name,
        inputs.f_name,
        inputs.family,
        inputs.degree,
        inputs.f_name_vec

    )

    verify_hdf5(
        v_ref, w_ref, h5_filename, inputs.name, inputs.f_name, inputs.family, inputs.degree, inputs.f_name_vec,
    )

    verify_xdmf(
        v_ref, w_ref, xdmf_filename, inputs.f_name, inputs.family, inputs.degree, inputs.f_name_vec,
    )
