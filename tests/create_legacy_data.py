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


def create_reference_data(h5_file: pathlib.Path, xdmf_file: pathlib.Path,
                          mesh_name: str, function_name: str,
                          family: str, degree: int) -> dolfin.Function:
    mesh = dolfin.UnitSquareMesh(3, 1)
    V = dolfin.FunctionSpace(mesh, family, degree)
    u = dolfin.Function(V)
    u.interpolate(dolfin.Expression("x[0]+3*x[1]", degree=degree))
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_file), "w") as hdf:
        hdf.write(mesh, mesh_name)
        hdf.write(u, function_name)
    with dolfin.XDMFFile(mesh.mpi_comm(), str(xdmf_file)) as xdmf:
        xdmf.write(mesh)
        xdmf.write_checkpoint(u, function_name, 0, dolfin.XDMFFile.Encoding.HDF5, append=True)

    with dolfin.XDMFFile(mesh.mpi_comm(), "test.xdmf") as xdmf:
        xdmf.write(mesh)
    return u


def verify_hdf5(u_ref: dolfin.Function, h5_file: pathlib.Path, mesh_name: str, function_name: str,
                family: str, degree: int):
    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), str(h5_file), "r") as hdf:
        hdf.read(mesh, mesh_name, False)
        V = dolfin.FunctionSpace(mesh, family, degree)
        w = dolfin.Function(V)
        hdf.read(w, function_name)

    assert np.allclose(w.vector().get_local(), u_ref.vector().get_local())


def verify_xdmf(u_ref: dolfin.Function, xdmf_file: pathlib.Path, function_name: str,
                family: str, degree: int):
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(mesh.mpi_comm(), str(xdmf_file)) as xdmf:
        xdmf.read(mesh)
        V = dolfin.FunctionSpace(mesh, family, degree)
        q = dolfin.Function(V)
        xdmf.read_checkpoint(q, function_name, 0)

    assert np.allclose(q.vector().get_local(), u_ref.vector().get_local())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--family", type=str, default="Lagrange")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="legacy", dest="dir")
    parser.add_argument("--mesh-name", type=str, default="mesh", dest="name")
    parser.add_argument("--function-name", type=str, default="mesh", dest="f_name")

    inputs = parser.parse_args()
    path = pathlib.Path(inputs.dir)
    if not os.path.exists(path):
        os.mkdir(path)
    h5_filename = path/f"{inputs.name}.h5"
    xdmf_filename = path/f"{inputs.name}_checkpoint.xdmf"

    u_ref = create_reference_data(h5_filename, xdmf_filename, inputs.name, inputs.f_name,
                                  inputs.family, inputs.degree)

    verify_hdf5(u_ref, h5_filename, inputs.name, inputs.f_name,
                inputs.family, inputs.degree)

    verify_xdmf(u_ref, xdmf_filename, inputs.f_name,
                inputs.family, inputs.degree)
