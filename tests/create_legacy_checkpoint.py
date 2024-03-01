# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


"""
Functions to create checkpoints with adios4dolfinx v0.7.x
"""

import argparse
import pathlib
from importlib.metadata import version

from mpi4py import MPI

import dolfinx
import numpy as np

import adios4dolfinx

a4d_version = version("adios4dolfinx")
assert (
    a4d_version < "0.7.2"
), f"Creating a legacy checkpoint requires adios4dolfinx < 0.7.2, you have {a4d_version}."


def f(x):
    values = np.zeros((2, x.shape[1]), dtype=np.float64)
    values[0] = x[0]
    values[1] = -x[1]
    return values


def write_checkpoint(filename, mesh, el, f):
    V = dolfinx.fem.FunctionSpace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=np.float64)
    uh.interpolate(f)

    adios4dolfinx.write_mesh(V.mesh, filename)
    adios4dolfinx.write_function(uh, filename)


def verify_checkpoint(filename, el, f):
    mesh = adios4dolfinx.read_mesh(
        MPI.COMM_WORLD, filename, "BP4", dolfinx.mesh.GhostMode.shared_facet
    )
    V = dolfinx.fem.FunctionSpace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=np.float64)
    adios4dolfinx.read_function(uh, filename)

    u_ex = dolfinx.fem.Function(V, dtype=np.float64)
    u_ex.interpolate(f)

    np.testing.assert_allclose(u_ex.x.array, uh.x.array, atol=1e-15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=str, default="legacy_checkpoint", dest="dir")

    inputs = parser.parse_args()
    path = pathlib.Path(inputs.dir)
    path.mkdir(exist_ok=True, parents=True)
    filename = path / "adios4dolfinx_checkpoint.bp"

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    el = ("N1curl", 3)
    write_checkpoint(filename, mesh,  el, f)
    MPI.COMM_WORLD.Barrier()
    verify_checkpoint(filename, el, f)
