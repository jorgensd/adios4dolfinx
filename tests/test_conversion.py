# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

import dolfinx
import argparse
from adios4dolfinx import read_mesh_from_legacy_checkpoint, write_mesh


def test_1():

    pass

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--file", type=str, dest="filename", default="file.h5",
#                         help="Path to mesh file (.h5)")
#     parser.add_argument("--path", type=str, dest="path", default="/func/func_0/mesh",
#                         help="Path in h5 file to mesh")
#     args = parser.parse_args()

#     # Read topology with h5py
#     # MPI.COMM_WORLD.Barrier()

#     # in_file = h5py.File(args.filename, "r", driver="mpio",
#     #                     comm=MPI.COMM_WORLD)
#     # in_file.close()

#     # Create ADIOS2 reader

#     with dolfinx.io.XDMFFile(mesh.comm, "test_mesh.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)

#     W = dolfinx.fem.FunctionSpace(mesh, ("CG", 2))

#     u = dolfinx.fem.Function(W)
#     u.interpolate(lambda x: x[0] + 3 * x[1])

#     # Open ADIOS2 Reader

#     write_mesh(mesh)
