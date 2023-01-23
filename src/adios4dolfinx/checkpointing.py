# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

import adios2
import numpy as np
#from mpi4py import MPI
import dolfinx
#from .utils import compute_local_range

__all__ = ["write_mesh"]


def write_mesh(mesh: dolfinx.mesh.Mesh, filename: str, engine: str = "BP4"):
    """
    Write a mesh to specified ADIOS2 format, see: https://adios2.readthedocs.io/en/stable/engines/engines.html
    for possible formats.

    Args:
        mesh: The mesh to write to file
        filename: Path to save mesh (without file-extension)
        engine: Adios2 Engine
    """
    comm = mesh.comm
    local_points = mesh.geometry.x
    num_xdofs_local = mesh.geometry.index_map().size_local
    num_xdofs_global = mesh.geometry.index_map().size_local
    local_range = mesh.geometry.index_map().local_range

    gdim = mesh.geometry.dim

    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshWriter")
    io.SetEngine(engine)
    outfile = io.Open(filename, adios2.Mode.Write)

    # Write geometry
    pointvar = io.DefineVariable(
        "Points", local_points[:num_xdofs_local, :], shape=[num_xdofs_global, local_points.shape[1]],
        start=[local_range[0], 0], count=[num_xdofs_local, local_points.shape[1]])
    outfile.Put(pointvar, local_points[:num_xdofs_local, :])

    # Write celltype
    io.DefineAttribute("CellType", mesh.topology.cell_name())

    # Write topology
    g_imap = mesh.geometry.index_map()
    g_dmap = mesh.geometry.dofmap
    l_start = g_imap.local_range[0]
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(
        mesh.topology.dim).size_global
    start_cell = mesh.topology.index_map(mesh.topology.dim).local_range[0]
    geom_layout = mesh.geometry.cmap.create_dof_layout()
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(
        mesh.geometry.dim)

    dofs_out = np.zeros(
        (num_cells_local, num_dofs_per_cell), dtype=np.int64)
    dofs_out[:, :] = g_dmap.array[:num_cells_local *
                                  num_dofs_per_cell].reshape(num_cells_local, num_dofs_per_cell) + l_start

    dvar = io.DefineVariable(
        "Dofmap", dofs_out, shape=[num_cells_global, num_dofs_per_cell],
        start=[start_cell, 0], count=[num_cells_local, num_dofs_per_cell])
    outfile.Put(dvar, dofs_out)
    outfile.PerformPuts()
    assert adios.RemoveIO("MeshWriter")
