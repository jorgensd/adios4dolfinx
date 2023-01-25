# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

import pathlib

import adios2
import basix
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from .utils import compute_local_range

__all__ = ["write_mesh", "read_mesh"]


def write_mesh(mesh: dolfinx.mesh.Mesh, filename: pathlib.Path, engine: str = "BP4"):
    """
    Write a mesh to specified ADIOS2 format, see:
    https://adios2.readthedocs.io/en/stable/engines/engines.html
    for possible formats.

    Args:
        mesh: The mesh to write to file
        filename: Path to save mesh (without file-extension)
        engine: Adios2 Engine
    """
    local_points = mesh.geometry.x
    num_xdofs_local = mesh.geometry.index_map().size_local
    num_xdofs_global = mesh.geometry.index_map().size_global
    local_range = mesh.geometry.index_map().local_range

    gdim = mesh.geometry.dim

    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshWriter")
    io.SetEngine(engine)
    outfile = io.Open(str(filename), adios2.Mode.Write)

    # Write geometry
    pointvar = io.DefineVariable(
        "Points", np.zeros((num_xdofs_local, gdim), dtype=np.float64), shape=[num_xdofs_global, gdim],
        start=[local_range[0], 0], count=[num_xdofs_local, gdim])
    outfile.Put(pointvar, local_points[:num_xdofs_local, :gdim])

    # Write celltype
    io.DefineAttribute("CellType", mesh.topology.cell_name())

    # Write basix properties
    io.DefineAttribute("Degree", np.array([mesh.geometry.cmap.degree], dtype=np.int32))
    io.DefineAttribute("LagrangeVariant",  np.array([mesh.geometry.cmap.variant], dtype=np.int32))

    # Write topology
    g_imap = mesh.geometry.index_map()
    g_dmap = mesh.geometry.dofmap
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(
        mesh.topology.dim).size_global
    start_cell = mesh.topology.index_map(mesh.topology.dim).local_range[0]
    geom_layout = mesh.geometry.cmap.create_dof_layout()
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(
        mesh.geometry.dim)

    dofs_out = np.zeros(
        (num_cells_local, num_dofs_per_cell), dtype=np.int64)

    dofs_out[:, :] = g_imap.local_to_global(
        g_dmap.array[:num_cells_local * num_dofs_per_cell
                     ]).reshape(num_cells_local, num_dofs_per_cell)

    dvar = io.DefineVariable(
        "Topology", dofs_out, shape=[num_cells_global, num_dofs_per_cell],
        start=[start_cell, 0], count=[num_cells_local, num_dofs_per_cell])
    outfile.Put(dvar, dofs_out)
    outfile.PerformPuts()
    outfile.EndStep()
    assert adios.RemoveIO("MeshWriter")


def read_mesh(comm: MPI.Comm, file: pathlib.Path, engine: str):
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("MeshReader")
    io.SetEngine(engine)
    infile = io.Open(str(file), adios2.Mode.Read)
    infile.BeginStep()

    # Get mesh cell type
    if "CellType" not in io.AvailableAttributes().keys():
        raise KeyError("Mesh cell type not found at CellType")
    celltype = io.InquireAttribute("CellType")
    cell_type = celltype.DataString()[0]

    # Get basix info
    if "LagrangeVariant" not in io.AvailableAttributes().keys():
        raise KeyError("Mesh LagrangeVariant not found")
    lvar = io.InquireAttribute("LagrangeVariant").Data()[0]
    if "Degree" not in io.AvailableAttributes().keys():
        raise KeyError("Mesh degree not found")
    degree = io.InquireAttribute("Degree").Data()[0]

    # Get mesh geometry
    if "Points" not in io.AvailableVariables().keys():
        raise KeyError("Mesh coordinates not found at Points")
    geometry = io.InquireVariable("Points")
    x_shape = geometry.Shape()
    geometry_range = compute_local_range(comm, x_shape[0])
    geometry.SetSelection([[geometry_range[0], 0], [
                          geometry_range[1]-geometry_range[0], x_shape[1]]])
    mesh_geometry = np.empty(
        (geometry_range[1]-geometry_range[0], x_shape[1]), dtype=np.float64)
    infile.Get(geometry, mesh_geometry, adios2.Mode.Deferred)

    # Get mesh topology (distributed)
    if "Topology" not in io.AvailableVariables().keys():
        raise KeyError("Mesh topology not found at Topology'")
    topology = io.InquireVariable("Topology")
    shape = topology.Shape()
    local_range = compute_local_range(comm, shape[0])
    topology.SetSelection([[local_range[0], 0], [
                          local_range[1]-local_range[0], shape[1]]])
    mesh_topology = np.empty(
        (local_range[1]-local_range[0], shape[1]), dtype=np.int64)
    infile.Get(topology, mesh_topology, adios2.Mode.Deferred)

    infile.PerformGets()
    infile.EndStep()
    assert adios.RemoveIO("MeshReader")

    # Create DOLFINx mesh
    element = basix.ufl_wrapper.create_vector_element(
        basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant(lvar),
        dim=mesh_geometry.shape[1], gdim=mesh_geometry.shape[1])
    domain = ufl.Mesh(element)

    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, mesh_topology, mesh_geometry, domain)
