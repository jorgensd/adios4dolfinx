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

__all__ = ["read_mesh_from_legacy_checkpoint", "read_mesh_from_legacy_h5"]


def read_mesh_from_legacy_h5(comm: MPI.Comm,
                             filename: pathlib.Path,
                             meshname: str) -> dolfinx.mesh.Mesh:

    # Create ADIOS2 reader
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("Mesh reader")
    io.SetEngine("HDF5")

    # Open ADIOS2 Reader
    infile = io.Open(str(filename), adios2.Mode.Read)
    # Get mesh topology (distributed)
    if f"{meshname}/topology" not in io.AvailableVariables().keys():
        raise KeyError(f"Mesh topology not found at '{meshname}/topology'")
    topology = io.InquireVariable(f"{meshname}/topology")
    shape = topology.Shape()
    local_range = compute_local_range(MPI.COMM_WORLD, shape[0])
    topology.SetSelection([[local_range[0], 0], [
                          local_range[1]-local_range[0], shape[1]]])

    mesh_topology = np.empty(
        (local_range[1]-local_range[0], shape[1]), dtype=np.int64)
    infile.Get(topology, mesh_topology, adios2.Mode.Sync)

    # Get mesh cell type
    if f"{meshname}/topology/celltype" not in io.AvailableAttributes().keys():
        raise KeyError(
            f"Mesh cell type not found at '{meshname}/topology/celltype'")
    celltype = io.InquireAttribute(f"{meshname}/topology/celltype")
    cell_type = celltype.DataString()[0]

    # Get mesh geometry
    if f"{meshname}/coordinates" not in io.AvailableVariables().keys():
        raise KeyError(
            f"Mesh coordintes not found at '{meshname}/coordinates'")
    geometry = io.InquireVariable(f"{meshname}/coordinates")
    shape = geometry.Shape()
    local_range = compute_local_range(MPI.COMM_WORLD, shape[0])
    geometry.SetSelection([[local_range[0], 0], [
                          local_range[1]-local_range[0], shape[1]]])
    mesh_geometry = np.empty(
        (local_range[1]-local_range[0], shape[1]), dtype=np.float64)
    infile.Get(geometry, mesh_geometry, adios2.Mode.Sync)

    assert adios.RemoveIO("Mesh reader")

    # Create DOLFINx mesh
    element = basix.ufl_wrapper.create_vector_element(
        basix.ElementFamily.P, cell_type, 1, basix.LagrangeVariant.equispaced,
        dim=mesh_geometry.shape[1], gdim=mesh_geometry.shape[1])
    domain = ufl.Mesh(element)
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, mesh_topology, mesh_geometry, domain)


def read_mesh_from_legacy_checkpoint(
        filename: str, cell_type: str = "tetrahedron") -> dolfinx.mesh.Mesh:
    """
    Read mesh from `h5`-file generated by legacy DOLFIN `XDMFFile.write_checkpoint`.
    Needs to get the `cell_type` as input, as legacy DOLFIN does not store the cell-type in the
    `h5`-file

    Args:
        filename: Path to `h5` file (with extension)
        celltype: String describing cell-type
    """

    adios = adios2.ADIOS(MPI.COMM_WORLD)
    io = adios.DeclareIO("Mesh reader")
    io.SetEngine("HDF5")

    # Open ADIOS2 Reader
    infile = io.Open(filename, adios2.Mode.Read)
    path = "func/func_0"
    # Get mesh topology (distributed)
    if f"{path}/topology" not in io.AvailableVariables().keys():
        raise KeyError(f"Mesh topology not found at '{path}topology'")
    topology = io.InquireVariable(f"{path}/topology")
    shape = topology.Shape()
    local_range = compute_local_range(MPI.COMM_WORLD, shape[0])
    topology.SetSelection([[local_range[0], 0], [
                          local_range[1]-local_range[0], shape[1]]])

    mesh_topology = np.empty(
        (local_range[1]-local_range[0], shape[1]), dtype=np.int32)
    infile.Get(topology, mesh_topology, adios2.Mode.Sync)

    # Get mesh geometry
    if f"{path}/geometry" not in io.AvailableVariables().keys():
        raise KeyError(
            f"Mesh geometry not found at '{path}/geometry'")
    geometry = io.InquireVariable(f"{path}/geometry")
    shape = geometry.Shape()
    local_range = compute_local_range(MPI.COMM_WORLD, shape[0])
    geometry.SetSelection([[local_range[0], 0], [
                          local_range[1]-local_range[0], shape[1]]])
    mesh_geometry = np.empty(
        (local_range[1]-local_range[0], shape[1]), dtype=np.float64)
    infile.Get(geometry, mesh_geometry, adios2.Mode.Sync)

    assert adios.RemoveIO("Mesh reader")

    # Create DOLFINx mesh
    element = basix.ufl_wrapper.create_vector_element(
        basix.ElementFamily.P, cell_type, 1, basix.LagrangeVariant.equispaced,
        dim=mesh_geometry.shape[1], gdim=mesh_geometry.shape[1])
    domain = ufl.Mesh(element)

    return dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, mesh_topology, mesh_geometry, domain)
