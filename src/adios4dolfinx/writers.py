# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


from pathlib import Path

from mpi4py import MPI

import adios2
import numpy as np

from .adios2_helpers import ADIOSFile, resolve_adios_scope
from .structures import FunctionData, MeshData

adios2 = resolve_adios_scope(adios2)


def write_mesh(
    filename: Path,
    comm: MPI.Intracomm,
    mesh: MeshData,
    engine: str = "BP4",
    mode: adios2.Mode = adios2.Mode.Write,
    time: float = 0.0,
    io_name: str = "MeshWriter",
):
    """
    Write a mesh to file using ADIOS2

    Parameters:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        engine: ADIOS2 engine to use
        mode: ADIOS2 mode to use (write or append)
        io_name: Internal name used for the ADIOS IO object
    """

    gdim = mesh.local_geometry.shape[1]
    adios = adios2.ADIOS(comm)
    with ADIOSFile(
        adios=adios, filename=filename, mode=mode, engine=engine, io_name=io_name
    ) as adios_file:
        adios_file.file.BeginStep()
        # Write geometry
        pointvar = adios_file.io.DefineVariable(
            "Points",
            mesh.local_geometry,
            shape=[mesh.num_nodes_global, gdim],
            start=[mesh.local_geometry_pos[0], 0],
            count=[mesh.local_geometry_pos[1] - mesh.local_geometry_pos[0], gdim],
        )
        adios_file.file.Put(pointvar, mesh.local_geometry, adios2.Mode.Sync)

        # Write celltype
        adios_file.io.DefineAttribute("CellType", mesh.cell_type)

        # Write basix properties
        adios_file.io.DefineAttribute("Degree", np.array([mesh.degree], dtype=np.int32))
        adios_file.io.DefineAttribute(
            "LagrangeVariant", np.array([mesh.lagrange_variant], dtype=np.int32)
        )

        # Write topology
        num_dofs_per_cell = mesh.local_topology.shape[1]
        dvar = adios_file.io.DefineVariable(
            "Topology",
            mesh.local_topology,
            shape=[mesh.num_cells_global, num_dofs_per_cell],
            start=[mesh.local_topology_pos[0], 0],
            count=[
                mesh.local_topology_pos[1] - mesh.local_topology_pos[0],
                num_dofs_per_cell,
            ],
        )

        adios_file.file.Put(dvar, mesh.local_topology)

        # Add time step to file
        t_arr = np.array([time], dtype=np.float64)
        time_var = adios_file.io.DefineVariable(
            "MeshTime",
            t_arr,
            shape=[1],
            start=[0],
            count=[1 if comm.rank == 0 else 0],
        )
        adios_file.file.Put(time_var, t_arr)

        adios_file.file.PerformPuts()
        adios_file.file.EndStep()


def write_function(
    filename: Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    engine: str = "BP4",
    mode: adios2.Mode = adios2.Mode.Append,
    time: float = 0.0,
    io_name: str = "FunctionWriter",
):
    """
    Write a function to file using ADIOS2

    Parameters:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        engine: ADIOS2 engine to use
        mode: ADIOS2 mode to use (write or append)
        time: Time stamp associated with function
        io_name: Internal name used for the ADIOS IO object
    """
    adios = adios2.ADIOS(comm)

    with ADIOSFile(
        adios=adios, filename=filename, mode=mode, engine=engine, io_name=io_name
    ) as adios_file:
        adios_file.file.BeginStep()
        # Add mesh permutations
        pvar = adios_file.io.DefineVariable(
            "CellPermutations",
            u.cell_permutations,
            shape=[u.num_cells_global],
            start=[u.local_cell_range[0]],
            count=[u.local_cell_range[1] - u.local_cell_range[0]],
        )
        adios_file.file.Put(pvar, u.cell_permutations)
        dofmap_var = adios_file.io.DefineVariable(
            f"{u.name}_dofmap",
            u.dofmap_array,
            shape=[u.global_dofs_in_dofmap],
            start=[u.dofmap_range[0]],
            count=[u.dofmap_range[1] - u.dofmap_range[0]],
        )
        adios_file.file.Put(dofmap_var, u.dofmap_array)

        xdofmap_var = adios_file.io.DefineVariable(
            f"{u.name}_XDofmap",
            u.dofmap_offsets,
            shape=[u.num_cells_global + 1],
            start=[u.local_cell_range[0]],
            count=[u.local_cell_range[1] - u.local_cell_range[0] + 1],
        )
        adios_file.file.Put(xdofmap_var, u.dofmap_offsets)

        val_var = adios_file.io.DefineVariable(
            f"{u.name}_values",
            u.values,
            shape=[u.num_dofs_global],
            start=[u.dof_range[0]],
            count=[u.dof_range[1] - u.dof_range[0]],
        )
        adios_file.file.Put(val_var, u.values)

        # Add time step to file
        t_arr = np.array([time], dtype=np.float64)
        time_var = adios_file.io.DefineVariable(
            f"{u.name}_time",
            t_arr,
            shape=[1],
            start=[0],
            count=[1 if comm.rank == 0 else 0],
        )
        adios_file.file.Put(time_var, t_arr)
        adios_file.file.PerformPuts()
        adios_file.file.EndStep()
