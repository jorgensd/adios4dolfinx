# Copyright (C) 2024-2026 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from pathlib import Path
from typing import Any, Literal

from mpi4py import MPI

import adios2
import dolfinx
import numpy as np
from packaging.version import Version

from .backends import FileMode, get_backend
from .backends.adios2.helpers import ADIOSFile, check_variable_exists, resolve_adios_scope
from .structures import FunctionData, MeshData

adios2 = resolve_adios_scope(adios2)


def prepare_meshdata_for_storage(mesh: dolfinx.mesh.Mesh, store_partition_info: bool) -> MeshData:
    """
    Helper function for extracting the required data from a distributed
    {py:class}`dolfinx.mesh.Mesh`.

    Args:
        mesh: The mesh
        store_partition_info: If one should store the partitioning info
    Returns:
        Data-container with the info that should be stored.
    """

    num_xdofs_local = mesh.geometry.index_map().size_local
    num_xdofs_global = mesh.geometry.index_map().size_global
    geometry_range = mesh.geometry.index_map().local_range
    gdim = mesh.geometry.dim

    # Convert local connectivity to globa l connectivity
    g_imap = mesh.geometry.index_map()
    g_dmap = mesh.geometry.dofmap
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    cell_range = mesh.topology.index_map(mesh.topology.dim).local_range
    cmap = mesh.geometry.cmap
    geom_layout = cmap.create_dof_layout()
    if hasattr(geom_layout, "num_entity_closure_dofs"):
        num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh.topology.dim)
    else:
        num_dofs_per_cell = len(geom_layout.entity_closure_dofs(mesh.topology.dim, 0))
    dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
    assert g_dmap.shape[1] == num_dofs_per_cell
    dofs_out[:, :] = np.asarray(
        g_imap.local_to_global(g_dmap[:num_cells_local, :].reshape(-1))
    ).reshape(dofs_out.shape)

    if store_partition_info:
        partition_processes = mesh.comm.size

        # Get partitioning
        if Version(dolfinx.__version__) > Version("0.9.0"):
            consensus_tag = 1202
            cell_map = mesh.topology.index_map(mesh.topology.dim).index_to_dest_ranks(consensus_tag)
        else:
            cell_map = mesh.topology.index_map(mesh.topology.dim).index_to_dest_ranks()
        num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
        cell_offsets = cell_map.offsets[: num_cells_local + 1]
        if cell_offsets[-1] == 0:
            cell_array = np.empty(0, dtype=np.int32)
        else:
            cell_array = cell_map.array[: cell_offsets[-1]]

        # Compute adjacency with current process as first entry
        ownership_array = np.full(num_cells_local + cell_offsets[-1], -1, dtype=np.int32)
        ownership_offset = cell_offsets + np.arange(len(cell_offsets), dtype=np.int32)
        ownership_array[ownership_offset[:-1]] = mesh.comm.rank
        insert_position = np.flatnonzero(ownership_array == -1)
        ownership_array[insert_position] = cell_array

        partition_map = dolfinx.common.IndexMap(mesh.comm, ownership_array.size)
        ownership_offset += partition_map.local_range[0]
        partition_range = partition_map.local_range
        partition_global = partition_map.size_global
    else:
        partition_processes = None
        ownership_array = None
        ownership_offset = None
        partition_range = None
        partition_global = None

    return MeshData(
        local_geometry=mesh.geometry.x[:num_xdofs_local, :gdim].copy(),
        local_geometry_pos=geometry_range,
        num_nodes_global=num_xdofs_global,
        local_topology=dofs_out,
        local_topology_pos=cell_range,
        num_cells_global=num_cells_global,
        cell_type=mesh.topology.cell_name(),
        degree=mesh.geometry.cmap.degree,
        lagrange_variant=mesh.geometry.cmap.variant,
        store_partition=store_partition_info,
        partition_processes=partition_processes,
        ownership_array=ownership_array,
        ownership_offset=ownership_offset,
        partition_range=partition_range,
        partition_global=partition_global,
    )


def write_mesh(
    filename: Path,
    comm: MPI.Intracomm,
    mesh_data: MeshData,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
    backend_args: dict[str, Any] | None = None,
    backend: Literal["adios2", "h5py"] = "adios2",
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
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    backend_cls.write_mesh(filename, comm, mesh_data, backend_args, mode, time)


def write_function(
    filename: Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    engine: str = "BP4",
    mode: FileMode = FileMode.append,
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

    cell_permutations_exists = False
    dofmap_exists = False
    XDofmap_exists = False
    if mode == adios2.Mode.Append:
        cell_permutations_exists = check_variable_exists(
            adios, filename, "CellPermutations", engine=engine
        )
        dofmap_exists = check_variable_exists(adios, filename, f"{u.name}_dofmap", engine=engine)
        XDofmap_exists = check_variable_exists(adios, filename, f"{u.name}_XDofmap", engine=engine)

    with ADIOSFile(
        adios=adios, filename=filename, mode=mode, engine=engine, io_name=io_name, comm=comm
    ) as adios_file:
        adios_file.file.BeginStep()

        if not cell_permutations_exists:
            # Add mesh permutations
            pvar = adios_file.io.DefineVariable(
                "CellPermutations",
                u.cell_permutations,
                shape=[u.num_cells_global],
                start=[u.local_cell_range[0]],
                count=[u.local_cell_range[1] - u.local_cell_range[0]],
            )
            adios_file.file.Put(pvar, u.cell_permutations)

        if not dofmap_exists:
            # Add dofmap
            dofmap_var = adios_file.io.DefineVariable(
                f"{u.name}_dofmap",
                u.dofmap_array,
                shape=[u.global_dofs_in_dofmap],
                start=[u.dofmap_range[0]],
                count=[u.dofmap_range[1] - u.dofmap_range[0]],
            )
            adios_file.file.Put(dofmap_var, u.dofmap_array)

        if not XDofmap_exists:
            # Add XDofmap
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
