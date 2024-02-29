# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from mpi4py import MPI

import adios2
import dolfinx
import numpy as np
import numpy.typing as npt

from .adios2_helpers import resolve_adios_scope
from .comm_helpers import numpy_to_mpi
from .utils import compute_local_range, index_owner, unroll_dofmap

adios2 = resolve_adios_scope(adios2)

__all__ = ["write_function_on_input_mesh", "write_mesh_input_order"]


def compute_insert_position(
    data_owner: npt.NDArray[np.int32],
    destination_ranks: npt.NDArray[np.int32],
    out_size: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """
    Giving a list of ranks, compute the local insert position for each rank in a list sorted by destination ranks.
    This function is used for packing data from a given process to its destination processes.

    Example:

        .. highlight:: python
        .. code-block:: python

            data_owner = [0, 1, 1, 0, 2, 3]
            destination_ranks = [2,0,3,1]
            out_size = [1, 2, 1, 2]
            insert_position = compute_insert_position(data_owner, destination_ranks, out_size)

        Insert position is then ``[1, 4, 5, 2, 0, 3]``
    """
    process_pos_indicator = data_owner.reshape(-1, 1) == destination_ranks

    # Compute offsets for insertion based on input size
    send_offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
    send_offsets[1:] = np.cumsum(out_size)
    assert send_offsets[-1] == len(data_owner)

    # Compute local insert index on each process
    proc_row, proc_col = np.nonzero(process_pos_indicator)
    cum_pos = np.cumsum(process_pos_indicator, axis=0)
    insert_position = cum_pos[proc_row, proc_col] - 1

    # Add process offset for each local index
    insert_position += send_offsets[proc_col]
    return insert_position


def unroll_insert_position(
    insert_position: npt.NDArray[np.int32], block_size: int
) -> npt.NDArray[np.int32]:
    """
    Unroll insert position by a block size

    Example:


        .. highlight:: python
        .. code-block:: python

            insert_position = [1, 4, 5, 2, 0, 3]
            unrolled_ip = unroll_insert_position(insert_position, 3)

        where ``unrolled_ip = [3, 4 ,5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 0, 1, 2, 9, 10, 11]``
    """
    unrolled_ip = np.repeat(insert_position, block_size) * block_size
    unrolled_ip += np.tile(np.arange(block_size), len(insert_position))
    return unrolled_ip


@dataclass
class MeshData:
    # 2 dimensional array of node coordinates
    local_geometry: npt.NDArray[np.floating]
    local_geometry_pos: Tuple[
        int, int
    ]  # Insert range on current process for geometry nodes
    num_nodes_global: int  # Number of nodes in global geometry array

    local_topology: npt.NDArray[
        np.int64
    ]  # 2 dimensional connecitivty array for mesh topology
    # Insert range on current process for topology
    local_topology_pos: Tuple[int, int]
    num_cells_global: int  # NUmber of cells in global topology

    cell_type: str
    degree: int
    lagrange_variant: int


@dataclass
class FunctionData:
    cell_permutations: npt.NDArray[np.uint32]
    local_cell_range: Tuple[int, int]
    num_cells_global: int
    dofmap_array: npt.NDArray[np.int64]
    dofmap_offsets: npt.NDArray[np.int64]
    dofmap_range: Tuple[int, int]
    global_dofs_in_dofmap: int
    values: npt.NDArray[np.floating]
    dof_range: Tuple[int, int]
    num_dofs_global: int


def create_original_mesh_data(mesh: dolfinx.mesh.Mesh) -> MeshData:
    """
    Store data locally on output process
    """

    # 1. Send cell indices owned by current process to the process which owned its input

    # Get the input cell index for cells owned by this process
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    original_cell_index = mesh.topology.original_cell_index[:num_owned_cells]

    # Compute owner of cells on this process based on the original cell index
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    output_cell_owner = index_owner(
        mesh.comm, original_cell_index, num_cells_global)
    local_cell_range = compute_local_range(mesh.comm, num_cells_global)

    # Compute outgoing edges from current process to outputting process
    # Computes the number of cells sent to each process at the same time
    cell_destinations, send_cells_per_proc = np.unique(
        output_cell_owner, return_counts=True
    )
    cell_to_output_comm = mesh.comm.Create_dist_graph(
        [mesh.comm.rank],
        [len(cell_destinations)],
        cell_destinations.tolist(),
        reorder=False,
    )
    cell_sources, cell_dests, _ = cell_to_output_comm.Get_dist_neighbors()
    assert np.allclose(cell_dests, cell_destinations)

    # Compute number of recieving cells
    recv_cells_per_proc = np.zeros_like(cell_sources, dtype=np.int32)
    if len(send_cells_per_proc) == 0:
        send_cells_per_proc = np.zeros(1, dtype=np.int32)
    if len(recv_cells_per_proc) == 0:
        recv_cells_per_proc = np.zeros(1, dtype=np.int32)
    send_cells_per_proc = send_cells_per_proc.astype(np.int32)
    cell_to_output_comm.Neighbor_alltoall(
        send_cells_per_proc, recv_cells_per_proc)
    assert recv_cells_per_proc.sum(
    ) == local_cell_range[1] - local_cell_range[0]
    # Pack and send cell indices (used for mapping topology dofmap later)
    cell_insert_position = compute_insert_position(
        output_cell_owner, cell_destinations, send_cells_per_proc
    )
    send_cells = np.empty_like(cell_insert_position, dtype=np.int64)
    send_cells[cell_insert_position] = original_cell_index
    recv_cells = np.empty(recv_cells_per_proc.sum(), dtype=np.int64)
    send_cells_msg = [send_cells, send_cells_per_proc, MPI.INT64_T]
    recv_cells_msg = [recv_cells, recv_cells_per_proc, MPI.INT64_T]
    cell_to_output_comm.Neighbor_alltoallv(send_cells_msg, recv_cells_msg)
    del send_cells_msg, recv_cells_msg, send_cells

    # Map received cells to the local index
    local_cell_index = recv_cells - local_cell_range[0]

    # 2. Create dofmap based on original geometry indices and re-order in the same order as original
    # cell indices on output process

    # Get original node index for all nodes (including ghosts) and convert dofmap to these indices
    original_node_index = mesh.geometry.input_global_indices
    _, num_nodes_per_cell = mesh.geometry.dofmap.shape
    local_geometry_dofmap = mesh.geometry.dofmap[:num_owned_cells, :]
    global_geometry_dofmap = original_node_index[local_geometry_dofmap.reshape(
        -1)]

    # Unroll insert position for geometry dofmap
    dofmap_insert_position = unroll_insert_position(
        cell_insert_position, num_nodes_per_cell
    )

    # Create and commmnicate connecitivity in original geometry indices
    send_geometry_dofmap = np.empty_like(
        dofmap_insert_position, dtype=np.int64)
    send_geometry_dofmap[dofmap_insert_position] = global_geometry_dofmap
    del global_geometry_dofmap
    send_sizes_dofmap = send_cells_per_proc * num_nodes_per_cell
    recv_sizes_dofmap = recv_cells_per_proc * num_nodes_per_cell
    recv_geometry_dofmap = np.empty(recv_sizes_dofmap.sum(), dtype=np.int64)
    send_geometry_dofmap_msg = [
        send_geometry_dofmap, send_sizes_dofmap, MPI.INT64_T]
    recv_geometry_dofmap_msg = [
        recv_geometry_dofmap, recv_sizes_dofmap, MPI.INT64_T]
    cell_to_output_comm.Neighbor_alltoallv(
        send_geometry_dofmap_msg, recv_geometry_dofmap_msg
    )
    del send_geometry_dofmap_msg, recv_geometry_dofmap_msg

    # Reshape dofmap and sort by original cell index
    recv_dofmap = recv_geometry_dofmap.reshape(-1, num_nodes_per_cell)
    sorted_recv_dofmap = np.empty_like(recv_dofmap)
    sorted_recv_dofmap[local_cell_index] = recv_dofmap

    # 3. Move geometry coordinates to input process
    # Compute outgoing edges from current process and create neighbourhood communicator
    # Also create number of outgoing cells at the same time
    num_owned_nodes = mesh.geometry.index_map().size_local
    num_nodes_global = mesh.geometry.index_map().size_global
    output_node_owner = index_owner(
        mesh.comm, original_node_index[:num_owned_nodes], num_nodes_global
    )

    node_destinations, send_nodes_per_proc = np.unique(
        output_node_owner, return_counts=True
    )
    geometry_to_owner_comm = mesh.comm.Create_dist_graph(
        [mesh.comm.rank],
        [len(node_destinations)],
        node_destinations.tolist(),
        reorder=False,
    )

    node_sources, node_dests, _ = geometry_to_owner_comm.Get_dist_neighbors()
    assert np.allclose(node_dests, node_destinations)

    # Compute send node insert positions
    send_nodes_position = compute_insert_position(
        output_node_owner, node_destinations, send_nodes_per_proc
    )
    unrolled_nodes_positiion = unroll_insert_position(send_nodes_position, 3)

    send_coordinates = np.empty_like(
        unrolled_nodes_positiion, dtype=mesh.geometry.x.dtype
    )
    send_coordinates[unrolled_nodes_positiion] = mesh.geometry.x[
        :num_owned_nodes, :
    ].reshape(-1)

    # Send and recieve geometry sizes
    send_coordinate_sizes = (send_nodes_per_proc * 3).astype(np.int32)
    recv_coordinate_sizes = np.zeros_like(node_sources, dtype=np.int32)
    geometry_to_owner_comm.Neighbor_alltoall(
        send_coordinate_sizes, recv_coordinate_sizes
    )

    # Send node coordinates
    recv_coordinates = np.empty(
        recv_coordinate_sizes.sum(), dtype=mesh.geometry.x.dtype
    )
    mpi_type = numpy_to_mpi[recv_coordinates.dtype.type]
    send_coord_msg = [send_coordinates, send_coordinate_sizes, mpi_type]
    recv_coord_msg = [recv_coordinates, recv_coordinate_sizes, mpi_type]
    geometry_to_owner_comm.Neighbor_alltoallv(send_coord_msg, recv_coord_msg)
    del send_coord_msg, recv_coord_msg

    # Send node ordering for reordering the coordinates on output process
    send_nodes = np.empty(num_owned_nodes, dtype=np.int64)
    send_nodes[send_nodes_position] = original_node_index[:num_owned_nodes]

    recv_indices = np.empty(recv_coordinate_sizes.sum() // 3, dtype=np.int64)
    send_nodes_msg = [send_nodes, send_nodes_per_proc, MPI.INT64_T]
    recv_nodes_msg = [recv_indices, recv_coordinate_sizes // 3, MPI.INT64_T]
    geometry_to_owner_comm.Neighbor_alltoallv(send_nodes_msg, recv_nodes_msg)

    # Compute local ording of received nodes
    local_node_range = compute_local_range(mesh.comm, num_nodes_global)
    recv_indices -= local_node_range[0]

    # Sort geometry based on input index and strip to gdim
    gdim = mesh.geometry.dim
    recv_nodes = recv_coordinates.reshape(-1, 3)
    geometry = np.empty_like(recv_nodes)
    geometry[recv_indices, :] = recv_nodes
    geometry = geometry[:, :gdim].copy()
    assert local_node_range[1] - local_node_range[0] == geometry.shape[0]
    cmap = mesh.geometry.cmap
    return MeshData(
        local_geometry=geometry,
        local_geometry_pos=local_node_range,
        num_nodes_global=num_nodes_global,
        local_topology=sorted_recv_dofmap,
        local_topology_pos=local_cell_range,
        num_cells_global=num_cells_global,
        cell_type=mesh.topology.cell_name(),
        degree=cmap.degree,
        lagrange_variant=cmap.variant,
    )


def write_mesh_input_order(
    mesh: dolfinx.mesh.Mesh, filename: Path, engine: str = "BP4"
):
    """
    Write mesh to checkpoint file in original input ordering
    """

    mesh_data = create_original_mesh_data(mesh)
    gdim = mesh_data.local_geometry.shape[1]
    assert gdim == mesh.geometry.dim
    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("OriginalMeshWriter")
    io.SetEngine("BP4")
    outfile = io.Open(str(filename), adios2.Mode.Write)

    # Write geometry
    pointvar = io.DefineVariable(
        "Points",
        mesh_data.local_geometry,
        shape=[mesh_data.num_nodes_global, gdim],
        start=[mesh_data.local_geometry_pos[0], 0],
        count=[mesh_data.local_geometry_pos[1] -
               mesh_data.local_geometry_pos[0], gdim],
    )
    outfile.Put(pointvar, mesh_data.local_geometry, adios2.Mode.Sync)

    # Write celltype
    io.DefineAttribute("CellType", mesh_data.cell_type)

    # Write basix properties
    io.DefineAttribute("Degree", np.array([mesh_data.degree], dtype=np.int32))
    io.DefineAttribute(
        "LagrangeVariant", np.array(
            [mesh_data.lagrange_variant], dtype=np.int32)
    )

    # Write topology
    num_dofs_per_cell = mesh_data.local_topology.shape[1]
    dvar = io.DefineVariable(
        "Topology",
        mesh_data.local_topology,
        shape=[mesh_data.num_cells_global, num_dofs_per_cell],
        start=[mesh_data.local_topology_pos[0], 0],
        count=[
            mesh_data.local_topology_pos[1] - mesh_data.local_topology_pos[0],
            num_dofs_per_cell,
        ],
    )

    outfile.Put(dvar, mesh_data.local_topology)
    outfile.PerformPuts()
    outfile.EndStep()
    outfile.Close()
    assert adios.RemoveIO("OriginalMeshWriter")


def create_function_data_on_original_mesh(u: dolfinx.fem.Function) -> FunctionData:
    """
    Create data object to save with ADIOS2
    """
    mesh = u.function_space.mesh

    # Compute what cells owned by current process should be sent to what output process
    # FIXME: Cache this
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    original_cell_index = mesh.topology.original_cell_index[:num_owned_cells]

    # Compute owner of cells on this process based on the original cell index
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    output_cell_owner = index_owner(
        mesh.comm, original_cell_index, num_cells_global)
    local_cell_range = compute_local_range(mesh.comm, num_cells_global)

    # Compute outgoing edges from current process to outputting process
    # Computes the number of cells sent to each process at the same time
    cell_destinations, send_cells_per_proc = np.unique(
        output_cell_owner, return_counts=True
    )
    cell_to_output_comm = mesh.comm.Create_dist_graph(
        [mesh.comm.rank],
        [len(cell_destinations)],
        cell_destinations.tolist(),
        reorder=False,
    )
    cell_sources, cell_dests, _ = cell_to_output_comm.Get_dist_neighbors()
    assert np.allclose(cell_dests, cell_destinations)

    # Compute number of recieving cells
    recv_cells_per_proc = np.zeros_like(cell_sources, dtype=np.int32)
    send_cells_per_proc = send_cells_per_proc.astype(np.int32)
    cell_to_output_comm.Neighbor_alltoall(
        send_cells_per_proc, recv_cells_per_proc)
    assert recv_cells_per_proc.sum(
    ) == local_cell_range[1] - local_cell_range[0]

    # Pack and send cell indices (used for mapping topology dofmap later)
    cell_insert_position = compute_insert_position(
        output_cell_owner, cell_destinations, send_cells_per_proc
    )
    send_cells = np.empty_like(cell_insert_position, dtype=np.int64)
    send_cells[cell_insert_position] = original_cell_index
    recv_cells = np.empty(recv_cells_per_proc.sum(), dtype=np.int64)
    send_cells_msg = [send_cells, send_cells_per_proc, MPI.INT64_T]
    recv_cells_msg = [recv_cells, recv_cells_per_proc, MPI.INT64_T]
    cell_to_output_comm.Neighbor_alltoallv(send_cells_msg, recv_cells_msg)
    del send_cells_msg, recv_cells_msg

    # Map received cells to the local index
    local_cell_index = recv_cells - local_cell_range[0]

    # Pack and send cell permutation info
    mesh.topology.create_entity_permutations()
    cell_permutation_info = mesh.topology.get_cell_permutation_info()[
        :num_owned_cells]
    send_perm = np.empty_like(send_cells, dtype=np.uint32)
    send_perm[cell_insert_position] = cell_permutation_info
    recv_perm = np.empty_like(recv_cells, dtype=np.uint32)
    send_perm_msg = [send_perm, send_cells_per_proc, MPI.UINT32_T]
    recv_perm_msg = [recv_perm, recv_cells_per_proc, MPI.UINT32_T]
    cell_to_output_comm.Neighbor_alltoallv(send_perm_msg, recv_perm_msg)
    cell_permutation_info = np.empty_like(recv_perm)
    cell_permutation_info[local_cell_index] = recv_perm

    # 2. Extract function data (array is the same, keeping global indices from DOLFINx)
    # Dofmap is moved by the original cell index similar to the mesh geometry dofmap
    dofmap = u.function_space.dofmap
    dmap = dofmap.list
    num_dofs_per_cell = dmap.shape[1]
    dofmap_bs = dofmap.bs
    index_map_bs = dofmap.index_map_bs

    # Unroll dofmap for block size
    unrolled_dofmap = unroll_dofmap(
        dofmap.list[:num_owned_cells, :], dofmap_bs)
    dmap_loc = (unrolled_dofmap // index_map_bs).reshape(-1)
    dmap_rem = (unrolled_dofmap % index_map_bs).reshape(-1)

    # Convert imap index to global index
    imap_global = dofmap.index_map.local_to_global(dmap_loc)
    dofmap_global = (imap_global * index_map_bs + dmap_rem).reshape(
        unrolled_dofmap.shape
    )
    num_dofs_per_cell = dofmap_global.shape[1]
    dofmap_insert_position = unroll_insert_position(
        cell_insert_position, num_dofs_per_cell
    )

    # Create and send array for global dofmap
    send_function_dofmap = np.empty(
        len(dofmap_insert_position), dtype=np.int64)
    send_function_dofmap[dofmap_insert_position] = dofmap_global.reshape(-1)
    send_sizes_dofmap = send_cells_per_proc * num_dofs_per_cell
    recv_size_dofmap = recv_cells_per_proc * num_dofs_per_cell
    recv_function_dofmap = np.empty(recv_size_dofmap.sum(), dtype=np.int64)
    cell_to_output_comm.Neighbor_alltoallv(
        [send_function_dofmap, send_sizes_dofmap, MPI.INT64_T],
        [recv_function_dofmap, recv_size_dofmap, MPI.INT64_T],
    )

    shaped_dofmap = recv_function_dofmap.reshape(
        local_cell_range[1] - local_cell_range[0], num_dofs_per_cell
    ).copy()
    final_dofmap = np.empty_like(shaped_dofmap)
    final_dofmap[local_cell_index] = shaped_dofmap
    final_dofmap = final_dofmap.reshape(-1)

    # Get offsets of dofmap
    num_cells_local = local_cell_range[1] - local_cell_range[0]
    num_dofs_local_dmap = num_cells_local * num_dofs_per_cell
    dofmap_imap = dolfinx.common.IndexMap(mesh.comm, num_dofs_local_dmap)
    local_dofmap_offsets = np.arange(num_cells_local + 1, dtype=np.int64)
    local_dofmap_offsets[:] *= num_dofs_per_cell
    local_dofmap_offsets[:] += dofmap_imap.local_range[0]

    num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
    num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
    local_range = (
        np.asarray(dofmap.index_map.local_range,
                   dtype=np.int64) * dofmap.index_map_bs
    )
    return FunctionData(
        cell_permutations=cell_permutation_info,
        local_cell_range=local_cell_range,
        num_cells_global=num_cells_global,
        dofmap_array=final_dofmap,
        dofmap_offsets=local_dofmap_offsets,
        values=u.x.array[:num_dofs_local].copy(),
        dof_range=local_range,
        num_dofs_global=num_dofs_global,
        dofmap_range=dofmap_imap.local_range,
        global_dofs_in_dofmap=dofmap_imap.size_global,
    )


def write_function_on_input_mesh(
    u: dolfinx.fem.Function,
    filename: Path,
    engine: str = "BP4",
    mode: adios2.Mode = adios2.Mode.Append,
    time: float = 0.0,
):
    mesh = u.function_space.mesh
    function_data = create_function_data_on_original_mesh(u)

    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("OriginalFunctionWriter")
    io.SetEngine(engine)
    outfile = io.Open(str(filename), mode)

    # Add mesh permutations
    pvar = io.DefineVariable(
        "CellPermutations",
        function_data.cell_permutations,
        shape=[function_data.num_cells_global],
        start=[function_data.local_cell_range[0]],
        count=[function_data.local_cell_range[1] -
               function_data.local_cell_range[0]],
    )
    outfile.Put(pvar, function_data.cell_permutations)
    dofmap_var = io.DefineVariable(
        f"{u.name}_dofmap",
        function_data.dofmap_array,
        shape=[function_data.global_dofs_in_dofmap],
        start=[function_data.dofmap_range[0]],
        count=[function_data.dofmap_range[1] - function_data.dofmap_range[0]],
    )
    outfile.Put(dofmap_var, function_data.dofmap_array)

    xdofmap_var = io.DefineVariable(
        f"{u.name}_XDofmap",
        function_data.dofmap_offsets,
        shape=[function_data.num_cells_global + 1],
        start=[function_data.local_cell_range[0]],
        count=[
            function_data.local_cell_range[1] -
            function_data.local_cell_range[0] + 1
        ],
    )
    outfile.Put(xdofmap_var, function_data.dofmap_offsets)

    val_var = io.DefineVariable(
        f"{u.name}_values",
        function_data.values,
        shape=[function_data.num_dofs_global],
        start=[function_data.dof_range[0]],
        count=[function_data.dof_range[1] - function_data.dof_range[0]],
    )
    outfile.Put(val_var, function_data.values)

    # Add time step to file
    t_arr = np.array([time], dtype=np.float64)
    time_var = io.DefineVariable(
        f"{u.name}_time",
        t_arr,
        shape=[1],
        start=[0],
        count=[1 if mesh.comm.rank == 0 else 0],
    )
    outfile.Put(time_var, t_arr)

    outfile.PerformPuts()
    outfile.EndStep()
    outfile.Close()
    assert adios.RemoveIO("OriginalFunctionWriter")
