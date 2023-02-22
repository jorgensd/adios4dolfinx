
import numpy as np
import numpy.typing as npt
import pathlib
from .utils import find_first, index_owner, compute_local_range, compute_dofmap_pos
from .adios2_helpers import read_array, read_dofmap, read_dofmap_new, read_cell_perms
from mpi4py import MPI
from typing import Tuple
import dolfinx.cpp
import dolfinx
__all__ = ["send_cells_and_receive_dofmap_index",
           "send_dofs_and_receive_values", "send_cells_and_cell_perms"]
"""
Helpers for sending and receiving values for checkpointing
"""


def send_dofmap_get_vals(comm: MPI.Intracomm,
                         source_ranks: npt.NDArray[np.int32],
                         dest_ranks: npt.NDArray[np.int32],
                         output_owners: npt.NDArray[np.int32],
                         input_cells: npt.NDArray[np.int64],
                         dofmap_pos: npt.NDArray[np.int32],
                         num_cells_global: np.int64,
                         values: npt.NDArray[np.float64],
                         dofmap_offsets: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    """
    Given a set of positions in input dofmap, give the global input index of this dofmap entry
    in input file.
    """

    # Compute amount of data to send to each process
    out_size = np.zeros(len(dest_ranks), dtype=np.int32)
    for owner in output_owners:
        proc_pos = find_first(owner, dest_ranks)
        out_size[proc_pos] += 1
        del proc_pos
    recv_size = np.zeros(len(source_ranks), dtype=np.int32)
    mesh_to_data_comm = comm.Create_dist_graph_adjacent(source_ranks.tolist(), dest_ranks.tolist(), reorder=False)
    # Send sizes to create data structures for receiving from NeighAlltoAllv
    mesh_to_data_comm.Neighbor_alltoall(out_size, recv_size)

    # Sort output for sending
    offsets = np.zeros(len(out_size)+1, dtype=np.intc)
    offsets[1:] = np.cumsum(out_size)
    out_cells = np.zeros(offsets[-1], dtype=np.int64)
    out_pos = np.zeros(offsets[-1], dtype=np.int32)
    count = np.zeros_like(out_size, dtype=np.int32)
    proc_to_dof = np.zeros_like(input_cells, dtype=np.int32)
    for i, owner in enumerate(output_owners):
        # Find relative position of owner in MPI communicator
        # Could be cached from previous run
        proc_pos = find_first(owner, dest_ranks)

        # Fill output data
        out_cells[offsets[proc_pos]+count[proc_pos]] = input_cells[i]
        out_pos[offsets[proc_pos]+count[proc_pos]] = dofmap_pos[i]

        # Compute map from global out position to relative position in proc
        proc_to_dof[offsets[proc_pos]+count[proc_pos]] = i
        count[proc_pos] += 1
        del proc_pos
    del count

    # Prepare data-structures for receiving
    total_incoming = sum(recv_size)
    inc_cells = np.zeros(total_incoming, dtype=np.int64)
    inc_pos = np.zeros(total_incoming, dtype=np.intc)

    # Compute incoming offset
    inc_offsets = np.zeros(len(recv_size)+1, dtype=np.intc)
    inc_offsets[1:] = np.cumsum(recv_size)

    # Send data
    s_msg = [out_cells, out_size, MPI.INT64_T]
    r_msg = [inc_cells, recv_size, MPI.INT64_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    s_msg = [out_pos, out_size, MPI.INT32_T]
    r_msg = [inc_pos, recv_size, MPI.INT32_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    local_input_range = compute_local_range(comm, num_cells_global)
    values_to_distribute = np.zeros_like(inc_pos, dtype=np.float64)
    for i, (cell, pos) in enumerate(zip(inc_cells, inc_pos)):
        l_cell = cell - local_input_range[0]
        values_to_distribute[i] = values[dofmap_offsets[l_cell]+pos]

    # Send input dofs back to owning process
    data_to_mesh_comm = comm.Create_dist_graph_adjacent(dest_ranks.tolist(), source_ranks.tolist(),
                                                        reorder=False)

    incoming_global_dofs = np.zeros(sum(out_size), dtype=np.float64)
    s_msg = [values_to_distribute, recv_size, MPI.DOUBLE]
    r_msg = [incoming_global_dofs, out_size, MPI.DOUBLE]
    data_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Sort incoming global dofs as they were inputted
    sorted_global_dofs = np.zeros_like(incoming_global_dofs, dtype=np.float64)
    assert len(incoming_global_dofs) == len(input_cells)
    for i in range(len(dest_ranks)):
        for j in range(out_size[i]):
            input_pos = offsets[i] + j
            sorted_global_dofs[proc_to_dof[input_pos]] = incoming_global_dofs[input_pos]
    return sorted_global_dofs


def send_cells_and_cell_perms(filename: pathlib.Path, comm: MPI.Intracomm,
                              source_ranks: npt.NDArray[np.int32],
                              dest_ranks: npt.NDArray[np.int32],
                              output_owners: npt.NDArray[np.int32],
                              input_cells: npt.NDArray[np.int64],
                              cell_perms: npt.NDArray[np.uint32],
                              num_cells_global: np.int64,
                              dofmap_path: str,
                              xdofmap_path: str,
                              engine: str,
                              u: dolfinx.fem.Function):
    """
    Send original cell indices + cell permutation for owned cells on current process
    to the one that has the input.

    Returns:
        A tuple (input_dofmap, cell_perm, input_perm, input_to_output), where `input_dofmap` is the
        part of the dofmap read in on the input process. `cell_perm` is the permutation for 
    """
    bs = u.function_space.dofmap.bs
    num_dofs_global = u.function_space.dofmap.index_map.size_global * u.function_space.dofmap.index_map_bs
    element = u.function_space.element
    # Compute amount of data to send to each process
    out_size = np.zeros(len(dest_ranks), dtype=np.int32)
    for owner in output_owners:
        proc_pos = find_first(owner, dest_ranks)
        out_size[proc_pos] += 1
        del proc_pos
    recv_size = np.zeros(len(source_ranks), dtype=np.int32)
    mesh_to_data_comm = comm.Create_dist_graph_adjacent(source_ranks.tolist(), dest_ranks.tolist(), reorder=False)
    # Send sizes to create data structures for receiving from NeighAlltoAllv
    mesh_to_data_comm.Neighbor_alltoall(out_size, recv_size)

    # Sort output for sending
    offsets = np.zeros(len(out_size)+1, dtype=np.intc)
    offsets[1:] = np.cumsum(out_size)
    out_cells = np.zeros(offsets[-1], dtype=np.int64)
    out_perm = np.zeros(offsets[-1], dtype=np.uint32)
    count = np.zeros_like(out_size, dtype=np.int32)
    proc_to_cell = np.zeros_like(input_cells, dtype=np.int32)
    for i, owner in enumerate(output_owners):
        # Find relative position of owner in MPI communicator
        # Could be cached from previous run
        proc_pos = find_first(owner, dest_ranks)

        # Fill output data
        out_cells[offsets[proc_pos]+count[proc_pos]] = input_cells[i]
        out_perm[offsets[proc_pos]+count[proc_pos]] = cell_perms[i]

        # Compute map from global out position to relative position in proc
        proc_to_cell[offsets[proc_pos]+count[proc_pos]] = i
        count[proc_pos] += 1
        del proc_pos
    del count

    # Prepare data-structures for receiving
    total_incoming = sum(recv_size)
    inc_cells = np.zeros(total_incoming, dtype=np.int64)
    inc_perm = np.zeros(total_incoming, dtype=np.uint32)

    # Compute incoming offset
    inc_offsets = np.zeros(len(recv_size)+1, dtype=np.intc)
    inc_offsets[1:] = np.cumsum(recv_size)

    # Send data
    s_msg = [out_cells, out_size, MPI.INT64_T]
    r_msg = [inc_cells, recv_size, MPI.INT64_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    s_msg = [out_perm, out_size, MPI.UINT32_T]
    r_msg = [inc_perm, recv_size, MPI.UINT32_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)
    # Something happens when source ranks is note [0,1] (but [1,0]). Have to look at this tomorrow
    print(MPI.COMM_WORLD.rank, "SENDING:", out_perm, out_size, "Receiving", inc_perm, recv_size, "\n",
          source_ranks, dest_ranks)

    # Read dofmap from file
    input_dofmap = read_dofmap_new(comm, filename, dofmap_path, xdofmap_path,
                                   num_cells_global, engine)

    # Compute owner of all dofs in dofmap
    dof_owner = index_owner(comm, input_dofmap.array, num_dofs_global)

    # Create MPI neigh comm to owner.
    # NOTE: USE NBX in C++
    unique_dof_owners = np.unique(dof_owner)
    mesh_to_dof_comm = comm.Create_dist_graph(
        [comm.rank], [len(unique_dof_owners)], unique_dof_owners, reorder=False)
    dof_source, dof_dest, _ = mesh_to_dof_comm.Get_dist_neighbors()

    local_values = send_dofs_and_receive_values(filename, "Values", engine,
                                                comm,   np.asarray(dof_source, dtype=np.int32),
                                                np.asarray(dof_dest, dtype=np.int32), input_dofmap.array, dof_owner)
    local_input_range = compute_local_range(comm, num_cells_global)
    input_local_cell_index = inc_cells - local_input_range[0]

    # Read input cell permutations
    input_perms = read_cell_perms(comm, filename, "CellPermutations", num_cells_global, engine)

    # print(MPI.COMM_WORLD.rank, inc_perm, input_perms, input_local_cell_index)

    # Permute perms coming from other process to match local ordering
    inc_perm = inc_perm[input_local_cell_index]

    # First invert input data to reference element then transform to current mesh
    for local_cell in input_local_cell_index:
        start, end = input_dofmap.offsets[local_cell], input_dofmap.offsets[local_cell+1]
        element.apply_inverse_dof_transformation(local_values[start:end], input_perms[local_cell], bs)
        element.apply_dof_transformation(local_values[start:end], inc_perm[local_cell], bs)
    # For each dof owned by a process, find the local position in the dofmap.
    V = u.function_space
    local_cells, dof_pos = compute_dofmap_pos(V)
    input_cells = V.mesh.topology.original_cell_index[local_cells]
    num_cells_global = V.mesh.topology.index_map(V.mesh.topology.dim).size_global
    owners = index_owner(V.mesh.comm, input_cells, num_cells_global)
    unique_owners = np.unique(owners)
    # FIXME: In C++ use NBX to find neighbourhood
    sub_comm = V.mesh.comm.Create_dist_graph(
        [V.mesh.comm.rank], [len(unique_owners)], unique_owners, reorder=False)
    source, dest, _ = sub_comm.Get_dist_neighbors()

    owned_values = send_dofmap_get_vals(comm, np.asarray(source, dtype=np.int32), np.asarray(dest, dtype=np.int32), owners, input_cells, dof_pos,
                                        num_cells_global, local_values, input_dofmap.offsets)

    u.x.array[:len(owned_values)] = owned_values
    u.x.scatter_forward()


def send_cells_and_receive_dofmap_index(filename: pathlib.Path, comm: MPI.Intracomm,
                                        source_ranks: npt.NDArray[np.int32],
                                        dest_ranks: npt.NDArray[np.int32],
                                        output_owners: npt.NDArray[np.int32],
                                        input_cells: npt.NDArray[np.int64],
                                        dofmap_pos: npt.NDArray[np.int32],
                                        num_cells_global: np.int64,
                                        dofmap_path: str,
                                        xdofmap_path: str,
                                        engine: str) -> npt.NDArray[np.int64]:
    """
    Given a set of positions in input dofmap, give the global input index of this dofmap entry
    in input file.
    """

    # Compute amount of data to send to each process
    out_size = np.zeros(len(dest_ranks), dtype=np.int32)
    for owner in output_owners:
        proc_pos = find_first(owner, dest_ranks)
        out_size[proc_pos] += 1
        del proc_pos
    recv_size = np.zeros(len(source_ranks), dtype=np.int32)
    mesh_to_data_comm = comm.Create_dist_graph_adjacent(source_ranks.tolist(), dest_ranks.tolist(), reorder=False)
    # Send sizes to create data structures for receiving from NeighAlltoAllv
    mesh_to_data_comm.Neighbor_alltoall(out_size, recv_size)

    # Sort output for sending
    offsets = np.zeros(len(out_size)+1, dtype=np.intc)
    offsets[1:] = np.cumsum(out_size)
    out_cells = np.zeros(offsets[-1], dtype=np.int64)
    out_pos = np.zeros(offsets[-1], dtype=np.int32)
    count = np.zeros_like(out_size, dtype=np.int32)
    proc_to_dof = np.zeros_like(input_cells, dtype=np.int32)
    for i, owner in enumerate(output_owners):
        # Find relative position of owner in MPI communicator
        # Could be cached from previous run
        proc_pos = find_first(owner, dest_ranks)

        # Fill output data
        out_cells[offsets[proc_pos]+count[proc_pos]] = input_cells[i]
        out_pos[offsets[proc_pos]+count[proc_pos]] = dofmap_pos[i]

        # Compute map from global out position to relative position in proc
        proc_to_dof[offsets[proc_pos]+count[proc_pos]] = i
        count[proc_pos] += 1
        del proc_pos
    del count

    # Prepare data-structures for receiving
    total_incoming = sum(recv_size)
    inc_cells = np.zeros(total_incoming, dtype=np.int64)
    inc_pos = np.zeros(total_incoming, dtype=np.intc)

    # Compute incoming offset
    inc_offsets = np.zeros(len(recv_size)+1, dtype=np.intc)
    inc_offsets[1:] = np.cumsum(recv_size)

    # Send data
    s_msg = [out_cells, out_size, MPI.INT64_T]
    r_msg = [inc_cells, recv_size, MPI.INT64_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    s_msg = [out_pos, out_size, MPI.INT32_T]
    r_msg = [inc_pos, recv_size, MPI.INT32_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Read dofmap from file
    input_dofs = read_dofmap(comm, filename, dofmap_path, xdofmap_path,
                             num_cells_global, engine,
                             inc_cells, inc_pos)
    # Send input dofs back to owning process
    data_to_mesh_comm = comm.Create_dist_graph_adjacent(dest_ranks.tolist(), source_ranks.tolist(),
                                                        reorder=False)

    incoming_global_dofs = np.zeros(sum(out_size), dtype=np.int64)
    s_msg = [input_dofs, recv_size, MPI.INT64_T]
    r_msg = [incoming_global_dofs, out_size, MPI.INT64_T]
    data_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Sort incoming global dofs as they were inputted
    sorted_global_dofs = np.zeros_like(incoming_global_dofs, dtype=np.int64)
    assert len(incoming_global_dofs) == len(input_cells)
    for i in range(len(dest_ranks)):
        for j in range(out_size[i]):
            input_pos = offsets[i] + j
            sorted_global_dofs[proc_to_dof[input_pos]] = incoming_global_dofs[input_pos]
    return sorted_global_dofs


def send_dofs_and_receive_values(
        filename: pathlib.Path, vector_path: str, engine: str,
        comm: MPI.Intracomm, source_ranks: npt.NDArray[np.int32],
        dest_ranks: npt.NDArray[np.int32], dofs: npt.NDArray[np.int64],
        dof_owner: npt.NDArray[np.int32]):

    mesh_to_data_comm = comm.Create_dist_graph_adjacent(source_ranks.tolist(),
                                                        dest_ranks.tolist(), reorder=False)

    # Send global dof number to input process
    dof_out_size = np.zeros_like(dest_ranks, dtype=np.int32)
    for owner in dof_owner:
        proc_pos = find_first(owner, dest_ranks)
        dof_out_size[proc_pos] += 1
    dof_recv_size = np.zeros_like(source_ranks, dtype=np.int32)
    mesh_to_data_comm.Neighbor_alltoall(dof_out_size, dof_recv_size)

    # Sort output for sending
    dofs_offsets = np.zeros(len(dof_out_size)+1, dtype=np.intc)
    dofs_offsets[1:] = np.cumsum(dof_out_size)
    out_dofs = np.zeros(dofs_offsets[-1], dtype=np.int64)
    dof_count = np.zeros_like(dof_out_size, dtype=np.int32)
    proc_to_local = np.zeros_like(dofs, dtype=np.int32)  # Map output to local dof
    for i, (dof, owner) in enumerate(zip(dofs, dof_owner)):
        proc_pos = find_first(owner, dest_ranks)
        out_dofs[dofs_offsets[proc_pos]+dof_count[proc_pos]] = dof
        proc_to_local[dofs_offsets[proc_pos]+dof_count[proc_pos]] = i
        dof_count[proc_pos] += 1

    input_dofs = np.zeros(sum(dof_recv_size), dtype=np.int64)
    s_msg = [out_dofs, dof_out_size, MPI.INT64_T]
    r_msg = [input_dofs, dof_recv_size, MPI.INT64_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    vals, start_pos = read_array(filename, vector_path, engine, comm)
    # Compute local dof input dof number (using local_range)
    input_vals = np.zeros(sum(dof_recv_size), dtype=np.float64)
    for i, dof in enumerate(input_dofs):
        input_vals[i] = vals[dof - start_pos]

    # Create reverse comm and send back
    dof_to_mesh_comm = comm.Create_dist_graph_adjacent(dest_ranks.tolist(),
                                                       source_ranks.tolist(), reorder=False)
    incoming_vals = np.zeros(sum(dof_out_size), dtype=np.float64)
    s_msg = [input_vals, dof_recv_size, MPI.DOUBLE]
    r_msg = [incoming_vals, dof_out_size, MPI.DOUBLE]
    dof_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Sort input according to local dof number
    sorted_vals = np.empty_like(incoming_vals, dtype=np.float64)
    for i in range(len(dest_ranks)):
        for j in range(dof_out_size[i]):
            sorted_vals[proc_to_local[dofs_offsets[i] + j]] = incoming_vals[dofs_offsets[i] + j]

    return sorted_vals
