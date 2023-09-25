from typing import Tuple

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from .utils import compute_local_range, find_first, valid_function_types


__all__ = [
    "send_dofmap_and_recv_values",
    "send_and_recv_cell_perm",
    "send_dofs_and_recv_values",
]
"""
Helpers for sending and receiving values for checkpointing
"""

numpy_to_mpi = {np.float64: MPI.DOUBLE, np.float32: MPI.FLOAT,
                np.complex64: MPI.COMPLEX, np.complex128: MPI.DOUBLE_COMPLEX}


def send_dofmap_and_recv_values(
    comm: MPI.Intracomm,
    source_ranks: npt.NDArray[np.int32],
    dest_ranks: npt.NDArray[np.int32],
    output_owners: npt.NDArray[np.int32],
    input_cells: npt.NDArray[np.int64],
    dofmap_pos: npt.NDArray[np.int32],
    num_cells_global: np.int64,
    values: npt.NDArray[valid_function_types],
    dofmap_offsets: npt.NDArray[np.int32],
) -> npt.NDArray[valid_function_types]:
    """
    Given a set of positions in input dofmap, give the global input index of this dofmap entry
    in input file.

    Args:
        comm: The MPI communicator to create the Neighbourhood-communicator from
        source_ranks: Ranks that will send dofmap indices to current process
        dest_ranks: Ranks that will receive dofmap indices from current process
        output_owners: The owners of each dofmap entry on this process. The unique set of these entries
            should be the same as the dest_ranks.
        input_cells: A cell associated with the degree of freedom sent (global index).
        dofmap_pos: The local position in the dofmap. I.e. `dof = dofmap.links(input_cells)[dofmap_pos]`
        num_cells_global: Number of global cells
        values: Values currently held by this process. These are ordered (num_cells_local, num_dofs_per_cell),
            flattened row-major.
        dofmap_offsets: Local dofmap offsets to access the correct `values`.

    Returns:
        Values corresponding to the dofs owned by this process.
    """

    # Compute amount of data to send to each process
    out_size = np.zeros(len(dest_ranks), dtype=np.int32)
    for owner in output_owners:
        proc_pos = find_first(owner, dest_ranks)
        out_size[proc_pos] += 1
        del proc_pos
    recv_size = np.zeros(len(source_ranks), dtype=np.int32)
    mesh_to_data_comm = comm.Create_dist_graph_adjacent(
        source_ranks.tolist(), dest_ranks.tolist(), reorder=False
    )
    # Send sizes to create data structures for receiving from NeighAlltoAllv
    mesh_to_data_comm.Neighbor_alltoall(out_size, recv_size)

    # Sort output for sending
    offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
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
        out_cells[offsets[proc_pos] + count[proc_pos]] = input_cells[i]
        out_pos[offsets[proc_pos] + count[proc_pos]] = dofmap_pos[i]

        # Compute map from global out position to relative position in proc
        proc_to_dof[offsets[proc_pos] + count[proc_pos]] = i
        count[proc_pos] += 1
        del proc_pos
    del count

    # Prepare data-structures for receiving
    total_incoming = sum(recv_size)
    inc_cells = np.zeros(total_incoming, dtype=np.int64)
    inc_pos = np.zeros(total_incoming, dtype=np.intc)

    # Compute incoming offset
    inc_offsets = np.zeros(len(recv_size) + 1, dtype=np.intc)
    inc_offsets[1:] = np.cumsum(recv_size)

    # Send data
    s_msg = [out_cells, out_size, MPI.INT64_T]
    r_msg = [inc_cells, recv_size, MPI.INT64_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    s_msg = [out_pos, out_size, MPI.INT32_T]
    r_msg = [inc_pos, recv_size, MPI.INT32_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)
    mesh_to_data_comm.Free()

    local_input_range = compute_local_range(comm, num_cells_global)
    values_to_distribute = np.zeros_like(inc_pos, dtype=values.dtype)
    for i, (cell, pos) in enumerate(zip(inc_cells, inc_pos)):
        l_cell = cell - local_input_range[0]
        values_to_distribute[i] = values[dofmap_offsets[l_cell] + pos]

    # Send input dofs back to owning process
    data_to_mesh_comm = comm.Create_dist_graph_adjacent(
        dest_ranks.tolist(), source_ranks.tolist(), reorder=False
    )

    incoming_global_dofs = np.zeros(sum(out_size), dtype=values.dtype)
    s_msg = [values_to_distribute, recv_size, numpy_to_mpi[values.dtype.type]]
    r_msg = [incoming_global_dofs, out_size, numpy_to_mpi[values.dtype.type]]
    data_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Sort incoming global dofs as they were inputted
    sorted_global_dofs = np.zeros_like(incoming_global_dofs, dtype=values.dtype)
    assert len(incoming_global_dofs) == len(input_cells)
    for i in range(len(dest_ranks)):
        for j in range(out_size[i]):
            input_pos = offsets[i] + j
            sorted_global_dofs[proc_to_dof[input_pos]] = incoming_global_dofs[input_pos]
    data_to_mesh_comm.Free()
    return sorted_global_dofs


def send_and_recv_cell_perm(
    cells: npt.NDArray[np.int64],
    perms: npt.NDArray[np.uint32],
    cell_owners: npt.NDArray[np.int32],
    comm: MPI.Intracomm,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.uint32]]:
    """
    Send global cell index and permutation to corresponding entry in `dest_ranks`.

    Args:
        cells: The global input index of the cell
        perms: The corresponding cell permutation of the cell
        cell_owners: The rank to send the i-th entry of cells and perms to
        comm: Rank of comm to generate neighbourhood communicator from
    """
    dest_ranks = np.unique(cell_owners)

    mesh_to_data = comm.Create_dist_graph(
        [comm.rank], [len(dest_ranks)], dest_ranks.tolist(), reorder=False
    )
    source, dest, _ = mesh_to_data.Get_dist_neighbors()

    # Compute amount of data to send to each process
    out_size = np.zeros_like(dest, dtype=np.int32)
    for owner in cell_owners:
        proc_pos = find_first(owner, np.asarray(dest, dtype=np.int32))
        out_size[proc_pos] += 1
        del proc_pos

    # Send sizes to create data structures for receiving from NeighAlltoAllv
    recv_size = np.zeros_like(source, dtype=np.int32)
    mesh_to_data.Neighbor_alltoall(out_size, recv_size)

    # Sort cells, perms for sending
    offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
    offsets[1:] = np.cumsum(out_size)
    assert offsets[-1] == len(cells)
    out_cells = np.zeros_like(cells, dtype=np.int64)
    out_perm = np.zeros_like(perms, dtype=np.uint32)
    count = np.zeros_like(out_size, dtype=np.int32)
    for i in range(len(cells)):
        # Find relative position of owner in MPI communicator
        # Could be cached from previous run
        proc_pos = find_first(cell_owners[i], np.asarray(dest_ranks, dtype=np.int32))

        # Fill output data
        out_cells[offsets[proc_pos] + count[proc_pos]] = cells[i]
        out_perm[offsets[proc_pos] + count[proc_pos]] = perms[i]
        count[proc_pos] += 1
        del proc_pos
    del count

    # Prepare data-structures for receiving
    total_incoming = sum(recv_size)
    inc_cells = np.zeros(total_incoming, dtype=np.int64)
    inc_perm = np.zeros(total_incoming, dtype=np.uint32)

    # Compute incoming offset
    inc_offsets = np.zeros(len(recv_size) + 1, dtype=np.intc)
    inc_offsets[1:] = np.cumsum(recv_size)

    # Send data
    s_msg = [out_cells, out_size, MPI.INT64_T]
    r_msg = [inc_cells, recv_size, MPI.INT64_T]
    mesh_to_data.Neighbor_alltoallv(s_msg, r_msg)

    s_msg = [out_perm, out_size, MPI.UINT32_T]
    r_msg = [inc_perm, recv_size, MPI.UINT32_T]
    mesh_to_data.Neighbor_alltoallv(s_msg, r_msg)
    mesh_to_data.Free()
    return inc_cells, inc_perm


def send_dofs_and_recv_values(
    input_dofmap: npt.NDArray[np.int64],
    dofmap_owners: npt.NDArray[np.int32],
    comm: MPI.Intracomm,
    input_array: npt.NDArray[valid_function_types],
    array_start: int,
):
    """
    Send a set of dofs (global index) to the process holding the DOF values to retrieve them.

    Args:
        input_dofmap: List of dofs (global index) that this process wants values for
        dofmap_owners: The process currently holding the values this process want to get.
        comm: MPI communicator
        input_array: Values for dofs
        array_start: The global starting index of `input_array`.
    """
    dest_ranks = np.unique(dofmap_owners)
    dofmap_to_values = comm.Create_dist_graph(
        [comm.rank], [len(dest_ranks)], dest_ranks.tolist(), reorder=False
    )

    source, dest, _ = dofmap_to_values.Get_dist_neighbors()

    # Compute amount of data to send to each process
    out_size = np.zeros_like(dest, dtype=np.int32)
    for owner in dofmap_owners:
        proc_pos = find_first(owner, dest_ranks)
        out_size[proc_pos] += 1
        del proc_pos

    # Send sizes to create data structures for receiving from NeighAlltoAllv
    recv_size = np.zeros_like(source, dtype=np.int32)
    dofmap_to_values.Neighbor_alltoall(out_size, recv_size)

    # Sort output for sending
    dofs_offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
    dofs_offsets[1:] = np.cumsum(out_size)
    out_dofs = np.zeros(dofs_offsets[-1], dtype=np.int64)
    dof_count = np.zeros_like(out_size, dtype=np.int32)
    proc_to_local = np.zeros_like(
        input_dofmap, dtype=np.int32
    )  # Map output to local dof
    for i, (dof, owner) in enumerate(zip(input_dofmap, dofmap_owners)):
        proc_pos = find_first(owner, dest_ranks)
        out_dofs[dofs_offsets[proc_pos] + dof_count[proc_pos]] = dof
        proc_to_local[dofs_offsets[proc_pos] + dof_count[proc_pos]] = i
        dof_count[proc_pos] += 1

    # Send input dofs to processes holding input array
    inc_dofs = np.zeros(sum(recv_size), dtype=np.int64)
    s_msg = [out_dofs, out_size, MPI.INT64_T]
    r_msg = [inc_dofs, recv_size, MPI.INT64_T]
    dofmap_to_values.Neighbor_alltoallv(s_msg, r_msg)
    dofmap_to_values.Free()

    # Send back appropriate input values
    sending_values = np.zeros(len(inc_dofs), dtype=input_array.dtype)
    for i, dof in enumerate(inc_dofs):
        sending_values[i] = input_array[dof - array_start]

    values_to_dofmap = comm.Create_dist_graph_adjacent(dest, source, reorder=False)
    inc_values = np.zeros_like(out_dofs, dtype=input_array.dtype)
    s_msg_rev = [sending_values, recv_size, numpy_to_mpi[input_array.dtype.type]]
    r_msg_rev = [inc_values, out_size, numpy_to_mpi[input_array.dtype.type]]
    values_to_dofmap.Neighbor_alltoallv(s_msg_rev, r_msg_rev)
    values_to_dofmap.Free()

    # Sort inputs according to local dof number (input process)
    values = np.empty_like(inc_values, dtype=input_array.dtype)

    for i in range(len(dest_ranks)):
        for j in range(out_size[i]):
            in_pos = dofs_offsets[i] + j
            values[proc_to_local[in_pos]] = inc_values[in_pos]
    return values
