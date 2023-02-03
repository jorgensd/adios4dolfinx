
import numpy as np
import numpy.typing as npt
import pathlib
from .utils import find_first
from .adios2_helpers import read_array, read_dofmap
from mpi4py import MPI

__all__ = ["send_cells_and_receive_dofmap_index",
           "send_dofs_and_receive_values"]
"""
Helpers for sending and receiving values for checkpointing
"""


def send_cells_and_receive_dofmap_index(filename: pathlib.Path, comm: MPI.Comm,
                                        source_ranks: npt.NDArray[np.int32],
                                        dest_ranks: npt.NDArray[np.int32],
                                        output_owners: npt.NDArray[np.int32],
                                        input_cells: npt.NDArray[np.int64],
                                        dofmap_pos: npt.NDArray[np.int32],
                                        num_cells_global: np.int64) -> npt.NDArray[np.int64]:
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
    mesh_to_data_comm = comm.Create_dist_graph_adjacent(source_ranks, dest_ranks, reorder=False)
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
    del count, proc_pos

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
    input_dofs = read_dofmap(comm, filename, "/mesh/cell_dofs", "/mesh/x_cell_dofs",
                             num_cells_global, "HDF5",
                             inc_cells, inc_pos)
    # Send input dofs back to owning process
    data_to_mesh_comm = comm.Create_dist_graph_adjacent(dest_ranks, source_ranks, reorder=False)

    incoming_global_dofs = np.zeros(sum(out_size), dtype=np.int64)
    s_msg = [input_dofs, recv_size, MPI.INT64_T]
    r_msg = [incoming_global_dofs, out_size, MPI.INT64_T]
    data_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Sort incoming global dofs as they were inputted
    sorted_global_dofs = np.zeros_like(incoming_global_dofs, dtype=np.int64)
    assert len(incoming_global_dofs) == len(input_cells)
    for i in range(len(dest_ranks)):
        pos = np.cumsum(out_size[:i])
        for j in range(out_size[i]):
            input_pos = offsets[i] + j
            sorted_global_dofs[proc_to_dof[input_pos]] = incoming_global_dofs[input_pos]
    return sorted_global_dofs


def send_dofs_and_receive_values(
        filename: pathlib.Path, vector: str, engine: str,
        comm: MPI.Comm, source_ranks: npt.NDArray[np.int32],
        dest_ranks: npt.NDArray[np.int32], dofs: npt.NDArray[np.int64],
        dof_owner: npt.NDArray[np.int32]):

    mesh_to_data_comm = comm.Create_dist_graph_adjacent(source_ranks, dest_ranks, reorder=False)

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

    vals, start_pos = read_array(filename, vector, engine, comm)
    # Compute local dof input dof number (using local_range)
    input_vals = np.zeros(sum(dof_recv_size), dtype=np.float64)
    for i, dof in enumerate(input_dofs):
        input_vals[i] = vals[dof - start_pos]

    # Create reverse comm and send back
    dof_to_mesh_comm = comm.Create_dist_graph_adjacent(dest_ranks, source_ranks, reorder=False)
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
