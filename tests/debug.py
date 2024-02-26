from mpi4py import MPI
import numpy as np
import adios4dolfinx.utils
import dolfinx
from pathlib import Path


comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(
    comm, 2, 2, ghost_mode=dolfinx.mesh.GhostMode.none)

num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
original_cell_index = domain.topology.original_cell_index[:num_cells_local]
num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
output_cell_owner = adios4dolfinx.utils.index_owner(domain.comm, original_cell_index, num_cells_global)
local_cell_range = adios4dolfinx.utils.compute_local_range(domain.comm, num_cells_global)

# Compute outgoing edges from current process and create neighbourhood communicator
# Also create number of outgoing cells at the same time
unique_output_owners, out_size = np.unique(output_cell_owner, return_counts=True)
topology_to_owner_comm = comm.Create_dist_graph(
  [domain.comm.rank], [len(unique_output_owners)], unique_output_owners, reorder=False)
source, dest, _ = topology_to_owner_comm.Get_dist_neighbors()
assert np.allclose(dest, unique_output_owners)

# Compute what local process each cell is sent to
cell_owners = output_cell_owner.reshape(-1, 1)
process_pos_indicator = (cell_owners == unique_output_owners)

# Compute offset for cell index
send_offsets = np.zeros(len(out_size)+1, dtype=np.intc)
send_offsets[1:] = np.cumsum(out_size)
assert send_offsets[-1] == len(cell_owners)

# Compute local insert index for each cell
proc_row, proc_col = np.nonzero(process_pos_indicator)
cum_pos = np.cumsum(process_pos_indicator, axis=0)
insert_position = cum_pos[proc_row, proc_col] - 1
insert_position += send_offsets[proc_col]

# Compute number of recieving cells
recv_size = np.zeros_like(source, dtype=np.int32)
topology_to_owner_comm.Neighbor_alltoall(out_size, recv_size)

# Pack cells and send
send_cells = original_cell_index[insert_position]
recv_cells = np.empty(recv_size.sum(), dtype=np.int64)
topology_to_owner_comm.Neighbor_alltoallv([send_cells, out_size, MPI.INT64_T], [recv_cells, recv_size, MPI.INT64_T])

local_cell_index = recv_cells - local_cell_range[0]

# Map local dofmap to its original indices (flattened)
original_node_index = domain.geometry.input_global_indices
geom_dofmap = domain.geometry.dofmap
global_geom_dofmap = original_node_index[geom_dofmap.reshape(-1)]

# Unroll insert position for dofmap
num_nodes_per_cell = geom_dofmap.shape[1]
insert_position_dofmap = np.repeat(insert_position, num_nodes_per_cell) * num_nodes_per_cell
insert_position_dofmap += np.tile(np.arange(num_nodes_per_cell), len(original_cell_index))

# Create send array for global dofmap
send_dofmap = global_geom_dofmap[insert_position_dofmap]

# Compute dofmap sizes
send_offsets_dofmap = send_offsets * num_nodes_per_cell
send_sizes_dofmap = out_size * num_nodes_per_cell

# Send size for dofmap
recv_size_dofmap = np.zeros_like(source, dtype=np.int32)
topology_to_owner_comm.Neighbor_alltoall(send_sizes_dofmap, recv_size_dofmap)
#
recv_offsets_dofmap = np.zeros(len(source)+1, dtype=np.intc)
recv_offsets_dofmap[1:] = np.cumsum(recv_size_dofmap)
recv_dofmap = np.empty(recv_offsets_dofmap[-1], dtype=np.int64)
topology_to_owner_comm.Neighbor_alltoallv(
  [send_dofmap, send_sizes_dofmap, MPI.INT64_T],
  [recv_dofmap, recv_size_dofmap, MPI.INT64_T]
)

# Permute recieved dofmap based on incoming cell index
sorted_recv_dofmap = recv_dofmap.reshape(-1, num_nodes_per_cell)[local_cell_index]


original_cell_index = domain.topology.original_cell_index
num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
output_cell_owner = adios4dolfinx.utils.index_owner(domain.comm, original_cell_index, num_cells_global)
local_cell_range = adios4dolfinx.utils.compute_local_range(domain.comm, num_cells_global)

# Compute outgoing edges from current process and create neighbourhood communicator
# Also create number of outgoing cells at the same time
num_owned_nodes = domain.geometry.index_map().size_local
output_node_owner = adios4dolfinx.utils.index_owner(domain.comm, original_node_index[:num_owned_nodes], domain.geometry.index_map().size_global)
unique_node_owners, out_size_node = np.unique(output_node_owner, return_counts=True)
geometry_to_owner_comm = comm.Create_dist_graph(
  [domain.comm.rank], [len(unique_node_owners)], unique_node_owners, reorder=False)


source_geom, dest_geom, _ = geometry_to_owner_comm.Get_dist_neighbors()
assert np.allclose(dest_geom, unique_node_owners)




#print(output_cell_owner, insert_position)
exit()
# print(domain.comm.rank, output_cell_owner, output_node_owner)



out_path_1_mesh = Path("mesh_1.bp")



# adios4dolfinx.write_mesh(domain, out_path_1_mesh, "BP4")

# mesh_1 = adios4dolfinx.read_mesh(MPI.COMM_WORLD, out_path_1_mesh, engine="BP4",
#                                  ghost_mode=dolfinx.mesh.GhostMode.none)


# out_path_1_function = Path("function_1.bp")
# adios4dolfinx.write_mesh(mesh_1, out_path_1_function, "BP4")


# mesh_2 = adios4dolfinx.read_mesh(MPI.COMM_WORLD, out_path_1_function, engine="BP4",
#                                  ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

# mesh_3 = adios4dolfinx.read_mesh(MPI.COMM_WORLD, out_path_1_mesh, engine="BP4",
#                                  ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

# print(mesh_2.geometry.index_map().size_local, mesh_3.geometry.index_map().size_local)   

# np.testing.assert_allclose(mesh_2.geometry.x[:num_nodes_local_2,:], mesh_3.geometry.x[:num_nodes_local_3,:])
# np.testing.assert_allclose(mesh_2.topology.original_cell_index, mesh_3.topology.original_cell_index)
# np.testing.assert_allclose(mesh_2.geometry.input_global_indices, mesh_3.geometry.input_global_indices)