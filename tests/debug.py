from mpi4py import MPI
import numpy as np
import adios4dolfinx.utils
import dolfinx
from pathlib import Path


comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_cube(
    comm, 2, 1, 2, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

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
send_cells = np.empty(len(insert_position), dtype=np.int64)
send_cells[insert_position] = original_cell_index
recv_cells = np.empty(recv_size.sum(), dtype=np.int64)
topology_to_owner_comm.Neighbor_alltoallv([send_cells, out_size, MPI.INT64_T], [recv_cells, recv_size, MPI.INT64_T])
local_cell_index = recv_cells - local_cell_range[0]


# Map local dofmap to its original indices (flattened)
original_node_index = domain.geometry.input_global_indices
geom_dofmap = domain.geometry.dofmap[:num_cells_local, :]
global_geom_dofmap = original_node_index[geom_dofmap.reshape(-1)]

# Unroll insert position for dofmap
num_nodes_per_cell = geom_dofmap.shape[1]
insert_position_dofmap = np.repeat(insert_position, num_nodes_per_cell) * num_nodes_per_cell
insert_position_dofmap += np.tile(np.arange(num_nodes_per_cell), len(original_cell_index))

# Create send array for global dofmap
send_dofmap = np.empty(len(insert_position_dofmap), dtype=np.int64)
send_dofmap[insert_position_dofmap] = global_geom_dofmap

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


sorted_recv_dofmap = recv_dofmap.reshape(-1, num_nodes_per_cell)[local_cell_index]

original_cell_index = domain.topology.original_cell_index
num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
output_cell_owner = adios4dolfinx.utils.index_owner(domain.comm, original_cell_index, num_cells_global)
local_cell_range = adios4dolfinx.utils.compute_local_range(domain.comm, num_cells_global)

# Compute outgoing edges from current process and create neighbourhood communicator
# Also create number of outgoing cells at the same time
num_owned_nodes = domain.geometry.index_map().size_local
output_node_owner = adios4dolfinx.utils.index_owner(domain.comm, original_node_index, domain.geometry.index_map().size_global)
unique_node_owners, out_size_node = np.unique(output_node_owner[:num_owned_nodes], return_counts=True)
geometry_to_owner_comm = comm.Create_dist_graph(
  [domain.comm.rank], [len(unique_node_owners)], unique_node_owners, reorder=False)


source_geom, dest_geom, _ = geometry_to_owner_comm.Get_dist_neighbors()
assert np.allclose(dest_geom, unique_node_owners)

# Compute send size for nodes
ono = output_node_owner[:num_owned_nodes].reshape(-1, 1)
process_pos_indicator = (ono == unique_node_owners)

# Compute offset for cell index
send_offsets = np.zeros(len(out_size_node)+1, dtype=np.intc)
send_offsets[1:] = np.cumsum(out_size_node)
assert send_offsets[-1] == num_owned_nodes

# Compute local insert index for each node
proc_row, proc_col = np.nonzero(process_pos_indicator)
cum_pos = np.cumsum(process_pos_indicator, axis=0)
insert_position = cum_pos[proc_row, proc_col] - 1
insert_position += send_offsets[proc_col]

insert_position_node = np.repeat(insert_position, 3) * 3
insert_position_node += np.tile(np.arange(3), num_owned_nodes)

send_nodes = np.empty(len(insert_position_node), dtype=domain.geometry.x.dtype)
send_nodes[insert_position_node] = domain.geometry.x[:num_owned_nodes,:].reshape(-1)

# Send and recieve geometry sizes
send_sizes = out_size_node * 3
recv_size = np.zeros_like(source_geom, dtype=np.int32)
geometry_to_owner_comm.Neighbor_alltoall(send_sizes, recv_size)

# Send nodes
recv_nodes = np.empty(recv_size.sum(), dtype=domain.geometry.x.dtype)
geometry_to_owner_comm.Neighbor_alltoallv([send_nodes, send_sizes, MPI.DOUBLE], [recv_nodes, recv_size, MPI.DOUBLE])

# Send node ordering
send_indices = np.empty(num_owned_nodes, dtype=np.int64)
send_indices[insert_position]= original_node_index[:num_owned_nodes]
recv_indices = np.empty(recv_size.sum()//3, dtype=np.int64)
geometry_to_owner_comm.Neighbor_alltoallv([send_indices, out_size_node, MPI.INT64_T], [recv_indices, recv_size//3, MPI.INT64_T])
local_node_range = adios4dolfinx.utils.compute_local_range(domain.comm, domain.geometry.index_map().size_global)
recv_indices -= local_node_range[0]
geometry = recv_nodes.reshape(-1, 3)[recv_indices]


# Create index map for cells
cell_imap = dolfinx.common.IndexMap(comm,local_cell_range[1]-local_cell_range[0], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))

# Create index map for geometry nodes
recv_dofmap_unique = np.unique(sorted_recv_dofmap.reshape(-1))
dofmap_node_owner = adios4dolfinx.utils.index_owner(domain.comm, recv_dofmap_unique, domain.geometry.index_map().size_global)
ghost_node_indices = (recv_dofmap_unique>= local_node_range[1]) | (recv_dofmap_unique < local_node_range[0])
ghost_nodes_owners = dofmap_node_owner[ghost_node_indices]
ghost_nodes = recv_dofmap_unique[ghost_node_indices]
node_imap = dolfinx.common.IndexMap(comm,local_node_range[1]-local_node_range[0], ghost_nodes, ghost_nodes_owners)

# Extract topology from geometry
global_recv_top = dolfinx.cpp.mesh.extract_topology(domain.topology.cell_type, domain.geometry.cmap.create_dof_layout(), sorted_recv_dofmap.reshape(-1))
local_recv_top = node_imap.global_to_local(global_recv_top).reshape(sorted_recv_dofmap.shape[0], -1)

new_topology = dolfinx.cpp.mesh.Topology(comm, domain.topology.cell_type)
new_topology.set_index_map(0, node_imap)
new_topology.set_index_map(domain.topology.dim, cell_imap)
new_topology.set_connectivity(dolfinx.graph.adjacencylist(local_recv_top), domain.topology.dim, 0)
new_topology.set_connectivity(dolfinx.graph.adjacencylist(np.arange(node_imap.size_local+node_imap.num_ghosts, dtype=np.int32).reshape(-1, 1)), 0, 0)

# Create cell permutations
new_topology.create_entity_permutations()
cell_permutation_info = new_topology.get_cell_permutation_info()


# # 
# import adios2
# adios = adios2.ADIOS(domin.comm)
# io = adios.DeclareIO("MeshWriter")
# io.SetEngine("BP4")
# outfile = io.Open(str("test_origina.bp"), adios2.Mode.Write)
#   # Write geometry
#   pointvar = io.DefineVariable(
#       "Points",
#       local_points,
#       shape=[num_xdofs_global, gdim],
#       start=[local_range[0], 0],
#       count=[num_xdofs_local, gdim],
#   )
#   outfile.Put(pointvar, local_points, adios2.Mode.Sync)

#     # Write celltype
#     io.DefineAttribute("CellType", mesh.topology.cell_name())

#     # Write basix properties
#     cmap = mesh.geometry.cmap
#     io.DefineAttribute("Degree", np.array([cmap.degree], dtype=np.int32))
#     io.DefineAttribute("LagrangeVariant", np.array([cmap.variant], dtype=np.int32))

#     # Write topology
#     g_imap = mesh.geometry.index_map()
#     g_dmap = mesh.geometry.dofmap
#     num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
#     num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
#     start_cell = mesh.topology.index_map(mesh.topology.dim).local_range[0]
#     geom_layout = cmap.create_dof_layout()
#     num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh.topology.dim)

#     dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
#     assert g_dmap.shape[1] == num_dofs_per_cell
#     dofs_out[:, :] = np.asarray(
#         g_imap.local_to_global(g_dmap[:num_cells_local, :].reshape(-1))
#     ).reshape(dofs_out.shape)

#     dvar = io.DefineVariable(
#         "Topology",
#         dofs_out,
#         shape=[num_cells_global, num_dofs_per_cell],
#         start=[start_cell, 0],
#         count=[num_cells_local, num_dofs_per_cell],
#     )
#     outfile.Put(dvar, dofs_out)

#     # Add mesh permutations
#     mesh.topology.create_entity_permutations()
#     cell_perm = mesh.topology.get_cell_permutation_info()
#     pvar = io.DefineVariable(
#         "CellPermutations",
#         cell_perm,
#         shape=[num_cells_global],
#         start=[start_cell],
#         count=[num_cells_local],
#     )
#     outfile.Put(pvar, cell_perm)
#     outfile.PerformPuts()
#     outfile.EndStep()
#     outfile.Close()
#     assert adios.RemoveIO("MeshWriter")




# print(domain.comm.rank, local_node_map.reshape(sorted_recv_dofmap.shape))
# Gather all indices on root0

# Use new mesh library to visualize with xdmffile




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