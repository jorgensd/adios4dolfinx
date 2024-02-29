import adios2
from mpi4py import MPI
import numpy as np
import adios4dolfinx.utils
import dolfinx
from pathlib import Path


def g(x):
    return (-x[1], x[0])


family = "N1curl"
degree = 3
# shape = (2, )
shape = None
comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_rectangle(
    comm, [[1, 3], [5, 7]], [5, 5], ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
V = dolfinx.fem.functionspace(domain, (family, degree, shape))
u = dolfinx.fem.Function(V)
u.interpolate(g)
u.name = "u"
# with dolfinx.io.VTXWriter(u.function_space.mesh.comm, "test.bp", [u], engine="BP4") as bp:
#    bp.write(0.0)
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
assert recv_size.sum() == local_cell_range[1] - local_cell_range[0]

# Pack cells and send
send_cells = np.empty(len(insert_position), dtype=np.int64)
send_cells[insert_position] = original_cell_index
recv_cells = np.empty(recv_size.sum(), dtype=np.int64)
topology_to_owner_comm.Neighbor_alltoallv([send_cells, out_size, MPI.INT64_T], [recv_cells, recv_size, MPI.INT64_T])
local_cell_index = recv_cells - local_cell_range[0]

# Pack and send cell permutation info
domain.topology.create_entity_permutations()
cell_permutation_info = domain.topology.get_cell_permutation_info()[:num_cells_local]
send_perm = np.empty_like(send_cells, dtype=np.uint32)
send_perm[insert_position] = cell_permutation_info
recv_perm = np.empty_like(recv_cells, dtype=np.uint32)
topology_to_owner_comm.Neighbor_alltoallv([send_perm, out_size, MPI.UINT32_T], [recv_perm, recv_size, MPI.UINT32_T])
cell_permutation_info = np.empty_like(recv_perm)
cell_permutation_info[local_cell_index] = recv_perm

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
recv_dofmap = recv_dofmap.reshape(-1, num_nodes_per_cell)
sorted_recv_dofmap = np.empty_like(recv_dofmap)
sorted_recv_dofmap[local_cell_index] = recv_dofmap
original_cell_index_full = domain.topology.original_cell_index
num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
output_cell_owner = adios4dolfinx.utils.index_owner(domain.comm, original_cell_index_full, num_cells_global)
local_cell_range = adios4dolfinx.utils.compute_local_range(domain.comm, num_cells_global)

# Compute outgoing edges from current process and create neighbourhood communicator
# Also create number of outgoing cells at the same time
num_owned_nodes = domain.geometry.index_map().size_local
output_node_owner = adios4dolfinx.utils.index_owner(
    domain.comm, original_node_index, domain.geometry.index_map().size_global)
unique_node_owners, out_size_node = np.unique(output_node_owner[:num_owned_nodes], return_counts=True)
geometry_to_owner_comm = comm.Create_dist_graph(
    [domain.comm.rank], [len(unique_node_owners)], unique_node_owners, reorder=False)
source_geom, dest_geom, _ = geometry_to_owner_comm.Get_dist_neighbors()
assert np.allclose(dest_geom, unique_node_owners)

# Compute send size for nodes
ono = output_node_owner[:num_owned_nodes].reshape(-1, 1)
process_pos_indicator = (ono == unique_node_owners)

# Compute offset for cell index
send_offsets_nodes = np.zeros(len(out_size_node)+1, dtype=np.intc)
send_offsets_nodes[1:] = np.cumsum(out_size_node)
assert send_offsets_nodes[-1] == num_owned_nodes

# Compute local insert index for each node
proc_row, proc_col = np.nonzero(process_pos_indicator)
cum_pos = np.cumsum(process_pos_indicator, axis=0)
insert_position_node = cum_pos[proc_row, proc_col] - 1
insert_position_node += send_offsets_nodes[proc_col]

insert_position_unrolled = np.repeat(insert_position_node, 3) * 3
insert_position_unrolled += np.tile(np.arange(3), num_owned_nodes)

send_nodes = np.empty(len(insert_position_unrolled), dtype=domain.geometry.x.dtype)
send_nodes[insert_position_unrolled] = domain.geometry.x[:num_owned_nodes, :].reshape(-1)

# Send and recieve geometry sizes
send_sizes = out_size_node * 3
recv_size = np.zeros_like(source_geom, dtype=np.int32)
geometry_to_owner_comm.Neighbor_alltoall(send_sizes, recv_size)

# Send nodes
recv_nodes = np.empty(recv_size.sum(), dtype=domain.geometry.x.dtype)
geometry_to_owner_comm.Neighbor_alltoallv([send_nodes, send_sizes, MPI.DOUBLE], [recv_nodes, recv_size, MPI.DOUBLE])

# Send node ordering
send_indices = np.empty(num_owned_nodes, dtype=np.int64)
send_indices[insert_position_node] = original_node_index[:num_owned_nodes]
recv_indices = np.empty(recv_size.sum()//3, dtype=np.int64)
geometry_to_owner_comm.Neighbor_alltoallv([send_indices, out_size_node, MPI.INT64_T], [
                                          recv_indices, recv_size//3, MPI.INT64_T])
local_node_range = adios4dolfinx.utils.compute_local_range(domain.comm, domain.geometry.index_map().size_global)
recv_indices -= local_node_range[0]
# Sort geometry based on input index
recv_nodes = recv_nodes.reshape(-1, 3)
geometry = np.empty_like(recv_nodes)
geometry[recv_indices, :] = recv_nodes
geometry = geometry[:, :domain.geometry.dim].copy()

# Create index map for cells
cell_imap = dolfinx.common.IndexMap(
    comm, local_cell_range[1]-local_cell_range[0], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32))


assert (local_node_range[1] - local_node_range[0] == geometry.shape[0])
#
fn = "test_origina.bp"
adios = adios2.ADIOS(domain.comm)
io = adios.DeclareIO("MeshWriter")
io.SetEngine("BP4")
outfile = io.Open(str(fn), adios2.Mode.Write)
# Write geometry
pointvar = io.DefineVariable(
    "Points",
    geometry,
    shape=[domain.geometry.index_map().size_global, domain.geometry.dim],
    start=[local_node_range[0], 0],
    count=[geometry.shape[0], geometry.shape[1]],
)
outfile.Put(pointvar, geometry, adios2.Mode.Sync)

# Write celltype
io.DefineAttribute("CellType", domain.topology.cell_name())

# Write basix properties
cmap = domain.geometry.cmap
io.DefineAttribute("Degree", np.array([cmap.degree], dtype=np.int32))
io.DefineAttribute("LagrangeVariant", np.array([cmap.variant], dtype=np.int32))

# Write topology
assert sorted_recv_dofmap.shape[0] == local_cell_range[1] - local_cell_range[0]
dvar = io.DefineVariable(
    "Topology",
    sorted_recv_dofmap,
    shape=[num_cells_global, sorted_recv_dofmap.shape[1]],
    start=[local_cell_range[0], 0],
    count=[local_cell_range[1]-local_cell_range[0], sorted_recv_dofmap.shape[1]],
)
outfile.Put(dvar, sorted_recv_dofmap)

# Add mesh permutations
pvar = io.DefineVariable(
    "CellPermutations",
    cell_permutation_info,
    shape=[num_cells_global],
    start=[local_cell_range[0]],
    count=[local_cell_range[1]-local_cell_range[0]],
)
outfile.Put(pvar, cell_permutation_info)


# Extract function data (array is the same, keeping global indices from DOLFINx)
# Dofmap is moved by the original cell index similar to the mesh geometry dofmap
dofmap = V.dofmap
dmap = dofmap.list
num_dofs_per_cell = dmap.shape[1]
dofmap_bs = dofmap.bs
index_map_bs = dofmap.index_map_bs

# Unroll dofmap for block size
unrolled_dofmap = adios4dolfinx.utils.unroll_dofmap(dofmap.list[:num_cells_local, :], dofmap_bs)
dmap_loc = (unrolled_dofmap // index_map_bs).reshape(-1)
dmap_rem = (unrolled_dofmap % index_map_bs).reshape(-1)


# Convert imap index to global index
imap_global = dofmap.index_map.local_to_global(dmap_loc)
dofmap_global = (imap_global * index_map_bs + dmap_rem).reshape(unrolled_dofmap.shape)
num_dofs_per_cell = dofmap_global.shape[1]
insert_position_function_dofmap = np.repeat(insert_position, num_dofs_per_cell) * num_dofs_per_cell
insert_position_function_dofmap += np.tile(np.arange(num_dofs_per_cell), len(original_cell_index))

# Create send array for global dofmap
send_function_dofmap = np.empty(len(insert_position_function_dofmap), dtype=np.int64)
send_function_dofmap[insert_position_function_dofmap] = dofmap_global.reshape(-1)

# Compute dofmap sizes
send_offsets_dofmap = send_offsets * num_dofs_per_cell
send_sizes_dofmap = out_size * num_dofs_per_cell

# Send size for dofmap
recv_size_dofmap = np.zeros_like(source, dtype=np.int32)
topology_to_owner_comm.Neighbor_alltoall(send_sizes_dofmap, recv_size_dofmap)
#
recv_offsets_dofmap = np.zeros(len(source)+1, dtype=np.intc)
recv_offsets_dofmap[1:] = np.cumsum(recv_size_dofmap)
recv_function_dofmap = np.empty(recv_offsets_dofmap[-1], dtype=np.int64)
topology_to_owner_comm.Neighbor_alltoallv(
    [send_function_dofmap, send_sizes_dofmap, MPI.INT64_T],
    [recv_function_dofmap, recv_size_dofmap, MPI.INT64_T]
)


shaped_dofmap = recv_function_dofmap.reshape(local_cell_range[1]-local_cell_range[0], num_dofs_per_cell).copy()
final_dofmap = np.empty_like(shaped_dofmap)
final_dofmap[local_cell_index] = shaped_dofmap

# Get offsets of dofmap
num_cells_local = local_cell_range[1] - local_cell_range[0]
num_dofs_local_dmap = num_cells_local * num_dofs_per_cell
dofmap_imap = dolfinx.common.IndexMap(domain.comm, num_dofs_local_dmap)
local_dofmap_offsets = np.arange(num_cells_local + 1, dtype=np.int64)
local_dofmap_offsets[:] *= num_dofs_per_cell
local_dofmap_offsets[:] += dofmap_imap.local_range[0]

dofmap_var = io.DefineVariable(
    f"{u.name}_dofmap",
    np.zeros(num_dofs_local_dmap, dtype=np.int64),
    shape=[dofmap_imap.size_global],
    start=[dofmap_imap.local_range[0]],
    count=[dofmap_imap.size_local],
)
outfile.Put(dofmap_var, final_dofmap)

xdofmap_var = io.DefineVariable(
    f"{u.name}_XDofmap",
    local_dofmap_offsets,
    shape=[num_cells_global + 1],
    start=[local_cell_range[0]],
    count=[num_cells_local + 1],
)
outfile.Put(xdofmap_var, local_dofmap_offsets)

# Write local part of vector
num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
local_start = dofmap.index_map.local_range[0] * dofmap.index_map_bs
val_var = io.DefineVariable(
    f"{u.name}_values",
    np.zeros(num_dofs_local, dtype=u.dtype),
    shape=[num_dofs_global],
    start=[local_start],
    count=[num_dofs_local],
)
outfile.Put(val_var, u.x.array[:num_dofs_local])


# Add time step to file
t_arr = np.array([0.0], dtype=np.float64)
time_var = io.DefineVariable(
    f"{u.name}_time",
    t_arr,
    shape=[1],
    start=[0],
    count=[1 if domain.comm.rank == 0 else 0],
)
outfile.Put(time_var, t_arr)

outfile.PerformPuts()
outfile.EndStep()
outfile.Close()
assert adios.RemoveIO("MeshWriter")

MPI.COMM_WORLD.Barrier()
# in_mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, fn, "BP4", dolfinx.mesh.GhostMode.none)
# V_in = dolfinx.fem.functionspace(in_mesh, (family, degree, shape))
in_mesh = domain
V_in = V

u_in = dolfinx.fem.Function(V_in)
u_in.name = "u"
adios4dolfinx.read_function(u_in, fn)
u_ex = dolfinx.fem.Function(V_in)
u_ex.interpolate(g)
u_ex.name = "exact"
# with dolfinx.io.VTXWriter(u_in.function_space.mesh.comm, "test_in.bp", [u_in], engine="BP4") as bp:
#    bp.write(0.0)
# xdmf.write_function(u_ex)
np.testing.assert_allclose(u_in.x.array, u_ex.x.array, atol=1e-13)


adios4dolfinx.write_mesh_input_order(domain, "test_in.bp", "BP4")
adios4dolfinx.write_function_on_input_mesh(u, "test_in.bp", "BP4")
MPI.COMM_WORLD.Barrier()


u_file = dolfinx.fem.Function(V_in)
u_file.name = "u"
adios4dolfinx.read_function(u_file, "test_in.bp")

np.testing.assert_allclose(u_file.x.array, u_ex.x.array, atol=1e-13)
# assert np.allclose(in_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*ufl.dx(domain=in_mesh))), op=MPI.SUM),
#                    domain.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*ufl.dx(domain=domain))), op=MPI.SUM))

# print(domain.comm.rank, local_node_map.reshape(sorted_recv_dofmap.shape))
# Gather all indices on root0

# Use new mesh library to visualize with xdmffile


# print(output_cell_owner, insert_position)
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
