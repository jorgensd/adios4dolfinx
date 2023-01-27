import pathlib

import adios2
import dolfinx
import numba
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from adios4dolfinx import (compute_local_range, index_owner,
                           read_mesh_from_legacy_h5)

comm = MPI.COMM_WORLD
path = (pathlib.Path("legacy")/"mesh.h5").absolute()
meshname = "/mesh"
mesh = read_mesh_from_legacy_h5(comm, path, meshname)
V = dolfinx.fem.FunctionSpace(mesh, ("DG", 2))
u = dolfinx.fem.Function(V)


# 1. Compute mesh to input cells
num_cells_global = mesh.topology.index_map(
    mesh.topology.dim).size_global
num_owned_dofs = V.dofmap.index_map.size_local
num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
dofmap_bs = V.dofmap.bs
imap_bs = V.dofmap.index_map_bs
local_cell = np.empty(num_owned_dofs*imap_bs, dtype=np.int32)  # Local cell index for each dof owned by process
dof_pos = np.empty(num_owned_dofs*imap_bs, dtype=np.int32)  # Position in dofmap for said dof
for c in range(num_owned_cells):
    for i, dof in enumerate(V.dofmap.cell_dofs(c)):
        for b in range(dofmap_bs):
            if dof*dofmap_bs + b < num_owned_dofs * imap_bs:
                local_cell[dof*dofmap_bs + b] = c
                dof_pos[dof*dofmap_bs + b] = i * dofmap_bs + b
input_cells = mesh.topology.original_cell_index[local_cell]

# 1.1 Compute mesh->input communicator
unique_input_cells = np.unique(input_cells)
owners = index_owner(mesh.comm, input_cells, num_cells_global)
unique_owners = np.unique(owners)
# print(f"Rank {mesh.comm.rank} send to {unique_owners}")
mesh_to_data_comm = mesh.comm.Create_dist_graph(
    [mesh.comm.rank], [len(unique_owners)], unique_owners, reorder=False)
source, dest, _ = mesh_to_data_comm.Get_dist_neighbors()
dest = np.asarray(dest, dtype=np.int32)


@numba.jit
def find_first(b: int, a: npt.NDArray[np.int32]):
    for i, ai in enumerate(a):
        if ai == b:
            return i


# 1.2 Send input cell, dof_pos to input comm
# Compute amount of data to send to each process
out_size = np.zeros(len(dest), dtype=np.int32)
for owner in owners:
    proc_pos = find_first(owner, dest)
    out_size[proc_pos] += 1
recv_size = np.zeros(len(source), dtype=np.int32)
mesh_to_data_comm.Neighbor_alltoall(out_size, recv_size)

# Sort output for sending
offsets = np.zeros(len(out_size)+1, dtype=np.intc)
offsets[1:] = np.cumsum(out_size)
out_org_cell = np.zeros(offsets[-1], dtype=np.int32)
out_org_pos = np.zeros(offsets[-1], dtype=np.int32)
count = np.zeros_like(out_size, dtype=np.int32)
proc_to_dof = np.zeros(num_owned_dofs*imap_bs, dtype=np.int32)
for i, owner in enumerate(owners):
    proc_pos = find_first(owner, dest)
    out_org_cell[offsets[proc_pos]+count[proc_pos]] = input_cells[i]
    out_org_pos[offsets[proc_pos]+count[proc_pos]] = dof_pos[i]
    proc_to_dof[offsets[proc_pos]+count[proc_pos]] = i
    count[proc_pos] += 1

cells_from_mesh = np.zeros(sum(recv_size), dtype=np.int32)
pos_from_mesh = np.zeros(sum(recv_size), dtype=np.intc)
in_offsets = np.zeros(len(recv_size)+1, dtype=np.intc)
in_offsets[1:] = np.cumsum(recv_size)
s_msg = [out_org_cell, out_size, MPI.INT32_T]
r_msg = [cells_from_mesh, recv_size, MPI.INT32_T]
mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)
s_msg = [out_org_pos, out_size, MPI.INT32_T]
r_msg = [pos_from_mesh, recv_size, MPI.INT32_T]
mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)
# print(f"{mesh.comm.rank} first send a total of {sum(out_size)} to {dest} distributed as {out_size}\n" +
#       f"and recieves {sum(recv_size)} distributed as {recv_size} from {source}")

# 2 Get global input dof numbering from Input dofmap and return to owner
# 2.1 Read in dofmap from infile
adios = adios2.ADIOS(comm)
io = adios.DeclareIO("Function reader")
io.SetEngine("HDF5")
infile = io.Open(str(path), adios2.Mode.Read)

# Compute how the cells has been partitioned when read from file
local_cell_range = compute_local_range(
    comm, mesh.topology.index_map(mesh.topology.dim).size_global)

if f"{meshname}/x_cell_dofs" not in io.AvailableVariables().keys():
    raise KeyError(f"Dof offsets not found at '{meshname}/x_cell_dofs'")
d_offsets = io.InquireVariable(f"{meshname}/x_cell_dofs")
shape = d_offsets.Shape()
assert len(shape) == 1
# As the offsets are one longer than the number of cells, we need to read in with an overlap
d_offsets.SetSelection([[local_cell_range[0]], [local_cell_range[1]+1-local_cell_range[0]]])
dofmap_offsets = np.empty(local_cell_range[1]+1-local_cell_range[0], dtype=np.dtype(d_offsets.Type().strip("_t")))
infile.Get(d_offsets, dofmap_offsets, adios2.Mode.Sync)
# Get the relevant part of the dofmap
if f"{meshname}/cell_dofs" not in io.AvailableVariables().keys():
    raise KeyError(f"Dof offsets not found at '{meshname}/cell_dofs'")
cell_dofs = io.InquireVariable(f"{meshname}/cell_dofs")
cell_dofs.SetSelection([[dofmap_offsets[0]], [dofmap_offsets[-1]-dofmap_offsets[0]]])
in_dofmap = np.empty(dofmap_offsets[-1]-dofmap_offsets[0], dtype=np.dtype(cell_dofs.Type().strip("_t")))
infile.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)
in_dofmap = in_dofmap.astype(np.int64)

# 2.2 Extract dofmap data
global_dofs = np.zeros_like(cells_from_mesh, dtype=np.int64)
for i, (cell, pos) in enumerate(zip(cells_from_mesh, pos_from_mesh.astype(np.uint64))):
    input_cell_pos = cell-local_cell_range[0]
    dofmap_pos = dofmap_offsets[input_cell_pos] + pos - dofmap_offsets[0]
    global_dofs[i] = in_dofmap[dofmap_pos]

# 2.3 Send global to number to dof owner
data_to_mesh_comm = mesh.comm.Create_dist_graph_adjacent(dest, source, reorder=False)

incoming_global_dofs = np.zeros(sum(out_size), dtype=np.int64)
s_msg = [global_dofs, recv_size, MPI.INT64_T]
r_msg = [incoming_global_dofs, out_size, MPI.INT64_T]
# print(f"{mesh.comm.rank} then send a total of {sum(recv_size)} distributed as {recv_size} to {dest2}\n"
#       + f"and recieves a total of {sum(out_size)} distributed as{out_size} from {source2}")
data_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

# Sort incoming global dofs as they were inputted
sorted_global_dofs = np.zeros_like(incoming_global_dofs, dtype=np.int64)
assert len(incoming_global_dofs) == num_owned_dofs*imap_bs
for i in range(len(dest)):
    pos = np.cumsum(out_size[:i])
    for j in range(out_size[i]):
        sorted_global_dofs[proc_to_dof[offsets[i] + j]] = incoming_global_dofs[offsets[i] + j]

# 3 Compute owner of global dof on distributed mesh
num_dof_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
owned_dof_input_owner = index_owner(mesh.comm, sorted_global_dofs, num_dof_global)

# 3.1 Create MPI neigh comm to owner.
unique_dof_owners = np.unique(owned_dof_input_owner)
mesh_to_dof_comm = mesh.comm.Create_dist_graph(
    [mesh.comm.rank], [len(unique_dof_owners)], unique_dof_owners, reorder=False)
dof_source, dof_dest, _ = mesh_to_dof_comm.Get_dist_neighbors()
dof_dest = np.asarray(dof_dest, dtype=np.int32)

# 3.2 Send global dof number to proc
dof_out_size = np.zeros(len(dof_dest), dtype=np.int32)
for owner in owned_dof_input_owner:
    proc_pos = find_first(owner, dof_dest)
    dof_out_size[proc_pos] += 1
dof_recv_size = np.zeros(len(dof_source), dtype=np.int32)
mesh_to_dof_comm.Neighbor_alltoall(dof_out_size, dof_recv_size)

# Sort output for sending
dofs_offsets = np.zeros(len(dof_out_size)+1, dtype=np.intc)
dofs_offsets[1:] = np.cumsum(dof_out_size)
out_dofs = np.zeros(dofs_offsets[-1], dtype=np.int64)
dof_count = np.zeros_like(dof_out_size, dtype=np.int32)
proc_to_local = np.zeros_like(sorted_global_dofs, dtype=np.int32)  # Map output to local dof
for i, (dof, owner) in enumerate(zip(sorted_global_dofs, owned_dof_input_owner)):
    proc_pos = find_first(owner, dof_dest)
    out_dofs[dofs_offsets[proc_pos]+dof_count[proc_pos]] = dof
    proc_to_local[dofs_offsets[proc_pos]+dof_count[proc_pos]] = i
    dof_count[proc_pos] += 1
input_dofs = np.zeros(sum(dof_recv_size), dtype=np.int64)
s_msg = [out_dofs, dof_out_size, MPI.INT64_T]
r_msg = [input_dofs, dof_recv_size, MPI.INT64_T]
mesh_to_dof_comm.Neighbor_alltoallv(s_msg, r_msg)


# 3.3 Read vector data
if f"{meshname}/vector_0" not in io.AvailableVariables().keys():
    raise KeyError(f"Dof offsets not found at '{meshname}/vector_0'")
func = io.InquireVariable(f"{meshname}/vector_0")
func_shape = func.Shape()
assert len(func_shape) == 1
func_range = compute_local_range(comm, func_shape[0])
func.SetSelection([[func_range[0]], [func_range[1]-func_range[0]]])
vals = np.empty(func_range[1]-func_range[0], dtype=np.dtype(func.Type().strip("_t")))
infile.Get(func, vals, adios2.Mode.Sync)

# Compute local dof input dof number (using local_range)
input_vals = np.zeros(sum(dof_recv_size), dtype=np.float64)
for i, dof in enumerate(input_dofs):
    pos = dof - func_range[0]
    input_vals[i] = vals[pos]

# 3.4 Create reverse comm and send back
dof_to_mesh_comm = mesh.comm.Create_dist_graph_adjacent(dof_dest, dof_source, reorder=False)
incoming_vals = np.zeros(sum(dof_out_size), dtype=np.float64)
s_msg = [input_vals, dof_recv_size, MPI.DOUBLE]
r_msg = [incoming_vals, dof_out_size, MPI.DOUBLE]
# print(f"{mesh.comm.rank} then send a total of {sum(recv_size)} distributed as {recv_size} to {dest2}\n"
#       + f"and recieves a total of {sum(out_size)} distributed as{out_size} from {source2}")
dof_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

# 4 Populate local vector
arr = u.x.array
for i in range(len(dof_dest)):
    pos = np.cumsum(dof_out_size[:i])
    for j in range(dof_out_size[i]):
        arr[proc_to_local[dofs_offsets[i] + j]] = incoming_vals[dofs_offsets[i] + j]

# 5 Scatter forward
u.x.scatter_forward()

with dolfinx.io.VTXWriter(mesh.comm, "u_checkpointed.bp", [u]) as vtx:
    vtx.write(0.)
