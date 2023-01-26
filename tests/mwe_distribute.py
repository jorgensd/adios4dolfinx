from IPython import embed
import numpy.typing as npt
import argparse
import pathlib

import adios2
import dolfinx
import h5py
import numpy as np
import ufl
from IPython import embed
from mpi4py import MPI
import numba

from adios4dolfinx import (compute_local_range, index_owner,
                           read_mesh_from_legacy_h5)

comm = MPI.COMM_WORLD
path = (pathlib.Path("legacy")/"mesh.h5").absolute()
meshname = "/mesh"
mesh = read_mesh_from_legacy_h5(comm, path, meshname)
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 2))
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
#print(f"Rank {mesh.comm.rank} send to {unique_owners}")
mesh_to_data_comm = mesh.comm.Create_dist_graph(
    [mesh.comm.rank], [len(unique_owners)], list(unique_owners), reorder=False)
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
for i, owner in enumerate(owners):
    proc_pos = find_first(owner, dest)
    out_org_cell[offsets[proc_pos]+count[proc_pos]] = input_cells[i]
    out_org_pos[offsets[proc_pos]+count[proc_pos]] = dof_pos[i]
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

# 2 Get global input dof numbering from Input dofmap and return to owner
# 2.1 Read in dofmap from infile

# 2.2 Extract dofmap data

# 2.3 Send global to number to dof owner

# 3 Compute owner of global dof on distributed mesh

# 3.1 Create MPI neigh comm to owner.

# 3.2 Send global dof number to proc

# 3.3 Compute local dof input dof number (using local_range)
# and create reverse comm and send back

# 4 Populate local vector

# 5 Scatter forward


print(mesh.comm.rank, dof_pos, owners, "Sending:", out_org_cell, out_org_pos, "to", dest, out_size,
      "\nRecv from", source, "amount:", recv_size, cells_from_mesh, pos_from_mesh)

exit()
adios = adios2.ADIOS(comm)
io = adios.DeclareIO("Function reader")
io.SetEngine("HDF5")
infile = io.Open(str(path), adios2.Mode.Read)

# Compute how the cells has been partitioned when read from file
local_cell_range = compute_local_range(
    comm, mesh.topology.index_map(mesh.topology.dim).size_global)


if f"{meshname}/x_cell_dofs" not in io.AvailableVariables().keys():
    raise KeyError(f"Dof offsets not found at '{meshname}/x_cell_dofs'")
offsets = io.InquireVariable(f"{meshname}/x_cell_dofs")
shape = offsets.Shape()
assert len(shape) == 1
# As the offsets are one longer than the number of cells, we need to read in with an overlap
offsets.SetSelection([[local_cell_range[0]], [local_cell_range[1]+1-local_cell_range[0]]])
dofmap_offsets = np.empty(local_cell_range[1]+1-local_cell_range[0], dtype=np.dtype(offsets.Type().strip("_t")))
infile.Get(offsets, dofmap_offsets, adios2.Mode.Sync)

# Get the relevant part of the dofmap
if f"{meshname}/cell_dofs" not in io.AvailableVariables().keys():
    raise KeyError(f"Dof offsets not found at '{meshname}/cell_dofs'")
cell_dofs = io.InquireVariable(f"{meshname}/cell_dofs")
cell_dofs.SetSelection([[dofmap_offsets[0]], [dofmap_offsets[-1]-dofmap_offsets[0]]])
in_dofmap = np.empty(dofmap_offsets[-1]-dofmap_offsets[0], dtype=np.dtype(cell_dofs.Type().strip("_t")))
infile.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)
in_dofmap = in_dofmap.astype(np.int64)


# Partition input function values
if f"{meshname}/vector_0" not in io.AvailableVariables().keys():
    raise KeyError(f"Dof offsets not found at '{meshname}/vector_0'")
func = io.InquireVariable(f"{meshname}/vector_0")
func_shape = func.Shape()
assert len(func_shape) == 1
func_range = compute_local_range(comm, func_shape[0])
func.SetSelection([[func_range[0]], [func_range[1]-func_range[0]]])
vals = np.empty(func_range[1]-func_range[0], dtype=np.dtype(func.Type().strip("_t")))
infile.Get(func, vals, adios2.Mode.Sync)


# Create neighborhood communicator from distributed mesh to input owner
mesh_to_data_comm = mesh.comm.Create_dist_graph(
    [mesh.comm.rank], [len(unique_owners)], list(unique_owners), reorder=False)
source, dest, _ = mesh_to_data_comm.Get_dist_neighbors()


# Create neighborhood comm from input owner to distributed mesh
data_to_mesh_comm = mesh.comm.Create_dist_graph_adjacent(dest, source, reorder=False)
#print(mesh.comm.rank, data_to_mesh_comm.Get_dist_neighbors())


out_data = np.full(len(unique_owners), mesh.comm.rank, dtype=np.int32)

#print("MESH INDICES", mesh.comm.rank, mesh.topology.original_cell_index)
# USE NBX tell rank with data that it should send something here.
#
# Create neighborhood communicator
#
# Send global original cell index to fetch data for
#
# Get


# if f"{meshname}/cell_dofs" not in io.AvailableVariables().keys():
#     raise KeyError(f"Dof offsets not found at '{meshname}/cell_dofs'")
# compute_local_range

# # As th
# ranks_with_dofmap = index_owner(comm, mesh.topology.original_cell_index, num_cells_global)

# print(comm.rank, local_cell_range, num_cells_global, mesh.topology.original_cell_index,
#     )

# # Open ADIOS2 Reader
# # Get number of dofs per cell (to get correct striding )

# # mesh_topology = np.empty(
# #     (local_range[1]-local_range[0], shape[1]), dtype=np.int64)
# # infile.Get(topology, mesh_topology, adios2.Mode.Sync)

# # Get mesh cell type
# if f"{meshname}/topology/celltype" not in io.AvailableAttributes().keys():
#     raise KeyError(
#         f"Mesh cell type not found at '{meshname}/topology/celltype'")
# celltype = io.InquireAttribute(f"{meshname}/topology/celltype")
# cell_type = celltype.DataString()[0]
