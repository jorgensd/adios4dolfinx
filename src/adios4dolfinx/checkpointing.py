# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from pathlib import Path

import adios2
import basix
import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from .adios2_helpers import (adios_to_numpy_dtype, read_array, read_cell_perms,
                             read_dofmap)
from .comm_helpers import (send_and_recv_cell_perm,
                           send_dofmap_and_recv_values,
                           send_dofs_and_recv_values)
from .utils import compute_dofmap_pos, compute_local_range, index_owner

__all__ = [
    "read_mesh",
    "write_function",
    "read_function",
    "write_mesh",
    "snapshot_checkpoint",
]


def snapshot_checkpoint(uh: dolfinx.fem.Function, file: Path, mode: adios2.Mode):
    """Read or write a snapshot checkpoint

    This checkpoint is only meant to be used on the same mesh during the same simulation.

    :param uh: The function to write data from or read to
    :param file: The file to write to or read from
    :param mode: Either read or write
    """
    # Create ADIOS IO
    adios = adios2.ADIOS(uh.function_space.mesh.comm)
    io_name = "SnapshotCheckPoint"
    io = adios.DeclareIO(io_name)
    io.SetEngine("BP4")
    if mode not in [adios2.Mode.Write, adios2.Mode.Read]:
        raise ValueError("Got invalid mode {mode}")
    adios_file = io.Open(str(file), mode)

    if mode == adios2.Mode.Write:
        dofmap = uh.function_space.dofmap
        num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
        local_dofs = uh.x.array[:num_dofs_local].copy()

        # Write to file
        adios_file.BeginStep()
        dofs = io.DefineVariable("dofs", local_dofs, count=[num_dofs_local])
        adios_file.Put(dofs, local_dofs, adios2.Mode.Sync)
        adios_file.EndStep()
    else:
        adios_file.BeginStep()
        in_variable = io.InquireVariable("dofs")
        in_variable.SetBlockSelection(uh.function_space.mesh.comm.rank)
        adios_file.Get(in_variable, uh.x.array, adios2.Mode.Sync)
        adios_file.EndStep()
        uh.x.scatter_forward()
    adios_file.Close()
    adios.RemoveIO(io_name)


def write_mesh(mesh: dolfinx.mesh.Mesh, filename: Path, engine: str = "BP4"):
    """
    Write a mesh to specified ADIOS2 format, see:
    https://adios2.readthedocs.io/en/stable/engines/engines.html
    for possible formats.

    Args:
        mesh: The mesh to write to file
        filename: Path to save mesh (without file-extension)
        engine: Adios2 Engine
    """
    num_xdofs_local = mesh.geometry.index_map().size_local
    num_xdofs_global = mesh.geometry.index_map().size_global
    local_range = mesh.geometry.index_map().local_range
    gdim = mesh.geometry.dim

    local_points = mesh.geometry.x[:num_xdofs_local, :gdim].copy()
    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshWriter")
    io.SetEngine(engine)
    outfile = io.Open(str(filename), adios2.Mode.Write)
    # Write geometry
    pointvar = io.DefineVariable(
        "Points",
        local_points,
        shape=[num_xdofs_global, gdim],
        start=[local_range[0], 0],
        count=[num_xdofs_local, gdim],
    )
    outfile.Put(pointvar, local_points, adios2.Mode.Sync)

    # Write celltype
    io.DefineAttribute("CellType", mesh.topology.cell_name())

    # Write basix properties
    cmaps = mesh.geometry.cmaps
    assert len(cmaps) == 1, "Does not support mixed cell type"
    io.DefineAttribute(
        "Degree", np.array([mesh.geometry.cmaps[0].degree], dtype=np.int32)
    )
    io.DefineAttribute(
        "LagrangeVariant", np.array([mesh.geometry.cmaps[0].variant], dtype=np.int32)
    )

    # Write topology
    g_imap = mesh.geometry.index_map()
    g_dmap = mesh.geometry.dofmap
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    start_cell = mesh.topology.index_map(mesh.topology.dim).local_range[0]
    geom_layout = mesh.geometry.cmaps[0].create_dof_layout()
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh.topology.dim)

    dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
    assert g_dmap.shape[1] == num_dofs_per_cell
    dofs_out[:, :] = g_imap.local_to_global(
        g_dmap[:num_cells_local, :].reshape(-1)
    ).reshape(dofs_out.shape)

    dvar = io.DefineVariable(
        "Topology",
        dofs_out,
        shape=[num_cells_global, num_dofs_per_cell],
        start=[start_cell, 0],
        count=[num_cells_local, num_dofs_per_cell],
    )
    outfile.Put(dvar, dofs_out)

    # Add mesh permutations
    mesh.topology.create_entity_permutations()
    cell_perm = mesh.topology.get_cell_permutation_info()
    pvar = io.DefineVariable(
        "CellPermutations",
        cell_perm,
        shape=[num_cells_global],
        start=[start_cell],
        count=[num_cells_local],
    )
    outfile.Put(pvar, cell_perm)
    outfile.PerformPuts()
    outfile.EndStep()
    outfile.Close()
    assert adios.RemoveIO("MeshWriter")


def read_function(u: dolfinx.fem.Function, filename: Path, engine: str = "BP4"):
    """
    Read checkpoint from file and fill it into `u`.

    Args:
        u: Function to fill
        filename: Path to checkpoint
        engine: ADIOS engine type used for reading
    """
    mesh = u.function_space.mesh
    comm = mesh.comm
    adios = adios2.ADIOS(comm)
    # ----------------------Step 1---------------------------------
    # Compute index of input cells and get cell permutation
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    input_cells = mesh.topology.original_cell_index[:num_owned_cells]
    mesh.topology.create_entity_permutations()
    cell_perm = mesh.topology.get_cell_permutation_info()[:num_owned_cells]

    # Compute mesh->input communicator
    # 1.1 Compute mesh->input communicator
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    owners = index_owner(mesh.comm, input_cells, num_cells_global)

    # -------------------Step 2------------------------------------
    # Send and receive global cell index and cell perm
    inc_cells, inc_perms = send_and_recv_cell_perm(
        input_cells, cell_perm, owners, mesh.comm
    )

    # -------------------Step 3-----------------------------------
    # Read dofmap from file and compute dof owners
    dofmap_path = "Dofmap"
    xdofmap_path = "XDofmap"
    input_dofmap = read_dofmap(
        adios, comm, filename, dofmap_path, xdofmap_path, num_cells_global, engine
    )
    # Compute owner of dofs in dofmap
    num_dofs_global = (
        u.function_space.dofmap.index_map.size_global
        * u.function_space.dofmap.index_map_bs
    )
    dof_owner = index_owner(comm, input_dofmap.array, num_dofs_global)

    # --------------------Step 4-----------------------------------
    # Read array from file and communicate them to input dofmap process
    array_path = "Values"
    input_array, starting_pos = read_array(adios, filename, array_path, engine, comm)
    recv_array = send_dofs_and_recv_values(
        input_dofmap.array, dof_owner, comm, input_array, starting_pos
    )

    # -------------------Step 5--------------------------------------
    # Invert permutation of input data based on input perm
    # Then apply current permutation to the local data
    element = u.function_space.element
    if element.needs_dof_transformations:
        bs = u.function_space.dofmap.bs

        # Read input cell permutations on dofmap process
        local_input_range = compute_local_range(comm, num_cells_global)
        input_local_cell_index = inc_cells - local_input_range[0]
        input_perms = read_cell_perms(
            adios, comm, filename, "CellPermutations", num_cells_global, engine
        )

        # First invert input data to reference element then transform to current mesh
        for i, l_cell in enumerate(input_local_cell_index):
            start, end = input_dofmap.offsets[l_cell], input_dofmap.offsets[l_cell + 1]
            element.apply_transpose_dof_transformation(
                recv_array[start:end], input_perms[l_cell], bs
            )
            element.apply_inverse_transpose_dof_transformation(
                recv_array[start:end], inc_perms[i], bs
            )
    # ------------------Step 6----------------------------------------
    # For each dof owned by a process, find the local position in the dofmap.
    V = u.function_space
    local_cells, dof_pos = compute_dofmap_pos(V)
    input_cells = V.mesh.topology.original_cell_index[local_cells]
    num_cells_global = V.mesh.topology.index_map(V.mesh.topology.dim).size_global
    owners = index_owner(V.mesh.comm, input_cells, num_cells_global)
    unique_owners = np.unique(owners)
    # FIXME: In C++ use NBX to find neighbourhood
    sub_comm = V.mesh.comm.Create_dist_graph(
        [V.mesh.comm.rank], [len(unique_owners)], unique_owners, reorder=False
    )
    source, dest, _ = sub_comm.Get_dist_neighbors()
    sub_comm.Free()

    owned_values = send_dofmap_and_recv_values(
        comm,
        np.asarray(source, dtype=np.int32),
        np.asarray(dest, dtype=np.int32),
        owners,
        input_cells,
        dof_pos,
        num_cells_global,
        recv_array,
        input_dofmap.offsets,
    )
    u.x.array[: len(owned_values)] = owned_values
    u.x.scatter_forward()


def read_mesh(
    comm: MPI.Intracomm, file: Path, engine: str, ghost_mode: dolfinx.mesh.GhostMode
) -> dolfinx.mesh.Mesh:
    """
    Read an ADIOS2 mesh into DOLFINx.

    Args:
        comm: The MPI communciator to distribute the mesh over
        file: Path to input file
        engine: ADIOS engine to use for reading (BP4, BP5 or HDF5)
        ghost_mode: Ghost mode to use for mesh
    Returns:
        The distributed mesh
    """
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("MeshReader")
    io.SetEngine(engine)
    infile = io.Open(str(file), adios2.Mode.Read)
    infile.BeginStep()

    # Get mesh cell type
    if "CellType" not in io.AvailableAttributes().keys():
        raise KeyError("Mesh cell type not found at CellType")
    celltype = io.InquireAttribute("CellType")
    cell_type = celltype.DataString()[0]

    # Get basix info
    if "LagrangeVariant" not in io.AvailableAttributes().keys():
        raise KeyError("Mesh LagrangeVariant not found")
    lvar = io.InquireAttribute("LagrangeVariant").Data()[0]
    if "Degree" not in io.AvailableAttributes().keys():
        raise KeyError("Mesh degree not found")
    degree = io.InquireAttribute("Degree").Data()[0]

    # Get mesh geometry
    if "Points" not in io.AvailableVariables().keys():
        raise KeyError("Mesh coordinates not found at Points")
    geometry = io.InquireVariable("Points")
    x_shape = geometry.Shape()
    geometry_range = compute_local_range(comm, x_shape[0])
    geometry.SetSelection(
        [[geometry_range[0], 0], [geometry_range[1] - geometry_range[0], x_shape[1]]]
    )
    mesh_geometry = np.empty(
        (geometry_range[1] - geometry_range[0], x_shape[1]), dtype=adios_to_numpy_dtype[geometry.Type()]
    )
    infile.Get(geometry, mesh_geometry, adios2.Mode.Deferred)
    # Get mesh topology (distributed)
    if "Topology" not in io.AvailableVariables().keys():
        raise KeyError("Mesh topology not found at Topology'")
    topology = io.InquireVariable("Topology")
    shape = topology.Shape()
    local_range = compute_local_range(comm, shape[0])
    topology.SetSelection(
        [[local_range[0], 0], [local_range[1] - local_range[0], shape[1]]]
    )
    mesh_topology = np.empty(
        (local_range[1] - local_range[0], shape[1]), dtype=np.int64
    )
    infile.Get(topology, mesh_topology, adios2.Mode.Deferred)

    infile.PerformGets()
    infile.EndStep()
    assert adios.RemoveIO("MeshReader")

    # Create DOLFINx mesh
    element = basix.ufl.element(
        basix.ElementFamily.P,
        cell_type,
        degree,
        basix.LagrangeVariant(lvar),
        shape=(mesh_geometry.shape[1],),
        gdim=mesh_geometry.shape[1],
    )
    domain = ufl.Mesh(element)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(ghost_mode)
    return dolfinx.mesh.create_mesh(
        comm, mesh_topology, mesh_geometry, domain, partitioner
    )


def write_function(
    u: dolfinx.fem.Function,
    filename: Path,
    engine: str = "BP4",
    mode: adios2.Mode = adios2.Mode.Append,
):
    """
    Write function checkpoint to file.

    Args:
        u: Function to write to file
        filename: Path to write to
        egine: ADIOS2 engine
        mode: Write or append.
    """
    dofmap = u.function_space.dofmap
    values = u.x.array
    mesh = u.function_space.mesh
    comm = mesh.comm

    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("FunctionWriter")
    io.SetEngine(engine)
    outfile = io.Open(str(filename), adios2.Mode.Append)
    # Write local part of vector
    num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
    num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
    local_start = dofmap.index_map.local_range[0] * dofmap.index_map_bs
    outfile.BeginStep()
    val_var = io.DefineVariable(
        "Values",
        np.zeros(num_dofs_local, dtype=u.dtype),
        shape=[num_dofs_global],
        start=[local_start],
        count=[num_dofs_local],
    )
    outfile.Put(val_var, values[:num_dofs_local])

    # Convert local dofmap into global_dofmap
    dmap = dofmap.list
    num_dofs_per_cell = dmap.shape[1]
    dofmap_bs = dofmap.bs
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_dofs_local_dmap = num_cells_local * num_dofs_per_cell * dofmap_bs
    dmap_loc = np.empty(num_dofs_local_dmap, dtype=np.int32)
    dmap_rem = np.empty(num_dofs_local_dmap, dtype=np.int32)
    index_map_bs = dofmap.index_map_bs
    # Unroll local dofmap and convert into index map index
    for c in range(num_cells_local):
        for i, dof in enumerate(dmap[c]):
            for b in range(dofmap_bs):
                dmap_loc[(num_dofs_per_cell * c + i) * dofmap_bs + b] = (
                    dof * dofmap_bs + b
                ) // index_map_bs
                dmap_rem[(num_dofs_per_cell * c + i) * dofmap_bs + b] = (
                    dof * dofmap_bs + b
                ) % index_map_bs
    local_dofmap_offsets = np.arange(num_cells_local + 1, dtype=np.int64)
    local_dofmap_offsets[:] *= num_dofs_per_cell * dofmap_bs
    # Convert imap index to global index
    imap_global = dofmap.index_map.local_to_global(dmap_loc)
    dofmap_global = np.empty_like(dmap_loc, dtype=np.int64)
    for i in range(num_dofs_local_dmap):
        dofmap_global[i] = imap_global[i] * index_map_bs + dmap_rem[i]

    # Get offsets of dofmap
    dofmap_imap = dolfinx.common.IndexMap(mesh.comm, num_dofs_local_dmap)
    dofmap_var = io.DefineVariable(
        "Dofmap",
        np.zeros(num_dofs_local_dmap, dtype=np.int64),
        shape=[dofmap_imap.size_global],
        start=[dofmap_imap.local_range[0]],
        count=[dofmap_imap.size_local],
    )
    outfile.Put(dofmap_var, dofmap_global)

    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    cell_start = mesh.topology.index_map(mesh.topology.dim).local_range[0]
    local_dofmap_offsets += dofmap_imap.local_range[0]
    xdofmap_var = io.DefineVariable(
        "XDofmap",
        np.zeros(num_cells_local + 1, dtype=np.int64),
        shape=[num_cells_global + 1],
        start=[cell_start],
        count=[num_cells_local + 1],
    )
    outfile.Put(xdofmap_var, local_dofmap_offsets)
    outfile.PerformPuts()
    outfile.EndStep()
    outfile.Close()
    assert adios.RemoveIO("FunctionWriter")
