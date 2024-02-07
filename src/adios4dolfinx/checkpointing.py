# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from pathlib import Path
from typing import Optional, Union

from mpi4py import MPI

import adios2
import basix
import dolfinx
import numpy as np
import ufl

from .adios2_helpers import (adios_to_numpy_dtype, read_array, read_cell_perms,
                             read_dofmap)
from .comm_helpers import (send_and_recv_cell_perm,
                           send_dofmap_and_recv_values,
                           send_dofs_and_recv_values)
from .utils import compute_dofmap_pos, compute_local_range, index_owner, unroll_dofmap

__all__ = [
    "read_mesh",
    "write_function",
    "read_function",
    "write_mesh",
    "snapshot_checkpoint",
    "read_meshtags",
    "write_meshtags"
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
    cmap = mesh.geometry.cmaps[0]
    io.DefineAttribute("Degree", np.array([cmap.degree], dtype=np.int32))
    io.DefineAttribute("LagrangeVariant", np.array([cmap.variant], dtype=np.int32))

    # Write topology
    g_imap = mesh.geometry.index_map()
    g_dmap = mesh.geometry.dofmap
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    start_cell = mesh.topology.index_map(mesh.topology.dim).local_range[0]
    geom_layout = cmap.create_dof_layout()
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh.topology.dim)

    dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
    assert g_dmap.shape[1] == num_dofs_per_cell
    dofs_out[:, :] = np.asarray(
        g_imap.local_to_global(g_dmap[:num_cells_local, :].reshape(-1))
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


def write_meshtags(filename: Union[Path, str], mesh: dolfinx.mesh.Mesh, meshtags: dolfinx.mesh.MeshTags,
                   engine: Optional[str] = "BP4"):
    """
    Write meshtags associated with input mesh to file.

    .. note::
        For this checkpoint to work, the mesh must be written to file using :func:`write_mesh`
        before calling this function.

    Args:
        filename: Path to save meshtags (with file-extension)
        mesh: The mesh associated with the meshtags
        meshtags: The meshtags to write to file
        engine: Adios2 Engine
    """
    tag_entities = meshtags.indices
    dim = meshtags.dim
    num_tag_entities_local = mesh.topology.index_map(dim).size_local
    local_tag_entities = tag_entities[tag_entities < num_tag_entities_local]
    local_values = meshtags.values[:len(local_tag_entities)]

    num_saved_tag_entities = len(local_tag_entities)
    local_start = mesh.comm.exscan(num_saved_tag_entities, op=MPI.SUM)
    local_start = local_start if mesh.comm.rank != 0 else 0
    global_num_tag_entities = mesh.comm.allreduce(num_saved_tag_entities, op=MPI.SUM)
    dof_layout = mesh.geometry.cmaps[0].create_dof_layout()
    num_dofs_per_entity = dof_layout.num_entity_closure_dofs(dim)

    entities_to_geometry = dolfinx.cpp.mesh.entities_to_geometry(
        mesh._cpp_object, dim, tag_entities, False)

    indices = mesh.geometry.index_map().local_to_global(entities_to_geometry.reshape(-1))

    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshTagWriter")
    io.SetEngine(engine)
    outfile = io.Open(str(filename), adios2.Mode.Append)
    # Write meshtag topology
    topology_var = io.DefineVariable(
        meshtags.name+"_topology",
        indices,
        shape=[global_num_tag_entities, num_dofs_per_entity],
        start=[local_start, 0],
        count=[num_saved_tag_entities, num_dofs_per_entity],
    )
    outfile.Put(topology_var, indices, adios2.Mode.Sync)

    # Write meshtag topology
    values_var = io.DefineVariable(
        meshtags.name+"_values",
        local_values,
        shape=[global_num_tag_entities],
        start=[local_start],
        count=[num_saved_tag_entities],
    )
    outfile.Put(values_var, local_values, adios2.Mode.Sync)

    # Write meshtag dim
    io.DefineAttribute(meshtags.name + "_dim", np.array([meshtags.dim], dtype=np.uint8))

    outfile.PerformPuts()
    outfile.EndStep()
    outfile.Close()


def read_meshtags(filename: str, mesh: dolfinx.mesh.Mesh, meshtag_name: str,
                  engine: str = "BP4") -> dolfinx.mesh.MeshTags:
    """
    Read meshtags from file and return a :class:`dolfinx.mesh.MeshTags` object.

    Args:
        filename: Path to meshtags file (with file-extension)
        mesh: The mesh associated with the meshtags
        meshtag_name: The name of the meshtag to read
        engine: Adios2 Engine
    Returns:
        The meshtags
    """
    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshTagsReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    # Get mesh cell type
    dim_attr_name = f"{meshtag_name}_dim"
    step = 0
    for i in range(infile.Steps()):
        infile.BeginStep()
        if dim_attr_name in io.AvailableAttributes().keys():
            step = i
            break
        infile.EndStep()
    if dim_attr_name not in io.AvailableAttributes().keys():
        raise KeyError(f"{dim_attr_name} not found in {filename}")

    m_dim = io.InquireAttribute(dim_attr_name)
    dim = int(m_dim.Data()[0])

    # Get mesh tags entites
    topology_name = f"{meshtag_name}_topology"
    for i in range(step, infile.Steps()):
        if i > step:
            infile.BeginStep()
        if topology_name in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if topology_name not in io.AvailableVariables().keys():
        raise KeyError(f"{topology_name} not found in {filename}")

    topology = io.InquireVariable(topology_name)
    top_shape = topology.Shape()
    topology_range = compute_local_range(mesh.comm, top_shape[0])

    topology.SetSelection(
        [[topology_range[0], 0], [topology_range[1] - topology_range[0], top_shape[1]]]
    )
    mesh_entities = np.empty((topology_range[1] - topology_range[0], top_shape[1]), dtype=np.int64)
    infile.Get(topology, mesh_entities, adios2.Mode.Deferred)

    # Get mesh tags values
    values_name = f"{meshtag_name}_values"
    if values_name not in io.AvailableVariables().keys():
        raise KeyError(f"{values_name} not found")

    values = io.InquireVariable(values_name)
    val_shape = values.Shape()
    assert val_shape[0] == top_shape[0]
    values.SetSelection(
        [[topology_range[0]], [topology_range[1] - topology_range[0]]]
    )
    tag_values = np.empty((topology_range[1] - topology_range[0]), dtype=np.int32)
    infile.Get(values, tag_values, adios2.Mode.Deferred)

    infile.PerformGets()
    infile.EndStep()
    infile.Close()
    assert adios.RemoveIO("MeshTagsReader")

    local_entities, local_values = dolfinx.cpp.io.distribute_entity_data(
        mesh._cpp_object, int(dim), mesh_entities, tag_values)
    mesh.topology.create_connectivity(dim, 0)
    mesh.topology.create_connectivity(dim, mesh.topology.dim)

    adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)

    local_values = np.array(local_values, dtype=np.int32)

    mt = dolfinx.mesh.meshtags_from_entities(mesh, int(dim), adj, local_values)
    mt.name = meshtag_name

    return mt


def read_function(u: dolfinx.fem.Function, filename: Path, engine: str = "BP4", time: float = 0.,
                  legacy: bool = False):
    """
    Read checkpoint from file and fill it into `u`.

    Args:
        u: Function to fill
        filename: Path to checkpoint
        engine: ADIOS engine type used for reading
        legacy: If checkpoint is from prior to time-dependent writing set to True
    """
    mesh = u.function_space.mesh
    comm = mesh.comm
    adios = adios2.ADIOS(comm)
    name = u.name
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
    if legacy:
        dofmap_path = "Dofmap"
        xdofmap_path = "XDofmap"
    else:
        dofmap_path = f"{name}_dofmap"
        xdofmap_path = f"{name}_XDofmap"
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
    if legacy:
        array_path = "Values"
    else:
        array_path = f"{name}_values"
    time_name = f"{name}_time"
    input_array, starting_pos = read_array(adios, filename, array_path, engine, comm, time, time_name,
                                           legacy=legacy)
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
            start, end = input_dofmap.offsets[l_cell:l_cell+2]
            # FIXME: Tempoary cast uint32 to integer as transformations doesn't support uint32 with the switch
            # to nanobind
            element.apply_transpose_dof_transformation(
                recv_array[int(start):int(end)], int(input_perms[l_cell]), bs
            )
            element.apply_inverse_transpose_dof_transformation(
                recv_array[int(start):int(end)], int(inc_perms[i]), bs
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
        raise KeyError(f"Mesh cell type not found at CellType in {file}")
    celltype = io.InquireAttribute("CellType")
    cell_type = celltype.DataString()[0]

    # Get basix info
    if "LagrangeVariant" not in io.AvailableAttributes().keys():
        raise KeyError(f"Mesh LagrangeVariant not found in {file}")
    lvar = io.InquireAttribute("LagrangeVariant").Data()[0]
    if "Degree" not in io.AvailableAttributes().keys():
        raise KeyError(f"Mesh degree not found in {file}")
    degree = io.InquireAttribute("Degree").Data()[0]

    # Get mesh geometry
    if "Points" not in io.AvailableVariables().keys():
        raise KeyError(f"Mesh coordinates not found at Points in {file}")
    geometry = io.InquireVariable("Points")
    x_shape = geometry.Shape()
    geometry_range = compute_local_range(comm, x_shape[0])
    geometry.SetSelection(
        [[geometry_range[0], 0], [geometry_range[1] - geometry_range[0], x_shape[1]]]
    )
    mesh_geometry = np.empty(
        (geometry_range[1] - geometry_range[0], x_shape[1]),
        dtype=adios_to_numpy_dtype[geometry.Type()],
    )
    infile.Get(geometry, mesh_geometry, adios2.Mode.Deferred)
    # Get mesh topology (distributed)
    if "Topology" not in io.AvailableVariables().keys():
        raise KeyError("Mesh topology not found at Topology in {file}")
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
        basix.LagrangeVariant(int(lvar)),
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
    time: float = 0.0
):
    """
    Write function checkpoint to file.

    Args:
        u: Function to write to file
        filename: Path to write to
        engine: ADIOS2 engine
        mode: Write or append.
        time: Time-stamp for simulation
    """
    dofmap = u.function_space.dofmap
    values = u.x.array
    mesh = u.function_space.mesh
    comm = mesh.comm

    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("FunctionWriter")
    io.SetEngine(engine)

    # If mode is append, check if we have written the function to file before
    name = u.name

    if mode == adios2.Mode.Append:
        # First open the file in read-mode to check if the function has been written before
        read_file = io.Open(str(filename), adios2.Mode.Read)
        io.SetEngine(engine)
        first_write = True
        for _ in range(read_file.Steps()):
            read_file.BeginStep()
            if name in io.AvailableAttributes():
                first_write = False
                break
            read_file.EndStep()
        read_file.Close()
    outfile = io.Open(str(filename), mode)
    io.DefineAttribute(name, name)
    outfile.BeginStep()

    # Add time step to file
    t_arr = np.array([time], dtype=np.float64)
    time_var = io.DefineVariable(
        f"{name}_time",
        t_arr,
        shape=[1],
        start=[0],
        count=[1 if mesh.comm.rank == 0 else 0],
    )
    outfile.Put(time_var, t_arr)

    # Write local part of vector
    num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
    num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
    local_start = dofmap.index_map.local_range[0] * dofmap.index_map_bs
    val_var = io.DefineVariable(
        f"{name}_values",
        np.zeros(num_dofs_local, dtype=u.dtype),
        shape=[num_dofs_global],
        start=[local_start],
        count=[num_dofs_local],
    )
    outfile.Put(val_var, values[:num_dofs_local])

    if not first_write:
        outfile.PerformPuts()
        outfile.EndStep()
        outfile.Close()
        assert adios.RemoveIO("FunctionWriter")
        return

    # Convert local dofmap into global_dofmap
    dmap = dofmap.list
    num_dofs_per_cell = dmap.shape[1]
    dofmap_bs = dofmap.bs
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_dofs_local_dmap = num_cells_local * num_dofs_per_cell * dofmap_bs
    index_map_bs = dofmap.index_map_bs

    # Unroll dofmap for block size
    unrolled_dofmap = unroll_dofmap(dofmap.list[:num_cells_local, :], dofmap_bs)
    dmap_loc = (unrolled_dofmap // index_map_bs).reshape(-1)
    dmap_rem = (unrolled_dofmap % index_map_bs).reshape(-1)

    local_dofmap_offsets = np.arange(num_cells_local + 1, dtype=np.int64)
    local_dofmap_offsets[:] *= num_dofs_per_cell * dofmap_bs

    # Convert imap index to global index
    imap_global = dofmap.index_map.local_to_global(dmap_loc)
    dofmap_global = imap_global * index_map_bs + dmap_rem

    # Get offsets of dofmap
    dofmap_imap = dolfinx.common.IndexMap(mesh.comm, num_dofs_local_dmap)
    dofmap_var = io.DefineVariable(
        f"{name}_dofmap",
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
        f"{name}_XDofmap",
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
