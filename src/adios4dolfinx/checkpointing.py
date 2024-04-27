# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import typing
from pathlib import Path

from mpi4py import MPI

import adios2
import basix
import dolfinx
import numpy as np
import ufl

from .adios2_helpers import (
    ADIOSFile,
    adios_to_numpy_dtype,
    read_adjacency_list,
    read_array,
    read_cell_perms,
    resolve_adios_scope,
)
from .comm_helpers import (
    send_and_recv_cell_perm,
    send_dofmap_and_recv_values,
    send_dofs_and_recv_values,
)
from .structures import FunctionData, MeshData
from .utils import (
    compute_dofmap_pos,
    compute_local_range,
    index_owner,
    unroll_dofmap,
    unroll_insert_position,
)
from .writers import write_function as _internal_function_writer
from .writers import write_mesh as _internal_mesh_writer

adios2 = resolve_adios_scope(adios2)

__all__ = [
    "read_mesh_data",
    "read_mesh",
    "write_function",
    "read_function",
    "write_mesh",
    "read_meshtags",
    "write_meshtags",
    "read_attributes",
    "write_attributes",
]


def write_attributes(
    filename: typing.Union[Path, str],
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    engine: str = "BP4",
):
    """Write attributes to file using ADIOS2.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attributes
        attributes: Dictionary of attributes to write to file
        engine: ADIOS2 engine to use
    """

    adios = adios2.ADIOS(comm)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Append,
        engine=engine,
        io_name="AttributesWriter",
    ) as adios_file:
        adios_file.file.BeginStep()

        for k, v in attributes.items():
            adios_file.io.DefineAttribute(f"{name}_{k}", v)

        adios_file.file.PerformPuts()
        adios_file.file.EndStep()


def read_attributes(
    filename: typing.Union[Path, str],
    comm: MPI.Intracomm,
    name: str,
    engine: str = "BP4",
) -> dict[str, np.ndarray]:
    """Read attributes from file using ADIOS2.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attributes
        engine: ADIOS2 engine to use
    Returns:
        The attributes
    """
    adios = adios2.ADIOS(comm)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        engine=engine,
        io_name="AttributesReader",
    ) as adios_file:
        adios_file.file.BeginStep()
        attributes = {}
        for k in adios_file.io.AvailableAttributes().keys():
            if k.startswith(f"{name}_"):
                a = adios_file.io.InquireAttribute(k)
                attributes[k[len(name) + 1 :]] = a.Data()
        adios_file.file.EndStep()
    return attributes


def write_meshtags(
    filename: typing.Union[Path, str],
    mesh: dolfinx.mesh.Mesh,
    meshtags: dolfinx.mesh.MeshTags,
    engine: str = "BP4",
    meshtag_name: typing.Optional[str] = None,
):
    """
    Write meshtags associated with input mesh to file.

    .. note::
        For this checkpoint to work, the mesh must be written to file
        using :func:`write_mesh` before calling this function.

    Args:
        filename: Path to save meshtags (with file-extension)
        mesh: The mesh associated with the meshtags
        meshtags: The meshtags to write to file
        engine: Adios2 Engine
        meshtag_name: Name of the meshtag. If None, the meshtag name is used.
    """
    tag_entities = meshtags.indices
    dim = meshtags.dim
    num_tag_entities_local = mesh.topology.index_map(dim).size_local
    local_tag_entities = tag_entities[tag_entities < num_tag_entities_local]
    local_values = meshtags.values[: len(local_tag_entities)]

    num_saved_tag_entities = len(local_tag_entities)
    local_start = mesh.comm.exscan(num_saved_tag_entities, op=MPI.SUM)
    local_start = local_start if mesh.comm.rank != 0 else 0
    global_num_tag_entities = mesh.comm.allreduce(num_saved_tag_entities, op=MPI.SUM)
    dof_layout = mesh.geometry.cmap.create_dof_layout()
    num_dofs_per_entity = dof_layout.num_entity_closure_dofs(dim)

    entities_to_geometry = dolfinx.cpp.mesh.entities_to_geometry(
        mesh._cpp_object, dim, tag_entities, False
    )

    indices = mesh.geometry.index_map().local_to_global(entities_to_geometry.reshape(-1))

    name = meshtag_name or meshtags.name

    adios = adios2.ADIOS(mesh.comm)
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Append,
        engine=engine,
        io_name="MeshTagWriter",
    ) as adios_file:
        adios_file.file.BeginStep()

        # Write meshtag topology
        topology_var = adios_file.io.DefineVariable(
            name + "_topology",
            indices,
            shape=[global_num_tag_entities, num_dofs_per_entity],
            start=[local_start, 0],
            count=[num_saved_tag_entities, num_dofs_per_entity],
        )
        adios_file.file.Put(topology_var, indices, adios2.Mode.Sync)

        # Write meshtag topology
        values_var = adios_file.io.DefineVariable(
            name + "_values",
            local_values,
            shape=[global_num_tag_entities],
            start=[local_start],
            count=[num_saved_tag_entities],
        )
        adios_file.file.Put(values_var, local_values, adios2.Mode.Sync)

        # Write meshtag dim
        adios_file.io.DefineAttribute(name + "_dim", np.array([meshtags.dim], dtype=np.uint8))

        adios_file.file.PerformPuts()
        adios_file.file.EndStep()


def read_meshtags(
    filename: typing.Union[Path, str],
    mesh: dolfinx.mesh.Mesh,
    meshtag_name: str,
    engine: str = "BP4",
) -> dolfinx.mesh.MeshTags:
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
    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        engine=engine,
        io_name="MeshTagsReader",
    ) as adios_file:
        # Get mesh cell type
        dim_attr_name = f"{meshtag_name}_dim"
        step = 0
        for i in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if dim_attr_name in adios_file.io.AvailableAttributes().keys():
                step = i
                break
            adios_file.file.EndStep()
        if dim_attr_name not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"{dim_attr_name} not found in {filename}")

        m_dim = adios_file.io.InquireAttribute(dim_attr_name)
        dim = int(m_dim.Data()[0])

        # Get mesh tags entites
        topology_name = f"{meshtag_name}_topology"
        for i in range(step, adios_file.file.Steps()):
            if i > step:
                adios_file.file.BeginStep()
            if topology_name in adios_file.io.AvailableVariables().keys():
                break
            adios_file.file.EndStep()
        if topology_name not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"{topology_name} not found in {filename}")

        topology = adios_file.io.InquireVariable(topology_name)
        top_shape = topology.Shape()
        topology_range = compute_local_range(mesh.comm, top_shape[0])

        topology.SetSelection(
            [
                [topology_range[0], 0],
                [topology_range[1] - topology_range[0], top_shape[1]],
            ]
        )
        mesh_entities = np.empty(
            (topology_range[1] - topology_range[0], top_shape[1]), dtype=np.int64
        )
        adios_file.file.Get(topology, mesh_entities, adios2.Mode.Deferred)

        # Get mesh tags values
        values_name = f"{meshtag_name}_values"
        if values_name not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"{values_name} not found")

        values = adios_file.io.InquireVariable(values_name)
        val_shape = values.Shape()
        assert val_shape[0] == top_shape[0]
        values.SetSelection([[topology_range[0]], [topology_range[1] - topology_range[0]]])
        tag_values = np.empty((topology_range[1] - topology_range[0]), dtype=np.int32)
        adios_file.file.Get(values, tag_values, adios2.Mode.Deferred)

        adios_file.file.PerformGets()
        adios_file.file.EndStep()

    local_entities, local_values = dolfinx.io.distribute_entity_data(
        mesh, int(dim), mesh_entities.astype(np.int32), tag_values
    )
    mesh.topology.create_connectivity(dim, 0)
    mesh.topology.create_connectivity(dim, mesh.topology.dim)

    adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)

    local_values = np.array(local_values, dtype=np.int32)

    mt = dolfinx.mesh.meshtags_from_entities(mesh, int(dim), adj, local_values)
    mt.name = meshtag_name

    return mt


def read_function(
    filename: typing.Union[Path, str],
    u: dolfinx.fem.Function,
    engine: str = "BP4",
    time: float = 0.0,
    legacy: bool = False,
    name: typing.Optional[str] = None,
):
    """
    Read checkpoint from file and fill it into `u`.

    Args:
        filename: Path to checkpoint
        u: Function to fill
        engine: ADIOS engine type used for reading
        time: Time-stamp associated with checkpoint
        legacy: If checkpoint is from prior to time-dependent writing set to True
        name: If not provided, `u.name` is used to search through the input file for the function
    """
    mesh = u.function_space.mesh
    comm = mesh.comm
    adios = adios2.ADIOS(comm)
    if name is None:
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
    inc_cells, inc_perms = send_and_recv_cell_perm(input_cells, cell_perm, owners, mesh.comm)

    # -------------------Step 3-----------------------------------
    # Read dofmap from file and compute dof owners
    if legacy:
        dofmap_path = "Dofmap"
        xdofmap_path = "XDofmap"
    else:
        dofmap_path = f"{name}_dofmap"
        xdofmap_path = f"{name}_XDofmap"
    input_dofmap = read_adjacency_list(
        adios, comm, filename, dofmap_path, xdofmap_path, num_cells_global, engine
    )
    # Compute owner of dofs in dofmap
    num_dofs_global = (
        u.function_space.dofmap.index_map.size_global * u.function_space.dofmap.index_map_bs
    )
    dof_owner = index_owner(comm, input_dofmap.array, num_dofs_global)

    # --------------------Step 4-----------------------------------
    # Read array from file and communicate them to input dofmap process
    if legacy:
        array_path = "Values"
    else:
        array_path = f"{name}_values"
    time_name = f"{name}_time"
    input_array, starting_pos = read_array(
        adios, filename, array_path, engine, comm, time, time_name, legacy=legacy
    )
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
        # Start by sorting data array by cell permutation
        num_dofs_per_cell = input_dofmap.offsets[1:] - input_dofmap.offsets[:-1]
        assert np.allclose(num_dofs_per_cell, num_dofs_per_cell[0])

        # Sort dofmap by input local cell index
        input_perms_sorted = input_perms[input_local_cell_index]
        unrolled_dofmap_position = unroll_insert_position(
            input_local_cell_index, num_dofs_per_cell[0]
        )
        dofmap_sorted_by_input = recv_array[unrolled_dofmap_position]

        # First invert input data to reference element then transform to current mesh
        element.Tt_apply(dofmap_sorted_by_input, input_perms_sorted, bs)
        element.Tt_inv_apply(dofmap_sorted_by_input, inc_perms, bs)
        # Compute invert permutation
        inverted_perm = np.empty_like(unrolled_dofmap_position)
        inverted_perm[unrolled_dofmap_position] = np.arange(
            len(unrolled_dofmap_position), dtype=inverted_perm.dtype
        )
        recv_array = dofmap_sorted_by_input[inverted_perm]

    # ------------------Step 6----------------------------------------
    # For each dof owned by a process, find the local position in the dofmap.
    V = u.function_space
    local_cells, dof_pos = compute_dofmap_pos(V)
    input_cells = V.mesh.topology.original_cell_index[local_cells]
    num_cells_global = V.mesh.topology.index_map(V.mesh.topology.dim).size_global
    owners = index_owner(V.mesh.comm, input_cells, num_cells_global)
    unique_owners, owner_count = np.unique(owners, return_counts=True)
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
        owner_count.astype(np.int32),
        input_cells,
        dof_pos,
        num_cells_global,
        recv_array,
        input_dofmap.offsets,
    )
    u.x.array[: len(owned_values)] = owned_values
    u.x.scatter_forward()


def read_mesh_data(
    filename: typing.Union[Path, str],
    comm: MPI.Intracomm,
    engine: str = "BP4",
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.shared_facet,
    time: float = 0.0,
    legacy: bool = False,
    read_from_partition: bool = False,
) -> tuple[np.ndarray, np.ndarray, ufl.Mesh, typing.Callable]:
    """
    Read an ADIOS2 mesh data for use with DOLFINx.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        engine: ADIOS engine to use for reading (BP4, BP5 or HDF5)
        ghost_mode: Ghost mode to use for mesh. If `read_from_partition`
            is set to `True` this option is ignored.
        time: Time stamp associated with mesh
        legacy: If checkpoint was made prior to time-dependent mesh-writer set to True
        read_from_partition: Read mesh with partition from file
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """
    adios = adios2.ADIOS(comm)

    with ADIOSFile(
        adios=adios,
        filename=filename,
        mode=adios2.Mode.Read,
        engine=engine,
        io_name="MeshReader",
    ) as adios_file:
        # Get time independent mesh variables (mesh topology and cell type info) first
        adios_file.file.BeginStep()
        # Get mesh topology (distributed)
        if "Topology" not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Mesh topology not found at Topology in {filename}")
        topology = adios_file.io.InquireVariable("Topology")
        shape = topology.Shape()
        local_range = compute_local_range(comm, shape[0])
        topology.SetSelection([[local_range[0], 0], [local_range[1] - local_range[0], shape[1]]])
        mesh_topology = np.empty((local_range[1] - local_range[0], shape[1]), dtype=np.int64)
        adios_file.file.Get(topology, mesh_topology, adios2.Mode.Deferred)

        # Check validity of partitioning information
        if read_from_partition:
            if "PartitionProcesses" not in adios_file.io.AvailableAttributes().keys():
                raise KeyError(f"Partitioning information not found in {filename}")
            par_num_procs = adios_file.io.InquireAttribute("PartitionProcesses")
            num_procs = par_num_procs.Data()[0]
            if num_procs != comm.size:
                raise ValueError(f"Number of processes in file ({num_procs})!=({comm.size=})")

        # Get mesh cell type
        if "CellType" not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"Mesh cell type not found at CellType in {filename}")
        celltype = adios_file.io.InquireAttribute("CellType")
        cell_type = celltype.DataString()[0]

        # Get basix info
        if "LagrangeVariant" not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"Mesh LagrangeVariant not found in {filename}")
        lvar = adios_file.io.InquireAttribute("LagrangeVariant").Data()[0]
        if "Degree" not in adios_file.io.AvailableAttributes().keys():
            raise KeyError(f"Mesh degree not found in {filename}")
        degree = adios_file.io.InquireAttribute("Degree").Data()[0]

        if not legacy:
            time_name = "MeshTime"
            for i in range(adios_file.file.Steps()):
                if i > 0:
                    adios_file.file.BeginStep()
                if time_name in adios_file.io.AvailableVariables().keys():
                    arr = adios_file.io.InquireVariable(time_name)
                    time_shape = arr.Shape()
                    arr.SetSelection([[0], [time_shape[0]]])
                    times = np.empty(time_shape[0], dtype=adios_to_numpy_dtype[arr.Type()])
                    adios_file.file.Get(arr, times, adios2.Mode.Sync)
                    if times[0] == time:
                        break
                if i == adios_file.file.Steps() - 1:
                    raise KeyError(
                        f"No data associated with {time_name}={time} found in {filename}"
                    )

                adios_file.file.EndStep()

            if time_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No data associated with {time_name}={time} found in {filename}")

        # Get mesh geometry
        if "Points" not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Mesh coordinates not found at Points in {filename}")
        geometry = adios_file.io.InquireVariable("Points")
        x_shape = geometry.Shape()
        geometry_range = compute_local_range(comm, x_shape[0])
        geometry.SetSelection(
            [
                [geometry_range[0], 0],
                [geometry_range[1] - geometry_range[0], x_shape[1]],
            ]
        )
        mesh_geometry = np.empty(
            (geometry_range[1] - geometry_range[0], x_shape[1]),
            dtype=adios_to_numpy_dtype[geometry.Type()],
        )
        adios_file.file.Get(geometry, mesh_geometry, adios2.Mode.Deferred)
        adios_file.file.PerformGets()
        adios_file.file.EndStep()

    # Create DOLFINx mesh
    element = basix.ufl.element(
        basix.ElementFamily.P,
        cell_type,
        degree,
        basix.LagrangeVariant(int(lvar)),
        shape=(mesh_geometry.shape[1],),
        dtype=mesh_geometry.dtype,
    )
    domain = ufl.Mesh(element)

    if read_from_partition:
        partition_graph = read_adjacency_list(
            adios, comm, filename, "PartitioningData", "PartitioningOffset", shape[0], engine
        )

        def partitioner(comm: MPI.Intracomm, n, m, topo):
            assert len(partition_graph.offsets) - 1 == topo.num_nodes
            return partition_graph
    else:
        partitioner = dolfinx.cpp.mesh.create_cell_partitioner(ghost_mode)

    return mesh_topology, mesh_geometry, domain, partitioner


def read_mesh(
    filename: typing.Union[Path, str],
    comm: MPI.Intracomm,
    engine: str = "BP4",
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.shared_facet,
    time: float = 0.0,
    legacy: bool = False,
    read_from_partition: bool = False,
) -> dolfinx.mesh.Mesh:
    """
    Read an ADIOS2 mesh into DOLFINx.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        engine: ADIOS engine to use for reading (BP4, BP5 or HDF5)
        ghost_mode: Ghost mode to use for mesh. If `read_from_partition`
            is set to `True` this option is ignored.
        time: Time stamp associated with mesh
        legacy: If checkpoint was made prior to time-dependent mesh-writer set to True
        read_from_partition: Read mesh with partition from file
    Returns:
        The distributed mesh
    """
    return dolfinx.mesh.create_mesh(
        comm,
        *read_mesh_data(
            filename,
            comm,
            engine=engine,
            ghost_mode=ghost_mode,
            time=time,
            legacy=legacy,
            read_from_partition=read_from_partition,
        ),
    )


def write_mesh(
    filename: Path,
    mesh: dolfinx.mesh.Mesh,
    engine: str = "BP4",
    mode: adios2.Mode = adios2.Mode.Write,
    time: float = 0.0,
    store_partition_info: bool = False,
):
    """
    Write a mesh to specified ADIOS2 format, see:
    https://adios2.readthedocs.io/en/stable/engines/engines.html
    for possible formats.

    Args:
        filename: Path to save mesh (without file-extension)
        mesh: The mesh to write to file
        engine: Adios2 Engine
        store_partition_info: Store mesh partitioning (including ghosting) to file
    """
    num_xdofs_local = mesh.geometry.index_map().size_local
    num_xdofs_global = mesh.geometry.index_map().size_global
    geometry_range = mesh.geometry.index_map().local_range
    gdim = mesh.geometry.dim

    # Convert local connectivity to globa l connectivity
    g_imap = mesh.geometry.index_map()
    g_dmap = mesh.geometry.dofmap
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    cell_range = mesh.topology.index_map(mesh.topology.dim).local_range
    cmap = mesh.geometry.cmap
    geom_layout = cmap.create_dof_layout()
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh.topology.dim)
    dofs_out = np.zeros((num_cells_local, num_dofs_per_cell), dtype=np.int64)
    assert g_dmap.shape[1] == num_dofs_per_cell
    dofs_out[:, :] = np.asarray(
        g_imap.local_to_global(g_dmap[:num_cells_local, :].reshape(-1))
    ).reshape(dofs_out.shape)

    if store_partition_info:
        partition_processes = mesh.comm.size

        # Get partitioning
        cell_map = mesh.topology.index_map(mesh.topology.dim).index_to_dest_ranks()
        num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
        cell_offsets = cell_map.offsets[: num_cells_local + 1]
        if cell_offsets[-1] == 0:
            cell_array = np.empty(0, dtype=np.int32)
        else:
            cell_array = cell_map.array[: cell_offsets[-1]]

        # Compute adjacency with current process as first entry
        ownership_array = np.full(num_cells_local + cell_offsets[-1], -1, dtype=np.int32)
        ownership_offset = cell_offsets + np.arange(len(cell_offsets), dtype=np.int32)
        ownership_array[ownership_offset[:-1]] = mesh.comm.rank
        insert_position = np.flatnonzero(ownership_array == -1)
        ownership_array[insert_position] = cell_array

        partition_map = dolfinx.common.IndexMap(mesh.comm, ownership_array.size)
        ownership_offset += partition_map.local_range[0]
        partition_range = partition_map.local_range
        partition_global = partition_map.size_global
    else:
        partition_processes = None
        ownership_array = None
        ownership_offset = None
        partition_range = None
        partition_global = None

    mesh_data = MeshData(
        local_geometry=mesh.geometry.x[:num_xdofs_local, :gdim].copy(),
        local_geometry_pos=geometry_range,
        num_nodes_global=num_xdofs_global,
        local_topology=dofs_out,
        local_topology_pos=cell_range,
        num_cells_global=num_cells_global,
        cell_type=mesh.topology.cell_name(),
        degree=mesh.geometry.cmap.degree,
        lagrange_variant=mesh.geometry.cmap.variant,
        store_partition=store_partition_info,
        partition_processes=partition_processes,
        ownership_array=ownership_array,
        ownership_offset=ownership_offset,
        partition_range=partition_range,
        partition_global=partition_global,
    )

    _internal_mesh_writer(
        filename,
        mesh.comm,
        mesh_data,
        engine,
        mode=mode,
        time=time,
        io_name="MeshWriter",
    )


def write_function(
    filename: typing.Union[Path, str],
    u: dolfinx.fem.Function,
    engine: str = "BP4",
    mode: adios2.Mode = adios2.Mode.Append,
    time: float = 0.0,
    name: typing.Optional[str] = None,
):
    """
    Write function checkpoint to file.

    Args:
        u: Function to write to file
        filename: Path to write to
        engine: ADIOS2 engine
        mode: Write or append.
        time: Time-stamp for simulation
        name: Name of function to write. If None, the name of the function is used.
    """
    dofmap = u.function_space.dofmap
    values = u.x.array
    mesh = u.function_space.mesh
    comm = mesh.comm
    mesh.topology.create_entity_permutations()
    cell_perm = mesh.topology.get_cell_permutation_info()
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    local_cell_range = mesh.topology.index_map(mesh.topology.dim).local_range
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global

    # Convert local dofmap into global_dofmap
    dmap = dofmap.list
    num_dofs_per_cell = dmap.shape[1]
    dofmap_bs = dofmap.bs
    num_dofs_local_dmap = num_cells_local * num_dofs_per_cell * dofmap_bs
    index_map_bs = dofmap.index_map_bs

    # Unroll dofmap for block size
    unrolled_dofmap = unroll_dofmap(dofmap.list[:num_cells_local, :], dofmap_bs)
    dmap_loc = (unrolled_dofmap // index_map_bs).reshape(-1)
    dmap_rem = (unrolled_dofmap % index_map_bs).reshape(-1)

    # Convert imap index to global index
    imap_global = dofmap.index_map.local_to_global(dmap_loc)
    dofmap_global = imap_global * index_map_bs + dmap_rem
    dofmap_imap = dolfinx.common.IndexMap(mesh.comm, num_dofs_local_dmap)

    # Compute dofmap offsets
    local_dofmap_offsets = np.arange(num_cells_local + 1, dtype=np.int64)
    local_dofmap_offsets[:] *= num_dofs_per_cell * dofmap_bs
    local_dofmap_offsets += dofmap_imap.local_range[0]

    num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
    local_dof_range = np.asarray(dofmap.index_map.local_range) * dofmap.index_map_bs
    num_dofs_local = local_dof_range[1] - local_dof_range[0]

    # Create internal data structure for function data to write to file
    function_data = FunctionData(
        cell_permutations=cell_perm[:num_cells_local].copy(),
        local_cell_range=local_cell_range,
        num_cells_global=num_cells_global,
        dofmap_array=dofmap_global,
        dofmap_offsets=local_dofmap_offsets,
        dofmap_range=dofmap_imap.local_range,
        global_dofs_in_dofmap=dofmap_imap.size_global,
        values=values[:num_dofs_local].copy(),
        dof_range=local_dof_range,
        num_dofs_global=num_dofs_global,
        name=name or u.name,
    )
    # Write to file
    fname = Path(filename)
    _internal_function_writer(fname, comm, function_data, engine, mode, time, "FunctionWriter")
