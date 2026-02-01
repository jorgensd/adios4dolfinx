# Copyright (C) 2023-2026 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from packaging.version import Version

from .backends import FileMode, ReadMode, get_backend
from .comm_helpers import (
    send_and_recv_cell_perm,
    send_dofmap_and_recv_values,
    send_dofs_and_recv_values,
)
from .structures import FunctionData, MeshTagsData
from .utils import (
    check_file_exists,
    compute_dofmap_pos,
    compute_local_range,
    index_owner,
    unroll_dofmap,
    unroll_insert_position,
)
from .writers import prepare_meshdata_for_storage
from .writers import write_function as _internal_function_writer
from .writers import write_mesh as _internal_mesh_writer

__all__ = [
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
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, typing.Any] | None = None,
    backend: str = "adios2",
):
    """Write attributes to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attributes
        attributes: Dictionary of attributes to write to file
        backend_args: Arguments for backend, for instance file type.
        backend: What backend to use for writing.
    """
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    backend_cls.write_attributes(filename, comm, name, attributes, backend_args)


def read_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, typing.Any] | None = None,
    backend: str = "adios2",
) -> dict[str, typing.Any]:
    """Read attributes from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attributes
        backend_args: Arguments for backend, for instance file type.
        backend: What backend to use for writing.
    Returns:
        The attributes
    """
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    return backend_cls.read_attributes(filename, comm, name, backend_args)


def read_timestamps(
    filename: Path | str,
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, typing.Any] | None = None,
    backend: str = "adios2",
) -> npt.NDArray[np.float64]:
    """
    Read time-stamps from a checkpoint file.

    Args:
        comm: MPI communicator
        filename: Path to file
        function_name: Name of the function to read time-stamps for
        backend_args: Arguments for backend, for instance file type.
        backend: What backend to use for writing.
    Returns:
        The time-stamps
    """
    check_file_exists(filename)
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    return backend_cls.read_timestamps(filename, comm, function_name, backend_args)


def write_meshtags(
    filename: Path | str,
    mesh: dolfinx.mesh.Mesh,
    meshtags: dolfinx.mesh.MeshTags,
    meshtag_name: typing.Optional[str] = None,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
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
        meshtag_name: Name of the meshtag. If None, the meshtag name is used.
        backend_args: Option to IO backend.
        backend: IO backend
    """

    # Extract data from meshtags (convert to global geometry node indices for each entity)
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
    if hasattr(dof_layout, "num_entity_closure_dofs"):
        num_dofs_per_entity = dof_layout.num_entity_closure_dofs(dim)
    else:
        num_dofs_per_entity = len(dof_layout.entity_closure_dofs(dim, 0))

    entities_to_geometry = dolfinx.cpp.mesh.entities_to_geometry(
        mesh._cpp_object, dim, local_tag_entities, False
    )

    indices = (
        mesh.geometry.index_map()
        .local_to_global(entities_to_geometry.reshape(-1))
        .reshape(entities_to_geometry.shape)
    )
    name = meshtag_name or meshtags.name

    tag_ct = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, dim, 0).name
    tag_data = MeshTagsData(
        values=local_values,
        num_entities_global=global_num_tag_entities,
        num_dofs_per_entity=num_dofs_per_entity,
        indices=indices,
        name=name,
        local_start=local_start,
        dim=meshtags.dim,
        cell_type=tag_ct,
    )

    # Get backend and default arguments
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    return backend_cls.write_meshtags(filename, mesh.comm, tag_data, backend_args=backend_args)


def read_meshtags(
    filename: Path | str,
    mesh: dolfinx.mesh.Mesh,
    meshtag_name: str,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
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
    check_file_exists(filename)
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    data = backend_cls.read_meshtags_data(filename, mesh.comm, meshtag_name, backend_args)

    local_entities, local_values = dolfinx.io.distribute_entity_data(
        mesh, int(data.dim), data.indices, data.values
    )
    mesh.topology.create_connectivity(data.dim, 0)
    mesh.topology.create_connectivity(data.dim, mesh.topology.dim)

    adj = dolfinx.graph.adjacencylist(local_entities)

    local_values = np.array(local_values, dtype=np.int32)

    mt = dolfinx.mesh.meshtags_from_entities(mesh, int(data.dim), adj, local_values)
    mt.name = meshtag_name
    return mt


def read_function(
    filename: Path | str,
    u: dolfinx.fem.Function,
    time: float = 0.0,
    name: str | None = None,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
):
    """
    Read checkpoint from file and fill it into `u`.

    Args:
        filename: Path to checkpoint
        u: Function to fill
        time: Time-stamp associated with checkpoint
        name: If not provided, `u.name` is used to search through the input file for the function
    """
    check_file_exists(filename)

    mesh = u.function_space.mesh
    comm = mesh.comm
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
    backend_cls = get_backend(backend)
    if backend_cls.read_mode == ReadMode.serial:
        owners = np.zeros(input_cells, dtype=np.int32)
    elif backend_cls.read_mode == ReadMode.parallel:
        num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
        owners = index_owner(mesh.comm, input_cells, num_cells_global)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")
    # -------------------Step 2------------------------------------
    # Send and receive global cell index and cell perm
    inc_cells, inc_perms = send_and_recv_cell_perm(input_cells, cell_perm, owners, mesh.comm)

    # -------------------Step 3-----------------------------------
    # Read dofmap from file and compute dof owners
    check_file_exists(filename)
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)

    input_dofmap = backend_cls.read_dofmap(filename, comm, name, backend_args)

    # Compute owner of dofs in dofmap
    if backend_cls.read_mode == ReadMode.serial:
        dof_owner = np.zeros(len(input_dofmap.array), dtype=np.int32)
    elif backend_cls.read_mode == ReadMode.parallel:
        num_dofs_global = (
            u.function_space.dofmap.index_map.size_global * u.function_space.dofmap.index_map_bs
        )
        dof_owner = index_owner(comm, input_dofmap.array.astype(np.int64), num_dofs_global)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

    # --------------------Step 4-----------------------------------
    # Read array from file and communicate them to input dofmap process
    input_array, starting_pos = backend_cls.read_dofs(filename, comm, name, time, backend_args)

    recv_array = send_dofs_and_recv_values(
        input_dofmap.array.astype(np.int64), dof_owner, comm, input_array, starting_pos
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
        input_perms = backend_cls.read_cell_perms(comm, filename, backend_args)

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

    if backend_cls.read_mode == ReadMode.serial:
        owners = np.zeros(len(input_cells), dtype=np.int32)
    elif backend_cls.read_mode == ReadMode.parallel:
        owners = index_owner(V.mesh.comm, input_cells, num_cells_global)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

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


def read_mesh(
    filename: Path | str,
    comm: MPI.Intracomm,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.shared_facet,
    time: float = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
) -> dolfinx.mesh.Mesh:
    """
    Read an ADIOS2 mesh into DOLFINx.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        ghost_mode: Ghost mode to use for mesh. If `read_from_partition`
            is set to `True` this option is ignored.
        time: Time stamp associated with mesh
        read_from_partition: Read mesh with partition from file
        backend_args: List of arguments to reader backend
    Returns:
        The distributed mesh
    """
    # Read in data in a distributed fashin
    check_file_exists(filename)
    backend_cls = get_backend(backend)
    backend_args = backend_cls.get_default_backend_args(backend_args)
    dist_in_data = backend_cls.read_mesh_data(
        filename,
        comm,
        time=time,
        read_from_partition=read_from_partition,
        backend_args=backend_args,
    )

    # Create DOLFINx mesh
    element = basix.ufl.element(
        basix.ElementFamily.P,
        dist_in_data.cell_type,
        dist_in_data.degree,
        basix.LagrangeVariant(int(dist_in_data.lvar)),
        shape=(dist_in_data.x.shape[1],),
        dtype=dist_in_data.x.dtype,
    )
    domain = ufl.Mesh(element)
    if (partition_graph := dist_in_data.partition_graph) is not None:

        def partitioner(comm: MPI.Intracomm, n, m, topo):
            assert len(topo[0]) % (len(partition_graph.offsets) - 1) == 0
            if Version(dolfinx.__version__) > Version("0.9.0"):
                return partition_graph._cpp_object
            else:
                return partition_graph
    else:
        partitioner = dolfinx.cpp.mesh.create_cell_partitioner(ghost_mode)
    return dolfinx.mesh.create_mesh(
        comm, cells=dist_in_data.cells, x=dist_in_data.x, e=domain, partitioner=partitioner
    )


def write_mesh(
    filename: Path,
    mesh: dolfinx.mesh.Mesh,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
    store_partition_info: bool = False,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
):
    """
    Write a mesh to file.

    Args:
        filename: Path to save mesh (without file-extension)
        mesh: The mesh to write to file

        store_partition_info: Store mesh partitioning (including ghosting) to file
    """
    mesh_data = prepare_meshdata_for_storage(mesh=mesh, store_partition_info=store_partition_info)

    _internal_mesh_writer(
        filename,
        mesh.comm,
        mesh_data=mesh_data,
        time=time,
        backend_args=backend_args,
        backend=backend,
        mode=mode,
    )


def write_function(
    filename: Path | str,
    u: dolfinx.fem.Function,
    time: float = 0.0,
    mode: FileMode = FileMode.append,
    name: str | None = None,
    backend_args: dict[str, Any] | None = None,
    backend: str = "adios2",
):
    """
    Write function checkpoint to file.

    Args:
        u: Function to write to file
        time: Time-stamp for simulation
        filename: Path to write to
        mode: Write or append.
        name: Name of function to write. If None, the name of the function is used.
        backend_args: Arguments to the IO backend.
        backend: The backend to use
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
    _internal_function_writer(
        fname, comm, function_data, time, backend_args=backend_args, backend=backend, mode=mode
    )
