# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import pathlib
import typing
from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

from .backends import ReadMode, get_backend
from .comm_helpers import send_dofs_and_recv_values
from .utils import (
    check_file_exists,
    compute_dofmap_pos,
    compute_insert_position,
    compute_local_range,
    index_owner,
)

__all__ = ["read_mesh_from_legacy_h5", "read_function_from_legacy_h5", "read_point_data"]


def map_dofmap(dofmap: dolfinx.graph.AdjacencyList, bs: int) -> npt.NDArray[np.int64]:
    """
    Map xxxyyyzzz to xyzxyz
    """

    in_dofmap = dofmap.array
    in_offsets = dofmap.offsets

    mapped_dofmap = np.empty_like(in_dofmap)
    for i in range(len(in_offsets) - 1):
        pos_begin, pos_end = (
            in_offsets[i] - in_offsets[0],
            in_offsets[i + 1] - in_offsets[0],
        )
        dofs_i = in_dofmap[pos_begin:pos_end]
        assert (pos_end - pos_begin) % bs == 0
        num_dofs_local = int((pos_end - pos_begin) // bs)
        for k in range(bs):
            for j in range(num_dofs_local):
                mapped_dofmap[int(pos_begin + j * bs + k)] = dofs_i[int(num_dofs_local * k + j)]
    return mapped_dofmap.astype(np.int64)


def send_cells_and_receive_dofmap_index(
    filename: pathlib.Path,
    comm: MPI.Intracomm,
    source_ranks: npt.NDArray[np.int32],
    dest_ranks: npt.NDArray[np.int32],
    dest_size: npt.NDArray[np.int32],
    output_owners: npt.NDArray[np.int32],
    input_cells: npt.NDArray[np.int64],
    dofmap_pos: npt.NDArray[np.int32],
    num_cells_global: np.int64,
    dofmap_path: str,
    xdofmap_path: str,
    bs: int,
    backend: str,
) -> npt.NDArray[np.int64]:
    """
    Given a set of positions in input dofmap, give the global input index of this dofmap entry
    in input file.
    """
    check_file_exists(filename)

    recv_size = np.zeros(len(source_ranks), dtype=np.int32)
    mesh_to_data_comm = comm.Create_dist_graph_adjacent(
        source_ranks.tolist(), dest_ranks.tolist(), reorder=False
    )
    # Send sizes to create data structures for receiving from NeighAlltoAllv
    mesh_to_data_comm.Neighbor_alltoall(dest_size, recv_size)

    # Sort output for sending and fill send data
    out_cells = np.zeros(len(output_owners), dtype=np.int64)
    out_pos = np.zeros(len(output_owners), dtype=np.int32)
    proc_to_dof = np.zeros_like(input_cells, dtype=np.int32)
    insertion_array = compute_insert_position(output_owners, dest_ranks, dest_size)
    out_cells[insertion_array] = input_cells
    out_pos[insertion_array] = dofmap_pos
    proc_to_dof[insertion_array] = np.arange(len(input_cells), dtype=np.int32)
    del insertion_array

    # Prepare data-structures for receiving
    total_incoming = sum(recv_size)
    inc_cells = np.zeros(total_incoming, dtype=np.int64)
    inc_pos = np.zeros(total_incoming, dtype=np.intc)

    # Send data
    s_msg = [out_cells, dest_size, MPI.INT64_T]
    r_msg = [inc_cells, recv_size, MPI.INT64_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)

    s_msg = [out_pos, dest_size, MPI.INT32_T]
    r_msg = [inc_pos, recv_size, MPI.INT32_T]
    mesh_to_data_comm.Neighbor_alltoallv(s_msg, r_msg)
    mesh_to_data_comm.Free()

    backend_cls = get_backend(backend)
    # Read dofmap from file
    backend_args = {"dofmap": dofmap_path, "offsets": xdofmap_path}
    if backend == "adios2":
        backend_args.update({"engine": "HDF5"})
    input_dofs = backend_cls.read_dofmap(filename, comm, name="", backend_args=backend_args)
    # Map to xyz
    mapped_dofmap = map_dofmap(input_dofs, bs).astype(np.int64)

    # Extract dofmap data
    local_cell_range = compute_local_range(comm, num_cells_global)
    input_cell_positions = inc_cells - local_cell_range[0]
    in_offsets = input_dofs.offsets
    read_pos = (in_offsets[input_cell_positions] + inc_pos - in_offsets[0]).astype(np.int32)
    input_dofs = mapped_dofmap[read_pos]
    del input_cell_positions, read_pos

    # Send input dofs back to owning process
    data_to_mesh_comm = comm.Create_dist_graph_adjacent(
        dest_ranks.tolist(), source_ranks.tolist(), reorder=False
    )

    incoming_global_dofs = np.zeros(sum(dest_size), dtype=np.int64)
    s_msg = [input_dofs, recv_size, MPI.INT64_T]
    r_msg = [incoming_global_dofs, dest_size, MPI.INT64_T]
    data_to_mesh_comm.Neighbor_alltoallv(s_msg, r_msg)

    # Sort incoming global dofs as they were inputted
    sorted_global_dofs = np.zeros_like(incoming_global_dofs, dtype=np.int64)
    assert len(incoming_global_dofs) == len(input_cells)
    sorted_global_dofs[proc_to_dof] = incoming_global_dofs
    data_to_mesh_comm.Free()
    return sorted_global_dofs


def read_mesh_from_legacy_h5(
    filename: pathlib.Path,
    comm: MPI.Intracomm,
    group: str,
    cell_type: str = "tetrahedron",
    backend: str = "adios2",
) -> dolfinx.mesh.Mesh:
    """
    Read mesh from `h5`-file generated by legacy DOLFIN `HDF5File.write` or `XDMF.write_checkpoint`.

    Args:
        comm: MPI communicator to distribute mesh over
        filename: Path to `h5` or `xdmf` file
        group: Name of mesh in `h5`-file
        cell_type: What type of cell type, by default tetrahedron.
    """
    # Make sure we use the HDF5File and check that the file is present
    check_file_exists(filename)

    backend_cls = get_backend(backend)
    mesh_topology, mesh_geometry, ct = backend_cls.read_legacy_mesh(filename, comm, group)
    if ct is not None:
        cell_type = ct
    # Create DOLFINx mesh
    element = basix.ufl.element(
        basix.ElementFamily.P,
        cell_type,
        1,
        basix.LagrangeVariant.equispaced,
        shape=(mesh_geometry.shape[1],),
    )
    domain = ufl.Mesh(element)

    return dolfinx.mesh.create_mesh(
        comm=MPI.COMM_WORLD, cells=mesh_topology, x=mesh_geometry, e=domain
    )


def read_function_from_legacy_h5(
    filename: pathlib.Path,
    comm: MPI.Intracomm,
    u: dolfinx.fem.Function,
    group: str = "mesh",
    step: typing.Optional[int] = None,
    backend: str = "adios2",
):
    """
    Read function from a `h5`-file generated by legacy DOLFIN `HDF5File.write`
    or `XDMF.write_checkpoint`.

    Args:
        comm : MPI communicator to distribute mesh over
        filename : Path to `h5` or `xdmf` file
        u : The function used to stored the read values
        group : Group within the `h5` file where the function is stored, by default "mesh"
        step : The time step used when saving the checkpoint. If not provided it will assume that
            the function is saved as a regular function (i.e with `HDF5File.write`)
        backend: The IO backend
    """

    # Make sure we use the HDF5File and check that the file is present
    filename = pathlib.Path(filename).with_suffix(".h5")
    if not filename.is_file():
        raise FileNotFoundError(f"File {filename} does not exist")

    V = u.function_space
    mesh = u.function_space.mesh
    if u.function_space.element.needs_dof_transformations:
        raise RuntimeError(
            "Function-spaces requiring dof permutations are not compatible with legacy data"
        )
    # ----------------------Step 1---------------------------------
    # Compute index of input cells, and position in input dofmap
    local_cells, dof_pos = compute_dofmap_pos(u.function_space)
    input_cells = mesh.topology.original_cell_index[local_cells]

    # Compute mesh->input communicator
    # 1.1 Compute mesh->input communicator
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    backend_cls = get_backend(backend)
    owners: npt.NDArray[np.int32]
    if backend_cls.read_mode == ReadMode.serial:
        owners = np.zeros(len(input_cells), dtype=np.int32)
    elif backend_cls.read_mode == ReadMode.parallel:
        owners = index_owner(V.mesh.comm, input_cells, num_cells_global)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

    unique_owners, owner_count = np.unique(owners, return_counts=True)
    # FIXME: In C++ use NBX to find neighbourhood
    _tmp_comm = mesh.comm.Create_dist_graph(
        [mesh.comm.rank], [len(unique_owners)], unique_owners, reorder=False
    )
    source, dest, _ = _tmp_comm.Get_dist_neighbors()
    _tmp_comm.Free()
    # Strip out any /
    group = group.strip("/")
    if step is not None:
        group = f"{group}/{group}_{step}"
        vector_group = "vector"
    else:
        vector_group = "vector_0"

    # ----------------------Step 2--------------------------------
    # Get global dofmap indices from input process
    bs = V.dofmap.bs
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    dofmap_indices = send_cells_and_receive_dofmap_index(
        filename,
        comm,
        np.asarray(source, dtype=np.int32),
        np.asarray(dest, dtype=np.int32),
        owner_count.astype(np.int32),
        owners,
        input_cells,
        dof_pos,
        num_cells_global,
        f"/{group}/cell_dofs",
        f"/{group}/x_cell_dofs",
        bs,
        backend=backend,
    )

    # ----------------------Step 3---------------------------------
    dof_owner: npt.NDArray[np.int32]
    if backend_cls.read_mode == ReadMode.serial:
        dof_owner = np.zeros(len(dofmap_indices), dtype=np.int32)
    elif backend_cls.read_mode == ReadMode.parallel:
        # Compute owner of global dof on distributed input data
        num_dof_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
        dof_owner = index_owner(comm=mesh.comm, indices=dofmap_indices, N=num_dof_global)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

    # Create MPI neigh comm to owner.
    # NOTE: USE NBX in C++

    # Read input data
    local_array, starting_pos = backend_cls.read_hdf5_array(
        comm, filename, f"/{group}/{vector_group}", backend_args=None
    )

    # Send global dof indices to correct input process, and receive value of given dof
    local_values = send_dofs_and_recv_values(
        dofmap_indices, dof_owner, comm, local_array, starting_pos
    )

    # ----------------------Step 4---------------------------------
    # Populate local part of array and scatter forward
    u.x.array[: len(local_values)] = local_values
    u.x.scatter_forward()


def create_geometry_function_space(mesh: dolfinx.mesh.Mesh, N: int) -> dolfinx.fem.FunctionSpace:
    """Reconstruct a vector space with the N components using the geometry dofmap to ensure
    a 1-1 mapping between mesh nodes and DOFs."""
    geom_imap = mesh.geometry.index_map()
    geom_dofmap = mesh.geometry.dofmap
    ufl_domain = mesh.ufl_domain()
    assert ufl_domain is not None
    sub_el = ufl_domain.ufl_coordinate_element().sub_elements[0]
    adj_list = dolfinx.cpp.graph.AdjacencyList_int32(geom_dofmap)

    value_shape: tuple[int, ...]
    if N == 1:
        ufl_el = sub_el
        value_shape = ()
    else:
        ufl_el = basix.ufl.blocked_element(sub_el, shape=(N,))
        value_shape = (N,)

    if ufl_el.dtype == np.float32:
        _fe_constructor = dolfinx.cpp.fem.FiniteElement_float32
        _fem_constructor = dolfinx.cpp.fem.FunctionSpace_float32
    elif ufl_el.dtype == np.float64:
        _fe_constructor = dolfinx.cpp.fem.FiniteElement_float64
        _fem_constructor = dolfinx.cpp.fem.FunctionSpace_float64
    else:
        raise RuntimeError(f"Unsupported type {ufl_el.dtype}")
    try:
        cpp_el = _fe_constructor(ufl_el.basix_element._e, block_shape=value_shape, symmetric=False)
    except TypeError:
        cpp_el = _fe_constructor(ufl_el.basix_element._e, block_size=N, symmetric=False)
    dof_layout = dolfinx.cpp.fem.create_element_dof_layout(cpp_el, [])
    cpp_dofmap = dolfinx.cpp.fem.DofMap(dof_layout, geom_imap, N, adj_list, N)

    # Create function space
    try:
        cpp_space = _fem_constructor(mesh._cpp_object, cpp_el, cpp_dofmap)
    except TypeError:
        cpp_space = _fem_constructor(mesh._cpp_object, cpp_el, cpp_dofmap, value_shape=value_shape)

    return dolfinx.fem.FunctionSpace(mesh, ufl_el, cpp_space)


def read_point_data(
    filename: Path | str,
    name: str,
    mesh: dolfinx.mesh.Mesh,
    time: float | None = None,
    backend_args: dict[str, Any] | None = None,
    backend: str = "xdmf",
) -> dolfinx.fem.Function:
    """Read data from the nodes of a mesh.

    Note:
        Backend has to implement {py:class}`adios4dolfinx.backends.read_cell_data`.

    Args:
        filename: Path to file
        name: Name of point data
        mesh: The corresponding :py:class:`dolfinx.mesh.Mesh`.
        time: Time-step to read from.

    Returns:
        A function in the space equivalent to the mesh
        coordinate element (up to shape).
    """

    backend_cls = get_backend(backend)
    dataset, local_range_start = backend_cls.read_point_data(
        filename=filename, name=name, comm=mesh.comm, time=time, backend_args=backend_args
    )

    num_components = dataset.shape[1]

    # Create appropriate function space (based on coordinate map)
    V = create_geometry_function_space(mesh, num_components)
    uh = dolfinx.fem.Function(V, name=name, dtype=dataset.dtype)
    # Assume that mesh is first order for now
    x_dofmap = mesh.geometry.dofmap
    igi = np.array(mesh.geometry.input_global_indices, dtype=np.int64)

    # This is dependent on how the data is read in. If distributed equally this is correct
    global_geom_input = igi[x_dofmap]

    if backend_cls.read_mode == ReadMode.parallel:
        num_nodes_global = mesh.geometry.index_map().size_global
        global_geom_owner = index_owner(mesh.comm, global_geom_input.reshape(-1), num_nodes_global)
    elif backend_cls.read_mode == ReadMode.serial:
        # This is correct if everything is read in on rank 0
        global_geom_owner = np.zeros(len(global_geom_input.flatten()), dtype=np.int32)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

    for i in range(num_components):
        arr_i = send_dofs_and_recv_values(
            global_geom_input.reshape(-1),
            global_geom_owner,
            mesh.comm,
            dataset[:, i],
            local_range_start,
        )
        dof_pos = x_dofmap.reshape(-1) * num_components + i
        uh.x.array[dof_pos] = arr_i
    uh.x.scatter_forward()
    return uh


def read_cell_data(
    filename: Path | str,
    name: str,
    mesh: dolfinx.mesh.Mesh,
    time: float | None = None,
    backend_args: dict[str, Any] | None = None,
    backend: str = "xdmf",
) -> dolfinx.fem.Function:
    """Read data from the nodes of a mesh.

    Note:
        Backend has to implement {py:class}`adios4dolfinx.backends.read_cell_data`.

    Args:
        filename: Path to file
        name: Name of point data
        mesh: The corresponding :py:class:`dolfinx.mesh.Mesh`.
        time: Time-step to read from.

    Returns:
        A function in a DG-0 space on the mesh. The cells not found in input is set to zero.
    """

    backend_cls = get_backend(backend)

    topology, dofs = backend_cls.read_cell_data(
        filename=filename, name=name, comm=mesh.comm, time=time, backend_args=backend_args
    )
    num_components = dofs.shape[1]
    shape: tuple[int, ...]
    if num_components == 1:
        shape = ()
    else:
        shape = (num_components,)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, shape))
    u = dolfinx.fem.Function(V, dtype=dofs.dtype)
    data_array = u.x.array.reshape(-1, num_components)
    for i in range(num_components):
        local_entities, local_values = dolfinx.io.distribute_entity_data(
            mesh, mesh.topology.dim, topology, dofs[:, i].copy()
        )
        adj = dolfinx.graph.adjacencylist(local_entities)
        order = np.arange(len(local_values), dtype=np.int32)
        mt = dolfinx.mesh.meshtags_from_entities(mesh, mesh.topology.dim, adj, order)
        data_array[mt.indices, i] = local_values[mt.values]
    u.x.scatter_forward()
    return u
