from __future__ import annotations

import itertools
import typing
from collections import ChainMap

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import pytest

import adios4dolfinx

root = 0
dtypes: list["str"] = ["float64", "float32"]  # Mesh geometry dtypes
write_comm: list[MPI.Intracomm] = [
    MPI.COMM_SELF,
    MPI.COMM_WORLD,
]  # Communicators for creating mesh
read_modes: list[dolfinx.mesh.GhostMode] = [
    dolfinx.mesh.GhostMode.none,
    dolfinx.mesh.GhostMode.shared_facet,
]
# Cell types of different dimensions
two_dimensional_cell_types: list[dolfinx.mesh.CellType] = [
    dolfinx.mesh.CellType.triangle,
    dolfinx.mesh.CellType.quadrilateral,
]
three_dimensional_cell_types: list[dolfinx.mesh.CellType] = [
    dolfinx.mesh.CellType.tetrahedron,
    dolfinx.mesh.CellType.hexahedron,
]

one_dim_combinations = itertools.product(dtypes, write_comm)
two_dim_combinations = itertools.product(dtypes, two_dimensional_cell_types, write_comm)
three_dim_combinations = itertools.product(dtypes, three_dimensional_cell_types, write_comm)


@pytest.fixture(params=one_dim_combinations, scope="module")
def mesh_1D(request):
    dtype, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_interval(write_comm, 8, dtype=np.dtype(dtype))
    return mesh


@pytest.fixture(params=two_dim_combinations, scope="module")
def mesh_2D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_square(
        write_comm, 10, 7, cell_type=cell_type, dtype=np.dtype(dtype)
    )
    return mesh


@pytest.fixture(params=three_dim_combinations, scope="module")
def mesh_3D(request):
    dtype, cell_type, write_comm = request.param
    mesh = dolfinx.mesh.create_unit_cube(
        write_comm, 5, 7, 3, cell_type=cell_type, dtype=np.dtype(dtype)
    )
    return mesh


def generate_reference_map(
    mesh: dolfinx.mesh.Mesh,
    meshtag: dolfinx.mesh.MeshTags,
    comm: MPI.Intracomm,
    root: int,
) -> typing.Optional[dict[str, tuple[int, npt.NDArray]]]:
    """
    Helper function to generate map from meshtag value to its corresponding index and midpoint.

    Args:
        mesh: The mesh
        meshtag: The associated meshtag
        comm: MPI communicator to gather the map from all processes with
        root (int): Rank to store data on
    Returns:
        Root rank returns the map, all other ranks return None
    """
    midpoints = dolfinx.mesh.compute_midpoints(mesh, meshtag.dim, meshtag.indices)
    e_map = mesh.topology.index_map(meshtag.dim)
    value_to_midpoint = {}
    for index, value in zip(meshtag.indices, meshtag.values):
        value_to_midpoint[value] = (
            int(e_map.local_range[0] + index),
            midpoints[index],
        )
    global_map = comm.gather(value_to_midpoint, root=root)
    if comm.rank == root:
        return dict(ChainMap(*global_map))  # type: ignore
    return None


@pytest.mark.parametrize("read_mode", read_modes)
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_checkpointing_meshtags_1D(mesh_1D, read_comm, read_mode, tmp_path):
    mesh = mesh_1D

    # Write unique mesh file for each combination of MPI communicator and dtype
    hash = f"{mesh.comm.size}_{mesh.geometry.x.dtype}"
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = fname / f"meshtags_1D_{hash}.bp"

    # If mesh communicator is more than a self communicator or serial write on all processes.
    # If serial or self communicator, only write on root rank
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh(filename, mesh, engine="BP4")
    else:
        if MPI.COMM_WORLD.rank == root:
            adios4dolfinx.write_mesh(filename, mesh, engine="BP4")

    # Create meshtags labeling each entity (of each co-dimension) with a
    # unique number (their initial global index).
    org_maps = []
    for dim in range(mesh.topology.dim + 1):
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        e_map = mesh.topology.index_map(dim)
        num_entities_local = e_map.size_local
        entities = np.arange(num_entities_local, dtype=np.int32)
        ft = dolfinx.mesh.meshtags(mesh, dim, entities, e_map.local_range[0] + entities)
        ft.name = f"entity_{dim}"

        # If parallel write on all processes, else write on root rank
        if mesh.comm.size != 1:
            adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")
            # Create map from mesh tag value to its corresponding index and midpoint
            org_map = generate_reference_map(mesh, ft, mesh.comm, root)
            org_maps.append(org_map)
        else:
            if MPI.COMM_WORLD.rank == root:
                adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")
                # Create map from mesh tag value to its corresponding index and midpoint
                org_map = generate_reference_map(mesh, ft, MPI.COMM_SELF, root)
                org_maps.append(org_map)
        del ft
    del mesh

    MPI.COMM_WORLD.Barrier()
    # Read mesh on testing communicator
    new_mesh = adios4dolfinx.read_mesh(filename, read_comm, engine="BP4", ghost_mode=read_mode)
    for dim in range(new_mesh.topology.dim + 1):
        # Read meshtags on all processes if testing communicator has multiple ranks
        # else read on root 0
        if read_comm.size != 1:
            new_ft = adios4dolfinx.read_meshtags(
                filename, new_mesh, meshtag_name=f"entity_{dim}", engine="BP4"
            )
            # Generate meshtags map from mesh tag value to its corresponding index and midpoint
            # and gather on root process
            read_map = generate_reference_map(new_mesh, new_ft, new_mesh.comm, root)
        else:
            if MPI.COMM_WORLD.rank == root:
                new_ft = adios4dolfinx.read_meshtags(
                    filename, new_mesh, meshtag_name=f"entity_{dim}", engine="BP4"
                )
                read_map = generate_reference_map(new_mesh, new_ft, read_comm, root)

        # On root process, check that midpoints are the same for each value in the meshtag
        if MPI.COMM_WORLD.rank == root:
            org_map = org_maps[dim]
            assert len(org_map) == len(read_map)
            for value, (_, midpoint) in org_map.items():
                _, read_midpoint = read_map[value]
                np.testing.assert_allclose(read_midpoint, midpoint)


@pytest.mark.parametrize("read_mode", read_modes)
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_checkpointing_meshtags_2D(mesh_2D, read_comm, read_mode, tmp_path):
    mesh = mesh_2D
    hash = f"{mesh.comm.size}_{mesh.topology.cell_name()}_{mesh.geometry.x.dtype}"
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = fname / f"meshtags_1D_{hash}.bp"
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh(filename, mesh, engine="BP4")
    else:
        if MPI.COMM_WORLD.rank == root:
            adios4dolfinx.write_mesh(filename, mesh, engine="BP4")

    org_maps = []
    for dim in range(mesh.topology.dim + 1):
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        e_map = mesh.topology.index_map(dim)
        num_entities_local = e_map.size_local
        entities = np.arange(num_entities_local, dtype=np.int32)
        ft = dolfinx.mesh.meshtags(mesh, dim, entities, e_map.local_range[0] + entities)
        ft.name = f"entity_{dim}"
        if mesh.comm.size != 1:
            adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")
            org_map = generate_reference_map(mesh, ft, mesh.comm, root)
            org_maps.append(org_map)
        else:
            if MPI.COMM_WORLD.rank == root:
                adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")
                org_map = generate_reference_map(mesh, ft, MPI.COMM_SELF, root)
                org_maps.append(org_map)
        del ft
    del mesh
    MPI.COMM_WORLD.Barrier()
    new_mesh = adios4dolfinx.read_mesh(filename, read_comm, engine="BP4", ghost_mode=read_mode)
    for dim in range(new_mesh.topology.dim + 1):
        if read_comm.size != 1:
            new_ft = adios4dolfinx.read_meshtags(
                filename, new_mesh, meshtag_name=f"entity_{dim}", engine="BP4"
            )
            read_map = generate_reference_map(new_mesh, new_ft, new_mesh.comm, root)
        else:
            if MPI.COMM_WORLD.rank == root:
                new_ft = adios4dolfinx.read_meshtags(
                    filename, new_mesh, meshtag_name=f"entity_{dim}", engine="BP4"
                )
                read_map = generate_reference_map(new_mesh, new_ft, read_comm, root)

        if MPI.COMM_WORLD.rank == root:
            org_map = org_maps[dim]
            assert len(org_map) == len(read_map)
            for value, (_, midpoint) in org_map.items():
                _, read_midpoint = read_map[value]
                np.testing.assert_allclose(read_midpoint, midpoint)


@pytest.mark.parametrize("read_mode", read_modes)
@pytest.mark.parametrize("read_comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_checkpointing_meshtags_3D(mesh_3D, read_comm, read_mode, tmp_path):
    mesh = mesh_3D
    hash = f"{mesh.comm.size}_{mesh.topology.cell_name()}_{mesh.geometry.x.dtype}"
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = fname / f"meshtags_1D_{hash}.bp"
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh(filename, mesh, engine="BP4")
    else:
        if MPI.COMM_WORLD.rank == root:
            adios4dolfinx.write_mesh(filename, mesh, engine="BP4")

    org_maps = []
    for dim in range(mesh.topology.dim + 1):
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        e_map = mesh.topology.index_map(dim)
        num_entities_local = e_map.size_local
        entities = np.arange(num_entities_local, dtype=np.int32)
        ft = dolfinx.mesh.meshtags(mesh, dim, entities, e_map.local_range[0] + entities)
        ft.name = f"entity_{dim}"

        if mesh.comm.size != 1:
            adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")
            org_map = generate_reference_map(mesh, ft, mesh.comm, root)
            org_maps.append(org_map)
        else:
            if MPI.COMM_WORLD.rank == root:
                adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")
                org_map = generate_reference_map(mesh, ft, MPI.COMM_SELF, root)
                org_maps.append(org_map)
        del ft
    del mesh

    MPI.COMM_WORLD.Barrier()
    new_mesh = adios4dolfinx.read_mesh(filename, read_comm, engine="BP4", ghost_mode=read_mode)
    for dim in range(new_mesh.topology.dim + 1):
        if read_comm.size != 1:
            new_ft = adios4dolfinx.read_meshtags(
                filename, new_mesh, meshtag_name=f"entity_{dim}", engine="BP4"
            )
            read_map = generate_reference_map(new_mesh, new_ft, new_mesh.comm, root)
        else:
            if MPI.COMM_WORLD.rank == root:
                new_ft = adios4dolfinx.read_meshtags(
                    filename, new_mesh, meshtag_name=f"entity_{dim}", engine="BP4"
                )
                read_map = generate_reference_map(new_mesh, new_ft, MPI.COMM_SELF, root)
        if MPI.COMM_WORLD.rank == root:
            org_map = org_maps[dim]
            assert len(org_map) == len(read_map)
            for value, (_, midpoint) in org_map.items():
                _, read_midpoint = read_map[value]
                np.testing.assert_allclose(read_midpoint, midpoint)
