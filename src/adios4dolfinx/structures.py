# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from dolfinx.graph import AdjacencyList

"""Internal library classes for storing mesh and function data"""
__all__ = ["MeshData", "FunctionData", "ReadMeshData", "MeshTagsData"]


@dataclass
class MeshData:
    """
    Container for distributed mesh data that will be stored to disk
    """

    #: Two-dimensional array of node coordinates
    local_geometry: npt.NDArray[np.float32] | npt.NDArray[np.float64]
    local_geometry_pos: tuple[int, int]  #: Insert range on current process for geometry nodes
    num_nodes_global: int  #: Number of nodes in global geometry array

    local_topology: npt.NDArray[np.int64]  #: Two-dimensional connectivity array for mesh topology
    #: Insert range on current process for topology
    local_topology_pos: tuple[int, int]
    num_cells_global: int  #: NUmber of cells in global topology

    cell_type: str  #: The cell type
    degree: int  #: Degree of underlying Lagrange element
    lagrange_variant: int  #: Lagrange-variant of DOFs

    # Partitioning_information
    store_partition: bool  #: Indicator if one should store mesh partitioning
    partition_processes: int | None = None  #: Number of processes in partition
    ownership_array: npt.NDArray[np.int32] | None = None  #: Ownership array for cells
    ownership_offset: npt.NDArray[np.int32] | None = None  #: Ownership offset for cells
    partition_range: tuple[int, int] | None = (
        None  #: Local insert position for partitioning information
    )
    partition_global: int | None = None  #: Global size of partitioning array


@dataclass
class FunctionData:
    """
    Container for distributed function data that will be written to file
    """

    cell_permutations: npt.NDArray[np.uint32]  #: Cell permutations for dofmap
    local_cell_range: tuple[int, int]  #: Range of cells on current process
    num_cells_global: int  #: Number of cells in global topology
    dofmap_array: npt.NDArray[np.int64]  #: Local function dofmap (using global indices)
    dofmap_offsets: npt.NDArray[np.int64]  #: Global dofmap offsets
    dofmap_range: tuple[int, int]  #: Range of dofmap on current process
    global_dofs_in_dofmap: int  #: Number of entries in global dofmap
    values: npt.NDArray[np.floating]  #: Local function values
    dof_range: tuple[int, int]  #: Range of local function values
    num_dofs_global: int  #: Number of global function values
    name: str  #: Name of function


@dataclass
class ReadMeshData:
    """Container containing data that will be read into DOLFINx"""

    #: Two-dimensional array containing global cell->node connectivity
    cells: npt.NDArray[np.int64]
    cell_type: str  #: The cell type of the mesh
    x: npt.NDArray[np.floating]  #: The mesh coordinates
    lvar: int  #: The Lagrange variant
    degree: int  #: The degree of the underlying Lagrange element
    #: Partitioning information per cell on the process
    partition_graph: AdjacencyList | None = None


@dataclass
class MeshTagsData:
    name: str  #: Name of tag
    values: npt.NDArray  # Array of values
    indices: npt.NDArray[np.int64]  # Global indices of the entities
    dim: int  # Topological dimension of the entities

    # Optional entries (used for writing to disk)
    num_entities_global: int | None = None  #: Global number of entities that will be written out
    num_dofs_per_entity: int | None = None  #: Number of dofs per entity
    #: Starting index in output array `(0<=local_start<num_entities_global)``
    local_start: int | None = None

    # Optional info to help visualization
    cell_type: str | None = None  #: The cell type
