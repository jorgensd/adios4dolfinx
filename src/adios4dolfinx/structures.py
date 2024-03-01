# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT


from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

"""Internal library classes for storing mesh and function data"""
__all__ = ["MeshData", "FunctionData"]


@dataclass
class MeshData:
    # 2 dimensional array of node coordinates
    local_geometry: npt.NDArray[np.floating]
    local_geometry_pos: Tuple[
        int, int
    ]  # Insert range on current process for geometry nodes
    num_nodes_global: int  # Number of nodes in global geometry array

    local_topology: npt.NDArray[
        np.int64
    ]  # 2 dimensional connecitivty array for mesh topology
    # Insert range on current process for topology
    local_topology_pos: Tuple[int, int]
    num_cells_global: int  # NUmber of cells in global topology

    cell_type: str
    degree: int
    lagrange_variant: int


@dataclass
class FunctionData:
    cell_permutations: npt.NDArray[np.uint32]  # Cell permutations for dofmap
    local_cell_range: Tuple[int, int]  # Range of cells on current process
    num_cells_global: int  # Number of cells in global topology
    dofmap_array: npt.NDArray[np.int64]  # Local function dofmap (using global indices)
    dofmap_offsets: npt.NDArray[np.int64]  # Global dofmap offsets
    dofmap_range: Tuple[int, int]  # Range of dofmap on current process
    global_dofs_in_dofmap: int  # Number of entries in global dofmap
    values: npt.NDArray[np.floating]  # Local function values
    dof_range: Tuple[int, int]  # Range of local function values
    num_dofs_global: int  # Number of global function values
    name: str  # Name of function
