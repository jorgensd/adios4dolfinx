# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

__all__ = ["compute_local_range", "index_owner", "compute_dofmap_pos", "unroll_dofmap"]
from typing import Tuple, Union

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt

valid_function_types = Union[np.float32, np.float64, np.complex64, np.complex128]
valid_real_types = Union[np.float32, np.float64]


def compute_local_range(comm: MPI.Intracomm, N: np.int64):
    """
    Divide a set of `N` objects into `M` partitions, where `M` is
    the size of the MPI communicator `comm`.

    NOTE: If N is not divisible by the number of ranks, the first `r`
    processes gets an extra value

    Returns the local range of values
    """
    rank = comm.rank
    size = comm.size
    n = N // size
    r = N % size
    # First r processes has one extra value
    if rank < r:
        return [rank * (n + 1), (rank + 1) * (n + 1)]
    else:
        return [rank * n + r, (rank + 1) * n + r]


def index_owner(
    comm: MPI.Intracomm, indices: npt.NDArray[np.int64], N: np.int64
) -> npt.NDArray[np.int32]:
    """
    Find which rank (local to comm) which owns an `index`, given that
    data of size `N` has been split equally among the ranks.

    NOTE: If `N` is not divisible by the number of ranks, the first `r`
    processes gets an extra value.
    """
    size = comm.size
    assert (indices < N).all()
    n = N // size
    r = N % size

    owner = np.empty_like(indices, dtype=np.int32)
    owner[indices < r * n + 1] = indices[indices < r * n + 1] // (n + 1)
    owner[indices >= r * n + 1] = r + (indices[indices >= r * n + 1] - r * (n + 1)) // n

    return owner


def unroll_dofmap(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = np.repeat(dofs, bs).reshape(
        num_cells, num_dofs_per_cell * bs) * bs
    unrolled_dofmap += np.tile(np.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap


def compute_dofmap_pos(
    V: dolfinx.fem.FunctionSpace,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Compute a map from each owned dof in the dofmap to a single cell owned by the
    process, and the relative position of the dof.

    :param V: The function space
    :returns: The tuple (`cells`, `dof_pos`) where each array is the size of the
        number of owned dofs (unrolled for block size)
    """
    dofs = V.dofmap.list
    mesh = V.mesh
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    dofmap_bs = V.dofmap.bs
    num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    local_cell = np.empty(
        num_owned_dofs, dtype=np.int32
    )  # Local cell index for each dof owned by process
    dof_pos = np.empty(
        num_owned_dofs, dtype=np.int32
    )  # Position in dofmap for said dof

    unrolled_dofmap = unroll_dofmap(dofs[:num_owned_cells, :], dofmap_bs)
    markers = unrolled_dofmap < num_owned_dofs
    local_indices = np.broadcast_to(np.arange(markers.shape[1]), markers.shape)
    cell_indicator = np.broadcast_to(
        np.arange(num_owned_cells, dtype=np.int32).reshape(-1, 1), (num_owned_cells, markers.shape[1]))
    indicator = unrolled_dofmap[markers].reshape(-1)
    local_cell[indicator] = cell_indicator[markers].reshape(-1)
    dof_pos[indicator] = local_indices[markers].reshape(-1)
    return local_cell, dof_pos
