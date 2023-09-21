# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

__all__ = ["compute_local_range", "index_owner", "compute_dofmap_pos", "find_first"]
from typing import Tuple

import dolfinx
import numba
import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from typing import Union

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


def compute_dofmap_pos(
    V: dolfinx.fem.FunctionSpaceBase,
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

    @numba.njit(cache=True)
    def compute_positions(
        local_cell: npt.NDArray[np.int32],
        dof_pos: npt.NDArray[np.int32],
        dofs: npt.NDArray[np.int32],
        dofmap_bs: int,
        num_owned_dofs: int,
        num_owned_cells: int,
    ):
        """
        Loop through each owned cell and every dof in that cell, and for all cells owned by the process
        attach a cell and a position in the dofmap to it
        """
        assert len(local_cell) == num_owned_dofs
        assert len(dof_pos) == num_owned_dofs

        for c in range(num_owned_cells):
            for i, dof in enumerate(dofs[c]):
                for b in range(dofmap_bs):
                    local_dof = dof * dofmap_bs + b
                    if local_dof < num_owned_dofs:
                        local_cell[local_dof] = c
                        dof_pos[local_dof] = i * dofmap_bs + b

    compute_positions(
        local_cell, dof_pos, dofs, dofmap_bs, num_owned_dofs, num_owned_cells
    )
    return local_cell, dof_pos


@numba.njit(cache=True)
def find_first(b: int, a: npt.NDArray[np.int32]):
    """
    Given a numpy array `a` return the first entry equal to `b`
    """
    for i, ai in enumerate(a):
        if ai == b:
            return i
