# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

__all__ = ["compute_local_range", "index_owner"]
from mpi4py import MPI
import numpy as np
import numpy.typing as npt


def compute_local_range(comm: MPI.Comm, N: int):
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
        return [rank*(n+1), (rank+1)*(n+1)]
    else:
        return [rank*n+r, (rank+1)*n + r]


def index_owner(comm: MPI.Comm, indices: npt.NDArray[np.int64], N: int) -> npt.NDArray[np.int64]:
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

    owner = np.empty_like(indices)
    owner[indices < r * n + 1] = indices[indices < r * n + 1] // (n+1)
    owner[indices >= r*n+1] = r + (indices[indices >= r*n+1] - r*(n+1)) // n

    return owner
