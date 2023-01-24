# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

__all__ = ["compute_local_range"]
from mpi4py import MPI


def compute_local_range(comm: MPI.Comm, N: int):
    """
    Divide a set of `N` objects into `M` partitions, where `M` is
    the size of the MPI communicator `comm`.

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
