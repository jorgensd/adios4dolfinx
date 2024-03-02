from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import adios4dolfinx


@pytest.mark.parametrize("comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_attributes(comm, tmp_path):
    attributes = {
        "a": np.array([1, 2, 3], dtype=np.uint8),
        "b": np.array([4, 5, 6], dtype=np.uint8),
    }
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / Path("attributes.bp")

    adios4dolfinx.write_attributes(comm=comm, filename=file, name="test", attributes=attributes)
    MPI.COMM_WORLD.Barrier()
    loaded_attributes = adios4dolfinx.read_attributes(comm=comm, filename=file, name="test")
    for k, v in loaded_attributes.items():
        assert np.allclose(v, attributes[k])
    for k, v in attributes.items():
        assert np.allclose(v, loaded_attributes[k])
