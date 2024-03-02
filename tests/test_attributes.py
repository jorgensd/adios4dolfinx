from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import adios4dolfinx


@pytest.mark.parametrize("comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_attributes(comm, tmp_path):
    attributes1 = {
        "a": np.array([1, 2, 3], dtype=np.uint8),
        "b": np.array([4, 5], dtype=np.uint8),
    }
    attributes2 = {
        "c": np.array([6], dtype=np.uint8),
        "d": np.array([7, 8, 9, 10], dtype=np.uint8),
    }
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / Path("attributes.bp")

    adios4dolfinx.write_attributes(comm=comm, filename=file, name="group1", attributes=attributes1)
    adios4dolfinx.write_attributes(comm=comm, filename=file, name="group2", attributes=attributes2)
    MPI.COMM_WORLD.Barrier()
    loaded_attributes1 = adios4dolfinx.read_attributes(comm=comm, filename=file, name="group1")
    loaded_attributes2 = adios4dolfinx.read_attributes(comm=comm, filename=file, name="group2")

    for k, v in loaded_attributes1.items():
        assert np.allclose(v, attributes1[k])
    for k, v in attributes1.items():
        assert np.allclose(v, loaded_attributes1[k])

    for k, v in loaded_attributes2.items():
        assert np.allclose(v, attributes2[k])
    for k, v in attributes2.items():
        assert np.allclose(v, loaded_attributes2[k])
