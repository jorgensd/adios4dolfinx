from mpi4py import MPI

import adios2
import numpy as np
import pytest
from packaging.version import parse as _v

import adios4dolfinx


@pytest.mark.parametrize("backend", ["h5py", "adios2"])
@pytest.mark.skipif(
    _v(np.__version__) >= _v("2.0.0") and _v(adios2.__version__) < _v("2.10.2"),
    reason="Cannot use numpy>=2.0.0 and adios2<2.10.2",
)
@pytest.mark.parametrize("comm", [MPI.COMM_SELF, MPI.COMM_WORLD])
def test_read_write_attributes(comm, backend, tmp_path):
    attributes1 = {
        "a": np.array([1, 2, 3], dtype=np.uint8),
        "b": np.array([4, 5], dtype=np.uint8),
    }
    attributes2 = {
        "c": np.array([6], dtype=np.uint8),
        "d": np.array([7, 8, 9, 10], dtype=np.float64),
    }
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    fname = fname / "atributes"
    suffix = ".bp" if backend == "h5py" else ".h5"
    file = fname.with_suffix(suffix)

    adios4dolfinx.write_attributes(
        comm=comm, filename=file, name="group1", attributes=attributes1, backend=backend
    )
    adios4dolfinx.write_attributes(
        comm=comm, filename=file, name="group2", attributes=attributes2, backend=backend
    )
    MPI.COMM_WORLD.Barrier()
    loaded_attributes1 = adios4dolfinx.read_attributes(
        comm=comm, filename=file, name="group1", backend=backend
    )
    loaded_attributes2 = adios4dolfinx.read_attributes(
        comm=comm, filename=file, name="group2", backend=backend
    )

    for k, v in loaded_attributes1.items():
        assert np.allclose(v, attributes1[k])
    for k, v in attributes1.items():
        assert np.allclose(v, loaded_attributes1[k])

    for k, v in loaded_attributes2.items():
        assert np.allclose(v, attributes2[k])
    for k, v in attributes2.items():
        assert np.allclose(v, loaded_attributes2[k])
