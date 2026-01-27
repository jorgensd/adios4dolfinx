from mpi4py import MPI

import dolfinx
import ipyparallel as ipp
import numpy as np
import numpy.typing
import pytest

import adios4dolfinx


@pytest.fixture(scope="module")
def cluster():
    cluster = ipp.Cluster(engines="mpi", n=2)
    rc = cluster.start_and_connect_sync()
    yield rc
    cluster.stop_cluster_sync()


@pytest.fixture(scope="function")
def write_function(tmp_path):
    def _write_function(mesh, el, f, dtype, name="uh", append: bool = False) -> str:
        V = dolfinx.fem.functionspace(mesh, el)
        uh = dolfinx.fem.Function(V, dtype=dtype)
        uh.interpolate(f)
        uh.name = name
        el_hash = (
            adios4dolfinx.utils.element_signature(V)
            .replace(" ", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        # Consistent tmp dir across processes
        f_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
        file_hash = f"{el_hash}_{np.dtype(dtype).name}"
        filename = f_path / f"mesh_{file_hash}.bp"
        if mesh.comm.size != 1:
            if not append:
                adios4dolfinx.write_mesh(filename, mesh)
            adios4dolfinx.write_function(filename, uh, time=0.0)
        else:
            if MPI.COMM_WORLD.rank == 0:
                if not append:
                    adios4dolfinx.write_mesh(filename, mesh)
                adios4dolfinx.write_function(filename, uh, time=0.0)

        return filename

    return _write_function


@pytest.fixture(scope="function")
def read_function():
    def _read_function(comm, el, f, path, dtype, name="uh"):
        engine = "BP4"
        mesh = adios4dolfinx.read_mesh(
            filename=path,
            comm=comm,
            backend="adios2",
            backend_args={"engine": engine},
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        )
        V = dolfinx.fem.functionspace(mesh, el)
        v = dolfinx.fem.Function(V, dtype=dtype)
        v.name = name
        adios4dolfinx.read_function(path, v, engine)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f)

        res = np.finfo(dtype).resolution
        np.testing.assert_allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

    return _read_function


@pytest.fixture(scope="function")
def get_dtype():
    def _get_dtype(in_dtype: np.dtype, is_complex: bool):
        dtype: numpy.typing.DTypeLike
        if in_dtype == np.float32:
            if is_complex:
                dtype = np.complex64
            else:
                dtype = np.float32
        elif in_dtype == np.float64:
            if is_complex:
                dtype = np.complex128
            else:
                dtype = np.float64
        else:
            raise ValueError("Unsuported dtype")
        return dtype

    return _get_dtype


@pytest.fixture(scope="function")
def write_function_time_dep(tmp_path):
    def _write_function_time_dep(mesh, el, f0, f1, t0, t1, dtype) -> str:
        V = dolfinx.fem.functionspace(mesh, el)
        uh = dolfinx.fem.Function(V, dtype=dtype)
        uh.interpolate(f0)
        el_hash = (
            adios4dolfinx.utils.element_signature(V)
            .replace(" ", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        file_hash = f"{el_hash}_{np.dtype(dtype).name}"
        # Consistent tmp dir across processes
        f_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
        filename = f_path / f"mesh_{file_hash}.bp"
        if mesh.comm.size != 1:
            adios4dolfinx.write_mesh(filename, mesh)
            adios4dolfinx.write_function(filename, uh, time=t0)
            uh.interpolate(f1)
            adios4dolfinx.write_function(filename, uh, time=t1)

        else:
            if MPI.COMM_WORLD.rank == 0:
                adios4dolfinx.write_mesh(filename, mesh)
                adios4dolfinx.write_function(filename, uh, time=t0)
                uh.interpolate(f1)
                adios4dolfinx.write_function(filename, uh, time=t1)

        return filename

    return _write_function_time_dep


@pytest.fixture(scope="function")
def read_function_time_dep():
    def _read_function_time_dep(comm, el, f0, f1, t0, t1, path, dtype):
        engine = "BP4"
        mesh = adios4dolfinx.read_mesh(
            filename=path,
            comm=comm,
            backend="adios2",
            backend_args={"engine": engine},
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        )
        V = dolfinx.fem.functionspace(mesh, el)
        v = dolfinx.fem.Function(V, dtype=dtype)

        adios4dolfinx.read_function(path, v, engine, time=t1)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f1)

        res = np.finfo(dtype).resolution
        assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

        adios4dolfinx.read_function(path, v, engine, time=t0)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f0)

        res = np.finfo(dtype).resolution
        assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

    return _read_function_time_dep
