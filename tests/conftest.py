import sys
import typing
from collections import ChainMap
from unittest.mock import patch

from mpi4py import MPI

import dolfinx
import ipyparallel as ipp
import numpy as np
import numpy.typing
import numpy.typing as npt
import pytest

import adios4dolfinx


def find_backends():
    backends = []
    try:
        import adios2

        if adios2.is_built_with_mpi:
            backends.append("adios2")
    except ModuleNotFoundError:
        pass
    try:
        import h5py

        if h5py.get_config().mpi:
            backends.append("h5py")
    except ModuleNotFoundError:
        pass
    return backends


@pytest.fixture(params=find_backends(), scope="function")
def backend(request):
    value = request.param
    if value == "adios2":
        # Mock h5py to test adios2 only
        with patch.dict(sys.modules, {"h5py": None}):
            yield value
    else:
        # Mock adios2 to test h5py only
        with patch.dict(sys.modules, {"adios2": None}):
            yield value


@pytest.fixture(scope="module")
def cluster():
    cluster = ipp.Cluster(engines="mpi", n=2)
    rc = cluster.start_and_connect_sync()
    yield rc
    cluster.stop_cluster_sync()


@pytest.fixture(scope="function")
def write_function(tmp_path):
    def _write_function(
        mesh,
        el,
        f,
        dtype,
        backend: typing.Literal["adios2", "h5py"],
        name="uh",
        append: bool = False,
    ) -> str:
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
        if backend == "adios2":
            suffix = ".bp"
        else:
            suffix = ".h5"

        filename = (f_path / f"mesh_{file_hash}").with_suffix(suffix)
        if mesh.comm.size != 1:
            if not append:
                adios4dolfinx.write_mesh(filename, mesh, backend=backend)
            adios4dolfinx.write_function(filename, uh, time=0.0, backend=backend)
        else:
            if MPI.COMM_WORLD.rank == 0:
                if not append:
                    adios4dolfinx.write_mesh(filename, mesh, backend=backend)
                adios4dolfinx.write_function(filename, uh, time=0.0, backend=backend)

        return filename

    return _write_function


@pytest.fixture(scope="function")
def read_function():
    def _read_function(
        comm, el, f, path, dtype, backend: typing.Literal["adios2", "h5py"], name="uh"
    ):
        mesh = adios4dolfinx.read_mesh(
            path,
            comm,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            backend=backend,
        )
        V = dolfinx.fem.functionspace(mesh, el)
        v = dolfinx.fem.Function(V, dtype=dtype)
        v.name = name
        adios4dolfinx.read_function(path, v, backend=backend)
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
    def _write_function_time_dep(
        mesh, el, f0, f1, t0, t1, dtype, backend: typing.Literal["adios2", "h5py"]
    ) -> str:
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
        if backend == "adios2":
            suffix = ".bp"
        else:
            suffix = ".h5"
        filename = (f_path / f"mesh_{file_hash}").with_suffix(suffix)
        if mesh.comm.size != 1:
            adios4dolfinx.write_mesh(filename, mesh, backend=backend)
            adios4dolfinx.write_function(filename, uh, time=t0, backend=backend)
            uh.interpolate(f1)
            adios4dolfinx.write_function(filename, uh, time=t1, backend=backend)

        else:
            if MPI.COMM_WORLD.rank == 0:
                adios4dolfinx.write_mesh(filename, mesh, backend=backend)
                adios4dolfinx.write_function(filename, uh, time=t0, backend=backend)
                uh.interpolate(f1)
                adios4dolfinx.write_function(filename, uh, time=t1, backend=backend)

        return filename

    return _write_function_time_dep


@pytest.fixture(scope="function")
def read_function_time_dep():
    def _read_function_time_dep(
        comm, el, f0, f1, t0, t1, path, dtype, backend: typing.Literal["adios2", "h5py"]
    ):
        mesh = adios4dolfinx.read_mesh(
            path,
            comm,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            backend=backend,
        )
        V = dolfinx.fem.functionspace(mesh, el)
        v = dolfinx.fem.Function(V, dtype=dtype)

        adios4dolfinx.read_function(path, v, time=t1, backend=backend)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f1)

        res = np.finfo(dtype).resolution
        assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

        adios4dolfinx.read_function(path, v, time=t0, backend=backend)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f0)

        res = np.finfo(dtype).resolution
        assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

    return _read_function_time_dep


def _generate_reference_map(
    mesh: dolfinx.mesh.Mesh,
    meshtag: dolfinx.mesh.MeshTags,
    comm: MPI.Intracomm,
    root: int,
) -> typing.Optional[dict[str, tuple[int, npt.NDArray]]]:
    """
    Helper function to generate map from meshtag value to its corresponding index and midpoint.

    Args:
        mesh: The mesh
        meshtag: The associated meshtag
        comm: MPI communicator to gather the map from all processes with
        root (int): Rank to store data on
    Returns:
        Root rank returns the map, all other ranks return None
    """
    mesh.topology.create_connectivity(meshtag.dim, mesh.topology.dim)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, meshtag.dim, meshtag.indices)
    e_map = mesh.topology.index_map(meshtag.dim)
    value_to_midpoint = {}
    for index, value in zip(meshtag.indices, meshtag.values):
        value_to_midpoint[value] = (
            int(e_map.local_range[0] + index),
            midpoints[index],
        )
    global_map = comm.gather(value_to_midpoint, root=root)
    if comm.rank == root:
        return dict(ChainMap(*global_map))  # type: ignore
    return None


@pytest.fixture
def generate_reference_map():
    return _generate_reference_map
