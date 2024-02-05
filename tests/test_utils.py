import pathlib

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing

import adios4dolfinx


def write_function(mesh, el, f, dtype, name="uh", append: bool = False) -> str:
    V = dolfinx.fem.functionspace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=dtype)
    uh.interpolate(f)
    uh.name = name
    el_hash = (
        V.element.signature()
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )

    file_hash = f"{el_hash}_{np.dtype(dtype).name}"
    filename = pathlib.Path(f"output/mesh_{file_hash}.bp")
    if mesh.comm.size != 1:
        if not append:
            adios4dolfinx.write_mesh(mesh, filename)
        adios4dolfinx.write_function(uh, filename, time=0.0)
    else:
        if MPI.COMM_WORLD.rank == 0:
            if not append:
                adios4dolfinx.write_mesh(mesh, filename)
            adios4dolfinx.write_function(uh, filename, time=0.0)

    return file_hash


def read_function(comm, el, f, hash, dtype, name="uh"):
    filename = f"output/mesh_{hash}.bp"
    engine = "BP4"
    mesh = adios4dolfinx.read_mesh(
        comm, filename, engine, dolfinx.mesh.GhostMode.shared_facet
    )
    V = dolfinx.fem.functionspace(mesh, el)
    v = dolfinx.fem.Function(V, dtype=dtype)
    v.name = name
    adios4dolfinx.read_function(v, filename, engine)
    v_ex = dolfinx.fem.Function(V, dtype=dtype)
    v_ex.interpolate(f)

    res = np.finfo(dtype).resolution
    assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)


def get_dtype(in_dtype: np.dtype, complex: bool):
    dtype: numpy.typing.DTypeLike
    if in_dtype == np.float32:
        if complex:
            dtype = np.complex64
        else:
            dtype = np.float32
    elif in_dtype == np.float64:
        if complex:
            dtype = np.complex128
        else:
            dtype = np.float64
    else:
        raise ValueError("Unsuported dtype")
    return dtype


def write_function_time_dep(mesh, el, f0, f1, t0, t1, dtype) -> str:
    V = dolfinx.fem.functionspace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=dtype)
    uh.interpolate(f0)
    el_hash = (
        V.element.signature()
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )
    file_hash = f"{el_hash}_{np.dtype(dtype).name}"
    filename = pathlib.Path(f"output/mesh_{file_hash}.bp")
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh(mesh, filename)
        adios4dolfinx.write_function(uh, filename, time=t0)
        uh.interpolate(f1)
        adios4dolfinx.write_function(uh, filename, time=t1)

    else:
        if MPI.COMM_WORLD.rank == 0:
            adios4dolfinx.write_mesh(mesh, filename)
            adios4dolfinx.write_function(uh, filename, time=t0)
            uh.interpolate(f1)
            adios4dolfinx.write_function(uh, filename, time=t1)

    return file_hash


def read_function_time_dep(comm, el, f0, f1, t0, t1, hash, dtype):
    filename = f"output/mesh_{hash}.bp"
    engine = "BP4"
    mesh = adios4dolfinx.read_mesh(
        comm, filename, engine, dolfinx.mesh.GhostMode.shared_facet
    )
    V = dolfinx.fem.functionspace(mesh, el)
    v = dolfinx.fem.Function(V, dtype=dtype)

    adios4dolfinx.read_function(v, filename, engine, time=t1)
    v_ex = dolfinx.fem.Function(V, dtype=dtype)
    v_ex.interpolate(f1)

    res = np.finfo(dtype).resolution
    assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

    adios4dolfinx.read_function(v, filename, engine, time=t0)
    v_ex = dolfinx.fem.Function(V, dtype=dtype)
    v_ex.interpolate(f0)

    res = np.finfo(dtype).resolution
    assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)
