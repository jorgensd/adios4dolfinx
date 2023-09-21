import pathlib

import dolfinx
import numpy as np
import numpy.typing
from mpi4py import MPI

import adios4dolfinx


def write_function(mesh, el, f, dtype) -> str:
    V = dolfinx.fem.FunctionSpace(mesh, el)
    uh = dolfinx.fem.Function(V, dtype=dtype)
    uh.interpolate(f)
    el_hash = (
        V.element.signature()
        .replace(" ", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )
    filename = pathlib.Path(f"output/mesh{el_hash}_{dtype}.bp")
    if mesh.comm.size != 1:
        adios4dolfinx.write_mesh(mesh, filename)
        adios4dolfinx.write_function(uh, filename)

    else:
        if MPI.COMM_WORLD.rank == 0:
            adios4dolfinx.write_mesh(mesh, filename)
            adios4dolfinx.write_function(uh, filename)
    return f"{el_hash}_{dtype}"


def read_function(comm, el, f, hash, dtype):
    filename = f"output/mesh{hash}.bp"
    engine = "BP4"
    mesh = adios4dolfinx.read_mesh(
        comm, filename, engine, dolfinx.mesh.GhostMode.shared_facet
    )
    V = dolfinx.fem.functionspace(mesh, el)
    v = dolfinx.fem.Function(V, dtype=dtype)
    adios4dolfinx.read_function(v, filename, engine)
    v_ex = dolfinx.fem.Function(V, dtype=dtype)
    v_ex.interpolate(f)

    res = np.finfo(dtype).resolution
    assert np.allclose(v.x.array, v_ex.x.array, atol=10*res, rtol=10*res)


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
