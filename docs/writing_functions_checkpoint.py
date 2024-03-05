# # Writing a function checkpoint
# In the previous sections, we have gone in to quite some detail as to how
# to store meshes with adios4dolfinx.
# This section will explain how to store functions, and how to read them back in.

# We start by creating a mesh and an appropriate function

from pathlib import Path

from mpi4py import MPI

import dolfinx
import ipyparallel as ipp

import adios4dolfinx
import logging
assert MPI.COMM_WORLD.size == 1, "This example should only be run with 1 MPI process"

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, nx=10, ny=10, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral
)

# Next, we create a function, and interpolate a polynomial function into the function space
el = "N1curl"
degree = 3
V = dolfinx.fem.functionspace(mesh, (el, degree))


def f(x):
    return -(x[1] ** 2), x[0] - 2 * x[1]


u = dolfinx.fem.Function(V)
u.interpolate(f)

# Next we start by storing the mesh

filename = Path("function_checkpoint.bp")
adios4dolfinx.write_mesh(filename, mesh)

# Next, we store the function to file, and associate it with a name.
# Note that we can also associate a time stamp with it, as done for meshes in
# [Writing time-dependent mesh checkpoint](./time_dependent_mesh)

adios4dolfinx.write_function(filename, u, time=0.3, name="my_curl_function")

# Next, we want to read the function back in (using multiple MPI processes)
# and check that the function is correct.


def read_function(filename: Path, timestamp: float):
    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import adios4dolfinx

    in_mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    W = dolfinx.fem.functionspace(in_mesh, (el, degree))
    u_ref = dolfinx.fem.Function(W)
    u_ref.interpolate(f)
    u_in = dolfinx.fem.Function(W)
    adios4dolfinx.read_function(filename, u_in, time=timestamp, name="my_curl_function")
    np.testing.assert_allclose(u_ref.x.array, u_in.x.array, atol=1e-14)
    print(
        f"{MPI.COMM_WORLD.rank + 1}/{MPI.COMM_WORLD.size}: ",
        f"Function read in correctly at time {timestamp}",
    )


with ipp.Cluster(engines="mpi", n=3, log_level=logging.ERROR) as cluster:
    cluster[:].push({"f": f, "el": el, "degree": degree})
    query = cluster[:].apply_async(read_function, filename, 0.3)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
