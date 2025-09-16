# # Snapshot checkpoint (non-persistent)
# The checkpoint method described in [Writing function checkpoints](./writing_functions_checkpoint)
# are *N-to-M*, meaning that you can write them out on N-processes and read them in on M processes.
#
# As discussed in that chapter, these checkpoints need to be associated with a mesh.
# This is because the function is defined on a specific function space, which in turn is
# defined on a specific mesh.
#
# However, there are certain scenarios where you simply want to store a checkpoint associated
# with the current mesh, that should only be possible to use during this simulation.
# An example use-case is when running an iterative solver, and wanting a fall-back mechanism that
# does not require extra RAM.

# In this example, we will demonstrate how to write a snapshot checkpoint to disk.

# First we define a function `f` that we want to represent in the function space

import logging
from pathlib import Path

import ipyparallel as ipp


def f(x):
    import numpy as np

    return np.sin(x[0]) + 0.1 * x[1]


# Next, we create a mesh and an appropriate function space and read and write from file
def read_write_snapshot(filename: Path):
    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import adios4dolfinx

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 7, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 5))
    u = dolfinx.fem.Function(V)
    u.interpolate(f)
    u.name = "Current_solution"
    # Next, we store the solution to file
    adios4dolfinx.snapshot_checkpoint(u, filename, adios4dolfinx.adios2_helpers.adios2.Mode.Write)

    # Next, we create a new function and load the solution into it
    u_new = dolfinx.fem.Function(V)
    u_new.name = "Read_solution"
    adios4dolfinx.snapshot_checkpoint(
        u_new, filename, adios4dolfinx.adios2_helpers.adios2.Mode.Read
    )

    # Next, we verify that the solution is correct
    np.testing.assert_allclose(u_new.x.array, u.x.array, atol=np.finfo(float).eps)

    print(f"{MPI.COMM_WORLD.rank + 1}/{MPI.COMM_WORLD.size}: Successfully wrote and read snapshot")


mesh_file = Path("snapshot.bp")

with ipp.Cluster(engines="mpi", n=3, log_level=logging.ERROR) as cluster:
    cluster[:].push({"f": f})
    query = cluster[:].apply_async(
        read_write_snapshot,
        mesh_file,
    )
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
