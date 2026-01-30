# # Time-dependent mesh checkpoints
# As we have seen in the previous examples, we store information about the connectivity,
# the coordinates of the mesh nodes,
# as well as a reference element. Note that the only thing that can change for a mesh
# during a simulation are the coordinate of the mesh nodes.
# In the following example, we will demonstrate how to write a time-dependent mesh
# checkpoint to disk.

# First, we create a simple function to compute the volume of a mesh

# +
import logging
from pathlib import Path

from mpi4py import MPI

import ipyparallel as ipp

import adios4dolfinx


def compute_volume(mesh, time_stamp):
    from mpi4py import MPI

    import dolfinx
    import ufl

    # Compute the volume of the mesh
    vol_form = dolfinx.fem.form(1 * ufl.dx(domain=mesh))
    vol_local = dolfinx.fem.assemble_scalar(vol_form)
    vol_glob = mesh.comm.allreduce(vol_local, op=MPI.SUM)
    if mesh.comm.rank == 0:
        print(f"{mesh.comm.rank + 1}/{mesh.comm.size} Time: {time_stamp} Mesh Volume: {vol_glob}")


def write_meshes(filename: Path):
    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import adios4dolfinx

    # Create a unit cube
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        3,
        6,
        5,
        cell_type=dolfinx.mesh.CellType.hexahedron,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )

    # Write mesh to file, associated with time stamp 1.5
    adios4dolfinx.write_mesh(filename, mesh, time=1.5)
    compute_volume(mesh, 1.5)
    mesh.geometry.x[:, 0] += 0.1 * mesh.geometry.x[:, 0]
    mesh.geometry.x[:, 1] += 0.3 * mesh.geometry.x[:, 1] * np.sin(mesh.geometry.x[:, 2])
    compute_volume(mesh, 3.3)
    # Write mesh to file, associated with time stamp 3.3
    # Note that we set the mode to append, as we have already created the file
    # and we do not want to overwrite the existing data
    adios4dolfinx.write_mesh(filename, mesh, time=3.3, mode=adios4dolfinx.FileMode.append)


# -

# We write the sequence of meshes to file

# +
mesh_file = Path("timedep_mesh.bp")
n = 3

with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    # Write mesh to file
    cluster[:].push({"compute_volume": compute_volume})
    query = cluster[:].apply_async(write_meshes, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
# -

# # Reading a time dependent mesh
# The only thing we need to do to read the mesh is to send in the associated time stamp,
# which we do by adding `time=time_stamp` when calling {py:func}`adios4dolfinx.read_mesh`.

second_mesh = adios4dolfinx.read_mesh(
    mesh_file, comm=MPI.COMM_WORLD, backend="adios2", backend_args={"engine": "BP4"}, time=3.3
)
compute_volume(second_mesh, 3.3)

first_mesh = adios4dolfinx.read_mesh(
    mesh_file, comm=MPI.COMM_WORLD, backend="adios2", backend_args={"engine": "BP4"}, time=1.5
)
compute_volume(first_mesh, 1.5)

# We observe that the volume of the mesh has changed, as we have perturbed the mesh
# between the two time stamps.
# We also note that we can read the meshes in on a different number of processes than
# we wrote them with and in a different order (as long as the time stamps are correct).
