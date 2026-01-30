# # Writing a mesh checkpoint
#
# In this example, we will demonstrate how to write a mesh checkpoint to disk.
#
# We start by creating a simple {py:func}`unit-square mesh<dolfinx.mesh.create_unit_square>`.

# +
import logging
from pathlib import Path

from mpi4py import MPI

import dolfinx
import ipyparallel as ipp

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
# -

# Note that when a mesh is created in DOLFINx, we send in a
# {py:class}`MPI communicator<mpi4py.MPI.Comm>`.
# The communicator is used to partition (distribute) the mesh across the available processes.
# This means that each process only have access to a sub-set of cells and nodes of the mesh.
# We can inspect these with the following commands:


def print_mesh_info(mesh: dolfinx.mesh.Mesh):
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    node_map = mesh.geometry.index_map()
    print(
        f"Rank {mesh.comm.rank}: number of owned cells {cell_map.size_local}",
        f", number of ghosted cells {cell_map.num_ghosts}\n",
        f"Number of owned nodes {node_map.size_local}",
        f", number of ghosted nodes {node_map.num_ghosts}",
    )


print_mesh_info(mesh)

# ## Create a distributed mesh
# Next, we can use {py:mod}`IPython parallel<ipyparallel>` to inspect a partitioned
# {py:class}`mesh<dolfinx.mesh.Mesh>`.
# We create a convenience function for creating a mesh that
# {py:attr}`shares cells<dolfinx.mesh.GhostMode.shared_facet>` on the boundary
# between two processes if `ghosted=True`.


def create_distributed_mesh(ghosted: bool, N: int = 10):
    """
    Create a distributed mesh with N x N cells. Share cells on process boundaries
    if ghosted is set to True
    """
    from mpi4py import MPI

    import dolfinx

    ghost_mode = dolfinx.mesh.GhostMode.shared_facet if ghosted else dolfinx.mesh.GhostMode.none
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=ghost_mode)
    print(f"{ghost_mode=}")
    print_mesh_info(mesh)


# Next we start up a new cluster with three engines.
# As we defined `print_mesh_info` locally on this process, we need to push it to all engines.

with ipp.Cluster(engines="mpi", n=3, log_level=logging.ERROR) as cluster:
    # Push print_mesh_info to all engines
    cluster[:].push({"print_mesh_info": print_mesh_info})

    # Create mesh with ghosted cells
    query_true = cluster[:].apply_async(create_distributed_mesh, True)
    query_true.wait()
    assert query_true.successful(), query_true.error
    print("".join(query_true.stdout))
    # Create mesh without ghosted cells
    query_false = cluster[:].apply_async(create_distributed_mesh, False)
    query_false.wait()
    assert query_false.successful(), query_false.error
    print("".join(query_false.stdout))

# ## Writing a mesh checkpoint
# The input data to a mesh is:
# - A geometry: the set of points in R^D that are part of each cell
# - A two-dimensional connectivity array: A list that indicates which nodes of the geometry
#    is part of each cell
# - A {py:func}`reference element<basix.ufl.element>`: Used for push data back and
#   forth from the reference element and computing Jacobians
# We now use {py:mod}`adios4dolfinx` to write a mesh to file.


def write_mesh(filename: Path):
    import subprocess

    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    # Create a simple unit square mesh
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 10, 10, cell_type=dolfinx.mesh.CellType.quadrilateral
    )

    # Write mesh checkpoint
    adios4dolfinx.write_mesh(filename, mesh, backend="adios2", backend_args={"engine": "BP4"})

    # Inspect checkpoint on rank 0 with `bpls`
    if mesh.comm.rank == 0:
        output = subprocess.run(["bpls", "-a", "-l", str(filename.absolute())], capture_output=True)
        print(output.stdout.decode("utf-8"))


# +
mesh_file = Path("mesh.bp")

with ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR) as cluster:
    # Write mesh to file
    query = cluster[:].apply_async(write_mesh, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
# -

# We observe that we have stored all the data needed to re-create the mesh in the file `mesh.bp`.
# We can therefore read it (to any number of processes) with {py:func}`adios4dolfinx.read_mesh`


def read_mesh(filename: Path):
    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    mesh = adios4dolfinx.read_mesh(
        filename,
        comm=MPI.COMM_WORLD,
        backend="adios2",
        backend_args={"engine": "BP4"},
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )
    print_mesh_info(mesh)


# ## Reading mesh checkpoints (N-to-M)
# We can now read the checkpoint on a different number of processes than we wrote it on.

with ipp.Cluster(engines="mpi", n=4, log_level=logging.ERROR) as cluster:
    # Write mesh to file
    cluster[:].push({"print_mesh_info": print_mesh_info})
    query = cluster[:].apply_async(read_mesh, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
