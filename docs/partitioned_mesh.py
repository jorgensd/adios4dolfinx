# # Storing mesh partition
# This data is re-ordered when reading in a mesh, as the mesh is partitioned.
# This means that when storing the mesh to disk from DOLFINx, the geometry and
# connectivity arrays are re-ordered.
# If we want to avoid to re-partition the mesh every time you run a simulation
# (on a fixed number of processes), one can store the partitioning of the mesh
# in the checkpoint. This is done by setting the flag `store_partition_info=True`
# when calling {py:func}`adios4dolfinx.write_mesh`.

# +
import logging
from pathlib import Path

import ipyparallel as ipp


def write_partitioned_mesh(filename: Path):
    import subprocess

    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    # Create a simple unit square mesh
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        10,
        10,
        cell_type=dolfinx.mesh.CellType.quadrilateral,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )

    # Write mesh checkpoint
    adios4dolfinx.write_mesh(
        filename, mesh, backend="adios2", backend_args={"engine": "BP4"}, store_partition_info=True
    )
    # Inspect checkpoint on rank 0 with `bpls`
    if mesh.comm.rank == 0:
        output = subprocess.run(["bpls", "-a", "-l", filename], capture_output=True)
        print(output.stdout.decode("utf-8"))


# -

# We inspect the partitioned mesh

# +
mesh_file = Path("partitioned_mesh.bp")
n = 3

with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    query = cluster[:].apply_async(write_partitioned_mesh, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
# -

# # Reading a partitioned mesh

# If we try to read the mesh in on a different number of processes, we will get an error.
# We illustrate this below, by first trying to read the mesh using partitioning information,
# which is done by setting the flag `read_from_partition=True` when calling
# {py:func}`adios4dolfinx.read_mesh`.


def read_partitioned_mesh(filename: Path, read_from_partition: bool = True):
    from mpi4py import MPI

    import adios4dolfinx

    prefix = f"{MPI.COMM_WORLD.rank + 1}/{MPI.COMM_WORLD.size}: "
    try:
        mesh = adios4dolfinx.read_mesh(
            filename,
            comm=MPI.COMM_WORLD,
            backend="adios2",
            backend_args={"engine": "BP4"},
            read_from_partition=read_from_partition,
        )
        print(f"{prefix} Mesh: {mesh.name} read successfully with {read_from_partition=}")
    except ValueError as e:
        print(f"{prefix} Caught exception: ", e)


with ipp.Cluster(engines="mpi", n=n + 1, log_level=logging.ERROR) as cluster:
    # Read mesh from file with different number of processes
    query = cluster[:].apply_async(read_partitioned_mesh, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))

# Read mesh from file with different number of processes (not using partitioning information).
# If we instead turn of `read_from_partition`, we can read the mesh on a
# different number of processes.

with ipp.Cluster(engines="mpi", n=n + 1, log_level=logging.ERROR) as cluster:
    query = cluster[:].apply_async(read_partitioned_mesh, mesh_file, False)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))

# Read mesh from file with same number of processes as was written,
# re-using partitioning information.

with ipp.Cluster(engines="mpi", n=n, log_level=logging.ERROR) as cluster:
    query = cluster[:].apply_async(read_partitioned_mesh, mesh_file, True)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
