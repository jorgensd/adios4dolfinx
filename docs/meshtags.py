# # Writing MeshTags data to a checkpoint file
# In many scenarios, the mesh used in a checkpoint is not trivial, and subdomains and sub-entities
# have been tagged with appropriate markers.
# As the mesh gets redistributed when read
# (see [Writing Mesh Checkpoint](./writing_mesh_checkpoint)),
# we need to store any tags together with this new mesh.

# As an example we will use a unit-cube, where each entity has been tagged with a unique index.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import ipyparallel as ipp
import numpy as np

import adios4dolfinx

assert MPI.COMM_WORLD.size == 1, "This example should only be run with 1 MPI process"

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, nx=3, ny=4, nz=5)

# We start by computing the unique global index of each (owned) entity in the mesh
# as well as its corresponding midpoint
entity_midpoints = {}
meshtags = {}
for i in range(mesh.topology.dim + 1):
    mesh.topology.create_entities(i)
    e_map = mesh.topology.index_map(i)

    # Compute midpoints of entities
    entities = np.arange(e_map.size_local, dtype=np.int32)
    entity_midpoints[i] = dolfinx.mesh.compute_midpoints(mesh, i, entities)
    # Associate each local index with its global index
    values = np.arange(e_map.size_local, dtype=np.int32) + e_map.local_range[0]
    meshtags[i] = dolfinx.mesh.meshtags(mesh, i, entities, values)

# We use adios4dolfinx to write the mesh and meshtags to file.
# We associate each meshtag with a name
filename = Path("mesh_with_meshtags.bp")
adios4dolfinx.write_mesh(filename, mesh)
for i, tag in meshtags.items():
    adios4dolfinx.write_meshtags(filename, mesh, tag, meshtag_name=f"meshtags_{i}")


# Next we want to read the meshtags in on a different number of processes,
# and check that the midpoints of each entity is still correct


def verify_meshtags(filename: Path):
    # We assume that entity_midpoints have been sent to the engine
    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import adios4dolfinx

    read_mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    prefix = f"{read_mesh.comm.rank + 1}/{read_mesh.comm.size}: "
    for i in range(read_mesh.topology.dim + 1):
        # Read mesh from file
        meshtags = adios4dolfinx.read_meshtags(filename, read_mesh, meshtag_name=f"meshtags_{i}")

        # Compute midpoints for all local entities on process
        midpoints = dolfinx.mesh.compute_midpoints(read_mesh, i, meshtags.indices)
        # Compare locally computed midpoint with reference data
        for global_pos, midpoint in zip(meshtags.values, midpoints):
            np.testing.assert_allclose(
                entity_midpoints[i][global_pos],
                midpoint,
                err_msg=f"{prefix}: Midpoint ({i , global_pos}) do not match",
            )
        print(f"{prefix} Matching of all entities of dimension {i} successful")

with ipp.Cluster(engines="mpi", n=3) as cluster:
    cluster[:].push({"entity_midpoints": entity_midpoints})
    query = cluster[:].apply_async(verify_meshtags, filename)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
