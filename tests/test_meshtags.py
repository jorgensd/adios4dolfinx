import adios4dolfinx
from mpi4py import MPI
import dolfinx
import numpy as np


mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

for dim in range(mesh.topology.dim+1):

    filename = f"meshtags_{dim}.bp"
    adios4dolfinx.write_mesh(mesh, filename, engine="BP4")

    mesh.topology.create_connectivity(dim, mesh.topology.dim)
    num_entities_local = mesh.topology.index_map(dim).size_local
    entities = np.arange(num_entities_local, dtype=np.int32)
    ft = dolfinx.mesh.meshtags(mesh, dim, entities, entities)
    ft.name = f"entity_{dim}"
    adios4dolfinx.write_meshtags(filename, mesh, ft, engine="BP4")

    with dolfinx.io.XDMFFile(mesh.comm, f"org_tags_{dim}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft, mesh.geometry)

    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.rank == 0:
        new_mesh = adios4dolfinx.read_mesh(MPI.COMM_SELF, filename, engine="BP4",
                                           ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
        new_ft = adios4dolfinx.read_meshtags(filename, new_mesh, meshtag_name=ft.name, engine="BP4")
        with dolfinx.io.XDMFFile(new_mesh.comm, f"new_tags{dim}.xdmf", "w") as xdmf:
            xdmf.write_mesh(new_mesh)
            xdmf.write_meshtags(new_ft, new_mesh.geometry)
