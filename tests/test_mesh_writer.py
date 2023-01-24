from adios4dolfinx import write_mesh, read_mesh
import dolfinx
from mpi4py import MPI
import pathlib
import time


def test_mesh_read_writer():

    file = pathlib.Path("test")

    N = 50
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
    MPI.COMM_WORLD.Barrier()

    start = time.perf_counter()
    write_mesh(mesh, file.with_suffix(".bp"), "BP4")
    end = time.perf_counter()
    print(f"Write ADIOS2 mesh: {end-start}")

    MPI.COMM_WORLD.Barrier()
    start = time.perf_counter()
    with dolfinx.io.XDMFFile(mesh.comm, "test.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    end = time.perf_counter()
    print(f"Write XDMF mesh: {end-start}")
    MPI.COMM_WORLD.Barrier()

    start = time.perf_counter()
    mesh_adios = read_mesh(MPI.COMM_WORLD, file.with_suffix(".bp"), "BP4")
    end = time.perf_counter()
    print(f"Read ADIOS2 mesh: {end-start}")
    MPI.COMM_WORLD.Barrier()

    start = time.perf_counter()
    with dolfinx.io.XDMFFile(mesh.comm, "test.xdmf", "r") as xdmf:
        mesh_xdmf = xdmf.read_mesh()
    end = time.perf_counter()
    print(f"Read XDMF mesh: {end-start}")

    for i in range(mesh.topology.dim+1):
        mesh_xdmf.topology.create_entities(i)
        mesh_adios.topology.create_entities(i)
        assert mesh_xdmf.topology.index_map(
            i).size_global == mesh_adios.topology.index_map(i).size_global
