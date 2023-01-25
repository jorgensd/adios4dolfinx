from adios4dolfinx import write_mesh, read_mesh
import dolfinx
from mpi4py import MPI
import pathlib
import time
import pytest


@pytest.mark.parametrize("encoder", ["BP4", "HDF5", "BP5"])
def test_mesh_read_writer(encoder):

    N = 100
    if "BP" in encoder:
        suffix = ".bp"
    elif "HDF5" in encoder:
        suffix = ".h5"
    else:
        raise ValueError("Unknown encoder")
    file = pathlib.Path(f"output/adios_mesh_{encoder}")
    xdmf_file = pathlib.Path("output/xdmf_mesh")
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
    MPI.COMM_WORLD.Barrier()

    start = time.perf_counter()
    write_mesh(mesh, file.with_suffix(suffix), encoder)
    end = time.perf_counter()
    print(f"Write ADIOS2 mesh: {end-start}")

    MPI.COMM_WORLD.Barrier()
    start = time.perf_counter()
    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
    end = time.perf_counter()
    print(f"Write XDMF mesh: {end-start}")
    MPI.COMM_WORLD.Barrier()

    start = time.perf_counter()
    mesh_adios = read_mesh(MPI.COMM_WORLD, file.with_suffix(suffix), encoder)
    end = time.perf_counter()
    print(f"Read ADIOS2 mesh: {end-start}")
    MPI.COMM_WORLD.Barrier()

    start = time.perf_counter()
    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "r") as xdmf:
        mesh_xdmf = xdmf.read_mesh()
    end = time.perf_counter()
    print(f"Read XDMF mesh: {end-start}")

    for i in range(mesh.topology.dim+1):
        mesh_xdmf.topology.create_entities(i)
        mesh_adios.topology.create_entities(i)
        assert mesh_xdmf.topology.index_map(
            i).size_global == mesh_adios.topology.index_map(i).size_global
