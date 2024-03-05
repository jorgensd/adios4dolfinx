# # Checkpoint on input mesh
# As we have discussed earlier, one can choose to store function data in a way that
# is N-to-M compatible by using `adios4dolfinx.write_checkpoint`.
# This stores the distributed mesh in it's current (partitioned) ordering, and does
# use the original input data ordering for the cells and connectivity.
# This means that you cannot use your original mesh (from `.xdmf` files) or mesh tags
# together with the checkpoint. The checkpoint has to store the mesh and associated
# mesh-tags.

# An optional way of store an N-to-M checkpoint is to store the function data in the same
# ordering as the mesh. The write operation will be more expensive, as it requires data
# communication to ensure contiguous data being written to the checkpoint.
# The method is exposed as `adios4dolfinx.write_function_on_input_mesh`.
# Below we will demostrate this method.

import logging
from pathlib import Path
from typing import Tuple

import ipyparallel as ipp


def locate_facets(x, tol=1.0e-12):
    return abs(x[0]) < tol


def create_xdmf_mesh(filename: Path):
    from mpi4py import MPI

    import dolfinx

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, locate_facets)
    facet_tag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, 1)
    facet_tag.name = "FacetTag"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tag, mesh.geometry)
    print(f"{mesh.comm.rank+1}/{mesh.comm.size} Mesh written to {filename.with_suffix('.xdmf')}")


mesh_file = Path("MyMesh.xdmf")
with ipp.Cluster(engines="mpi", n=4, log_level=logging.ERROR) as cluster:
    # Create a mesh and write to XDMFFile
    cluster[:].push({"locate_facets": locate_facets})
    query = cluster[:].apply_async(create_xdmf_mesh, mesh_file)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# Next, we will create a function on the mesh and write it to a checkpoint.
def f(x):
    return (x[0] + x[1]) * (x[0] < 0.5), x[1], x[2] - x[1]


def write_function(
    mesh_filename: Path, function_filename: Path, element: Tuple[str, int, Tuple[int,]]
):
    from mpi4py import MPI

    import dolfinx

    import adios4dolfinx

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh()
    V = dolfinx.fem.functionspace(mesh, element)
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    adios4dolfinx.write_function_on_input_mesh(
        function_filename.with_suffix(".bp"),
        u,
        mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write,
        time=0.0,
        name="Output",
    )
    print(
        f"{mesh.comm.rank+1}/{mesh.comm.size} Function written to ",
        f"{function_filename.with_suffix('.bp')}",
    )


element = ("DG", 4, (3,))
function_file = Path("MyFunction.bp")
with ipp.Cluster(engines="mpi", n=2, log_level=logging.ERROR) as cluster:
    # Read in mesh and write function to file
    cluster[:].push({"f": f})
    query = cluster[:].apply_async(write_function, mesh_file, function_file, element)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))


# Finally, we will read in the mesh from file and the function from the checkpoint
# and compare it with the analytical solution.


def verify_checkpoint(
    mesh_filename: Path, function_filename: Path, element: Tuple[str, int, Tuple[int,]]
):
    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import adios4dolfinx

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
        in_mesh = xdmf.read_mesh()
    V = dolfinx.fem.functionspace(in_mesh, element)
    u_in = dolfinx.fem.Function(V)
    adios4dolfinx.read_function(function_filename.with_suffix(".bp"), u_in, time=0.0, name="Output")

    # Compute exact interpolation
    u_ex = dolfinx.fem.Function(V)
    u_ex.interpolate(f)

    np.testing.assert_allclose(u_in.x.array, u_ex.x.array)
    print(
        "Successfully read checkpoint onto mesh on rank ",
        f"{in_mesh.comm.rank + 1}/{in_mesh.comm.size}",
    )


with ipp.Cluster(engines="mpi", n=5, log_level=logging.ERROR) as cluster:
    # Read in mesh and write function to file
    cluster[:].push({"f": f})
    query = cluster[:].apply_async(verify_checkpoint, mesh_file, function_file, element)
    query.wait()
    assert query.successful(), query.error
    print("".join(query.stdout))
