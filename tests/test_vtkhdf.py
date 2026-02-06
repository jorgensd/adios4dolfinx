from mpi4py import MPI

import numpy as np
import pytest
import ufl
from dolfinx.fem import Function, assemble_scalar, form
from dolfinx.io.vtkhdf import write_cell_data, write_mesh, write_point_data
from dolfinx.mesh import CellType, compute_midpoints, create_unit_cube, locate_entities, meshtags

import adios4dolfinx


def f(x, t):
    return x[0] - 2 * x[1] + x[2] * t


def g(x, t):
    return x[0], 2 * x[1], -x[2] * t


@pytest.mark.parametrize(
    "cell_type", [CellType.tetrahedron, CellType.hexahedron, CellType.tetrahedron]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_timedep_mesh(dtype, tmp_path, cell_type):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    # Write temporal data
    mesh = create_unit_cube(comm, 5, 5, 5, dtype=dtype, cell_type=cell_type)
    ref_vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=mesh), dtype=dtype)), op=MPI.SUM
    )
    ref_surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=mesh), dtype=dtype)), op=MPI.SUM
    )

    # Write temporal data
    from pathlib import Path

    tmp_path = Path(".")
    filename = tmp_path / f"timedep_mesh_{cell_type.name}_{dtype}.vtkhdf"
    adios4dolfinx.write_mesh(filename, mesh, time=0.3, backend="vtkhdf")
    mesh.geometry.x[:, 0] += 0.05 * mesh.geometry.x[:, 0]
    mesh.geometry.x[:, 1] *= 1.1 + np.sin(mesh.geometry.x[:, 0])

    ref_pert_vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=mesh), dtype=dtype)), op=MPI.SUM
    )
    ref_pert_surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=mesh), dtype=dtype)), op=MPI.SUM
    )

    adios4dolfinx.write_mesh(
        filename,
        mesh,
        time=0.5,
        backend="vtkhdf",
        mode=adios4dolfinx.FileMode.append,
    )

    in_mesh = adios4dolfinx.read_mesh(filename, comm, time=0.5, backend="vtkhdf")
    pert_vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    pert_surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    assert np.isclose(pert_vol, ref_pert_vol)
    assert np.isclose(pert_surf, ref_pert_surf)

    in_mesh = adios4dolfinx.read_mesh(filename, comm, time=0.3, backend="vtkhdf")
    vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    assert np.isclose(vol, ref_vol)
    assert np.isclose(surf, ref_surf)


@pytest.mark.parametrize("cell_type", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_point_data(dtype, tmp_path, cell_type):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    # Write temporal data
    mesh = create_unit_cube(comm, 5, 5, 5, dtype=dtype, cell_type=cell_type)
    filename = tmp_path / f"point_data_{cell_type.name}.vtkhdf"
    write_mesh(str(filename), mesh)
    t = np.linspace(0.1, 1.2, 25)
    num_nodes_local = mesh.geometry.index_map().size_local
    for ti in t:
        point_data = f(mesh.geometry.x.T[:, :num_nodes_local], ti)
        write_point_data(str(filename), mesh, point_data, float(ti))
    comm.barrier()

    grid = adios4dolfinx.read_mesh(filename=filename, comm=comm, backend="vtkhdf")
    # Since we shuffle time we need to shuffle in the same way on each process
    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tj in t:
        u = adios4dolfinx.read_point_data(
            filename=filename, name="u", mesh=grid, time=tj, backend="vtkhdf"
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 10 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: f(x, tj))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)

    # Test blocked data as well (with shuffled input timestep)
    blocked_file = filename.with_stem(filename.stem + "_blocked")
    write_mesh(str(blocked_file), mesh)
    for tj in t:
        point_data = np.asarray(g(mesh.geometry.x.T[:, :num_nodes_local], tj)).T.flatten()
        write_point_data(str(blocked_file), mesh, point_data, float(tj))
    comm.barrier()

    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tk in t:
        u = adios4dolfinx.read_point_data(
            filename=blocked_file, name="u", mesh=grid, time=tk, backend="vtkhdf"
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 10 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: g(x, tk))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)


@pytest.mark.parametrize("cell_type", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_cell_data(dtype, tmp_path, cell_type):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    # Write temporal data
    mesh = create_unit_cube(comm, 5, 5, 5, dtype=dtype, cell_type=cell_type)
    filename = tmp_path / f"cell_data_{cell_type.name}.vtkhdf"
    write_mesh(str(filename), mesh)

    t = np.linspace(0.1, 1.2, 25)
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    midpoints = compute_midpoints(
        mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32)
    )
    for ti in t:
        cell_data = f(midpoints.T, ti)
        write_cell_data(str(filename), mesh, cell_data, float(ti))
    comm.barrier()

    grid = adios4dolfinx.read_mesh(filename=filename, comm=comm, backend="vtkhdf")
    # Since we shuffle time we need to shuffle in the same way on each process
    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tj in t:
        u = adios4dolfinx.read_cell_data(
            filename=filename, name="u", mesh=grid, time=tj, backend="vtkhdf"
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 10 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: f(x, tj))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)

    # Test blocked data as well (with shuffled input timestep)
    blocked_file = filename.with_stem(filename.stem + "_blocked")
    write_mesh(str(blocked_file), mesh)
    for tj in t:
        cell_data = np.asarray(g(midpoints.T, tj)).T.flatten()
        write_cell_data(str(blocked_file), mesh, cell_data, float(tj))
    comm.barrier()

    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tk in t:
        u = adios4dolfinx.read_cell_data(
            filename=blocked_file, name="u", mesh=grid, time=tk, backend="vtkhdf"
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 10 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: g(x, tk))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_meshtags(dtype, tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()
    from pathlib import Path

    tmp_path = Path("testdata")
    filename = tmp_path / "meshtags.vtkhdf"

    def left_cells(x):
        return x[0] <= 0.5

    mesh = create_unit_cube(comm, 3, 3, 3, dtype=dtype, cell_type=CellType.hexahedron)
    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.write,
        time=1.0,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )
    dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, mesh.topology.dim)
    cells = locate_entities(mesh, dim, left_cells)
    ct = meshtags(mesh, dim, cells, cells)
    mesh.geometry.x[:, 0] *= 1.1 + np.sin(mesh.geometry.x[:, 1])
    adios4dolfinx.write_meshtags(
        filename, mesh, ct, "CellTags", backend_args={"name": "hex"}, backend="vtkhdf"
    )
    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.append,
        time=2.5,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )

    # mesh = create_unit_cube(comm, 7, 3, 5, dtype=dtype, cell_type=CellType.tetrahedron)
    # adios4dolfinx.write_mesh(filename, mesh, mode=adios4dolfinx.FileMode.append, time=1.0, backend_args={"name": "tet"}, backend="vtkhdf")
    # dim = mesh.topology.dim
    # mesh.topology.create_connectivity(dim, mesh.topology.dim)
    # cells = locate_entities(mesh, dim, left_cells)
    # ct = meshtags(mesh, dim, cells, cells)
    # adios4dolfinx.write_meshtags(filename, mesh,ct, "CellTags", backend_args={"name": "tet"}, backend="vtkhdf")
