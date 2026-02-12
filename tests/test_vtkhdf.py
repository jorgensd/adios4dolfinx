from mpi4py import MPI

import numpy as np
import pytest
import ufl
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.io.vtkhdf import write_cell_data, write_mesh, write_point_data
from dolfinx.mesh import CellType, compute_midpoints, create_unit_cube, create_unit_square, meshtags

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
    filename = tmp_path / f"timedep_mesh_{cell_type.name}_{np.dtype(dtype).name}.vtkhdf"
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
    filename = tmp_path / f"point_data_{cell_type.name}_{np.dtype(dtype).name}.vtkhdf"
    write_mesh(str(filename), mesh)
    t = np.linspace(0.1, 1.2, 25)
    num_nodes_local = mesh.geometry.index_map().size_local
    for ti in t:
        point_data = f(mesh.geometry.x.T[:, :num_nodes_local], ti)
        write_point_data(str(filename), mesh, point_data, float(ti))
    comm.barrier()

    grid = adios4dolfinx.read_mesh(filename=filename, comm=comm, time=None, backend="vtkhdf")
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
    filename = tmp_path / f"cell_data_{cell_type.name}_{np.dtype(dtype).name}.vtkhdf"
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

    grid = adios4dolfinx.read_mesh(filename=filename, comm=comm, time=None, backend="vtkhdf")
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
def test_write_meshtags(dtype, tmp_path, generate_reference_map):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    filename = tmp_path / f"meshtags_{np.dtype(dtype).name}.vtkhdf"

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
    cmap = mesh.topology.index_map(dim)
    cells = np.arange(cmap.size_local, dtype=np.int32)
    ct = meshtags(mesh, dim, cells, cells + cmap.local_range[0])
    adios4dolfinx.write_meshtags(
        filename, mesh, ct, "CellTags", backend_args={"name": "hex"}, backend="vtkhdf"
    )
    root = 0

    org_map = generate_reference_map(mesh, ct, comm, root)
    # Move mesh
    mesh.geometry.x[:, 0] *= 1 + 0.2 * np.sin(mesh.geometry.x[:, 1])

    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.append,
        time=2.5,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )

    # Add stationary meshtags (after time loop)
    mesh = create_unit_cube(comm, 7, 3, 5, dtype=dtype, cell_type=CellType.tetrahedron)
    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.append,
        time=1.0,
        backend_args={"name": "tet"},
        backend="vtkhdf",
    )
    dim = mesh.topology.dim
    org_maps = {}
    mesh.geometry.x[:, 0] *= 2.0 + mesh.geometry.x[:, 1]
    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.append,
        time=2.5,
        backend_args={"name": "tet"},
        backend="vtkhdf",
    )
    for dim in range(mesh.topology.dim + 1):
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        entities = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
        et = meshtags(mesh, dim, entities, entities + mesh.topology.index_map(dim).local_range[0])
        adios4dolfinx.write_meshtags(
            filename, mesh, et, f"{dim}tags", backend_args={"name": "tet"}, backend="vtkhdf"
        )
        org_maps[dim] = generate_reference_map(mesh, et, comm, root)

    tol = 10 * np.finfo(dtype).eps
    # Read in hex grid from second time step
    hex_mesh = adios4dolfinx.read_mesh(
        filename, comm, time=1.0, backend_args={"name": "hex"}, backend="vtkhdf"
    )
    hex_tag = adios4dolfinx.read_meshtags(
        filename, hex_mesh, "CellTags", backend_args={"name": "hex"}, backend="vtkhdf"
    )
    read_map = generate_reference_map(hex_mesh, hex_tag, comm, root)
    # On root process, check that midpoints are the same for each value in the meshtag
    if MPI.COMM_WORLD.rank == root:
        assert len(org_map) == len(read_map)
        for value, (_, midpoint) in org_map.items():
            _, read_midpoint = read_map[value]
            np.testing.assert_allclose(read_midpoint, midpoint, atol=tol)

    # Read tet grid from second time step
    tet_mesh = adios4dolfinx.read_mesh(
        filename, comm, time=2.5, backend_args={"name": "tet"}, backend="vtkhdf"
    )
    for dim in range(mesh.topology.dim + 1):
        tet_tag = adios4dolfinx.read_meshtags(
            filename, tet_mesh, f"{dim}tags", backend_args={"name": "tet"}, backend="vtkhdf"
        )
        read_map = generate_reference_map(tet_mesh, tet_tag, comm, root)
        if MPI.COMM_WORLD.rank == root:
            assert len(org_maps[dim]) == len(read_map)
            for value, (_, midpoint) in org_maps[dim].items():
                _, read_midpoint = read_map[value]
                np.testing.assert_allclose(read_midpoint, midpoint, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_pointdata(dtype, tmp_path):
    tol = 10 * np.finfo(dtype).eps

    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    filename = tmp_path / "point_data.vtkhdf"

    mesh = create_unit_cube(comm, 3, 3, 3, dtype=dtype, cell_type=CellType.hexahedron)

    def f(x, t):
        return (x[0] + np.sin(x[1]) + np.cos(x[0] * t), x[2] + x[1] - t)

    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.write,
        time=1.0,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )

    f_name = "point_data"
    V = functionspace(mesh, ("Lagrange", 2, (2,)))
    u = Function(V, dtype=dtype, name=f_name)
    u.interpolate(lambda x: f(x, 1.0))
    adios4dolfinx.write_point_data(
        filename,
        u,
        mode=adios4dolfinx.FileMode.append,
        time=1.0,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )

    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.append,
        time=2.0,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )
    u.interpolate(lambda x: f(x, 2.0))
    adios4dolfinx.write_point_data(
        filename,
        u,
        mode=adios4dolfinx.FileMode.append,
        time=2.0,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )

    # Read in hex grid from second time step
    hex_mesh = adios4dolfinx.read_mesh(
        filename, comm, time=2.0, backend_args={"name": "hex"}, backend="vtkhdf"
    )
    u_end = adios4dolfinx.read_point_data(
        filename,
        mesh=hex_mesh,
        name=f_name,
        time=2.0,
        backend_args={"name": "hex"},
        backend="vtkhdf",
    )
    u_ref = Function(u_end.function_space, dtype=dtype)
    u_ref.interpolate(lambda x: f(x, 2.0))
    np.testing.assert_allclose(u_end.x.array, u_ref.x.array, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_celldata(dtype, tmp_path):
    tol = 10 * np.finfo(dtype).eps

    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    filename = tmp_path / "cell_data.vtkhdf"

    mesh = create_unit_cube(comm, 3, 3, 3, dtype=dtype, cell_type=CellType.tetrahedron)

    def f(x, t):
        return (x[0] + np.sin(x[1]) + np.cos(x[0] * t), x[1] + t, x[0] - t)

    t_0 = 2.2
    t_1 = 3.0

    backend_args = {"name": "Grid"}
    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.write,
        time=t_0,
        backend_args=backend_args,
        backend="vtkhdf",
    )

    f_name = "Data"
    V = functionspace(mesh, ("DG", 0, (3,)))
    u = Function(V, dtype=dtype, name=f_name)
    u.interpolate(lambda x: f(x, t_0))
    adios4dolfinx.write_cell_data(
        filename,
        u,
        mode=adios4dolfinx.FileMode.append,
        time=t_0,
        backend_args=backend_args,
        backend="vtkhdf",
    )

    adios4dolfinx.write_mesh(
        filename,
        mesh,
        mode=adios4dolfinx.FileMode.append,
        time=t_1,
        backend_args=backend_args,
        backend="vtkhdf",
    )
    u.interpolate(lambda x: f(x, t_1))
    adios4dolfinx.write_cell_data(
        filename,
        u,
        mode=adios4dolfinx.FileMode.append,
        time=t_1,
        backend_args=backend_args,
        backend="vtkhdf",
    )

    for t in [t_1, t_0]:
        grid = adios4dolfinx.read_mesh(
            filename, comm, time=t, backend_args=backend_args, backend="vtkhdf"
        )
        u_end = adios4dolfinx.read_cell_data(
            filename,
            mesh=grid,
            name=f_name,
            time=t,
            backend_args=backend_args,
            backend="vtkhdf",
        )
        u_ref = Function(u_end.function_space, dtype=dtype)
        u_ref.interpolate(lambda x: f(x, t))
        np.testing.assert_allclose(u_end.x.array, u_ref.x.array, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_mix_data(dtype, tmp_path):
    tol = 10 * np.finfo(dtype).eps

    mesh = create_unit_square(MPI.COMM_WORLD, 5, 7, dtype=dtype)

    def f(x, t):
        return x[0] + t * x[1]

    def g(x, t):
        return x[1] - x[0] * t**2

    ts = [0.1, 0.3, 0.4, 0.5]

    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, name="points", dtype=dtype)
    z = Function(V, name="some_other_array", dtype=dtype)
    Q = functionspace(mesh, ("DG", 0))
    q = Function(Q, name="cells", dtype=dtype)
    tmp_path = mesh.comm.bcast(tmp_path, root=0)
    filename = tmp_path / "mixed_data.vtkhdf"

    backend_args = {"name": "MyGrid"}
    for i, t in enumerate(ts):
        if np.isclose(t, 0.1):
            mode = adios4dolfinx.FileMode.write
        else:
            mode = adios4dolfinx.FileMode.append
        mesh.geometry.x[:] *= 1 + 0.1 * t
        adios4dolfinx.write_mesh(
            filename, mesh, mode=mode, time=t, backend="vtkhdf", backend_args=backend_args
        )
        u.interpolate(lambda x: f(x, t))
        q.interpolate(lambda x: g(x, t))
        adios4dolfinx.write_point_data(
            filename,
            u,
            time=t,
            mode=adios4dolfinx.FileMode.append,
            backend_args=backend_args,
            backend="vtkhdf",
        )
        adios4dolfinx.write_point_data(
            filename,
            z,
            time=t,
            mode=adios4dolfinx.FileMode.append,
            backend_args=backend_args,
            backend="vtkhdf",
        )

        mesh_in = adios4dolfinx.read_mesh(
            filename, MPI.COMM_WORLD, time=t, backend_args=backend_args, backend="vtkhdf"
        )
        u_in = adios4dolfinx.read_point_data(
            filename, name=u.name, mesh=mesh_in, time=t, backend_args=backend_args, backend="vtkhdf"
        )
        u_ref = Function(u_in.function_space, dtype=dtype)
        u_ref.interpolate(lambda x: f(x, t))
        np.testing.assert_allclose(u_ref.x.array, u_in.x.array, atol=tol)
        c_step = i
        if not np.isclose(t, 0.3):
            adios4dolfinx.write_cell_data(
                filename,
                q,
                time=t,
                backend_args=backend_args,
                mode=adios4dolfinx.FileMode.append,
                backend="vtkhdf",
            )
        else:
            # Read in mesh from previous step as geometry adapts,
            # while reading data from current step.
            c_step = i - 1
            mesh_in = adios4dolfinx.read_mesh(
                filename,
                MPI.COMM_WORLD,
                time=ts[c_step],
                backend_args=backend_args,
                backend="vtkhdf",
            )
        q_in = adios4dolfinx.read_cell_data(
            filename, name=q.name, mesh=mesh_in, time=t, backend_args=backend_args, backend="vtkhdf"
        )
        q_ref = Function(q_in.function_space, dtype=dtype)
        q_ref.interpolate(lambda x: g(x, ts[c_step]))
        np.testing.assert_allclose(q_ref.x.array, q_in.x.array, atol=tol)
