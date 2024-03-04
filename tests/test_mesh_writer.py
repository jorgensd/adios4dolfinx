from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from adios4dolfinx import read_mesh, write_mesh
from adios4dolfinx.adios2_helpers import adios2


@pytest.mark.parametrize("encoder, suffix", [("BP4", ".bp"), ("HDF5", ".h5"), ("BP5", ".bp")])
@pytest.mark.parametrize(
    "ghost_mode", [dolfinx.mesh.GhostMode.shared_facet, dolfinx.mesh.GhostMode.none]
)
@pytest.mark.parametrize("store_partition", [True, False])
def test_mesh_read_writer(encoder, suffix, ghost_mode, tmp_path, store_partition):
    N = 7
    # Consistent tmp dir across processes
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / f"adios_mesh_{encoder}_{store_partition}"
    xdmf_file = fname / "xdmf_mesh_{encode}_{ghost_mode}_{store_partition}"
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=ghost_mode)

    write_mesh(file.with_suffix(suffix), mesh, encoder, store_partition_info=store_partition)
    mesh.comm.Barrier()
    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
    mesh.comm.Barrier()

    mesh_adios = read_mesh(
        file.with_suffix(suffix),
        MPI.COMM_WORLD,
        engine=encoder,
        ghost_mode=ghost_mode,
        read_from_partition=store_partition,
    )
    mesh_adios.comm.Barrier()
    if store_partition:

        def compute_distance_matrix(points_A, points_B, tol=1e-12):
            points_A_e = np.expand_dims(points_A, 1)
            points_B_e = np.expand_dims(points_B, 0)
            distances = np.sum(np.square(points_A_e - points_B_e), axis=2)
            return distances < tol

        cell_map = mesh.topology.index_map(mesh.topology.dim)
        new_cell_map = mesh_adios.topology.index_map(mesh_adios.topology.dim)
        assert cell_map.size_local == new_cell_map.size_local
        assert cell_map.num_ghosts == new_cell_map.num_ghosts
        midpoints = dolfinx.mesh.compute_midpoints(
            mesh,
            mesh.topology.dim,
            np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32),
        )
        new_midpoints = dolfinx.mesh.compute_midpoints(
            mesh_adios,
            mesh_adios.topology.dim,
            np.arange(new_cell_map.size_local + new_cell_map.num_ghosts, dtype=np.int32),
        )
        # Check that all points in owned by initial mesh is owned by the new mesh
        # (might be locally reordered)
        owned_distances = compute_distance_matrix(
            midpoints[: cell_map.size_local], new_midpoints[: new_cell_map.size_local]
        )
        np.testing.assert_allclose(np.sum(owned_distances, axis=1), 1)
        # Check that all points that are ghosted in original mesh is ghosted on the
        # same process in the new mesh
        ghost_distances = compute_distance_matrix(
            midpoints[cell_map.size_local :], new_midpoints[new_cell_map.size_local :]
        )
        np.testing.assert_allclose(np.sum(ghost_distances, axis=1), 1)

    mesh.comm.Barrier()

    with dolfinx.io.XDMFFile(mesh.comm, xdmf_file.with_suffix(".xdmf"), "r") as xdmf:
        mesh_xdmf = xdmf.read_mesh(ghost_mode=ghost_mode)

    for i in range(mesh.topology.dim + 1):
        mesh.topology.create_entities(i)
        mesh_xdmf.topology.create_entities(i)
        mesh_adios.topology.create_entities(i)
        assert (
            mesh_xdmf.topology.index_map(i).size_global
            == mesh_adios.topology.index_map(i).size_global
        )

    # Check that integration over different entities are consistent
    measures = (
        [ufl.ds, ufl.dx] if ghost_mode is dolfinx.mesh.GhostMode.none else [ufl.ds, ufl.dS, ufl.dx]
    )
    for measure in measures:
        c_adios = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_adios)))
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        c_xdmf = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_xdmf)))
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_xdmf, MPI.SUM),
        )
        assert np.isclose(
            mesh_adios.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )


@pytest.mark.parametrize("encoder, suffix", [("BP4", ".bp"), ("BP5", ".bp")])
@pytest.mark.parametrize(
    "ghost_mode", [dolfinx.mesh.GhostMode.shared_facet, dolfinx.mesh.GhostMode.none]
)
@pytest.mark.parametrize("store_partition", [True, False])
def test_timedep_mesh(encoder, suffix, ghost_mode, tmp_path, store_partition):
    # Currently unsupported, unclear why ("HDF5", ".h5"),
    N = 13
    # Consistent tmp dir across processes
    fname = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    file = fname / f"adios_time_dep_mesh_{encoder}"
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=ghost_mode)

    def u(x):
        return np.asarray([x[0] + 0.1 * np.sin(x[1]), 0.2 * np.cos(x[1]), x[2]])

    write_mesh(
        file.with_suffix(suffix),
        mesh,
        encoder,
        mode=adios2.Mode.Write,
        time=0.0,
        store_partition_info=store_partition,
    )
    delta_x = u(mesh.geometry.x.T).T
    mesh.geometry.x[:] += delta_x
    write_mesh(file.with_suffix(suffix), mesh, encoder, mode=adios2.Mode.Append, time=3.0)
    mesh.geometry.x[:] -= delta_x

    mesh_first = read_mesh(
        file.with_suffix(suffix),
        MPI.COMM_WORLD,
        encoder,
        ghost_mode,
        time=0.0,
        read_from_partition=store_partition,
    )
    mesh_first.comm.Barrier()

    # Check that integration over different entities are consistent
    measures = [ufl.ds, ufl.dx]
    if ghost_mode == dolfinx.mesh.GhostMode.shared_facet:
        measures.append(ufl.dx)
    for measure in measures:
        c_adios = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_first)))
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        assert np.isclose(
            mesh_first.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )

    mesh.geometry.x[:] += delta_x
    mesh_second = read_mesh(
        file.with_suffix(suffix),
        MPI.COMM_WORLD,
        encoder,
        ghost_mode,
        time=3.0,
        read_from_partition=store_partition,
    )
    mesh_second.comm.Barrier()
    measures = [ufl.ds, ufl.dx]
    if ghost_mode == dolfinx.mesh.GhostMode.shared_facet:
        measures.append(ufl.dx)
    for measure in measures:
        c_adios = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh_second)))
        c_ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * measure(domain=mesh)))
        assert np.isclose(
            mesh_second.comm.allreduce(c_adios, MPI.SUM),
            mesh.comm.allreduce(c_ref, MPI.SUM),
        )
