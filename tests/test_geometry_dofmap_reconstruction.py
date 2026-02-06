from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pytest

from adios4dolfinx.readers import create_geometry_function_space


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("value_shape", [(), (1,), (4,)])
@pytest.mark.parametrize("N", [3, 20])
@pytest.mark.parametrize("M", [8, 9])
@pytest.mark.parametrize(
    "ghost_mode", [dolfinx.mesh.GhostMode.shared_facet, dolfinx.mesh.GhostMode.none]
)
def test_dofmap_construction(cell_type, value_shape, N, M, ghost_mode):
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, N, M, cell_type=cell_type, ghost_mode=ghost_mode
    )
    el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    if value_shape == ():
        b_el = el
    else:
        b_el = basix.ufl.blocked_element(el, value_shape)

    MPI.COMM_WORLD.barrier()

    V_ref = dolfinx.fem.functionspace(mesh, b_el)

    MPI.COMM_WORLD.barrier()

    bs = int(np.prod(value_shape))
    V = create_geometry_function_space(mesh, bs)
    assert V.dofmap.bs == V_ref.dofmap.bs
    assert V.dofmap.bs == bs
    np.testing.assert_allclose(V.dofmap.list, V_ref.dofmap.list)
