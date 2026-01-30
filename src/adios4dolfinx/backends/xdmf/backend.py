"""
Module that uses DOLFINx/H%py to import XDMF files.
"""

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from mpi4py import MPI

import basix
import dolfinx
import numpy as np

from adios4dolfinx.comm_helpers import send_dofs_and_recv_values
from adios4dolfinx.structures import ReadMeshData
from adios4dolfinx.utils import check_file_exists, compute_local_range, index_owner

from .. import ReadMode

read_mode = ReadMode.parallel


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Parameters:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {}
    return args


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: float,
    read_from_partition: bool,
    backend_args: dict[str, Any] | None,
) -> ReadMeshData:
    """Read mesh data from file.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh data read from file
    """
    check_file_exists(filename)
    with dolfinx.io.XDMFFile(comm, filename, "r") as file:
        cell_shape, cell_degree = file.read_cell_type()
        cells = file.read_topology_data()
        x = file.read_geometry_data()
    return ReadMeshData(
        cells=cells,
        cell_type=cell_shape.name,
        x=x,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=cell_degree,
    )


def read_point_data(
    filename: Path | str, name: str, mesh: dolfinx.mesh.Mesh
) -> dolfinx.fem.Function:
    # Find function with name u in xml tree
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    func_node = root.find(".//Attribute[@Name='u']")
    data_node = func_node.find(".//DataItem")
    global_shape = data_node.attrib["Dimensions"].split(" ")
    func_path = data_node.text
    data_file, data_loc = func_path.split(":")
    data_path = filename.parent / data_file
    import h5py

    if h5py.h5.get_config().mpi:
        h5file = h5py.File(data_path, "r", driver="mpio", comm=mesh.comm)
    data = h5file[data_loc]

    for s1, s2 in zip(data.shape, global_shape, strict=True):
        assert int(s1) == int(s2)
    lr = compute_local_range(mesh.comm, data.shape[0])
    local_range_start = lr[0]
    dataset = data[slice(*lr), :]
    num_components = dataset.shape[1]
    h5file.close()

    # NOTE: THe below should be moved out of backend.

    # Create appropriate function space (based on coordinate map)
    if num_components == 1:
        shape = ()
    else:
        shape = (num_components,)
    element = basix.ufl.element(
        basix.ElementFamily.P,
        mesh.topology.cell_name(),
        mesh.geometry.cmap.degree,
        mesh.geometry.cmap.variant,
        shape=shape,
        dtype=mesh.geometry.x.dtype,
    )

    # Assumption: Same doflayout for geometry and function space, cannot test in python
    V = dolfinx.fem.functionspace(mesh, element)
    uh = dolfinx.fem.Function(V, name=name, dtype=dataset.dtype)
    # Assume that mesh is first order for now
    x_dofmap = mesh.geometry.dofmap
    igi = np.array(mesh.geometry.input_global_indices, dtype=np.int64)

    # This is dependent on how the data is read in. If distributed equally this is correct
    global_geom_input = igi[x_dofmap]
    from adios4dolfinx.backends import get_backend

    backend_cls = get_backend("xdmf")
    if backend_cls.read_mode == ReadMode.parallel:
        num_nodes_global = mesh.geometry.index_map().size_global
        global_geom_owner = index_owner(mesh.comm, global_geom_input.reshape(-1), num_nodes_global)
    elif backend_cls.read_mode == ReadMode.serial:
        # This is correct if everything is read in on rank 0
        global_geom_owner = np.zeros(len(global_geom_input.flatten()), dtype=np.int32)
    else:
        raise NotImplementedError(f"{backend_cls.read_mode} not implemented")

    for i in range(num_components):
        arr_i = send_dofs_and_recv_values(
            global_geom_input.reshape(-1),
            global_geom_owner,
            mesh.comm,
            dataset[:, i],
            local_range_start,
        )
        dof_pos = x_dofmap.reshape(-1) * num_components + i
        uh.x.array[dof_pos] = arr_i

    return uh
