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
import numpy.typing as npt

from adios4dolfinx.comm_helpers import send_dofs_and_recv_values
from adios4dolfinx.structures import FunctionData, MeshData, MeshTagsData, ReadMeshData
from adios4dolfinx.utils import check_file_exists, compute_local_range, index_owner

from .. import FileMode, ReadMode

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


def read_attributes(
    self,
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read attributes from file.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attribute group
        backend_args: Arguments to backend

    Returns:
        Dictionary of attributes read from file
    """
    raise NotImplementedError("The XDMF backend cannot read attributes.")


def read_timestamps(
    self,
    filename: Path | str,
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64]:
    """Read timestamps from file.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        function_name: Name of the function to read timestamps for
        backend_args: Arguments to backend

    Returns:
        Numpy array of timestamps read from file
    """
    raise NotImplementedError("The XDMF backend cannot read timestamps.")


def write_mesh(
    self,
    filename: Path | str,
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None,
    mode: FileMode,
    time: float,
):
    """
    Write a mesh to file.

    Parameters:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        backend_args: Arguments to backend
        mode: File-mode to store the mesh
        time: Time stamp associated with the mesh
    """
    raise NotImplementedError("The XDMF backend cannot write meshes.")


def write_meshtags(
    self,
    filename: str | Path,
    comm: MPI.Intracomm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None,
):
    """Write mesh tags to file.

    Parameters:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        data: Internal data structure for the mesh tags to save to file
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The XDMF backend cannot write meshtags.")


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> MeshTagsData:
    """Read mesh tags from file.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the mesh tags to read
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh tags read from file
    """
    raise NotImplementedError("The XDMF backend cannot read meshtags.")


def read_dofmap(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dolfinx.graph.AdjacencyList:
    """Read the dofmap of a function with a given name.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofmap for
        backend_args: Arguments to backend

    Returns:
        Dofmap as an AdjacencyList
    """
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


def read_dofs(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    time: float,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
    """Read the dofs (values) of a function with a given name from a given timestep.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofs for
        time: Time stamp associated with the function to read
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of degrees of freedom (with respect to input data)
        and the global starting point on the process.
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


def read_cell_perms(
    comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None
) -> npt.NDArray[np.uint32]:
    """
    Read cell permutation from file with given communicator,
    Split in continuous chunks based on number of cells in the input data.

    Parameters:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of permutations (with respect to input data)
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


def write_function(
    filename: Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """
    Write a function to file.

    Parameters:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        time: Time stamp associated with function
        mode: File-mode to store the function
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Intracomm, group: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], str | None]:
    """Read in the mesh topology, geometry and (optionally) cell type from a
    legacy DOLFIN HDF5-file.

    Parameters:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        group: Group in HDF5 file where mesh is stored

    Returns:
        Tuple containing:
            - Topology as a (num_cells, num_vertices_per_cell) array of global vertex indices
            - Geometry as a (num_vertices, geometric_dimension) array of vertex coordinates
            - Cell type as a string (e.g. "tetrahedron") or None if not found
    """
    raise NotImplementedError("The XDMF backend cannot read legacy DOLFIN meshes.")


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    """Create a snapshot checkpoint of a dolfinx function.

    Parameters:
        filename: Path to file to read from
        mode: File-mode to store the function
        u: dolfinx function to create a snapshot checkpoint for
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


def read_hdf5_array(
    comm: MPI.Intracomm,
    filename: Path | str,
    group: str,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read an array from an HDF5 file.

    Parameters:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        group: Group in HDF5 file where array is stored
        backend_args: Arguments to backend

    Returns:
        Tuple containing:
            - Numpy array read from file
            - Global starting point on the process.
                Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The XDMF backend cannot read HDF5 arrays")
