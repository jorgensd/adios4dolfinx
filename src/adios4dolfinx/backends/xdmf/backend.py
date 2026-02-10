"""
Module that uses DOLFINx/H%py to import XDMF files.
"""

import contextlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from adios4dolfinx.structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from adios4dolfinx.utils import check_file_exists, compute_local_range

from .. import FileMode, ReadMode

read_mode = ReadMode.parallel


@contextlib.contextmanager
def h5pyfile(h5name, filemode="r", force_serial: bool = False, comm=None):
    """Context manager for opening an HDF5 file with h5py.

    Args:
        h5name: The name of the HDF5 file.
        filemode: The file mode.
        force_serial: Force serial access to the file.
        comm: The MPI communicator

    """
    import h5py

    if comm is None:
        comm = MPI.COMM_WORLD

    if h5py.h5.get_config().mpi and not force_serial:
        h5file = h5py.File(h5name, filemode, driver="mpio", comm=comm)
    else:
        if comm.size > 1 and not force_serial:
            raise ValueError(
                f"h5py is not installed with MPI support, while using {comm.size} processes.",
                "If you really want to do this, turn on the `force_serial` flag.",
            )
        h5file = h5py.File(h5name, filemode)

    try:
        yield h5file
    finally:
        if h5file.id:
            h5file.close()


def extract_function_names_and_timesteps(filename: Path | str) -> dict[str, list[str]]:
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    mesh_nodes = root.findall(".//Grid[@CollectionType='Temporal']")
    function_names = []
    for mesh in mesh_nodes:
        function_names.append(mesh.attrib["Name"])

    time_stamps: dict[str, list[str]] = {name: [] for name in function_names}
    for name in function_names:
        time_steps = root.findall(f".//Grid[@Name='{name}']")
        for time in time_steps:
            step = time.find(".//Time")
            if step is not None:
                val = step.attrib["Value"]
                time_stamps[name].append(val)
    for name in function_names:
        float_steps = np.argsort(np.array(list(set(time_stamps[name])), dtype=np.float64))
        time_stamps[name] = np.array(list(set(time_stamps[name])), dtype=str)[float_steps].tolist()
    return time_stamps


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {}
    return args


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: str | float | None,
    read_from_partition: bool,
    backend_args: dict[str, Any] | None,
) -> ReadMeshData:
    """Read mesh data from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh data read from file
    """
    assert not read_from_partition
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
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: float | str | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read data from the nodes of a mesh.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments

    Returns:
       Data local to process (contiguous, no mpi comm) and local start range
    """
    # Find function with name u in xml tree
    check_file_exists(filename)
    filename = Path(filename)

    tree = ElementTree.parse(filename)
    root = tree.getroot()
    backend_args = get_default_backend_args(backend_args)
    if time is not None:
        time_steps = root.findall(f".//Grid[@Name='{name}']")
        time_found = False
        for time_node in time_steps:
            step_node = time_node.find(".//Time")
            assert isinstance(step_node, ElementTree.Element)
            if step_node.attrib["Value"] == time:
                time_found = True
                break
        func_node = time_node.find(f".//Attribute[@Name='{name}']")
        if not time_found:
            raise RuntimeError(f"Function {name} at time={time} not found in {filename}")
    else:
        func_node = root.find(f".//Attribute[@Name='{name}']")
    assert isinstance(func_node, ElementTree.Element)
    data_node = func_node.find(".//DataItem")
    assert isinstance(data_node, ElementTree.Element)
    global_shape = data_node.attrib["Dimensions"].split(" ")
    func_path = data_node.text
    assert isinstance(func_path, str)
    data_file, data_loc = func_path.split(":")
    data_path = filename.parent / data_file
    with h5pyfile(data_path, "r", comm=comm) as h5file:
        data = h5file[data_loc]
        for s1, s2 in zip(data.shape, global_shape, strict=True):
            assert int(s1) == int(s2)
        lr = compute_local_range(comm, data.shape[0])
        local_range_start = lr[0]
        dataset = data[slice(*lr), :]
    return dataset, local_range_start


def read_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read attributes from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attribute group
        backend_args: Arguments to backend

    Returns:
        Dictionary of attributes read from file
    """
    raise NotImplementedError("The XDMF backend cannot read attributes.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read timestamps from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        function_name: Name of the function to read timestamps for
        backend_args: Arguments to backend

    Returns:
        Numpy array of timestamps read from file
    """
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    time_stamps = []
    time_steps = root.findall(f".//Grid[@Name='{function_name}']")
    for time in time_steps:
        step = time.find(".//Time")
        if step is not None:
            val = step.attrib["Value"]
            time_stamps.append(val)
    float_steps = np.argsort(np.array(list(set(time_stamps)), dtype=np.float64))
    return np.array(list(set(time_stamps)), dtype=str)[float_steps].tolist()


def write_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None,
):
    """Write attributes to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attribute group
        attributes: Dictionary of attributes to write
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The XDMF backend cannot write attributes.")


def write_mesh(
    filename: Path | str,
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None,
    mode: FileMode,
    time: float,
):
    """
    Write a mesh to file.

    Args:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        backend_args: Arguments to backend
        mode: File-mode to store the mesh
        time: Time stamp associated with the mesh
    """
    raise NotImplementedError("The XDMF backend cannot write meshes.")


def write_meshtags(
    filename: str | Path,
    comm: MPI.Intracomm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None,
):
    """Write mesh tags to file.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
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


def read_function_names(
    filename: Path | str, comm: MPI.Intracomm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read all function names from a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        A list of function names.
    """
    # Find all functions in xml tree
    check_file_exists(filename)
    filename = Path(filename)

    tree = ElementTree.parse(filename)
    root = tree.getroot()
    backend_args = get_default_backend_args(backend_args)
    # Functions in checkpoint format
    checkpoint_funcs = root.findall(".//Attribute[@ItemType='FiniteElementFunction']")
    names = [func.attrib["Name"] for func in checkpoint_funcs]
    # Temporal funcs
    temporal_funcs = root.findall(".//Grid[@GridType='Collection']")
    for func in temporal_funcs:
        names.append(func.attrib["Name"])
    return list(set(names))


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read data from the cells of a mesh.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments
    Returns:
        A tuple (topology, dofs) where topology contains the
        vertex indices of the cells, dofs the degrees of
        freedom within that cell.
    """
    # Find function with name u in xml tree
    check_file_exists(filename)
    filename = Path(filename)

    tree = ElementTree.parse(filename)
    root = tree.getroot()
    backend_args = get_default_backend_args(backend_args)
    if time is not None:
        time_steps = root.findall(f".//Grid[@Name='{name}']")
        time_found = False
        for time_node in time_steps:
            step_node = time_node.find(".//Time")
            assert isinstance(step_node, ElementTree.Element)
            if np.isclose(float(step_node.attrib["Value"]), time):
                time_found = True
                break
        func_node = time_node.find(f".//Attribute[@Name='{name}']")
        if not time_found:
            raise RuntimeError(f"Function {name} at time={time} not found in {filename}")
    else:
        func_node = root.find(f".//Attribute[@Name='{name}']")
    assert func_node is not None
    if func_node.attrib["ItemType"] == "FiniteElementFunction":
        if (family := func_node.attrib["ElementFamily"]) != "DG" or (
            degree := int(func_node.attrib["ElementDegree"])
        ) != 0:
            raise ValueError(
                f"Cannot read in finite element function ({family}, {degree}) as cell data."
            )
        # Get vector sub-element
        vec_el = None
        for node in func_node.iter():
            comp = node.text
            assert comp is not None
            if "vector" in comp:
                vec_el = node
                break
        assert vec_el is not None
        dof_dimensions = np.array(vec_el.attrib["Dimensions"].split(" "), dtype=np.int32)
        vtxt = vec_el.text
        assert vtxt is not None
        vector_file, vector_h5path = vtxt.split(":")

        grid_node = root.find(f".//Attribute[@Name='{name}']/..")
        assert grid_node is not None
        topology = grid_node.find("./Topology")
        assert topology is not None
        ttext = topology[0].text
        assert ttext is not None
        topology_file, topology_h5path = ttext.split(":")

        with h5pyfile(filename.parent / vector_file, "r", comm=comm) as h5_mesh:
            data_loc = h5_mesh[vector_h5path]
            data_shape = data_loc.shape[0]
            assert int(np.prod(data_shape)) == int(np.prod(dof_dimensions))
            local_range = compute_local_range(comm, data_shape)
            vec_dofs = data_loc[slice(*local_range)]

        with h5pyfile(filename.parent / topology_file, "r", comm=comm) as h5_top:
            data_loc = h5_top[topology_h5path]
            top_data_shape = data_loc.shape[0]
            assert dof_dimensions[0] == top_data_shape
            local_range = compute_local_range(comm, data_shape)
            top_dofs = data_loc[slice(*local_range)].astype(np.int64)

        return top_dofs, vec_dofs
    else:
        raise NotImplementedError("Not implemented yet.")


def write_data(
    filename: Path | str,
    point_data: ArrayData,
    comm: MPI.Intracomm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a 2D-array to file (distributed across proceses with MPI).


    Args:
        filename: Path to file
        point_data: Data to write to file
        time: Time stamp
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("XDMF has not implemented this yet")
