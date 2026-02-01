from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt

from ..structures import FunctionData, MeshData, MeshTagsData, ReadMeshData

__all__ = ["FileMode", "IOBackend", "get_backend"]


class ReadMode(Enum):
    serial = 10  # This means that all data is read in on root rank

    # Total number of data P, num processes = i + 1.
    # All processes reads at least `P // (i+1)` items
    # The first j=P%(i+1) processes reads `P // (i+1) + 1` items
    # ```python
    # def compute_partitioning(P, J):
    #     min_num = P // J
    #     num_per_proc = np.full(J, min_num)
    #     rem = P % J
    #     num_per_proc[:int(rem)] += 1
    #     assert(sum(num_per_proc)) == P
    #     return num_per_proc
    # ```
    parallel = 20


class FileMode(Enum):
    """Filen mode used for opening files."""

    append = 10  #: Append data to file
    write = 20  #: Write data to file
    read = 30  #: Read data from file


# See https://peps.python.org/pep-0544/#modules-as-implementations-of-protocols
class IOBackend(Protocol):
    read_mode: ReadMode

    def get_default_backend_args(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Get default backend arguments given a set of input arguments.

        Parameters:
            arguments: Input backend arguments

        Returns:
            Updated backend arguments
        """

    def write_attributes(
        self,
        filename: Path | str,
        comm: MPI.Intracomm,
        name: str,
        attributes: dict[str, np.ndarray],
        backend_args: dict[str, Any] | None,
    ):
        """Write attributes to file.

        Parameters:
            filename: Path to file to write to
            comm: MPI communicator used in storage
            name: Name of the attribute group
            attributes: Dictionary of attributes to write
            backend_args: Arguments to backend
        """

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

    def read_mesh_data(
        self,
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

    def read_meshtags_data(
        self,
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

    def read_dofmap(
        self,
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

    def read_dofs(
        self,
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

    def read_cell_perms(
        self, comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None
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

    def write_function(
        self,
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

    def read_legacy_mesh(
        self, filename: Path | str, comm: MPI.Intracomm, group: str
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

    def snapshot_checkpoint(
        self,
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

    def read_hdf5_array(
        self,
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

    def read_point_data(
        filename: Path | str, name: str, mesh: dolfinx.mesh.Mesh
    ) -> dolfinx.fem.Function:
        """Read data from te nodes of a mesh.

        Parameters:
            filename: Path to file
            name: Name of point data
            mesh: The corresponding :py:class:`dolfinx.mesh.Mesh`.

        Returns:
            A function in the space equivalent to the mesh
            coordinate element (up to shape).
        """
        ...


def get_backend(backend: str) -> IOBackend:
    """Get backend class from backend name.

    Parameters:
        backend: Name of the backend to get

    Returns:
        Backend class
    """
    if backend == "h5py":
        from .h5py import backend as H5PYInterface

        return H5PYInterface
    elif backend == "adios2":
        from .adios2 import backend as ADIOS2Interface

        return ADIOS2Interface
    elif backend == "pyvista":
        from .pyvista import backend as PYVISTAInterface

        return PYVISTAInterface
    elif backend == "xdmf":
        from .xdmf import backend as XDMFInterface

        return XDMFInterface
    else:
        return import_module(backend)
