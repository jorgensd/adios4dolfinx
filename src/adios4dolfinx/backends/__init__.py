from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, Union

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

from ..structures import MeshData

__all__ = ["FileMode", "IOBackend", "get_backend"]


class FileMode(Enum):
    append = 10
    write = 20
    read = 30


# See https://peps.python.org/pep-0544/#modules-as-implementations-of-protocols
class IOBackend(Protocol):
    def get_default_backend_args(self, arguments: dict[str, Any] | None) -> dict[str, Any]: ...

    def write_attributes(
        self,
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        attributes: dict[str, np.ndarray],
        backend_args: dict[str, Any] | None,
    ): ...

    def read_attributes(
        self,
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        backend_args: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    def read_timestamps(
        self,
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        function_name: str,
        backend_args: dict[str, Any] | None,
    ) -> npt.NDArray[np.float64]: ...

    def write_mesh(
        self,
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        mesh: MeshData,
        backend_args: dict[str, Any] | None = None,
        mode: FileMode = FileMode.write,
        time: float = 0.0,
    ):
        """
        Write a mesh to file.

        Parameters:
            comm: MPI communicator used in storage
            mesh: Internal data structure for the mesh data to save to file
            filename: Path to file to write to
            backend_args: Arguments to backend
            mode: File-mode to store the mesh
        """
        ...

    # read_function
    # read_mesh
    # read_meshtags
    # read_timestamps
    # write_function
    # write_meshtags
    # read_function_from_legacy_h5
    # read_mesh_from_legacy_h5
    # write_function_on_input_mesh
    # write_mesh_input_order
    # snapshot_checkpoint


def get_backend(backend: Literal["h5py", "adios2"]) -> IOBackend:
    if backend == "h5py":
        from .h5py import backend as H5PYInterface

        return H5PYInterface
    elif backend == "adios2":
        from .adios2 import backend as ADIOS2Interface

        return ADIOS2Interface
    else:
        raise NotImplementedError(f"Backend: {backend} not implemented")
