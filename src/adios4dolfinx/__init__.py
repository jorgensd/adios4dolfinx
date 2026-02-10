# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Top-level package for ADIOS2Wrappers."""

from importlib.metadata import metadata

from .backends import FileMode, get_backend
from .checkpointing import (
    read_attributes,
    read_function,
    read_function_names,
    read_mesh,
    read_meshtags,
    read_timestamps,
    write_attributes,
    write_cell_data,
    write_function,
    write_mesh,
    write_meshtags,
    write_point_data,
)
from .original_checkpoint import write_function_on_input_mesh, write_mesh_input_order
from .readers import (
    read_cell_data,
    read_function_from_legacy_h5,
    read_mesh_from_legacy_h5,
    read_point_data,
)
from .snapshot import snapshot_checkpoint

meta = metadata("adios4dolfinx")
__version__ = meta["Version"]
__author__ = meta.get("Author", "")
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "FileMode",
    "write_meshtags",
    "read_meshtags",
    "read_cell_data",
    "read_mesh",
    "write_mesh",
    "read_function_from_legacy_h5",
    "read_mesh_from_legacy_h5",
    "write_function",
    "read_function",
    "snapshot_checkpoint",
    "write_function_on_input_mesh",
    "write_mesh_input_order",
    "write_attributes",
    "write_data",
    "read_attributes",
    "read_function_names",
    "read_point_data",
    "read_timestamps",
    "get_backend",
    "write_cell_data",
    "write_point_data",
]
