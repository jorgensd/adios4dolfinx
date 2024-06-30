# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Top-level package for ADIOS2Wrappers."""

from importlib.metadata import metadata

from .checkpointing import (
    read_attributes,
    read_function,
    read_mesh,
    read_meshtags,
    write_attributes,
    write_function,
    write_mesh,
    write_meshtags,
)
from .legacy_readers import read_function_from_legacy_h5, read_mesh_from_legacy_h5
from .original_checkpoint import write_function_on_input_mesh, write_mesh_input_order
from .snapshot import snapshot_checkpoint

meta = metadata("adios4dolfinx")
__version__ = meta["Version"]
try:
    __author__ = meta["Author"]
except KeyError:
    pass
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "write_meshtags",
    "read_meshtags",
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
    "read_attributes",
]
