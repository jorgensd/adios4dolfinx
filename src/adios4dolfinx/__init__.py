# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Top-level package for ADIOS2Wrappers."""
from importlib.metadata import metadata

from .checkpointing import (
    read_function,
    read_mesh,
    write_function,
    write_mesh,
    snapshot_checkpoint,
)
from .legacy_readers import (
    read_function_from_legacy_h5,
    read_mesh_from_legacy_h5,
)

meta = metadata("adios4dolfinx")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "read_mesh",
    "write_mesh",
    "read_function_from_legacy_h5",
    "read_mesh_from_legacy_h5",
    "write_function",
    "read_function",
    "snapshot_checkpoint",
]
