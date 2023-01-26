# Copyright (C) 2023 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Top-level package for ADIOS2Wrappers."""
from importlib.metadata import metadata

from .checkpointing import read_mesh, write_mesh
from .legacy_readers import (read_mesh_from_legacy_checkpoint,
                             read_mesh_from_legacy_h5)
from .utils import compute_local_range, index_owner

meta = metadata("adios4dolfinx")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["read_mesh", "write_mesh",
           "read_mesh_from_legacy_checkpoint", "read_mesh_from_legacy_h5",
           "compute_local_range", "index_owner"]
