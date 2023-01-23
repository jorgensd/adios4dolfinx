"""Top-level package for ADIOS2Wrappers."""
from .checkpointing import write_mesh, read_mesh_from_legacy_checkpoint
from importlib.metadata import metadata

meta = metadata("adios4dolfinx")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["write_mesh", "read_mesh_from_legacy_checkpoint"]
