# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from pathlib import Path

import dolfinx

from .adios2_helpers import resolve_adios_scope

import adios2
adios2 = resolve_adios_scope(adios2)

__all__ = [
    "snapshot_checkpoint",
]


def snapshot_checkpoint(uh: dolfinx.fem.Function, file: Path, mode: adios2.Mode):
    """Read or write a snapshot checkpoint

    This checkpoint is only meant to be used on the same mesh during the same simulation.

    :param uh: The function to write data from or read to
    :param file: The file to write to or read from
    :param mode: Either read or write
    """
    # Create ADIOS IO
    adios = adios2.ADIOS(uh.function_space.mesh.comm)
    io_name = "SnapshotCheckPoint"
    io = adios.DeclareIO(io_name)
    io.SetEngine("BP4")
    if mode not in [adios2.Mode.Write, adios2.Mode.Read]:
        raise ValueError("Got invalid mode {mode}")
    adios_file = io.Open(str(file), mode)

    if mode == adios2.Mode.Write:
        dofmap = uh.function_space.dofmap
        num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
        local_dofs = uh.x.array[:num_dofs_local].copy()

        # Write to file
        adios_file.BeginStep()
        dofs = io.DefineVariable("dofs", local_dofs, count=[num_dofs_local])
        adios_file.Put(dofs, local_dofs, adios2.Mode.Sync)
        adios_file.EndStep()
    else:
        adios_file.BeginStep()
        in_variable = io.InquireVariable("dofs")
        in_variable.SetBlockSelection(uh.function_space.mesh.comm.rank)
        adios_file.Get(in_variable, uh.x.array, adios2.Mode.Sync)
        adios_file.EndStep()
        uh.x.scatter_forward()
    adios_file.Close()
    adios.RemoveIO(io_name)
