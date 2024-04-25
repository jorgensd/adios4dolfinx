# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of adios4dolfinx
#
# SPDX-License-Identifier:    MIT

from pathlib import Path

import adios2
import dolfinx

from .adios2_helpers import ADIOSFile, resolve_adios_scope

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

    if mode not in [adios2.Mode.Write, adios2.Mode.Read]:
        raise ValueError("Got invalid mode {mode}")
    # Create ADIOS IO
    adios = adios2.ADIOS(uh.function_space.mesh.comm)
    with ADIOSFile(
        adios=adios,
        filename=file,
        mode=mode,
        io_name="SnapshotCheckPoint",
        engine="BP4",
    ) as adios_file:
        if mode == adios2.Mode.Write:
            dofmap = uh.function_space.dofmap
            num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
            local_dofs = uh.x.array[:num_dofs_local].copy()

            # Write to file
            adios_file.file.BeginStep()
            dofs = adios_file.io.DefineVariable("dofs", local_dofs, count=[num_dofs_local])
            adios_file.file.Put(dofs, local_dofs, adios2.Mode.Sync)
            adios_file.file.EndStep()
        else:
            adios_file.file.BeginStep()
            in_variable = adios_file.io.InquireVariable("dofs")
            in_variable.SetBlockSelection(uh.function_space.mesh.comm.rank)
            adios_file.file.Get(in_variable, uh.x.array, adios2.Mode.Sync)
            adios_file.file.EndStep()
            uh.x.scatter_forward()
