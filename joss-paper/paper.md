---
title: 'ADIOS4DOLFINx: A framework for checkpointing in FEniCS'
tags:
  - Python
  - finite element simulations
  - checkpointing
authors:
  - name: Jørgen Schartum Dokken
    orcid: 0000-0001-6489-8858
    corresponding: true
    affiliation: 1
affiliations:
 - name: Simula Research Laboratory
   index: 1
date: 6 March 2024
bibliography: paper.bib

---

# Summary

We introduce a checkpointing framework for the latest version of the FEniCS project, known as DOLFINx.
The framework leverages the data-centric approach of DOLFINx along with a state of the art adaptable Input/Output system called ADIOS2.
Several variations of checkpointing are supported, including *N-to-M* checkpointing of function data, storage of mesh partitioning information for N-to-N checkpointing and snapshot checkpointing for RAM reduction during simulation. All MPI operations are using MPI-3 Neighborhood collectives.

# Statement of need

The ability to start, stop and resume simulations is becoming increasingly important with the growing use of supercomputers for solving scientific and engineering problems.
A rising number of large scale problems are deployed on high performance, memory distributed computing systems and users tend to run more demanding simulations.
These are often non-linear and time-dependent, which typically amounts to thousands of CPU hours.
As it might uncover bugs and unphysical solutions, the ability to run parts of the simulation, inspect the result and then resume simulation becomes a key factor to enable efficient development.
If this is discovered early on, the simulation can be terminated saving the developer time, money and energy-usage.

The proposed framework enables users of the FEniCS project [@Baratta:2023] to store solutions during simulation, and read them in at their convenience to resume simulations at a later stage.
Several checkpointing methods are implemented, including *N-to-M* checkpointing, which means saving data from a program executed with N processes, and loading it back in on M processes.

Functionality for *N-to-M* checkpointing was implemented for the old version of DOLFIN by [@Habera:2018].
However, this functionality is not present in the newest version of the FEniCS Project [@Baratta:2023].
The storage principles in the ADIOS4DOLFINx are based on the ideas present in this implementation.
However, the implementation for non-Lagrangian finite element spaces vastly differs, due to the usage of dof-permutations [@Scroggs:2022].
Additionally, all global MPI-calls in the old implementation have been reimplemented with scalable MPI-communication using the MPI-3 Neighborhood Collectives [@MPI-Forum:2012].

The framework also extends the checkpointing functionality with special routines for storing partitioning information for *N-to-N* checkpointing, as well as very lightweight snapshot checkpoints.
The difference in  implementation on non-Lagrangian spaces also distinguishes this framework from *N-to-M* checkpointing in Firedrake [@Rathgeber:2016; @Ham:2024].

# Functionality

The software as written as a Python-extension to DOLFINx, which can be installed using the Python Package installer `pip` directly from the Github repository or using the [ADIOS4DOLFINx - Python Package Index](https://pypi.org/project/adios4dolfinx/).

The following features are supported:

- Snapshot checkpointing
- *N-to-M* checkpointing with mesh storage 
- *N-to-M* checkpointing without mesh storage
- *N-to-N* checkpointing storing partitioning information

A *snapshot checkpoint* is a checkpoint that is only valid during the run of a simulation.
It is lightweight (only stores the local portion of the global dof array to file), and is stored using the *Local Array* feature in ADIOS2 [@Godoy:2020] to store data local to the MPI process.
This feature is intended for use-cases where many solutions have to be aggregated to the end of a simulation to some post-processing step, or as a fall-back mechanism when restarting a divergence iterative solver.

A *N-to-M* checkpoint is a checkpoint that can be written with N processes and read back in with M processes.
Two versions of this checkpoint is supported; One where storage of the mesh is required and without mesh storage.
The reasoning for such a split is that when a mesh is read into DOLFINx and passed to an appropriate partitioner, the ordering mesh nodes (coordinates) and connectivity (cells) is changed.
Writing these back into *global arrays* requires MPI communication to ensure contiguous writing of data.

THe *N-to-M* checkpoint with mesh storage exclusively write contiguous chunks of data owned by the current process, to an ADIOS2 *Global Array* that can be read in with a different number of processes at a later stage.
This operation requires no MPI-communication for writing such checkpoints.

In many cases, the input mesh might stem from an external mesh generator and is stored together with mesh entity markers in an external file, for instance an XDMF-file.
To avoid duplication of this mesh data, a stand-alone file that can be associated with the XDMF file for a later restart can be created.
This method requires some MPI neighborhood collective calls to move data from the process that currently owns it, to the relevant process for writing it out into a *Global Array* in contiguous chunks.
Both *N-to-M* checkpoint routines uses the `read_checkpoint` function to read in a function checkpoint.

In certain scenarios, mesh partitioning might be time-consuming, as a developer is running the same problem over and over again with the same number of processes.
As DOLFINx supports custom partitioning [@Baratta:2023], we use this feature to read in partition data from a previous run.
As opposed to the checkpoints in the old version of DOLFIN, these checkpoints handle any ghosting, that being a custom ghosting provided by the user, or the shared-facet mode in DOLFINx.

# Examples
A large variety of examples covering all the functions in adios4dolfinx is available at [https://jorgensd.github.io/adios4dolfinx](https://jorgensd.github.io/adios4dolfinx).

# Acknowledgements

We acknowledge the valuable feedback on the documentation and manuscript by Thomas M. Surowiec and Halvor Herlyng. 
Additionally, we acknowledge the scientific discussion regarding feature development and code contributions by Henrik N. Finsberg and Francesco Ballarin.


# References