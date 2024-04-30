---
title: "ADIOS4DOLFINx: A framework for checkpointing in FEniCS"
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
  - name: Simula Research Laboratory, Oslo, Norway
    index: 1
date: 30 April 2024
bibliography: paper.bib
---

# Summary

We introduce ADIOS4DOLFINx, a checkpointing framework for the latest version of the FEniCS project, known as DOLFINx.
DOLFINx is a general framework for solving partial differential equations using the finite element method.
The input to simulations using the finite element method is the computational domain (mesh), mesh markers, initial conditions, and boundary conditions.
To be able to restart a simulation at any point, one has to have the capability to read in all of the aforementioned variables.
The adios4dolfinx package implements all of these operations, using the Message Passing Interface (MPI) for communication across multiple processes and ADIOS2 for writing/reading data to/from file.
In particular, the functionality of adios4dolfinx includes *N-to-M*-checkpointing, which means that one can store a result of a simulation that was generated with N number of processes,
and read it into a program running on M processes.

# Statement of need

The ability to start, stop, and resume simulations is becoming increasingly important with the growing use of supercomputers for solving scientific and engineering problems.
A rising number of large-scale problems are deployed on high-performance, distributed-memory computing systems and users tend to run more demanding simulations.
These are often non-linear and time-dependent, which typically amounts to thousands of CPU hours.
As it might uncover bugs and unphysical solutions, the ability to run parts of the simulation, inspect the result, and then resume simulation becomes a key factor to enable efficient development.
If this is discovered early on, the simulation can be terminated, saving the developer time, money and energy usage.

ADIOS4DOLFINx enables users of the FEniCS project [@Baratta:2023] to store solutions during simulation, and read them in at their convenience to resume simulations at a later stage.
Several checkpointing methods are implemented, including *N-to-M* checkpointing, which means saving data from a program executed with N processes, and loading it back in on M processes.

Functionality for *N-to-M* checkpointing was implemented for the old version of DOLFIN by @Habera:2018.
However, this functionality is not present in the newest version of the FEniCS Project [@Baratta:2023].
The storage principles in the ADIOS4DOLFINx are based on the ideas present in this implementation.
However, the implementation for non-Lagrangian finite element spaces vastly differs, due to the usage of dof-permutations [@Scroggs:2022].
Additionally, all global MPI calls in the old implementation have been reimplemented with scalable MPI communication using the MPI-3 Neighborhood Collectives [@MPI-Forum:2012].

The framework introduces several new methods for storing partitioning information for *N-to-N* checkpointing with arbitrary ghosting, as well as very lightweight snapshot checkpoints.
A similar framework for *N-to-M* checkpointing was implemented by @Ham:2024 for the finite element framework Firedrake [@Rathgeber:2016].
This frameworks differs from the one used in ADIOS4DOLFINx in several ways due to the different internal structures of DOLFINx and Firedrake.

# Functionality

The software is written as a Python-extension to DOLFINx, which can be installed using the Python Package installer `pip` directly from the GitHub repository or using the [ADIOS4DOLFINx](https://pypi.org/project/adios4dolfinx/) from the Python Package Index (PyPI).
The following features are supported:

- Snapshot checkpointing
- *N-to-M* checkpointing with mesh storage 
- *N-to-M* checkpointing without mesh storage
- *N-to-N* checkpointing storing partitioning information

A *snapshot checkpoint* is a checkpoint that is only valid during the run of a simulation.
It is lightweight (only storing the local portion of the global dof array to file), and is stored using the *Local Array* feature in ADIOS2 [@Godoy:2020] to store data local to the MPI process.
This feature is intended for use cases where many solutions have to be aggregated to the end of a simulation to some post-processing step, or as a fall-back mechanism when restarting a diverging iterative solver.

A *N-to-M* checkpoint is a checkpoint that can be written with N processes and read back in with M processes.
Two versions of this checkpoint are supported: one where storage of the mesh is required and one without mesh storage.
The reasoning for such a split is that when a mesh is read into DOLFINx and passed to an appropriate partitioner, the ordering mesh nodes (coordinates) and connectivity (cells) is changed.
Writing these back into *global arrays* requires MPI communication to ensure contiguous writing of data.

The *N-to-M* checkpoint with mesh storage exclusively writes contiguous chunks of data owned by the current process to an ADIOS2 *Global Array* that can be read in with a different number of processes at a later stage.
This operation requires no MPI communication.

In many cases, the input mesh might stem from an external mesh generator and is stored together with mesh entity markers in an external file, for instance an XDMF file.
To avoid duplication of this mesh data, a standalone file that can be associated with the XDMF file for a later restart can be created.
This method requires some MPI neighborhood collective calls to move data from the process that currently owns it to the relevant process for that stores it as a *Global Array* in contiguous chunks.
Both *N-to-M* checkpoint routines use the same API to read in checkpoints at a later instance.

In certain scenarios, mesh partitioning might be time consuming, as a developer is running the same problem over and over again with the same number of processes.
As DOLFINx supports custom partitioning [@Baratta:2023], we use this feature to read in partition data from a previous run.
As opposed to the checkpoints in the old version of DOLFIN, these checkpoints handle any ghosting, that being a custom ghosting provided by the user, or the shared-facet mode provided by DOLFINx.

# Examples
A large variety of examples covering all the functions in ADIOS4DOLFINx is available at [https://jorgensd.github.io/adios4dolfinx](https://jorgensd.github.io/adios4dolfinx).

# Acknowledgements

We acknowledge the valuable feedback on the documentation and manuscript by Thomas M. Surowiec and Halvor Herlyng and packaging support by Min Ragan-Kelley.
Additionally, we acknowledge the scientific discussion regarding feature development and code contributions by Francesco Ballarin, Henrik N. Finsberg, and Nathan Sime.

# References
