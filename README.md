# ADIOS2Wrappers for DOLFINx

[![MIT](https://img.shields.io/github/license/jorgensd/adios4dolfinx)](LICENSE)
[Read Latest Documentation](https://jsdokken.com/adios4dolfinx/)

This is an extension for [DOLFINx](https://github.com/FEniCS/dolfinx/) to checkpoint meshes, meshtags and functions using [ADIOS2](https://adios2.readthedocs.io/en/latest/).

The code uses the adios2 Python-wrappers to write DOLFINx objects to file, supporting N-to-M (_recoverable_) and N-to-N (_snapshot_) checkpointing.
See: [Checkpointing in DOLFINx - FEniCS 23](https://jsdokken.com/checkpointing-presentation/#/) for more information.

For scalability, the code uses [MPI Neighbourhood collectives](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node200.htm) for communication across processes.

## Installation

### Docker

ADIOS2 is installed in the official DOLFINx containers.

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --name=dolfinx-checkpoint ghcr.io/fenics/dolfinx/dolfinx:nightly
```

### Conda

To use with conda (DOLFINx release v0.7.0 works with v0.7.3 of ADIOS4DOLFINx)

```bash
conda create -n dolfinx-checkpoint python=3.10
conda activate dolfinx-checkpoint
conda install -c conda-forge fenics-dolfinx pip adios2
python3 -m pip install git+https://github.com/jorgensd/adios4dolfinx@v0.7.2
```

## Functionality

### DOLFINx

- Reading and writing meshes, using `adios4dolfinx.read/write_mesh`
- Reading and writing meshtags associated to meshes `adios4dolfinx.read/write_meshtags`
- Reading checkpoints for any element (serial and parallel, arbitrary number of functions and timesteps per file). Use `adios4dolfinx.read/write_function`.
- Writing standalone function checkpoints relating to "original meshes", i.e. meshes read from `XDMFFile`. Use `adios4dolfinx.write_function_on_input_mesh` for this.
- Store mesh partitioning and re-read the mesh with this information, avoiding calling SCOTCH, Kahip or Parmetis.

> [!IMPORTANT]  
> For checkpoints written with `write_function` to be valid, you first have to store the mesh with `write_mesh` to the checkpoint file.

> [!IMPORTANT]  
> A checkpoint file supports multiple functions and multiple time steps, as long as the functions are associated with the same mesh

> [!IMPORTANT]  
> Only one mesh per file is allowed

### Backwards compatibility

> [!WARNING]
> If you are using v0.7.2, you are adviced to upgrade to v0.7.3, as it contains som crucial fixes for openmpi.

### Legacy DOLFIN

Only checkpoints for `Lagrange` or `DG` functions are supported from legacy DOLFIN

- Reading meshes from the DOLFIN HDF5File-format
- Reading checkpoints from the DOLFIN HDF5File-format (one checkpoint per file only)
- Reading checkpoints from the DOLFIN XDMFFile-format (one checkpoint per file only, and only uses the `.h5` file)

See the [API](./docs/api) for more information.

## Long term plan

The long term plan is to get this library merged into DOLFINx (rewritten in C++ with appropriate Python-bindings).

# Contributor guidelines
When contributing to this repository, please first [create an issue](https://github.com/jorgensd/adios4dolfinx/issues/new/choose) containing information about the missing feature or the bug that you would like to fix. Here you can discuss the change you want to make with the maintainers of the repository.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## New contributor guide

To get an overview of the project, read the [documentation](https://jorgensd.github.io/adios4dolfinx). Here are some resources to help you get started with open source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

## Pull Request Process


### Pull Request

- When you're finished with the changes, create a pull request, also known as a PR. It is also OK to create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) from the very beginning. Once you are done you can click on the ["Ready for review"] button. You can also [request a review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) from one of the maintainers.
- Don't forget to [link PR to the issue that you opened ](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a team member will review your proposal. We may ask questions or request for additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://lab.github.com/githubtraining/managing-merge-conflicts) to help you resolve merge conflicts and other issues.
- Please make sure that all tests are passing, github pages renders nicely, and code coverage are are not lower than before your contribution. You see the different github action workflows by clicking the "Action" tab in the GitHub repository.

Note that for a pull request to be accepted, it has to pass all the tests on CI, which includes:
- `mypy`: typechecking
- `ruff`: Code formatting
- `pytest`: Successfull execution of all tests in the `tests` folder.


### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.
