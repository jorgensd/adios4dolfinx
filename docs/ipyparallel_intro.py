# # Introduction to IPython parallel
# The following demos heavily rely on IPython-parallel to illustrate how checkpointing works when
# using multiple MPI processes.
# We illustrate what happens in parallel by launching three MPI processes
# using [ipyparallel](https://ipyparallel.readthedocs.io/en/latest/)

import logging

import ipyparallel as ipp


def hello_mpi():
    # We define all imports inside the function as they have to be launched on the remote engines
    from mpi4py import MPI

    print(f"Hello from rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size - 1}")


with ipp.Cluster(engines="mpi", n=3, log_level=logging.ERROR) as cluster:
    # We send the query to run the function `hello_mpi` on all engines
    query = cluster[:].apply_async(hello_mpi)
    # We wait for all engines to finish
    query.wait()
    # We check that all engines exited successfully
    assert query.successful(), query.error
    # We print the output from each engine
    print("".join(query.stdout))
