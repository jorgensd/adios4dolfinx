from ghcr.io/fenics/dolfinx/dolfinx:nightly

ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3/dist-packages"

WORKDIR /src
COPY . adios4dolfinx

RUN python3 -m pip install -e ./adios4dolfinx