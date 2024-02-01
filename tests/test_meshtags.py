import adios4dolfinx
from mpi4py import MPI
import dolfinx
import numpy as np
import adios2


def write_meshtags(filename: str, mesh: dolfinx.mesh.Mesh, meshtags: dolfinx.mesh.MeshTags, engine: str = "BP4"):
    adios4dolfinx.write_mesh(mesh, filename, engine=engine)
    facets = meshtags.indices

    num_facets_local = mesh.topology.index_map(mesh.topology.dim-1).size_local
    local_facets = boundary_facets[facets < num_facets_local]
    local_values = meshtags.values[:len(local_facets)]

    num_saved_facets = len(local_facets)
    local_start = mesh.comm.exscan(num_saved_facets, op=MPI.SUM)
    local_start = local_start if mesh.comm.rank != 0 else 0
    global_num_facets = mesh.comm.allreduce(num_saved_facets, op=MPI.SUM)
    dof_layout = mesh.geometry.cmap.create_dof_layout()
    num_dofs_per_entity = dof_layout.num_entity_closure_dofs(mesh.topology.dim-1)

    entities_to_geometry = dolfinx.cpp.mesh.entities_to_geometry(
        mesh._cpp_object, mesh.topology.dim-1, boundary_facets, False)

    indices = mesh.geometry.index_map().local_to_global(entities_to_geometry.reshape(-1))

    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshTagWriter")
    io.SetEngine(engine)
    outfile = io.Open(str(filename), adios2.Mode.Append)
    # Write meshtag topology
    topology_var = io.DefineVariable(
        meshtags.name+"_topology",
        indices,
        shape=[global_num_facets, num_dofs_per_entity],
        start=[local_start, 0],
        count=[num_saved_facets, num_dofs_per_entity],
    )
    outfile.Put(topology_var, indices, adios2.Mode.Sync)

    # Write meshtag topology
    values_var = io.DefineVariable(
        meshtags.name+"_values",
        local_values,
        shape=[global_num_facets],
        start=[local_start],
        count=[num_saved_facets],
    )
    outfile.Put(values_var, local_values, adios2.Mode.Sync)

    # Write meshtag dim
    io.DefineAttribute(meshtags.name + "_dim", np.array([meshtags.dim], dtype=np.uint8))

    outfile.PerformPuts()
    outfile.EndStep()
    outfile.Close()


def read_meshtags(filename: str, mesh: dolfinx.mesh.Mesh, meshtag_name: str, engine: str = "BP4"):
    adios = adios2.ADIOS(mesh.comm)
    io = adios.DeclareIO("MeshTagsReader")
    io.SetEngine(engine)
    infile = io.Open(str(filename), adios2.Mode.Read)

    # Get mesh cell type
    dim_attr_name = f"{meshtag_name}_dim"

    if dim_attr_name not in io.AvailableAttributes().keys():
        raise KeyError(f"{dim_attr_name} nt found")

    m_dim = io.InquireAttribute(dim_attr_name)
    dim = m_dim.Data()[0]
    # Get mesh tags entites

    topology_name = f"{meshtag_name}_topology"
    for i in range(infile.Steps()):
        infile.BeginStep()
        if topology_name in io.AvailableVariables().keys():
            break
        infile.EndStep()
    if topology_name not in io.AvailableVariables().keys():
        raise KeyError(f"{topology_name} not found")

    topology = io.InquireVariable(topology_name)
    top_shape = topology.Shape()
    topology_range = adios4dolfinx.utils.compute_local_range(mesh.comm, top_shape[0])

    topology.SetSelection(
        [[topology_range[0], 0], [topology_range[1] - topology_range[0], top_shape[1]]]
    )
    mesh_entities = np.empty((topology_range[1] - topology_range[0], top_shape[1]), dtype=np.int64)
    infile.Get(topology, mesh_entities, adios2.Mode.Deferred)

    # Get mesh tags values
    values_name = f"{meshtag_name}_values"
    if values_name not in io.AvailableVariables().keys():
        raise KeyError(f"{values_name} not found")

    values = io.InquireVariable(values_name)
    val_shape = values.Shape()
    assert val_shape[0] == top_shape[0]
    values.SetSelection(
        [[topology_range[0]], [topology_range[1] - topology_range[0]]]
    )
    tag_values = np.empty((topology_range[1] - topology_range[0]), dtype=np.int32)
    infile.Get(values, tag_values, adios2.Mode.Deferred)

    infile.PerformGets()
    infile.EndStep()
    assert adios.RemoveIO("MeshTagsReader")
    local_entities, local_values = dolfinx.cpp.io.distribute_entity_data(
        mesh._cpp_object, dim, mesh_entities, tag_values)
    mesh.topology.create_connectivity(dim, 0)
    mesh.topology.create_connectivity(dim, mesh.topology.dim)

    adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)

    local_values = np.array(local_values, dtype=np.int32)

    ft = dolfinx.mesh.meshtags_from_entities(mesh, int(dim), adj, local_values)
    ft.name = meshtag_name

    return ft


filename = "meshtags.bp"
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
ft = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, boundary_facets, boundary_facets)
ft.name = "facets"
write_meshtags(filename, mesh, ft)


with dolfinx.io.XDMFFile(mesh.comm, "org_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)


MPI.COMM_WORLD.Barrier()
new_mesh = adios4dolfinx.read_mesh(MPI.COMM_WORLD, filename, engine="BP4",
                                   ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
new_ft = read_meshtags(filename, new_mesh, meshtag_name=ft.name, engine="BP4")
with dolfinx.io.XDMFFile(new_mesh.comm, "new_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(new_mesh)
    xdmf.write_meshtags(new_ft, new_mesh.geometry)
