from mpi4py import MPI
import dolfinx
import numpy as np
import adios4dolfinx
import ufl
import basix.ufl

def read_function(u, mesh, input_dofmap, starting_pos, input_array, original_cell_index, parent_permutations):

    input_dofmap = dolfinx.graph.adjacencylist(input_dofmap)

    comm = u.function_space.mesh.comm
      # ----------------------Step 1---------------------------------
    # Compute index of input cells and get cell permutation
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    if original_cell_index is None:
        original_cell_index = mesh.topology.original_cell_index
    input_cells = original_cell_index[:num_owned_cells]

    mesh.topology.create_entity_permutations()
    cell_perm = mesh.topology.get_cell_permutation_info()[:num_owned_cells]

    # Compute mesh->input communicator
    # 1.1 Compute mesh->input communicator
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    owners = adios4dolfinx.utils.index_owner(comm, input_cells, num_cells_global)

    # -------------------Step 2------------------------------------
    # Send and receive global cell index and cell perm
    inc_cells, inc_perms = adios4dolfinx.comm_helpers.send_and_recv_cell_perm(input_cells, cell_perm, owners, mesh.comm)

    # Compute owner of dofs in dofmap
    num_dofs_global = (
        u.function_space.dofmap.index_map.size_global * u.function_space.dofmap.index_map_bs
    )
    dof_owner = adios4dolfinx.utils.index_owner(mesh.comm, input_dofmap.array, num_dofs_global)

    # --------------------Step 4-----------------------------------
    # Read array from file and communicate them to input dofmap process

    recv_array = adios4dolfinx.comm_helpers.send_dofs_and_recv_values(
        input_dofmap.array, dof_owner, comm, input_array, starting_pos
    )

    # -------------------Step 5--------------------------------------
    # Invert permutation of input data based on input perm
    # Then apply current permutation to the local data
    # element = u.function_space.element
    # if element.needs_dof_transformations:
    #     bs = u.function_space.dofmap.bs

    #     # Read input cell permutations on dofmap process
    #     local_input_range = compute_local_range(comm, num_cells_global)
    #     input_local_cell_index = inc_cells - local_input_range[0]
    #     if legacy:
    #         perm_name = "CellPermutations"
    #     else:
    #         perm_name = f"{name}CellPermutations"
    #     input_perms = read_cell_perms(adios, comm, filename, perm_name, num_cells_global, engine)
    #     # Start by sorting data array by cell permutation
    #     num_dofs_per_cell = input_dofmap.offsets[1:] - input_dofmap.offsets[:-1]
    #     assert np.allclose(num_dofs_per_cell, num_dofs_per_cell[0])

    #     # Sort dofmap by input local cell index
    #     input_perms_sorted = input_perms[input_local_cell_index]
    #     unrolled_dofmap_position = unroll_insert_position(
    #         input_local_cell_index, num_dofs_per_cell[0]
    #     )
    #     dofmap_sorted_by_input = recv_array[unrolled_dofmap_position]

    #     # First invert input data to reference element then transform to current mesh
    #     element.Tt_apply(dofmap_sorted_by_input, input_perms_sorted, bs)
    #     element.Tt_inv_apply(dofmap_sorted_by_input, inc_perms, bs)
    #     # Compute invert permutation
    #     inverted_perm = np.empty_like(unrolled_dofmap_position)
    #     inverted_perm[unrolled_dofmap_position] = np.arange(
    #         len(unrolled_dofmap_position), dtype=inverted_perm.dtype
    #     )
    #     recv_array = dofmap_sorted_by_input[inverted_perm]

    # ------------------Step 6----------------------------------------
    # For each dof owned by a process, find the local position in the dofmap.
    V = u.function_space
    local_cells, dof_pos = adios4dolfinx.utils.compute_dofmap_pos(V)
    if len(local_cells) == 0:
        input_cells = np.empty(0, dtype=np.int64)
    else:
        input_cells = original_cell_index[local_cells]
    num_cells_global = V.mesh.topology.index_map(V.mesh.topology.dim).size_global
    owners = adios4dolfinx.utils.index_owner(V.mesh.comm, input_cells, num_cells_global)
    unique_owners, owner_count = np.unique(owners, return_counts=True)
    # FIXME: In C++ use NBX to find neighbourhood
    sub_comm = V.mesh.comm.Create_dist_graph(
        [V.mesh.comm.rank], [len(unique_owners)], unique_owners, reorder=False
    )
    source, dest, _ = sub_comm.Get_dist_neighbors()
    sub_comm.Free()
    owned_values = adios4dolfinx.comm_helpers.send_dofmap_and_recv_values(
        comm,
        np.asarray(source, dtype=np.int32),
        np.asarray(dest, dtype=np.int32),
        owners,
        owner_count.astype(np.int32),
        input_cells,
        dof_pos,
        num_cells_global,
        recv_array,
        input_dofmap.offsets,
    )
    u.x.array[: len(owned_values)] = owned_values
    u.x.scatter_forward()



def read_submesh(mesh, dim, mesh_entities, topology_start):

    # Convert the original mesh entities and their input index to a meshtags, as this is needed for the submesh creation
    tag_values = np.arange(topology_start, topology_start + len(mesh_entities), dtype=np.int64)

    local_entities, local_values = dolfinx.io.distribute_entity_data(
            mesh, int(dim), mesh_entities.astype(np.int32), tag_values
        )
    mesh.topology.create_connectivity(dim, 0)
    mesh.topology.create_connectivity(dim, mesh.topology.dim)

    adj = dolfinx.graph.adjacencylist(local_entities)
    mt = dolfinx.mesh.meshtags_from_entities(mesh, int(dim), adj, local_values.astype(np.int64))

    # Given the parent mesh entities and their input index, we can create a submesh
    entity_map = mesh.topology.index_map(dim)
    vec = dolfinx.la.vector(entity_map, dtype=np.int64)
    vec.array[:] = -1
    vec.array[mt.indices] = mt.values
    vec.scatter_forward()
    submesh_cells = np.flatnonzero(vec.array >= 0)
    submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(mesh, dim, submesh_cells)
    num_submesh_cells = submesh.topology.index_map(submesh.topology.dim).size_local + submesh.topology.index_map(submesh.topology.dim).num_ghosts
    cm = cell_map.sub_topology_to_topology(np.arange(num_submesh_cells, dtype=np.int32), False)
    submesh_input_indices = vec.array[cm]
    return submesh, cell_map, vertex_map, node_map, submesh_input_indices

def permute_submesh_dofmap(sub_space, global_sub_dofmap, parent_mesh, cell_map, inverse):
    sub_e = sub_space.element.basix_element
    submesh = sub_space.mesh
    p_e = basix.ufl.element(sub_e.family, parent_mesh.ufl_domain().ufl_coordinate_element().basix_element.cell_type, sub_e.degree, lagrange_variant=sub_e.lagrange_variant, discontinuous=False)
    num_submesh_cells = submesh.topology.index_map(submesh.topology.dim).size_local
    parent_mesh.topology.create_entity_permutations()
    cell_info = parent_mesh.topology.get_cell_permutation_info()
    parent_facets = cell_map.sub_topology_to_topology(np.arange(num_submesh_cells, dtype=np.int64), False)
    parent_mesh.topology.create_connectivity(parent_mesh.topology.dim-1, parent_mesh.topology.dim)
    f_to_c = parent_mesh.topology.connectivity(parent_mesh.topology.dim-1, parent_mesh.topology.dim)
    parent_mesh.topology.create_connectivity(parent_mesh.topology.dim, parent_mesh.topology.dim-1)
    c_to_f = parent_mesh.topology.connectivity(parent_mesh.topology.dim, parent_mesh.topology.dim-1)
    itg_data = np.full((num_submesh_cells, 2),-1, dtype=np.int32)
    for i, facet in enumerate(parent_facets):
        cell = f_to_c.links(facet)[0]
        facets = c_to_f.links(cell)
        pos = np.where(facets == facet)[0][0]
        itg_data[i, :] = (cell, pos)
    if inverse:
        func = p_e.basix_element.inv_permute_subentity_closure
    else:
        func = p_e.basix_element.permute_subentity_closure
    for i, (cell, pos) in enumerate(itg_data):
        permuting = np.arange(sub_space.element.basix_element.dim, dtype=np.int32)
        func(permuting, cell_info[cell], sub_space.element.basix_element.cell_type, int(pos))
        global_sub_dofmap[i, :] = global_sub_dofmap[i, permuting]


def pack_submesh_data(mesh, sub_function, cell_map, node_map):
    V_sub = sub_function.function_space
    submesh = V_sub.mesh

    # First pack the submesh geometry dofmap, as it is easy to read with distribute_entity_data
    submesh_geometry_dm = submesh.geometry.dofmap
    parent_geometry_dm = node_map[submesh_geometry_dm]
    global_parent_geom = mesh.geometry.index_map().local_to_global(parent_geometry_dm.flatten())
    global_parent_geom = global_parent_geom.reshape(submesh_geometry_dm.shape)
    # sub_cell_dim = submesh.topology.dim
    # submesh_cell_map = submesh.topology.index_map(sub_cell_dim)
    # submesh_cell_insert_pos = submesh_cell_map.local_range[0]

    # Next permute the function space dofmap of the submesh to match the parent mesh dofmap, as this is needed for the function values to be correctly read

    
    # Unroll dofmap and permute dofs for each cell in the submesh
    dofmap = V_sub.dofmap
    dofmap_bs = V_sub.dofmap.index_map_bs
    index_map_bs = V_sub.dofmap.index_map_bs

    # Unroll dofmap for block size
    num_sub_cells_local  = submesh.topology.index_map(submesh.topology.dim).size_local
    unrolled_dofmap = adios4dolfinx.utils.unroll_dofmap(dofmap.list[:num_sub_cells_local, :], dofmap_bs)
    dmap_loc = (unrolled_dofmap // index_map_bs).reshape(-1)
    dmap_rem = (unrolled_dofmap % index_map_bs).reshape(-1)

    # Convert imap index to global index
    imap_global = dofmap.index_map.local_to_global(dmap_loc)
    dofmap_global = imap_global * index_map_bs + dmap_rem
    dofmap_global = dofmap_global.reshape(unrolled_dofmap.shape)
    num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
    insert_pos_cells = submesh.topology.index_map(submesh.topology.dim).local_range[0]
    insert_func_pos = dofmap.index_map.local_range[0] * dofmap_bs

    mesh.topology.create_entity_permutations()
    return global_parent_geom, insert_pos_cells, dofmap_global, insert_func_pos, sub_function.x.array[:num_dofs_local]




def test_write_submesh_codim1(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(
        comm, 4,1,1, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    def locate_cells(x):
        return np.isclose(x[0], 0.0)#(x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2 < 0.2

    dim = mesh.topology.dim-1
    cells = dolfinx.mesh.locate_entities(mesh, dim, locate_cells)

    submesh, cell_map, _, node_map = dolfinx.mesh.create_submesh(mesh, dim, cells)

    f_path = mesh.comm.bcast(tmp_path, root=0)
    outfile = f_path / "submesh_codim1.bp"

    # We use standard XDMF format to write the mesh
    with dolfinx.io.XDMFFile(mesh.comm, outfile.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)

    def f(x):
        return x[1] -x[2]

    el = ("Lagrange", 1)
    V_sub = dolfinx.fem.functionspace(submesh, el)
    u_sub = dolfinx.fem.Function(V_sub, name="u_sub")
    u_sub.interpolate(f)

     # "store" data to file for submesh creation, but do not actually write the submesh to file
    global_parent_geom, insert_pos_cells, dofmap_global, insert_func_pos, sub_function_array = pack_submesh_data(mesh, u_sub, cell_map, node_map)

    # We now read the submesh again
    with dolfinx.io.XDMFFile(mesh.comm, outfile.with_suffix(".xdmf"), "r") as xdmf:
        new_mesh = xdmf.read_mesh()
    new_submesh, cell_m, v_m, n_m, sbmsh_ici = read_submesh(new_mesh, dim, global_parent_geom, insert_pos_cells)
    print(f"Submesh input cells:", submesh.topology.index_map(submesh.topology.dim).size_local, "New submesh cells:", new_submesh.topology.index_map(new_submesh.topology.dim).size_local)

    V_new = dolfinx.fem.functionspace(new_submesh, el)
    u_sub_new = dolfinx.fem.Function(V_new, name="u_sub_new")
    new_mesh.topology.create_entity_permutations()
    cell_perms = new_mesh.topology.get_cell_permutation_info()
    read_function(u_sub_new, new_submesh, dofmap_global, insert_func_pos, sub_function_array, sbmsh_ici,
                  cell_perms)

    u_ref = dolfinx.fem.Function(V_new, name="u_ref")
    u_ref.interpolate(f)
    breakpoint()

    tol = 100 * np.finfo(dolfinx.default_scalar_type).eps
    print(u_sub_new)
    np.testing.assert_allclose(u_sub_new.x.array - u_ref.x.array, 0, atol=tol)

    L2_squared = dolfinx.fem.form(ufl.inner(u_sub_new-u_ref, u_sub_new - u_ref) * ufl.dx)
    L2_local = dolfinx.fem.assemble_scalar(L2_squared)
    L2_global = np.sqrt(new_submesh.comm.allreduce(L2_local, op=MPI.SUM))

    np.testing.assert_allclose(L2_global, 0.0, atol=tol)
