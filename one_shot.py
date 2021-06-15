import numpy as np
from dolfin import *


def one_shot(data, marking_functions, state_bcs, multiplier_bcs, alpha):
    '''
    Here we with to solve: min_{u, f} (1/2)*norm(u - data)**2 + alpha/2*inner(f', f')*dx
    subject to

        -Delta u = f
               u = g on state_bcs['dirichlet'] regions
       grad(u).n = h on stata_bcs['neumann'] regions 
    '''
    V = data.function_space()
    mesh = V.mesh()
    # For integration we will use ...
    facet_f = entity_functions[mesh.topology().dim()-1]
    point_f = entity_functions[0]
    # ... the surface measure
    ds = Measure('ds', domain=mesh, subdomain_data=facet_f)
    # Point measure, here we only care about tagged/no tagged destinction
    point_tags = point_f.array()
    point_tags[point_tags > 0] = 1
    dP = Measure('dP', domain=mesh, subdomain_data=point_f)

    # Let's have state, control and multiplier in the same space
    Welm = MixedElement([V.ufl_element()]*3)
    W = FunctionSpace(mesh, Welm)

    u, f, lm = TrialFunctions(W)
    v, g, dlm = TestFunctions(W)
      
    a = (inner(u, v)*dP(1)                                       + inner(grad(v), grad(lm))*dx
                                         +alpha*inner(grad(f), grad(g))*dx -inner(g, lm)*dx
         +inner(grad(u), grad(dlm))*dx  - inner(f, dlm)*dx)
    # FIXME: there should be neumann terms here somewhere
    L = inner(data, v)*dP(1)
    # State Dirichlet
    bcs = [DirichletBC(W.sub(0), value, facet_f, tag) for tag, value in state_bcs['dirichlet'].items()]
    # LM Dirichlet
    bcs.extend([DirichletBC(W.sub(2), value, facet_f, tag) for tag, value in multiplier_bcs['dirichlet'].items()])

    wh = Function(W)
    solve(a == L, wh, bcs)

    uh, fh, lmh = wh.split(deepcopy=True)

    for x in (uh, fh, lmh):
        assert np.all(~np.isnan(x.vector().get_local()))
    
    return uh, fh, lmh

# --------------------------------------------------------------------

if __name__ == '__main__':
    from gmsh_mesh import Disk, gmsh_mesh
    import sys
    
    data = np.load('./xyz_example_data.npz')

    points = np.c_[data['x'], data['y']]
    data_points = data['z'].flatten()

    center = np.mean(points, axis=0)
    outer_r = 1.1*np.max(np.linalg.norm(points-center, 2, axis=1))
    inner_r = 0.9*np.min(np.linalg.norm(points-center, 2, axis=1))

    print(inner_r, outer_r, center)

    disk = Disk(center, in_radius=inner_r, out_radius=outer_r)
    mesh, entity_functions, inside_points = gmsh_mesh(points,
                                                      bounding_shape=disk,
                                                      argv=sys.argv)

    vertex_function = entity_functions[0]
    vertex_values = vertex_function.array()

    nnz_idx, = np.where(vertex_values > 0)  # Dolfin mesh index
    gmsh_idx = vertex_values[nnz_idx] - 1  # We based the physical tag on order

    V = FunctionSpace(mesh, 'CG', 1)
    File('mesh.pvd') << mesh
    File('facets.pvd') << entity_functions[1]
    u_data = Function(V)

    v2d = vertex_to_dof_map(V)
    values = u_data.vector().get_local()
    values[v2d[nnz_idx]] = data_points[gmsh_idx]
    u_data.vector().set_local(values)

    # Let's solve it
    state, control, multiplier = one_shot(u_data, entity_functions,
                                          state_bcs={'dirichlet': {1: Constant(1)},
                                                     'neumann': {2: Constant(0)}},
                                          multiplier_bcs={'neumann': {1: Constant(0), 2: Constant(0)},
                                                          'dirichlet': {}},
                                          alpha=Constant(1E0))

    File('data.pvd') << u_data
    File('state.pvd') << state
    File('control.pvd') << control
    File('multipler.pvd') << multiplier
