from gmsh_interpot import msh_gmsh_model, mesh_from_gmsh
import numpy as np
import gmsh
import sys


def circle_mesh(embed_points, resolution, center=None, radius=None, scale_radius=1.1):
    '''Circle mesh with embedded points'''
    nembed_points, gdim = embed_points.shape

    assert gdim == 2
    # Take center of mass
    if center is None:
        center = np.mean(embed_points, axis=0)
        
    if radius is None:
        radius = np.max(np.linalg.norm(embed_points-center, 2, axis=1))
        radius = radius*scale_radius
    else:
        radius = radius*scale_radius

    # All embed_points should be inside
    assert np.all(np.linalg.norm(embed_points-center, 2, axis=1)**2 < radius**2)

    gmsh.initialize(sys.argv)

    model = gmsh.model
    factory = model.occ

    cx, cy = center
    circle = factory.addCircle(cx, cy, 0, radius)
    loop = factory.addCurveLoop([circle])
    circle = factory.addPlaneSurface([loop])

    factory.synchronize()

    model.addPhysicalGroup(2, [circle], tag=1)
    
    bdry = model.getBoundary([(2, circle)])

    for tag, curve in enumerate(bdry, 1):
        model.addPhysicalGroup(1, [curve[1]], tag)

    # Embed_Points in surface we want to keep track of
    point_tags = [factory.addPoint(*point, z=0) for point in embed_points]

    factory.synchronize()    
    for phys_tag, tag in enumerate(point_tags, 1):
        model.addPhysicalGroup(0, [tag], phys_tag)
    model.mesh.embed(0, point_tags, 2, circle)

    factory.synchronize()

    # NOTE: if you want to see it first
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
        
    nodes, topologies = msh_gmsh_model(model,
                                       2,
                                       # Globally refine
                                       number_options={'Mesh.CharacteristicLengthFactor': resolution})
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

    gmsh.finalize()

    return mesh, entity_functions

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    
    # NOTE: `gmsh_mesh.py -clscale 0.5`
    # will perform global refinement (halving sizes), 0.25 is even finer etc

    cx, cy = 0., 0.
    # Synthetic
    r = np.random.rand(1000)
    th = 2*np.pi*np.random.rand(1000)
    x, y = cx + r*np.sin(th), cy + r*np.cos(th)
    
    embed_points = np.c_[x, y]
    data_points = 2*x**2 + 3*y**2

    # NOTE: We want all the points to be strictly inside the boundary
    mesh, entity_functions = circle_mesh(embed_points, resolution=0.25,
                                         center=np.array([cx, cy]),
                                         radius=1,
                                         scale_radius=1.1)
    
    mesh_coordinates = mesh.coordinates()
    # Let's check point embedding
    vertex_function = entity_functions[0]
    vertex_values = vertex_function.array()

    nnz_idx, = np.where(vertex_values > 0)  # Dolfin mesh index
    gmsh_idx = vertex_values[nnz_idx] - 1  # We based the physical tag on order 

    assert np.linalg.norm(mesh_coordinates[nnz_idx] - embed_points[gmsh_idx]) < 1E-13

    # Populating a P1 function
    V = df.FunctionSpace(mesh, 'CG', 1)
    f = df.Function(V)

    v2d = df.vertex_to_dof_map(V)
    values = f.vector().get_local()
    values[v2d[nnz_idx]] = data_points[gmsh_idx]
    f.vector().set_local(values)

    assert all(abs(f(x) - target) < 1E-13 for x, target in zip(embed_points, data_points))

    # Of course, the function is incomplete
    df.File('gmsh_foo.pvd') << f
