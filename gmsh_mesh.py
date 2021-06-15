from gmsh_interpot import msh_gmsh_model, mesh_from_gmsh
import numpy as np
import gmsh
import sys


class Shape:
    def is_inside(self, points):
        '''Indices of inside points'''
        pass

    def insert_gmsh(self, model, factory):
        '''Insert myself into model using factory'''
        pass


class Circle(Shape):
    '''Circle enclosing points'''
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return 'Circle'

    def is_inside(self, points):
        return np.where(np.linalg.norm(points-self.center, 2, axis=1)**2 < self.radius**2)

    def insert_gmsh(self, model, factory):
        cx, cy = self.center
        circle = factory.addCircle(cx, cy, 0, self.radius)
        loop = factory.addCurveLoop([circle])
        circle = factory.addPlaneSurface([loop])

        factory.synchronize()
        
        model.addPhysicalGroup(2, [circle], tag=1)
        
        bdry = model.getBoundary([(2, circle)])
        
        for tag, curve in enumerate(bdry, 1):
            model.addPhysicalGroup(1, [curve[1]], tag)
            
        # Return the surface that embeds points
        return circle

    
class Disk(Shape):
    '''Circle enclosing points'''
    def __init__(self, center, in_radius, out_radius):
        assert in_radius < out_radius
        self.center = center
        self.in_radius = in_radius
        self.out_radius = out_radius

    def __repr__(self):
        return 'Disk'

    def is_inside(self, points):
        return np.where(np.logical_and(np.linalg.norm(points-self.center, 2, axis=1)**2 < self.out_radius**2,
                                       np.linalg.norm(points-self.center, 2, axis=1)**2 > self.in_radius**2))

    def insert_gmsh(self, model, factory):
        cx, cy = self.center
        
        out_circle = factory.addCircle(cx, cy, 0, self.out_radius)
        in_circle = factory.addCircle(cx, cy, 0, self.in_radius)

        factory.synchronize()

        oloop = factory.addCurveLoop([out_circle])
        iloop = factory.addCurveLoop([in_circle])        

        disk = factory.addPlaneSurface([oloop, iloop])

        factory.synchronize()
        
        model.addPhysicalGroup(2, [disk], tag=1)
        
        bdry = [p[1] for p in model.getBoundary([(2, disk)])]
        # NOTE: we have 2 surfaces marked for boundary conditions
        # The inside shall be tagged as 1 and outside as 2
        _, maybe_in = factory.getEntitiesInBoundingBox(cx-1.001*self.in_radius, cy-1.001*self.in_radius, -10,
                                                       cx+1.001*self.in_radius, cy+1.001*self.in_radius, 10, dim=1)[0]

        if bdry[0] != maybe_in:
            bdry = reversed(bdry)

        for tag, curve in enumerate(bdry, 1):
            model.addPhysicalGroup(1, [curve], tag)
            
        # Return the surface that embeds points
        return disk

# def rectangle(model, factory, embed_points, ll=None, ur=None, scale_radius=1.1):
#     '''Rectangle enclosing points'''
#     # Take center of mass
#     if ll is None:
#         ll = np.min(embed_points, axis=0)
        
#     if ur is None:
#         ur = np.max(embed_points, axis=0)

#     dx = ur - ll

#     ll += -0.1*scale_radius*dx
#     ur += 0.1*scale_radius*dx
#     dx = ur - ll

#     # All embed_points should be inside
#     assert np.all(np.logical_and(embed_points[:, 0] > ll[0], embed_points[:, 0] < ur[0]))
#     assert np.all(np.logical_and(embed_points[:, 1] > ll[1], embed_points[:, 1] < ur[1]))    

#     square = factory.addRectangle(x=ll[0], y=ll[1], z=0, dx=dx[0], dy=dx[1])

#     factory.synchronize()
#     bdry = model.getBoundary([(2, square)])
#     model.addPhysicalGroup(2, [square], tag=1)

#     for tag, curve in enumerate(bdry, 1):
#         model.addPhysicalGroup(1, [curve[1]], tag)

#     # Return the surface that embeds points
#     return square


# Generic
def gmsh_mesh(embed_points, resolution, bounding_shape):
    '''Mesh bounded by bounded shape with embedded points'''
    nembed_points, gdim = embed_points.shape

    assert gdim == 2

    gmsh.initialize(sys.argv)

    model = gmsh.model
    factory = model.occ

    # How to bound the points returing tag of embedding surface
    bding_surface = bounding_shape.insert_gmsh(model, factory)

    inside_points, = bounding_shape.is_inside(embed_points)

    embed_points = embed_points[inside_points]
    # Embed_Points in surface we want to keep track of
    point_tags = [factory.addPoint(*point, z=0) for point in embed_points]

    factory.synchronize()    
    for phys_tag, tag in enumerate(point_tags, 1):
        model.addPhysicalGroup(0, [tag], phys_tag)
    model.mesh.embed(0, point_tags, 2, bding_surface)

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

    return mesh, entity_functions, inside_points

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    
    # NOTE: `gmsh_mesh.py -clscale 0.5`
    # will perform global refinement (halving sizes), 0.25 is even finer etc

    cx, cy = 0., 0.
    # Synthetic
    r = 2*np.random.rand(1000)
    # NOTE: here we will have points with radius 2 but our domain will
    # only have radius 1. To avoid points on the boundary we kick some
    # out
    tol = 0.05
    in_radius, out_radius = 0.2, 1.0
    
    r = r[np.logical_and(np.logical_or(r < out_radius+tol, r > out_radius-tol),   # Outradius
                         np.logical_or(r < in_radius+tol, r > in_radius-tol))]  # Inradius
    th = 2*np.pi*np.random.rand(len(r))
    x, y = cx + r*np.sin(th), cy + r*np.cos(th)
    
    embed_points = np.c_[x, y]
    data_points = 2*x**2 + 3*y**2

    center = np.array([cx, cy])
    bounding_shape = Disk(center=center, out_radius=out_radius, in_radius=in_radius)

    # NOTE: We want all the points to be strictly inside the boundary
    mesh, entity_functions, inside_points = gmsh_mesh(embed_points, resolution=0.125,
                                                      bounding_shape=bounding_shape)
    
    mesh_coordinates = mesh.coordinates()
    # Let's check point embedding
    vertex_function = entity_functions[0]
    vertex_values = vertex_function.array()

    nnz_idx, = np.where(vertex_values > 0)  # Dolfin mesh index
    gmsh_idx = vertex_values[nnz_idx] - 1  # We based the physical tag on order

    # NOTE: account for points that have been kicked out
    embed_points = embed_points[inside_points]
    data_points = data_points[inside_points]

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
    df.File(f'gmsh_foo_{bounding_shape}.pvd') << f

    # Let's also make sure that inside is labeled as 1 and outside is
    # labeled as 2
    _, e2v = mesh.init(1, 0), mesh.topology()(1, 0)
    facet_f = entity_functions[1].array()
    
    inner_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == 1)[0]]))
    inner_vertices = mesh_coordinates[inner_indices]
    assert np.all(np.linalg.norm(inner_vertices - center, 2, axis=1) - in_radius) < 1E-10

    outer_indices = np.unique(np.hstack([e2v(e) for e in np.where(facet_f == 2)[0]]))
    outer_vertices = mesh_coordinates[outer_indices]
    assert np.all(np.linalg.norm(outer_vertices - center, 2, axis=1) - out_radius) < 1E-10    

    # TODO:
    # - rectangle
    # - rectnagle hole
    # - one shot approach
