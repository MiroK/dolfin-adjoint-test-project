from scipy.spatial import Delaunay
import dolfin as df
import ufl


def delaunay_mesh(points_2d):
    '''Delaunay mesh from points'''
    _, gdim = points_2d.shape
    assert gdim == 2

    tri = Delaunay(points_2d)

    return build_mesh(tri.points, tri.simplices)


def build_mesh(vertices, cells):
    '''Simplex mesh from coordinates and cell-vertex connectivity'''
    nvertices, gdim = vertices.shape

    ncells, tdim_ = cells.shape
    tdim = tdim_ - 1

    mesh = df.Mesh()
    editor = df.MeshEditor()

    cell_type = {1: 'interval',
                 2: 'triangle',
                 3: 'tetrahedron'}[tdim]
    cell_type = ufl.Cell(cell_type, gdim)

    editor.open(mesh, str(cell_type), tdim, gdim)            

    editor.init_vertices(nvertices)
    editor.init_cells(ncells)

    for vi, x in enumerate(vertices):
        editor.add_vertex(vi, x)

    for ci, c in enumerate(cells):
        editor.add_cell(ci, c)
    
    editor.close()

    return mesh

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    data_path = './xyz_example_data.npz'
    data = np.load(data_path)

    points_2d = np.c_[data['x'], data['y']]
    mesh = delaunay_mesh(points_2d)
    # NOTE: for embedding data it is useful that we preserve node order
    assert np.linalg.norm(mesh.coordinates() - points_2d) < 1E-10
    # so we can just populate P1 function with
    V = df.FunctionSpace(mesh, 'CG', 1)    
    d2v = df.dof_to_vertex_map(V)

    f = df.Function(V)
    f.vector().set_local(data['z'][d2v]) 

    df.File('del_foo.pvd') << f
