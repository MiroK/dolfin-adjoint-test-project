from fenics import *
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

def refine_mesh(mesh):

    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
    for cell in cells(mesh):
        point = cell.midpoint()
        x =  point.x()
        y =  point.y()
        r = np.sqrt(x**2 + y**2)
        if r < 0.05:
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

    return mesh

def rectangular_mesh(hole):

    R_star = 141.        # characteristic length [um]
    M_star = 1.0e-3      # characteristic M [mmHg/um**2]

    R_ves = 6/R_star      # vessel radius

    grid = Rectangle(Point(-1,-1), Point(1,1))
    
    if hole:
        vessel = Circle(Point(0,0), R_ves)
        domain = grid - vessel
    else: 
        domain = grid
    
    resolution = 30
    mesh = generate_mesh(domain, resolution)

#    for n in range(4):
#        mesh = refine_mesh(mesh)

    return mesh

if __name__ == "__main__":
   
    hole = True 
    mesh = rectangular_mesh(hole)
    
    mesh_file = File("rectangular_mesh.xml")
    mesh_file << mesh
    plot(mesh)
    plt.show()
