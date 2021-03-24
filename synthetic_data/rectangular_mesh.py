from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

def rectangular_mesh():

    R_star = 141.        # characteristic length [um]
    M_star = 1.0e-3      # characteristic M [mmHg/um**2]

    Rart = 6/R_star      # vessel radius

    grid = Rectangle(Point(-1,-1), Point(1,1))
    vessel = Circle(Point(0,0), Rart)
    
    domain = grid - vessel
    
    resolution = 300
    mesh = generate_mesh(domain, resolution)
    
    return mesh

if __name__ == "__main__":
    
    mesh = rectangular_mesh()
    
    mesh_file = File("rectangular_mesh.xml")
    mesh_file << mesh
    #plot(mesh)
    #plt.show()

