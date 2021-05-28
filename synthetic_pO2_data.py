import matplotlib.pyplot as plt
import numpy as np
from fenics2nparray import *
from dolfin_adjoint_test_problem import create_synthetic_pO2_data
from dolfin_adjoint import *
from scipy.io import savemat

d = 0.014
N = np.ceil(2/d)
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
Nx = len(x)
Ny = len(y)
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx-1, Ny-1)
mesh_file = File("synthetic_mesh.xml")
mesh_file << mesh

hole = False
sigma = 1
p, p_noisy, V, W, bc, p_ves, R_ves = create_synthetic_pO2_data(mesh, hole, sigma)
p = fenics2nparray(p, p_ves, x, y)
p_noisy = fenics2nparray(p_noisy, p_ves, x, y)
np.savez('synthetic_data/pO2_data_sigma_'+str(sigma)+'_Ld', \
    p=p, p_noisy=p_noisy, 
    x=x, y=y, d=d, p_ves=p_ves, R_ves=R_ves)
mdic = {"p":p, "p_noisy":p_noisy, "x":x, "y":y, "d":d, "p_ves":p_ves, \
        "R_ves":R_ves}
savemat('synthetic_data/pO2_data_sigma_'+str(sigma)+'_Ld.mat', mdic)

