import matplotlib.pyplot as plt
import numpy as np
from fenics2nparray import *
from dolfin_adjoint_test_problem import create_synthetic_pO2_data

hole = True
sigma = 1
p, p_noisy, V, W, bc, p_ves, R_ves = create_synthetic_pO2_data(hole, sigma)

d = 0.007
x = np.arange(-1, 1, d)
y = np.arange(-1, 1, d)
p, r = fenics2nparray(p, p_ves, R_ves, x, y, [0,0])
p_noisy, r = fenics2nparray(p_noisy, p_ves, R_ves, x, y, [0,0])
np.savez('synthetic_data/pO2_data', \
    p=p, p_noisy=p_noisy, 
    r=r, x=x, y=y, d=d, p_ves=p_ves, R_ves=R_ves)
