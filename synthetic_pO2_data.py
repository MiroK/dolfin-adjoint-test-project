import matplotlib.pyplot as plt
import numpy as np
from fenics2nparray import *
from dolfin_adjoint_test_problem import create_synthetic_pO2_data

hole = False
sigma = 0
p, p_noisy, V, W, bc, p_ves, R_ves = create_synthetic_pO2_data(hole, sigma)

d = 0.007
N = np.ceil(2/d)
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
p, r = fenics2nparray(p, p_ves, R_ves, x, y, [0,0])
p_noisy, r = fenics2nparray(p_noisy, p_ves, R_ves, x, y, [0,0])
np.savez('synthetic_data/pO2_data', \
    p=p, p_noisy=p_noisy, 
    r=r, x=x, y=y, d=d, p_ves=p_ves, R_ves=R_ves)
