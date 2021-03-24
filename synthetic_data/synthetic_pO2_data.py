from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from fenics2nparray import *

def synthetic_pO2_data(mesh, sigma):

    R_star = 141.        # characteristic length [um]
    M_star = 1.0e-3      # characteristic M [mmHg/um**2]

    Rart = 6/R_star      # vessel radius
    pO2art = 80/(M_star*R_star**2)   # pO2 at vessel wall 
    M = 1.0

    V = FunctionSpace(mesh, 'CG', 1)
    p = TrialFunction(V)
    M = Constant(M)
    v = TestFunction(V)
    form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
    (a,L) = system(form)

    def boundary(x, on_boundary):
        eps = 0.01
        r = np.sqrt(x[0]**2 + x[1]**2)
        b = ((r < Rart+eps) and on_boundary)
        return b

    bcs = DirichletBC(V, pO2art, boundary)

    pO2 = Function(V)
    solve(a==L, pO2, bcs)

    noise = sigma*np.random.randn(np.size(np.array(pO2.vector())))
    pO2_noisy = pO2.copy(deepcopy=True)
    pO2_noisy.vector()[:] = np.array(pO2.vector()) + noise

    return pO2, pO2_noisy, pO2art, Rart

if __name__ == "__main__":

    #mesh = Mesh("circular_mesh.xml")
    mesh = Mesh("rectangular_mesh.xml")
    sigma = 5e-4
    pO2, pO2_noisy, pO2art, Rart = synthetic_pO2_data(mesh, sigma)
    
    d = 0.007
    x = np.arange(-1, 1.0001, d)
    y = np.arange(-1, 1.0001, d)
    pO2, r = fenics2nparray(pO2, pO2art, Rart, x, y, [0,0])
    pO2_noisy, r = fenics2nparray(pO2_noisy, pO2art, Rart, x, y, [0,0])
    np.savez('pO2_data', \
            pO2=pO2, pO2_noisy=pO2_noisy, 
            r=r, x=x, y=y, d=d)
#    file1 = File("pO2.pvd")
#    file1 << pO2
#    file2 = File("pO2_noisy.pvd")
#    file2 << pO2_noisy
#    c = plot(pO2_noisy)
#    plt.colorbar(c)
#    plt.show()
