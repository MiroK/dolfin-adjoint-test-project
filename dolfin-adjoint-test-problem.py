from fenics import *
from mshr import *
import numpy as np
import scipy.io as sio
from dolfin_adjoint import *
import moola
np.random.seed(1)

def read_pO2_from_file(filename):

    ### import data
    data = np.load(filename)
    Nx = len(data['x'])
    Ny = len(data['y'])
    N = int(Nx*Ny)
    p_exact_data = data['pO2']
    p_noisy_data = data['pO2_noisy']
    
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx-1, Ny-1)
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)

    d2v = dof_to_vertex_map(V)
    
    p_exact_vector = np.reshape(p_exact_data, (1, N))
    p = Function(V)
    p.vector()[:] = p_exact_vector[0][d2v]
    
    p_noisy_vector = np.reshape(p_noisy_data, (1, N))
    p_noisy = Function(V)
    p_noisy.vector()[:] = p_noisy_vector[0][d2v]

    ### add hole
    mesh = Mesh("rectangular_mesh.xml")
    
    R_ves = data['R_ves']
    p_ves = data['p_ves']
    
    # interpolate
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)
    p = interpolate(p, V)
    p_noisy = interpolate(p_noisy, V)
    
    def boundary(x, on_boundary):
        eps = 0.1
        r = np.sqrt(x[0]**2 + x[1]**2)
        b = ((r < R_ves+eps) and on_boundary)
        return b

    bc = DirichletBC(V, p_ves, boundary)
    
    return p, p_noisy, V, W, bc 
    
def create_synthetic_pO2_data():

    R_star = 141.       # characteristic length [um]
    M_star = 1.0e-3     # charcateristic M [mmHg/um**2]

    R_ves = 6/R_star                # vessel radius
    p_ves = 80./(M_star*R_star**2)  # pO2 at vessel wall
    M = 1
    #sigma = 5e-4
    sigma = 1/(M_star*R_star**2)    # noise

    mesh = Mesh("rectangular_mesh.xml")
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)

    def boundary(x, on_boundary):
        eps = 0.1
        r = np.sqrt(x[0]**2 + x[1]**2)
        b = ((r < R_ves+eps) and on_boundary)
        return b

    bc = DirichletBC(V, p_ves, boundary)

    ### Solve the noiseless system to find the true p
    p = TrialFunction(V)
    v = TestFunction(V)
    M = Constant(M)
    a = inner(grad(p), grad(v))*dx
    L = -M*v*dx
    p = Function(V)
    solve(a == L, p, bc)

    ### Create noisy system
    N = V.dim()
    noise = sigma*np.random.randn(N)
    p_noisy = p.copy(deepcopy=True)
    p_noisy.vector()[:] += noise

    return p, p_noisy, V, W, bc

def estimate_M(p_data, V, W, bc, alpha):

    # Solve forward problem (needed for moola)
    p = TrialFunction(V)
    v = TestFunction(V)
    M = Function(W, name='Control')
    a = inner(grad(p), grad(v))*dx
    L = -M*v*dx
    p = Function(V, name='State')
    solve(a == L, p, bc)

    ### Set up the functional:
    control = Control(M)
    functional = (0.5*inner(p_data-p,p_data-p) + (alpha/2)*inner(grad(M), grad(M)))*dx
    J = assemble(functional)
    rf = ReducedFunctional(J, control)

    ### Solve
    M_opt = minimize(rf, options={"disp":True})

    ### Check solution
    p_opt = TrialFunction(V)
    a = inner(grad(p_opt), grad(v))*dx
    L = -M_opt*v*dx
    p_opt = Function(V)
    solve(a == L, p_opt, bc)

    return p_opt, M_opt

if __name__ == "__main__":

    #alpha = 1e-8
    alpha = 1e-3

    #filename = 'pO2_data_low_sigma.npz'
    filename = 'pO2_data_high_sigma.npz'
    #filename = 'pO2_data_high_d.npz'
    p_exact, p_noisy, V, W, bc = read_pO2_from_file(filename)
    
    #p_exact, p_noisy, V, W, bc = create_synthetic_pO2_data()
    
    p_opt, M_opt = estimate_M(p_noisy, V, W, bc, alpha)
    
    e1 = errornorm(p_exact, p_noisy)
    e2 = errornorm(p_exact, p_opt)
    print("Error in noisy signal: ", e1)
    print("Error in restored signal: ", e2)

    ### Save solutions
    file1 = File("results/p_noisy.pvd")
    file1 << p_noisy
    file2 = File("results/p_exact.pvd")
    file2 << p_exact
    file3 = File("results/p_optimal.pvd")
    file3 << p_opt
    file4 = File("results/M_optimal.pvd")
    file4 << M_opt
