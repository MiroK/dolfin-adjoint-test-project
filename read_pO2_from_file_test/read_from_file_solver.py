from fenics import *
from mshr import *
import numpy as np
import scipy.io as sio
from dolfin_adjoint import *
import moola
import scipy.io as sio
np.random.seed(1)
#set_log_level(ERROR)
import matplotlib.pyplot as plt
#from scipy.misc import imread

def add_holes(filename, p, p_noisy, p_solution):
    ### interpolate pO2 values from mesh wo/ holes to mesh w/ holes
    
    # load data
    data = sio.loadmat(filename)
    corners = data['corners']
    hole_coor = data['hole_coor']
    r_ves = int(data['r_ves'][0][0])
    p_ves = data['p_ves'][0]
    
    # new mesh
    r = Rectangle(Point(corners[0][0], corners[0][1]), Point(corners[1][0], corners[1][1]))
    circle1 = Circle(Point(int(hole_coor[0][0]), int(hole_coor[0][1])), r_ves)
    circle2 = Circle(Point(int(hole_coor[1][0]), int(hole_coor[1][1])), r_ves)
    domain = r - circle1 - circle2
    mesh = generate_mesh(domain, 150)
   
    # interpolate
    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    
    p = interpolate(p, V, annotate=False)
    p_noisy = interpolate(p_noisy, V, annotate=False)
    p_solution = interpolate(p_solution, V, annotate=False)
    
    # boundary conditions 
    def boundary1(x, on_boundary):
        r = np.sqrt((x[0]-hole_coor[0][0])**2 + (x[1]-hole_coor[0][1])**2)
        b = ((r < r_ves+DOLFIN_EPS) and on_boundary)
        return b

    def boundary2(x, on_boundary):
        r = np.sqrt((x[0]-hole_coor[1][0])**2 + (x[1]-hole_coor[1][1])**2)
        b = ((r < r_ves+DOLFIN_EPS) and on_boundary)
        return b
    
    def boundary3(x, on_boundary):
        return on_boundary

    bc1 = DirichletBC(V, p_ves[0], boundary1)
    bc2 = DirichletBC(V, p_ves[1], boundary2)
    bc3 = DirichletBC(V, p_solution, boundary3)

    bcs = [bc1, bc2, bc3] 

    return p, p_noisy, p_solution, bcs, V, W, v

def import_pO2grid(filename):
    ### This is a hacky way of importing a pO2 grid to FEniCS.
    
    data = np.load(filename)
    Nx = len(data['x'])
    Ny = len(data['y'])
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx-1, Ny-1)
    p_data = data['pO2']
    p_vector = np.reshape(p_data, (int(float(Nx)*Ny),1) )
    p_noisy_data = data['pO2_noisy']
    p_noisy_vector = np.reshape(p_noisy_data, (int(float(Nx)*Ny),1) )

    V = FunctionSpace(mesh, 'CG', 1)
    W = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)

    d2v = dof_to_vertex_map(V)
    p = Function(V)
    p_noisy = Function(V)
    p.vector()[:] = p_vector[d2v]
    p_noisy.vector()[:] = p_noisy_vector[d2v]
    
    p_solution = p.copy(deepcopy=True)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, p_solution, boundary)

    return p, p_noisy, p_solution, bc, V, W, v, Nx, Ny

def inverse_poisson(filename, holes, maxiter):

    # maa fikse wo holes navnene
    p, p_noisy, p_solution, bc, V_without_holes, W_without_holes, v, Nx, Ny = import_pO2grid(filename)
    if holes:
        p, p_noisy, p_solution, bc, V, W, v = add_holes(filename, p, p_noisy, p_solution)
        p = Function(V, name='State')
        M = Function(W, name='Control')
    else:
        p = Function(V_without_holes, name='State')
        M = Function(W_without_holes, name='Control')

    ### Solve forward problem (needed for moola)
    form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
    solve(form==0, p, bc)

    ### Set up the functional:
    l = 1
    eps = 1e-8
    control = Control(M)
    def func(p, M, l, eps):
        return (0.5*inner(p_noisy-p,p_noisy-p) + l*sqrt(inner(nabla_grad(M), nabla_grad(M))+eps))*dx

    J = Functional(func(p,M,l,eps))
    rf = ReducedFunctional(J, control)
    problem = MoolaOptimizationProblem(rf)
    M_moola = moola.DolfinPrimalVector(M, inner_product="L2")
    solver = moola.BFGS(problem, M_moola, options={'jtol': 0,
                                                   'rjtol': 1e-8,
                                                   'gtol': 1e-9,
                                                   'Hinit': "default",
                                                   'maxiter': maxiter,
                                                   'mem_lim': 10})
    ### Solve
    sol = solver.solve()

    ### Check solution
    M_opt = sol['control'].data
    if holes:
        p_opt = Function(V)
    else: 
        p_opt = Function(V_without_holes)
    form_opt = (inner(nabla_grad(p_opt), nabla_grad(v)) + M_opt*v )*dx
    solve(form_opt==0, p_opt, bc)

    ### reshape M
#    if holes: # extrapolate
#        parameters['allow_extrapolation'] = True
#        u = Function(W)
#        u.assign(M_opt)
#        u = project(u, W_without_holes)
#        M_array = u.vector().array()
#    else:
#        M_array = M_opt.vector().array()     
#    M_array = M_array[vertex_to_dof_map(W_without_holes)]
#    M_grid = np.reshape(M_array, (Nx,Ny))

    return p_solution, p_noisy, p_opt, M_opt, M_grid


if __name__ == "__main__":

    filename = 'pO2_data.npz'

    p_solution, p_noisy, p_opt, M_opt, M_grid = inverse_poisson(filename, holes=False, maxiter=50)

    e1 = errornorm(p_noisy, p_solution)
    e2 = errornorm(p_opt, p_solution)

    print("Error in noisy signal: ", e1)
    print("Error in restored signal: ", e2)

    ### Save solutions
    file1 = File("p_noisy.pvd")
    file1 << p_noisy
    file2 = File("p_exact.pvd")
    file2 << p_solution
    file3 = File("p_optimal.pvd")
    file3 << p_opt
    file4 = File("M_opt.pvd")
    file4 << M_opt
