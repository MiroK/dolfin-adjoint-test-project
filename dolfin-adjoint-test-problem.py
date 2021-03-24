from fenics import *
from mshr import *
import numpy as np
import scipy.io as sio
from dolfin_adjoint import *
import moola
np.random.seed(1)
#set_log_level(ERROR)

### first we set up the system:
R_star = 141.       # characteristic length [um]
M_star = 1.0e-3     # charcateristic M [mmHg/um**2]

R_ves = 6/R_star        # vessel radius
p_ves = 80./(M_star*R_star**2)  # pO2 at vessel wall
#sigma = 5e-4
sigma = 1/(M_star*R_star**2)    # noise

mesh = Mesh("rectangular_mesh.xml")
V = FunctionSpace(mesh, 'CG', 1)
W = FunctionSpace(mesh, 'DG', 0)

def boundary1(x, on_boundary):
    r = np.sqrt(x[0]**2 + x[1]**2)
    b = ((r < R_ves+DOLFIN_EPS) and on_boundary)
    return b

bc1 = DirichletBC(V, p_ves, boundary1)
bcs = bc1

### Solve the noiseless system to find the true p
C = 1
p = Function(V)
M = Constant(C)
v = TestFunction(V)
form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
solve(form==0, p, bcs)
p_solution = p.copy(deepcopy=True)

### Create noisy system
noise = sigma*np.random.randn(np.size(np.array(p.vector())))
p_noisy = p.copy(deepcopy=True)
p_noisy.vector()[:] = np.array(p.vector()) + noise

e1 = errornorm(p,p_noisy)

# Solve forward problem (needed for moola)
p = Function(V, name='State')
M = Function(W, name='Control')
form = (inner(nabla_grad(p), nabla_grad(v)) + M*v )*dx
solve(form==0, p, bcs)

### Set up the functional:
l = 1
eps = 1e-8
control = Control(M)
def func(p, M, l, eps):
    return (0.5*inner(p_noisy-p,p_noisy-p) + l*sqrt(inner(nabla_grad(M), nabla_grad(M))+eps))*dx

#J = Functional(func(p,M,l,eps))
J = assemble(func(p,M,l,eps))
rf = ReducedFunctional(J, control)
problem = MoolaOptimizationProblem(rf)
M_moola = moola.DolfinPrimalVector(M, inner_product="L2")
solver = moola.BFGS(problem, M_moola, options={'jtol': 0,
                                               'rjtol': 1e-12,
                                               'gtol': 1e-9,
                                               'Hinit': "default",
                                               'maxiter': 100,
                                               'mem_lim': 10})
### Solve
sol = solver.solve()

### Check solution
M_opt = sol['control'].data
p_opt = Function(V)
form_opt = (inner(nabla_grad(p_opt), nabla_grad(v)) + M_opt*v )*dx
solve(form_opt==0, p_opt,bcs)

e2 = errornorm(p_opt, p_solution)

print("Error in noisy signal: ", e1)
print("Error in restored signal: ", e2)

### Save solutions
file1 = File("results/p_noisy.pvd")
file1 << p_noisy
file2 = File("results/p_exact.pvd")
file2 << p_solution
file3 = File("results/p_optimal.pvd")
file3 << p_opt
file4 = File("results/M_optimal.pvd")
file4 << M_opt
