from fenics import *
from mshr import *
import numpy as np
import scipy.io as sio
from dolfin_adjoint import *
import moola
np.random.seed(1)

### first we set up the system:
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

bcs = DirichletBC(V, p_ves, boundary)

### Solve the noiseless system to find the true p
p = TrialFunction(V)
v = TestFunction(V)
M = Constant(M)
a = inner(grad(p), grad(v))*dx
L = -M*v*dx
p = Function(V)
solve(a == L, p, bcs)
p_solution = p.copy(deepcopy=True)

### Create noisy system
L = -M*v*dx
p = Function(V)
solve(a == L, p, bcs)
p_solution = p.copy(deepcopy=True)

### Create noisy system
N = V.dim()
noise = sigma*np.random.randn(N)
p_noisy = p.copy(deepcopy=True)
p_noisy.vector()[:] += noise

e1 = errornorm(p,p_noisy)

# Solve forward problem (needed for moola)
p = TrialFunction(V)
M = Function(W, name='Control')
a = inner(grad(p), grad(v))*dx
L = -M*v*dx
p = Function(V, name='State')
solve(a == L, p, bcs)

### Set up the functional:
alpha = 1e-8
control = Control(M)
functional = (0.5*inner(p_noisy-p,p_noisy-p) + (alpha/2)*inner(grad(M), grad(M)))*dx
J = assemble(functional)
rf = ReducedFunctional(J, control)

### Solve
M_opt = minimize(rf, options={"disp":True})

### Check solution
p_opt = TrialFunction(V)
a = inner(grad(p_opt), grad(v))*dx
L = -M_opt*v*dx
p_opt = Function(V)
solve(a == L, p_opt, bcs)

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
