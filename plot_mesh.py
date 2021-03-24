from fenics import *
import matplotlib.pyplot as plt

mesh = Mesh("rectangular_mesh.xml")
plot(mesh)
plt.show()
