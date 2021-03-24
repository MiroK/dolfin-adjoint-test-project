import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

data = np.load('pO2_data.npz')

pO2 = data['pO2']
pO2_noisy = data['pO2_noisy']
x = data['x']
y = data['y']
r = data['r']

plt.figure(1)
X, Y = np.meshgrid(x, y)
cmap = cm.Reds
f = plt.imshow(pO2_noisy, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap=cm.get_cmap(cmap))
cbar = plt.colorbar(f)
plt.show()
