import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

data = np.load('synthetic_data/pO2_data_sigma_0.npz')

p = data['p']
p_noisy = data['p_noisy']
x = data['x']
y = data['y']

plt.figure(1)
X, Y = np.meshgrid(x, y)
cmap = cm.Reds
#f = plt.imshow(p, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap=cm.get_cmap(cmap))
f = plt.imshow(p_noisy, extent=[min(x), max(x), min(y), max(y)], origin='lower', cmap=cm.get_cmap(cmap))
cbar = plt.colorbar(f)
plt.show()
