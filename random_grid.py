import matplotlib.pyplot as plt
import numpy as np

N = 512

spin = np.random.choice([-1, 1], size = (N, N))

plt.imshow(spin, cmap = 'gray', vmin = -1, vmax = 1)

plt.show()