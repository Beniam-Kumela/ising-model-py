# Import modules
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Initialize starting conditions
J = 1
kb = 1
steps = 1024
magnetization = []
T = np.linspace(0.5, 4, 100)


# Define metropolis algorithm
def metropolis(spin, t, N):
    for i in range(N):
        for j in range(N):
            up = spin[(i - 1) % N, j]
            down = spin[(i + 1) % N, j]
            left = spin[i, (j - 1) % N]
            right = spin[i, (j + 1) % N]

            neighbors = up + down + left + right
            current = spin[i, j]

            E = (-J / 2) * current * neighbors
            dE = -2 * E

            if dE < 0:
                spin[i, j] = -1 * current
            elif np.exp(-dE / (kb * t)) > np.random.rand():
                spin[i, j] = -1 * current
                
    return spin

# Calculate average magnetization 
def calculate_mag(spin):
    M = np.mean(spin)
    return M

# Create general graph function
def graphM(N, c):
    magnetization = []
    for t_idx in tqdm(range(len(T))):
        spin = np.random.choice([-1, 1], size = (N, N))
        t = T[t_idx]
        magnetizations = []
        for i in range(steps):
            spin = metropolis(spin, t, N)
        for i in range(steps):
            spin = metropolis(spin, t, N)
            M = calculate_mag(spin)
            magnetizations.append(abs(M))
            
        m = np.mean(magnetizations)
        magnetization.append(m)
    
    plt.plot(T, magnetization, c, label=f"{N} grid")

fig, ax = plt.subplots()

# Run for different lattice sizes and show results
graphM(4, "b")
graphM(8, "r")
graphM(16, "g")

ax.set_xlabel("T")
ax.set_ylabel("|M|")
ax.set_title("Magnetization vs Temperature")
ax.legend()

plt.show()