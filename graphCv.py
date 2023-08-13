# Import modules
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#Initialize starting conditions
N = 10
J = 1
kb = 1
steps = 1024
specific_heats = []
T = np.linspace(0.5, 3.5, 100)
spin = np.random.choice([-1, 1], size = (N, N))

# Define metropolis algorithm
def metropolis(spin, t):
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

# Calculate energy given current lattice
def calculate_energy(spin):
    total_E = 0
    for i in range(N):
        for j in range(N):
            up = spin[(i - 1) % N, j]
            down = spin[(i + 1) % N, j]
            left = spin[i, (j - 1) % N]
            right = spin[i, (j + 1) % N]

            neighbors = up + down + left + right
            current = spin[i, j]

            E = (-J / 2) * current * neighbors
            
            total_E += E
            
    return total_E

# Iterate through temperature range
for t_idx in tqdm(range(len(T))):
    t = T[t_idx]
    energies = []
    for i in range(steps):
        spin = metropolis(spin, t)
    for i in range(steps):
        spin = metropolis(spin, t)
        E = calculate_energy(spin)
        energies.append(E)
    
    specific_heat = np.var(energies) / (N * N * t ** 2)
    specific_heats.append(specific_heat)

#Plot collected data
fig, ax = plt.subplots()
ax.plot(T, specific_heats)
ax.set_xlabel("T")
ax.set_ylabel("Cv / N ^ 2")
ax.set_title(f"Specific heat vs Temperature for {N} by {N} grid")
plt.show()