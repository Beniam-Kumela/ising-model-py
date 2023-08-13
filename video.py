# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# User dialogue
print('''
Welcome to the Ising Model Animator!
The Ising Model is one of the simplest mathematical models from which phase transitions can be observed.
We will create an NxN matrix of random spin up and spin down particles.
It will then be quenched as we observe order arising from chaos.
As the system cools to 'Curie Temperature' clumps start to form, representing spontaneous magnetization.
Let's get started by initializing conditions!
    ''')

# User input
N = int(input("Enter grid size: "))
T = float(input("Enter temperature: "))
F = int(input("Enter frames: "))

# Initialize remaining conditions
J = 1
kb = 1
steps = 10
spin = np.random.choice([-1, 1], size=(N, N))
E = []
M = []

# Define metropolis algorithm
def metropolis(spin):
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
            dE = -2 * E

            if dE < 0:
                spin[i, j] = -1 * current
            elif np.exp(-dE / (kb * T)) > np.random.rand():
                spin[i, j] = -1 * current
            
            total_E += E 
             
    m = np.mean(spin)
            
    return spin, total_E, m   
    
# Define animation function
def animate(frame):
    global spin
    for i in range(steps):
        spin, total_E, m = metropolis(spin)
        E.append(total_E)
        M.append(m)
        
    im = plt.imshow(spin, cmap = 'gray', vmin = -1, vmax = 1)
    plt.title(f"{N} by {N} grid for {frame*10} MC Moves") 
    return im

# Initialize plots
fig, ax = plt.subplots()
ax.axis("off")

# Compile and save animation
anim = FuncAnimation(fig, animate, frames = tqdm(range(F)), interval = 100, repeat = True)
anim.save(f"Ising Model for {T} T, {F} frames, {N} grid.mp4", writer = "ffmpeg")
print(f"Animation was successfully saved as 'Ising Model for {T} T, {F} frames, {N} grid.mp4' within the program directory.")

# Graph magnetization and energy over time
fig, ax = plt.subplots(1, 2)
time = np.arange(0, len(E))

ax[0].scatter(time, E, s = 0.5)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Total System Energy')

ax[1].scatter(time, M, s = 0.5)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Average System Magnetization')

plt.tight_layout()
plt.savefig(f"Plot for {T} T, {F} frames, {N} grid.png")
print(f"Graph was successfully saved as 'Plot for {T} T, {F} frames, {N} grid.png' within the program directory.")
plt.close()