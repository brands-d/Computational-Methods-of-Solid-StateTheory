import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from library import Grid, dirac_solver, abs_square_spinor

np.set_printoptions(precision=3)

# Settings
L = 10
delta = 1 / np.sqrt(2), 1, 1
m, V = 0, 0
t_end = 1000

# Initial condition
# Generate Gaussian Input for the real part of the u spinor component
x, y = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L))
d = np.sqrt(x ** 2 + y ** 2)
mu, sigma = 0, 0.05
u_0 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)), dtype=np.complex_)
u_1 = np.zeros((L, L), dtype=np.complex_)
u = Grid([u_0, u_1], L, type_='u')

v_0 = np.zeros((L, L), dtype=np.complex_)
v_1 = np.zeros((L, L), dtype=np.complex_)
v = Grid([v_0, v_1], L, type_='v')

fig, axs = plt.subplots(1, 2)
axs[0].imshow(abs_square_spinor(u, v))
# Calculation
u, v = dirac_solver(u, v, m, V, delta, t_end)

axs[1].imshow(abs_square_spinor(u, v))
plt.show()
