import numpy as np
import matplotlib.pyplot as plt
from library import Spinor, dirac_solver

np.set_printoptions(precision=3)

# Settings
L = 100
delta = 1 / np.sqrt(2), 1, 1
m, V = 1, 100
t_end = 700

# Initial condition
# Generate Gaussian Input for the real part of the u spinor component
x, y = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L - 1))
d = np.sqrt(x ** 2 + y ** 2)
mu, sigma = 0, 0.25

u = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)), dtype=np.complex_)
v = np.zeros(u.shape, dtype=np.complex_)
s = Spinor(u, v, L)

# print(abs(s))
fig, axs = plt.subplots(1, 3)
axs[0].imshow(abs((u + v).reshape(L - 1, L)) ** 2)
axs[1].imshow(abs(s))

# Calculation
s = dirac_solver(s, m, V, delta, t_end)
# print(abs(s))
axs[2].imshow(abs(s))

plt.show()
