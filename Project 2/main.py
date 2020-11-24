import types
import numpy as np
import matplotlib.pyplot as plt
from library import Spinor, DiracSolver

np.set_printoptions(precision=3)

# Settings
L = 250
t_end = 50
m = 1


def V(t, x, y):
    values = np.zeros(x.shape)
    values[x < 0.3] = 1
    return values


# Initial condition
# Generate Gaussian Input for the real part of the u spinor component
x, y = np.meshgrid(np.linspace(-1, 1, L), np.linspace(-1, 1, L - 1))
d = np.sqrt(x ** 2 + y ** 2)
mu, sigma = 0, 0.025

u = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)), dtype=np.complex_)
v = np.zeros(u.shape, dtype=np.complex_)
s_0 = Spinor(u, v, L)

# Calculation
ds = DiracSolver(s_0, m, V)
s_t = ds.solve(t_end)

# Plot
psi_t = abs(s_t)
psi_0 = abs(s_0)
v_min = np.amin([np.amin(psi_0), np.amin(psi_t)])
v_max = np.amax([np.amax(psi_0), np.amax(psi_t)])
fig, axs = plt.subplots(1, 2)

axs[0].imshow(psi_0, extent=[-1, 1, -1, 1])
axs[0].set_title(r't = {:d}'.format(0), fontsize=20)
axs[0].set(xlabel='x [arb. units]', ylabel='y [arb. units]')

im = axs[1].imshow(psi_t, extent=[-1, 1, -1, 1])
axs[1].set_title(r't = {:d}'.format(t_end), fontsize=20)
axs[1].set(xlabel='x [arb. units]', ylabel='y [arb. units]')

if not isinstance(m, types.FunctionType) and not isinstance(V,
                                                            types.FunctionType):
    plt.suptitle(r'$\left|\Psi\right|^2$; $m_z={:d}$; $\hat{{V}}={:d}$; '
                 r'$i\hbar c=1$'.format(m, V), fontsize=24)
else:
    plt.suptitle(r'$\left|\Psi\right|^2$; $i\hbar c=1$'.format(m, V),
                 fontsize=24)

fig.canvas.set_window_title('Staggered Grid Solution of the Dirac Equation')

#fig.colorbar(im, ax=axs.ravel().tolist())
plt.show()
