import types
import numpy as np
import matplotlib.pyplot as plt
from library import Spinor, DiracSolver
import matplotlib.animation as animation

np.set_printoptions(precision=3)

# Settings
L = 250
t_end = 200
m = 0.8
V = 1


def V(t, x, y):
    values = 3 * np.ones(x.shape)
    values = x
    return values


# Initial condition
# Generate Gaussian Input for the real part of the u spinor component
x, y = np.meshgrid(np.linspace(-1, 1, L), np.linspace(1, -1, L - 1))
d = np.sqrt(x ** 2 + y ** 2)
mu, sigma = 0, 0.1

u = np.exp(-((d - 0.3) ** 2 / (2.0 * sigma ** 2)), dtype=np.complex_)
v = 1j*np.exp(-((d + 0.3) ** 2 / (2.0 * sigma ** 2)),
           dtype=np.complex_)
s_0 = Spinor(u, v, L)

# Calculation
ds = DiracSolver(s_0, m, V)
save_points = [*list(range(t_end)), -1]
s_t = ds.solve(t_end, saves_points=save_points)

# # Plot
# fig, axs = plt.subplots(2, int(len(save_points) / 2))
# for i, s in enumerate(s_t):
#     axs.flatten()[i].imshow(abs(s[1]), extent=[-1, 1, 1, -1])
#     axs.flatten()[i].set_title(r't = {:.3f}'.format(s[0]), fontsize=20)
#     axs.flatten()[i].set(xlabel='x [arb. units]', ylabel='y [arb. units]')
#
# if not isinstance(m, types.FunctionType) and not isinstance(V,
#                                                             types.FunctionType):
#     plt.suptitle(r'$\left|\Psi\right|^2$; $m_z={:d}$; $\hat{{V}}={:d}$; '
#                  r'$i\hbar c=1$'.format(m, V), fontsize=24)
# else:
#     plt.suptitle(r'$\left|\Psi\right|^2$; $i\hbar c=1$'.format(m, V),
#                  fontsize=24)
#
# fig.canvas.set_window_title('Staggered Grid Solution of the Dirac Equation')

ims = []
for s in s_t:
    im = plt.imshow(abs(s[1]), extent=[-1, 1, 1, -1], animated=True)
    ims.append([im])

fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=25,
                                blit=True, repeat_delay=0)

plt.show()
