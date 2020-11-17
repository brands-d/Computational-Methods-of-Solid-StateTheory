from advection import solve_advection_pde
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fig, ax = plt.subplots()

u = signal.gaussian(1000, 25)[400:]
x = list(range(len(u)))
t_end = 3000
c, dt, dx = 0.1, 0.1, 1

# u = signal.gaussian(100, 2.5)[40:]
# x = list(range(len(u)))
# t_end = 300
# c, dt, dx = 0.1, 0.1, 1

# Initial
ax.plot(x, u, label=fr'Initial Impulse')

# Exact
time, result = solve_advection_pde(u, c, t_end=t_end, dt=dt,
                                   dx=dx, method='exact')
ax.plot(x, result[-1], label=fr'Exact Solution')

# First Order Downwind
time, result = solve_advection_pde(u, c, t_end=t_end, dt=dt,
                                   dx=dx, method='first_order_downwind')
ax.plot(x, result[-1], label=fr'First Order Downwind', linestyle='--',
        marker='x', markersize=5)

# Lax Wendroff
time, result = solve_advection_pde(u, c, t_end=t_end, dt=dt,
                                   dx=dx, method='lax_wendroff')
ax.plot(x, result[-1], label=fr'Lax-Wendroff', linestyle='--',
        marker='o', markersize=5)

# Lax Friedrich
time, result = solve_advection_pde(u, c, t_end=t_end, dt=dt,
                                   dx=dx, method='lax_friedrichs')
ax.plot(x, result[10], label=fr'Lax-Friedrichs', linestyle='--',
        marker='d', markersize=5)

# Leap-Frog
time, result = solve_advection_pde(u, c, t_end=t_end, dt=dt,
                                   dx=dx, method='leap_frog')
ax.plot(x, result[-1], label=fr'Leap-Frog', linestyle='--',
        marker='s', markersize=5)

# # Crank Nicolson
# time, result = solve_advection_pde(u, c, t_end=t_end, dt=dt,
#                                    dx=dx, method='crank_nicolson')
# ax.plot(x, result[-1], label=fr'Crank-Nicolson', linestyle='--',
#         marker='*', markersize=5)

ax.set(xlabel=r'Position', ylabel=fr'Intensity',
       title=r'Different Schemes: $\Delta t = {0:.0f}$; $r = {1:.3f}$'.format(t_end, c * dt / dx))
ax.grid()
ax.legend()
plt.show()
