import numpy as np
from library import grid, dirac_solver

np.set_printoptions(precision=3)

# Settings
L = 25
delta = 0.25, 1, 1
m, V = 10,10
t_end = 50

# Initial condition
u_0 = np.zeros(L**2, dtype=np.complex_)
u_0[5] = 1
u_1 = np.zeros((L - 1)**2, dtype=np.complex_)
u_1[4] = 1
u = grid(u_0, u_1, [(L, L), (L - 1, L - 1)])

v_0 = np.zeros(L * (L - 1), dtype=np.complex_)
v_0[5] = 1
v_1 = np.zeros(L * (L - 1), dtype=np.complex_)
v_1[4] = 0
v = grid(v_0, v_1, [(L, L - 1), (L - 1, L)])

# print('Initial Condition:')
# print('u:\n', u)
# print('v:\n', v)

# Calculation
u, v = dirac_solver(u, v, m, V, delta, t_end)

# print('Result:')
# print('u:\n', u)
# print('v:\n', v)
