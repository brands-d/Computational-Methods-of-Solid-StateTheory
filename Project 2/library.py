import time
import numpy as np
from scipy.interpolate import interpolate

ihc = 1j


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Execution Time ({0}): {1:.5f} seconds'.format(
            func.__name__, end_time - start_time))
        return result

    return wrapper


class SpinorComponent:

    def __init__(self, x, L):
        self.L = L
        self.x = x

    def __getitem__(self, i):
        values = np.nan * np.zeros(self.x.shape, dtype=np.complex_)
        mask = np.invert(np.isnan(i))
        values[mask] = self.x.flatten()[i[mask].astype(int)]
        return values

    def __abs__(self):
        return np.real(self.x * np.conjugate(self.x))

    def __rmul__(self, other):
        self.x *= other
        return self

    def __add__(self, other):
        mask = np.invert(np.isnan(other))
        self.x[mask] += other[mask]
        return self

    def __sub__(self, other):
        self.__add__(-other)
        return self

    def __repr__(self):
        return self.x.__repr__()

    @classmethod
    def get_all_neighbours(cls, L):
        num_points = L * (L - 1)
        n = [SpinorComponent.get_neighbour(i, L) for i in range(num_points)]

        return np.array(n).reshape(L - 1, L, 4).transpose((1, 0, 2))

    @classmethod
    def get_neighbour(cls, i, L):
        if SpinorComponent.is_boundary_point(i, L):
            n = [np.nan, np.nan, np.nan, np.nan]
        else:
            n = [i - L, i + L, i + 1, i - 1]
        # Upper, lower, right-side, left-side neighbour index
        return n

    @classmethod
    def is_boundary_point(cls, i, L):
        if i < L or i >= L * (L - 1) - L:
            return True
        elif i % L == 0 or i % L == L - 1:
            return True
        else:
            return False

    @classmethod
    def reg_to_staggered_index(cls, i, L):
        row = int(i / L)
        col = i - L * row
        grid = 'u' if (row % 2 == 0) == (col % 2 == 0) else 'v'
        idx = int(i / 2) if i % 2 == 0 else int((i - 1) / 2)
        return grid, idx

    @classmethod
    def staggered_indicies(cls, L, type_):
        num_points = L * (L - 1)
        idx = []
        # Indices of the u component grid points
        for i in range(num_points):
            grid, j = SpinorComponent.reg_to_staggered_index(i, L)
            if grid == type_:
                idx.append(j)

        return idx

    @classmethod
    def split_reg(cls, x, L):
        u = SpinorComponent.reg_to_staggered(x, L, 'u')
        v = SpinorComponent.reg_to_staggered(x, L, 'v')
        return u, v

    @classmethod
    def reg_to_staggered(cls, x, L, type_):
        idx = SpinorComponent.staggered_indicies(L, type_)
        return x.flatten()[idx]

    @classmethod
    def staggered_to_reg(self, x, L):
        grid = np.zeros(L * (L - 1), dtype=np.complex_)
        u_idx = SpinorComponent.staggered_indicies(L, 'u')
        v_idx = SpinorComponent.staggered_indicies(L, 'v')
        grid[u_idx] = x['u']
        grid[v_idx] = x['v']

        return grid.reshape(L - 1, L)


class Spinor:

    def __init__(self, u, v, L):
        self.u = SpinorComponent(u, L)
        self.v = SpinorComponent(v, L)
        self.L = self.u.L

    def __abs__(self):
        return abs(self.u) + abs(self.v)

    def __iter__(self):
        return iter((self.u, self.v))

    def __repr__(self):
        return 'u:\n' + self.u.__repr__() + '\nv:\n' + self.v.__repr__()


@timeit
def dirac_solver(spinor, m, V, delta, t_end):
    dt, dx, dy = delta
    time = np.arange(0, t_end, dt)
    u, v = spinor
    f = [(1 / dt - (m + V) / ihc) ** (-1), 1 / dt + (m + V) / ihc,
         (1 / dt - (V - m) / ihc) ** (-1), 1 / dt + (V - m) / ihc]
    # Factor for half a time-step
    f_2 = [(2 / dt - (m + V) / ihc) ** (-1), 2 / dt + (m + V) / ihc,
           (2 / dt - (V - m) / ihc) ** (-1), 2 / dt + (V - m) / ihc]
    n = SpinorComponent.get_all_neighbours(spinor.L).T

    def advance_u(u, v, n, f):
        up_down = (v[n[0]] - v[n[1]]) / dy
        right_left = (v[n[3]] - v[n[2]]) * 1j / dx
        u = f[0] * (f[1] * u - up_down - right_left)
        return u

    def advance_v(u, v, n, f):
        up_down = (u[n[0]] - u[n[1]]) / dy
        right_left = (u[n[3]] - u[n[2]]) * 1j / dx
        v = f[2] * (f[3] * v - up_down + right_left)
        return v

    advance_v(u, v, n, f_2)

    for _ in time:
        advance_u(u, v, n, f)
        advance_v(u, v, n, f)

    advance_u(u, v, n, f_2)

    return spinor
