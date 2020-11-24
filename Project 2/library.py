import time
import types
import copy
import numpy as np
from scipy.interpolate import interpolate


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

    def mul(self, other):
        self.x *= other
        return self

    def add(self, other):
        x = 1
        mask = np.invert(np.isnan(other))
        self.x[mask] += other[mask]
        return self

    def sub(self, other):
        self.add(-other)
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


class Spinor:

    def __init__(self, u, v, L=None):
        if L is None:
            self.u = u
            self.v = v
            self.L = u.L
        else:
            self.u = SpinorComponent(u, L)
            self.v = SpinorComponent(v, L)
            self.L = L

    def __abs__(self):
        return abs(self.u) + abs(self.v)

    def __iter__(self):
        return iter((self.u, self.v))

    def __repr__(self):
        return 'u:\n' + self.u.__repr__() + '\nv:\n' + self.v.__repr__()


class DiracSolver:

    def __init__(self, spinor, m, V):
        self.spinor = spinor
        self.n = SpinorComponent.get_all_neighbours(spinor.L).T
        self.k_u, self.k_v = self.get_factors(m, V, spinor.L)

    @timeit
    def solve(self, t_end, delta=(1 / np.sqrt(2), 1, 1)):
        u, v = copy.deepcopy(self.spinor)
        dt, dx, dy = delta
        r = dt / np.array([dx, dy])
        time = np.arange(0, t_end, dt)

        self.advance_u(u, v, self.n, self.k_u(0), -r / 2, -dt / 2)

        for t in time:
            self.advance_u(u, v, self.n, self.k_u(t), r, dt)
            self.advance_v(u, v, self.n, self.k_v(t + dt / 2), r, dt)

        self.advance_u(u, v, self.n, self.k_u(t_end), r / 2, dt / 2)

        return Spinor(u, v)

    @classmethod
    def advance_u(cls, u, v, n, k, r, dt):
        u = u.mul(1 + k * dt / 2)
        u = u.sub(cls.y_diff(v, n, r[1]))
        u = u.sub(cls.x_diff(v, n, r[0]))
        return u.mul((1 - k * dt / 2) ** (-1))

    @classmethod
    def advance_v(cls, u, v, n, k, r, dt):
        v = v.mul((1 + k * dt / 2))
        v = v.sub(cls.y_diff(u, n, r[1]))
        v = v.add(cls.x_diff(u, n, r[0]))
        return v.mul((1 - k * dt / 2) ** (-1))

    @classmethod
    def x_diff(cls, x, n, r):
        return (x[n[3]] - x[n[2]]) * 1j * r

    @classmethod
    def y_diff(cls, x, n, r):

        return (x[n[0]] - x[n[1]]) * r

    @classmethod
    def get_factors(cls, m, V, L):
        ihc = 1j
        x, y = np.meshgrid(np.linspace(-1, 1, L),
                           np.linspace(-1, 1, L - 1))

        if isinstance(m, types.FunctionType):
            if isinstance(V, types.FunctionType):
                k_u = lambda t: (m(t, x, y) + V(t, x, y)) / ihc
                k_v = lambda t: (V(t, x, y) - m(t, x, y)) / ihc
            else:
                k_u = lambda t: (m(t, x, y) + V) / ihc
                k_v = lambda t: (V - m(t, x, y)) / ihc
        else:
            if isinstance(V, types.FunctionType):
                k_u = lambda t: (m + V(t, x, y)) / ihc
                k_v = lambda t: (V(t, x, y) - m) / ihc
            else:
                k_u = lambda t: (m + V) / ihc
                k_v = lambda t: (V - m) / ihc

        return k_u, k_v
