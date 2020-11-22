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


def complex_interp(x, n):
    n_diag, n_para = n
    real_diag = np.real(x[n_diag])
    real_para = np.real(x[n_para])
    imag_diag = np.imag(x[n_diag])
    imag_para = np.imag(x[n_para])

    v_real = (np.mean(real_diag) + np.sqrt(2) * np.mean(real_para)) / 2
    v_imag = (np.mean(imag_diag) + np.sqrt(2) * np.mean(imag_para)) / 2
    return v_real + 1j * v_imag


class SpinorComponent:

    def __init__(self, x, L, type_):
        self.L = L
        self.type_ = type_

        self.x = SpinorComponent.reg_to_staggered(x, L, type_)

    def __getitem__(self, i):
        if isinstance(i, int):
            return np.nan if i is np.nan else self.x[i]

        else:
            value = np.zeros(len(i), dtype=np.complex_) * np.nan
            mask = np.invert(np.isnan(i))
            value[mask] = self.x[i[mask].astype(int)]
            return value

    def __setitem__(self, i, value):
        self.x[i] = value

    def __abs__(self):
        x = SpinorComponent.staggered_to_reg(self.x, self.L, self.type_)
        return np.real(x * np.conjugate(x))

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
    def get_other_grid_neighbour(cls, i, L, type_):

        # k is aux index. Is 1 for even layer on staggered grid, else 0.
        k = int(i / L) % 2
        if SpinorComponent.is_boundary_point(i, L, type_):
            n = [np.nan, np.nan, np.nan, np.nan]
        elif type_ == 'u':
            n = [i - L, i + L, i + k, i - (1 - k)]
        else:
            n = [i - L, i + L, i - k + 1, i - k]

        # Upper, lower, right-side, left-side neighbour index
        return n

    def get_same_grid_neighbour(i, L, type_):
        if type_ == 'u':
            is_left = i % (2 * L) == 0
            is_almost_left = i % (2 * L) == 0
            is_right = i % (2 * L) == 2 * L - 1
            is_almost_right = i % L == 2
        else:
            is_left = (i + L) % (2 * L) == 0
            is_almost_left = i % L == 0
            is_right = (i + L) % (2 * L) == 2 * L - 1
            is_almost_right = i % (2 * L) == 2 * L - 1

        if is_left or is_right or is_almost_left or is_almost_right:
            if is_left or is_right:
                n_diag = [i + L, i - L]
            else:
                n_diag = [i + L, i + L - 1, i - L, i - L - 1]

            if is_left:
                n_para = [i + 1, i + 2 * L, i - 2 * L]
            elif is_right:
                n_para = [i + 2 * L, i - 1, i - 2 * L]
            elif is_almost_left:
                n_para = [i + 2 * L, i - 1, i - 2 * L]
            else:
                n_para = [i + 2 * L, i - 1, i - 2 * L]
        else:
            n_diag = [i + L, i + L - 1, i - L, i - L - 1]
            n_para = [i + 1, i + 2 * L, i - 1, i - 2 * L]

        n_diag = np.array(n_diag)
        n_para = np.array(n_para)
        n_diag = n_diag[np.logical_and(n_diag >= 0, n_diag < 2 * L ** 2 - L)]
        n_para = n_para[np.logical_and(n_para >= 0, n_para < 2 * L ** 2 - L)]

        return [n_diag, n_para]

    @classmethod
    def is_boundary_point(cls, i, L, type_):

        # Tests for upper or lower lattice boundary
        if i < L or i >= 2 * (L ** 2 - L):
            return True
        elif type_ == 'u' and i % (2 * L) in [0, 2 * L - 1]:
            return True
        elif type_ == 'v' and (i + L) % (2 * L) in [0, 2 * L - 1]:
            return True

    @classmethod
    def staggered_index(cls, L, type_):
        num_points = 2 * L * (2 * L - 1)
        idx = list(range(num_points))
        # Indices of the u component grid points
        u_idx = [i if int(i / (2 * L)) % 2 == 0 else i + 1 for i in
                 range(0, num_points, 2)]

        if type_ == 'u':
            return u_idx
        else:
            return np.delete(idx, u_idx)

    @classmethod
    def reg_to_staggered(cls, x, L, type_):
        idx = SpinorComponent.staggered_index(L, type_)
        return x.flatten()[idx]

    @classmethod
    def staggered_to_reg(self, x, L, type_):
        grid = np.zeros(2 * L * (2 * L - 1), dtype=np.complex_)
        exist_idx = SpinorComponent.staggered_index(L, type_)
        missing_idx = SpinorComponent.staggered_index(L, 'u' if type_ == 'v'
        else 'v')

        grid[exist_idx] = x
        for i, j in zip(missing_idx, range(2 * L ** 2 - L)):
            n = SpinorComponent.get_same_grid_neighbour(j, L, type_)
            grid[i] = complex_interp(x, n)

        return grid.reshape(2 * L - 1, 2 * L)

    def get_all_other_grid_neighbours(self):
        n = [SpinorComponent.get_other_grid_neighbour(i, self.L, self.type_)
             for i in
             range(len(self.x))]
        return np.array(n)


class Spinor:

    def __init__(self, u, v, L):
        self.u = SpinorComponent(u, L, type_='u')
        self.v = SpinorComponent(v, L, type_='v')

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
    n_u = u.get_all_other_grid_neighbours().T
    n_v = v.get_all_other_grid_neighbours().T

    def advance_u(u, v, n):
        up_down = (v[n[0]] - v[n[1]]) / dy
        right_left = (v[n[3]] - v[n[2]]) * 1j / dx
        u = f[0] * (f[1] * u - up_down - right_left)
        return u

    def advance_v(u, v, n):
        up_down = (u[n[0]] - u[n[1]]) / dy
        right_left = (u[n[3]] - u[n[2]]) * 1j / dx
        v = f[2] * (f[3] * v - up_down + right_left)
        return v

    for _ in time:
        advance_u(u, v, n_u)
        advance_v(u, v, n_v)

    return spinor
