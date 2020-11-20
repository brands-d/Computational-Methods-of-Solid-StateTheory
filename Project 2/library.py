import time
import copy
import numpy as np
from scipy.interpolate import interpolate

hc = 1 / 2


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print('Execution Time ({0}): {1:.5f} seconds'.format(
            func.__name__, end_time - start_time))

        return result

    return wrapper


class Grid:

    def __init__(self, x, L, type_='u'):

        if type_ == 'u':
            self.x = [x[0], Grid.grid_transform(x[1], (L - 1, L - 1), L)]
        else:
            self.x = [Grid.grid_transform(x[0], (L - 1, L), L),
                      Grid.grid_transform(x[1], (L, L - 1), L)]

        self.L = L

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):

        if self.i < len(self.x[0].flatten()):
            i = self.i
            self.i += 1

            return [0, i]

        elif self.i < (len(self.x[0]) + len(self.x[1])):
            i = self.i
            self.i += 1

            return [1, i - len(self.x[0])]

        else:
            raise StopIteration

    def __getitem__(self, i):

        return self.x[i]

    def __str__(self):

        return str(self.x[0]) + '\n\n' + str(self.x[1])

    def __abs__(self):

        # Regrid them to the regular grid and sum them
        L = self.L
        x = np.sum([Grid.grid_transform(self.x[i], (L, L), L) for i in [0, 1]],
                   axis=0)

        return (x * np.conj(x)).real.astype(np.float)

    @classmethod
    def grid_transform(cls, u, new_shape, L):

        axis_old = [None, None]
        axis_new = [None, None]
        for i in [0, 1]:
            axis_old[i] = Grid.grid_to_axis(u.shape[i], L)
            if u.shape[i] == new_shape[i]:
                axis_new[i] = axis_old[i]

            else:
                axis_new[i] = Grid.grid_to_axis(new_shape[i], L)

        real = interpolate.interp2d(axis_old[1], axis_old[0], np.real(u),
                                    bounds_error=False)(*axis_new)
        imag = interpolate.interp2d(axis_old[1], axis_old[0], np.imag(u),
                                    bounds_error=False)(*axis_new)

        return real + 1j * imag.astype(np.complex_)

    @classmethod
    def grid_to_axis(cls, grid_points, L):
        half_step = 1 / (L - 1)
        diff = (L - grid_points) * half_step
        axis = np.linspace(-1 + diff, 1 - diff, grid_points, endpoint=True,
                           dtype=np.float)

        return axis

    def get(self, index):

        x = self.x[index[0]]
        idx = np.unravel_index(index[1], x.shape)
        value = x[idx]

        return value

    def set(self, index, value):

        x = self.x[index[0]]
        idx = np.unravel_index(index[1], x.shape)

        x[idx] = value


@timeit
def dirac_solver(u, v, m, V, delta, t_end):
    dt, dx, dy = delta
    L = len(u[0])
    time = np.arange(0, t_end, dt)
    f = 2 * 1j * hc
    f0 = (1 / dt - (m + V) / f) ** (-1)
    f1 = 1 / dt + (m + V) / f
    f2 = (1 / dt - (V - m) / f) ** (-1)
    f3 = 1 / dt + (V - m) / f

    for _ in time:

        # Advance points in u
        for i in u:
            n = get_neighbour(i, 'u', L)
            if n:
                a = u.get(i)
                a = v.get(n[0])
                a = v.get(n[1])
                a = v.get(n[3])
                a = v.get(n[2])
                value = f0 * (f1 * u.get(i) - (v.get(n[0]) - v.get(n[1])) /
                              dy - 1j * (v.get(n[3]) - v.get(n[2])) / dx)
                u.set(i, value)

        for i in v:
            n = get_neighbour(i, 'v', L)
            if n:
                value = f2 * (f3 * v.get(i) - (u.get(n[0]) - u.get(n[1])) /
                              dy + 1j * (u.get(n[3]) - u.get(n[2])) / dx)
                v.set(i, value)

    return u, v


def get_neighbour(index, grid, L):
    index, i = index

    if grid == 'u':
        if index == 0:
            if i < L or i >= L * (L - 1) or i % L == 0 or i % L == L - 1:
                # boundary point
                return False

            else:
                temp = i - int(i / L)
                return [[1, i - L], [1, i], [0, temp - 1], [0, temp]]

        else:
            temp = i + int(i / (L - 1))
            return [[0, i], [0, i + L - 1], [1, temp], [1, temp + 1]]

    elif grid == 'v':
        if index == 0:
            if i < (L - 1) or i >= (L - 1) ** 2:
                # boundary point
                return False

            else:
                temp = i + int(i / (L - 1))
                return [[1, i - (L - 1)], [1, i], [0, temp], [0, temp + 1]]

        else:
            if i % L == 0 or i % L == L - 1:
                # boundary point
                return False

            else:
                temp = i - int(i / L)
                return [[0, i], [0, i + L], [1, temp - 1], [1, temp]]


def abs_square_spinor(u, v):
    abs_ = [abs(x) for x in [u, v]]

    return np.sqrt(np.sum(abs_, axis=0))
