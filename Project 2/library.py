import enum
import time
import numpy as np

hc = 1 / 2

global is_timed
is_timed = False


def timeit(func):

    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print('Execution Time ({0}): {1:.5f} seconds'.format(
            func.__name__, end_time - start_time))

        return result

    return wrapper


class grid():

    def __init__(self, x_0, x_1, shapes):

        self.x = [x_0, x_1]
        self.shapes = shapes

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):

        if self.i < len(self.x[0]):
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

        return str(self.x[0].reshape(self.shapes[0])) + \
            '\n\n' + str(self.x[1].reshape(self.shapes[1]))

    def get(self, index):

        return self.x[index[0]][index[1]]

    def set(self, index, value):

        self.x[index[0]][index[1]] = value


@timeit
def dirac_solver(u, v, m, V, delta, t_end):

    dt, dx, dy = delta
    L = int(np.sqrt(len(u[0])))
    time = np.arange(0, t_end, dt)
    f0 = (1 / dt - (m + V) / (2 * 1j * hc))**(-1)
    f1 = 1 / dt + (m + V) / (2 * 1j * hc)
    f2 = (1 / dt - (V - m) / (2 * 1j * hc))**(-1)
    f3 = 1 / dt + (V - m) / (2 * 1j * hc)

    for _ in time:

        # Advance points in u
        for i in u:
            n = get_neighbour(i, 'u', L)
            if n:
                value = f0 * (f1 * u.get(i) - (v.get(n[0]) - v.get(n[1])) /
                              dy - 1j * (v.get(n[3]) - v.get(n[2])) / dx)
                u.set(i, value)

        for i in v:
            n = get_neighbour(i, 'v', L)
            if n:
                value = f2 * (f3 * v.get(i) - (u.get(n[0]) - u.get(n[1])) /
                              dy - 1j * (u.get(n[3]) - u.get(n[2])) / dx)
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
            if i < (L - 1) or i >= (L - 1)**2:
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
