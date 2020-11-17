import numpy as np
import time

global is_timed
is_timed = True


def timeit(func):

    def wrapper(*args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print('Execution Time ({0}): {1:.5f} seconds'.format(
            func.__name__, end_time - start_time))

        return result

    return wrapper


@timeit
def solve_advection_pde(u, c, t_end=1, dt=1, dx=1, method='crank_nicolson'):

    f = eval(method)
    r = c * dt / dx

    time = np.arange(0, t_end, dt)
    result = [u]

    for t in time[1:]:
        if method == 'exact':
            result.append(exact(u, t, c))

        else:
            result.append(f(result[-1], r))

    return time, result


def exact(u, t, c):

    shift = int(t * c)
    result = u.copy()

    if shift == 0:
        return result

    else:
        result[shift:] = u[:-shift]
        result[:shift] = result[0]

    return result


def first_order_downwind(u, r):

    result = u.copy()

    if r < 0:
        result[:-1] = u[:-1] - r * (u[1:] - u[:-1])

    else:
        result[1:] = u[1:] - r * (u[1:] - u[:-1])

    return result


def lax_wendroff(u, r):

    result = u.copy()
    result[1:-1] = u[1:-1] - (r / 2) * (u[2:] - u[:-2]) + \
        (r**2 / 2) * (u[2:] - 2 * u[1:-1] + u[:-2])

    return result


def lax_friedrichs(u, r):

    result = u.copy()
    result[1:-1] = (u[2:] * (1 - r) + u[:-2] * (1 + r)) / 2

    return result


def leap_frog(u, r):

    result = u.copy()
    result[1:] = u[1:] - r * (u[1:] - u[:-1])

    return result


def crank_nicolson(u, r):

    result = u.copy()

    for i in range(1, len(u) - 1):
        result[i] = u[i] - r * (u[i + 1] - u[i - 1] +
                                result[i + 1] - result[i - 1]) / 4

    return result
