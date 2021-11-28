from abc import ABC, abstractmethod
import numpy as np
from bisect import bisect_left
from typing import List


def apply_tridiagonal(lower: np.array, diag: np.array, upper: np.array, arg: np.array, out: np.array):
    """
        Computes out = A * arg for tridiagonal A

        :param lower: lower-diagonal part of A
        :param diag: diagonal part of A
        :param upper: upper-diagonal part of A
        :param arg: argument
        :param out: output
    """
    out[0] = diag[0] * arg[0] + upper[0] * arg[1]
    out[-1] = lower[-1] * arg[-2] + diag[-1] * arg[-1]
    for i, (up, mid, down) in enumerate(zip(arg, arg[1:], arg[2:]), 1):
        out[i] = lower[i-1] * up + diag[i] * mid + upper[i] * down


def solve_tridiagonal(lower, diag, upper, upper_tmp, rhs_tmp, rhs, sol):
    """
    solves A * sol = rhs with tridiagonal A

    :param lower: (n-1) lower-diagonal part of A
    :param diag: (n) diagonal part of A
    :param upper: (n-1) upper-diagonal part of A
    :param rhs: (n) right-hand-side
    :param sol: (n) solution
    :param rhs_tmp: (n) temporary buffer of the same size as rhs
    :param upper_tmp: (n-1) temporary buffer of the same size as upper

    implements Thomas algorithm (rus. metod progonki)
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    Complexity: 8*N-6, where N = Mesh.size
    """
    # Forward sweep (6*N-4 operations)
    N = len(sol)
    upper_tmp[0] = upper[0] / diag[0]
    rhs_tmp[0] = rhs[0] / diag[0]
    for k in range(1, N-1):
        denominator = diag[k] - lower[k - 1] * upper_tmp[k - 1]
        upper_tmp[k] = upper[k] / denominator
        rhs_tmp[k] = (rhs[k] - lower[k - 1] * rhs_tmp[k - 1]) / denominator
    rhs_tmp[-1] = (rhs[-1] - lower[-1] * rhs_tmp[-2]) \
        / (diag[-1] - lower[-1] * upper_tmp[-2])

    # Back substitution (2N-2 operations)
    sol[-1] = rhs_tmp[-1]
    for k in range(N-2, -1, -1):
        sol[k] = rhs_tmp[k] - upper_tmp[k] * sol[k + 1]


def bilin_interp(xs: np.array, ys: np.array, values: np.array, x: float, y: float) -> float:
    """
    Bilinear interpolation
    :param xs: grid in x direction
    :param ys: grid in y direction
    :param values: matrix containing values [[f(x,y) for y in ys] for x in xs]]
    :param x: x coordinate for interpolation
    :param y: y coordinate for interpolation
    :return: bilinearly interpolated value f(x,y)
    """
    ix = bisect_left(xs, x) - 1
    iy = bisect_left(ys, y) - 1
    x1, x2 = xs[ix:ix+2]
    y1, y2 = ys[iy:iy+2]
    z11, z12 = values[ix, iy:iy+2]
    z21, z22 = values[ix+1, iy:iy+2]
    return ( z11 * (x2 - x) * (y2  - y)
             + z12 * (x2 - x) * (y - y1)
             + z21 * (x - x1) * (y2 - y)
             + z22 * (x - x1) * (y - y1) ) \
            / ((x2 - x1) * (y2 - y1))

