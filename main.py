from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from math import exp
import numpy as np
from bisect import bisect_left

Vector = List[float]
Matrix = List[List[float]]


def linspace(first: float, last: float, numpoints: int) -> List[float]:
    step = (last - first) / (numpoints - 1)
    res = [first]
    for i in range(1, numpoints):
        res += [res[-1] + step]
    return res


def bilin_interp(xs: Vector, ys: Vector, values: Matrix, x: float, y: float) -> float:
    ix = bisect_left(xs, x) - 1
    iy = bisect_left(ys, y) - 1
    x1, x2 = xs[ix:ix+2]
    y1, y2 = ys[iy:iy+2]
    z11, z12 = values[ix, iy:iy+2]
    z21, z22 = values[ix+1, iy:iy+2]
    return ( z11 * (x2 - x) * (y2  - y) \
             + z12 * (x2 - x) * (y - y1) \
             + z21 * (x - x1) * (y2 - y) \
             + z22 * (x - x1) * (y - y1) ) \
            / ((x2 - x1) * (y2 - y1))


def zeros(d1, d2):
    return [[0.0 for _ in range(d2)] for _ in range(d1)]

class Curve(ABC):
    @abstractmethod
    def df(self, t: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def fwd(self, t1: float, t2: float) -> float:
        raise NotImplementedError


class Mesh2D(ABC):
    def __init__(self) -> None:
        self.xs : List[float] = []
        self.ys : List[float] = []

    def zeros(self) -> List[List[float]]:
        return [[0.0 for _ in self.xs] for _ in self.ys]

    def eval(self, f: Callable[[float, float], float], output: List[List[float]]) -> None:
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                output[i][j] = f(x, y)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.xs), len(self.ys)

    @abstractmethod
    def interpolate(self, values: List[List[float]], x: float, y: float) -> float:
        raise NotImplementedError


class CheyetteProduct(ABC):
    @abstractmethod
    def payoff(self, x: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def inner_value(self, t: float, x: float, y: float) -> float:
        raise NotImplementedError


class CheyetteProcess(ABC):
    """
    dot x = mu_x dt + gamma_x dW[t]
    dot y = mu_y dt

    L = L_x + L_y
    L_x = mu_x * d_x + 0.5 * gamma_x**2 d_x**2 - 0.5 * r
    L_y = mu_y * d_y
    """
    def __init__(self, curve: Curve):
        self.curve = curve

    @abstractmethod
    def mu_x(self, t: float, x: float, y: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def gamma_x(self, t: float, x: float, y:float) -> float:
        raise NotImplementedError

    @abstractmethod
    def mu_y(self, t: float, x: float, y: float) -> float:
        raise NotImplementedError

    def r(self, t: float, x: float):
        return self.curve.fwd(0, t) + x

    @abstractmethod
    def G(self, t: float, T: float):
        raise NotImplementedError

    def df(self, t: float, T: float, x: float, y: float) -> float:
        return self.curve.df(t) * exp(-self.G(t, T)*x - 0.5*self.G(t, T)**2 * y)

    def annuity(self, t: float, x: float, y: float, underlying_times: List[float]) -> float:
        return sum( (t2 - t1 ) * self.df(t, t2, x, y)
                    for t1, t2 in zip(underlying_times, underlying_times[1:]))

    def swap_value(self, t: float, x: float, y: float, strike: float, underlying_times: List[float]) -> float:
        return (self.df(t, underlying_times[0], x, y) - self.df(t, underlying_times[-1], x, y)
                - strike * self.annuity(t, x, y, underlying_times))


class CheyetteStepping(ABC):
    @abstractmethod
    def do_one_step(self, mid_time, old_values, new_values,
                    x_lower_boundary_values, x_upper_boundary_values,
                    y_lower_boundary_values, y_upper_boundary_values) -> None:
        raise NotImplementedError


class CheyetteEngine:

    def __init__(self, t_step: float, end_time: float, mesh: Mesh2D, product: CheyetteProduct,
                 stepping_method: CheyetteStepping):
        self.t_step = t_step
        self.evolution_times = np.arange(end_time, -t_step, -t_step)
        self.stepping_method = stepping_method
        self.mesh = mesh
        self.product = product
        self.values = [[product.inner_value(0.0, x, y) for y in mesh.ys] for x in mesh.xs]
        self.bc = CheyetteDirichletBC(self.evolution_times, mesh, product)

    def solve(self):
        for i, this_time, next_time in enumerate(zip(self.evolution_times, self.evolution_times[1:])):
            mid_time = 0.5 * (this_time + next_time)
            self.stepping_method.do_one_step(mid_time, self.values, self.values,
                                             self.bc.x_lower_boundary_values[i], self.bc.x_upper_boundary_values[i],
                                             self.bc.y_lower_boundary_values[i], self.bc.y_upper_boundary_values[i])
        return self.mesh.interpolate(self.values, 0.0, 0.0)


class UniformMesh2D(Mesh2D):
    def __init__(self, xlim: Tuple[float, float], ylim: Tuple[float, float], x_freq: int, y_freq: int) -> None:
        self.xs = np.linspace(xlim[0], xlim[1], x_freq)
        self.ys = np.linspace(ylim[0], ylim[1], y_freq)

        self.x_step = self.xs[1] - self.xs[0]
        self.y_step = self.ys[1] - self.ys[0]

    def interpolate(self, values: Matrix, x: float, y: float) -> float:
        pass


class CheyetteDirichletBC:
    def __init__(self, evolution_times: List[float], mesh: Mesh2D, product: CheyetteProduct):
        self.evolution_times = evolution_times
        self.mesh = mesh
        self.product = product

        self.x_lower_boundary_values = zeros(len(evolution_times)-1, mesh.shape[1])
        self.x_upper_boundary_values = zeros(len(evolution_times)-1, mesh.shape[1])
        self.y_lower_boundary_values = zeros(len(evolution_times)-1, mesh.shape[0])
        self.y_upper_boundary_values = zeros(len(evolution_times)-1, mesh.shape[0])

        self.precompute_x_boundary_values()
        self.precompute_y_boundary_values()

    def precompute_x_boundary_values(self) -> None:
        x_first = self.mesh.xs[0]
        x_last = self.mesh.xs[-1]
        for i, t in enumerate(self.evolution_times[:-1]):
            for j, y in enumerate(self.mesh.ys):
                self.x_upper_boundary_values[i][j] = self.product.inner_value(t, x_first, y)
                self.x_lower_boundary_values[i][j] = self.product.inner_value(t, x_last, y)

    def precompute_y_boundary_values(self) -> None:
        # this is more easy for Douglas-Rachford, but evolved for Peaceman-Rachford
        # we use an approximate boundary value which is exact if the coefs are time-independent
        y_first = self.mesh.ys[0]
        y_last = self.mesh.ys[-1]
        for i, (t1, t2) in zip(self.evolution_times, self.evolution_times[1:]):
            for j, x in self.mesh.xs:
                mid_time = 0.5 * (t1 + t2)
                self.y_upper_boundary_values[i][j] = self.product.inner_value(mid_time, x, y_first)
                self.y_lower_boundary_values[i][j] = self.product.inner_value(mid_time, x, y_last)


class CheyetteSwaption(CheyetteProduct):
    def __init__(self, strike: float, exercise_time: float, underlying_times: List[float]) -> None:
        self.strike = strike
        self.exercise_time = exercise_time
        self.underlying_times = underlying_times


class CheyetteVasicekProcess(CheyetteProcess):
    def __init__(self, mean_rev: float, local_vol: float):
        self.mean_rev = mean_rev
        self.local_vol = local_vol


class FlatCurve(Curve):
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def df(self, t: float) -> float:
        return exp(- t * self.rate)

    def fwd(self, t1: float, t2: float) -> float:
        return self.rate
    



class CheyetteOperator:
    def __init__(self, process: CheyetteProcess, mesh: UniformMesh2D, t_step: float):
        self.t_step = t_step
        self.process = process
        self.mesh = mesh
        self.shape = mesh.shape

        self.mu_x = zeros(*mesh.shape)
        self.mu_y = zeros(*mesh.shape)
        self.gamma_x_squared = zeros(*mesh.shape)
        self.r = zeros(*mesh.shape)

        # Initialization of a Dirichlet operator
        self.x_lower_diag = [0.0] * (self.shape[0]-1)
        self.x_diag = [1.0] * self.shape[0]
        self.x_upper_diag = [0.0] * (self.shape[0]-1)

        self.y_lower_diag = [0.0] * (self.shape[1]-1)
        self.y_diag = [1.0] * self.shape[1]
        self.y_upper_diag = [0.0] * (self.shape[1]-1)

        # allocating memory for the solver
        self.x_upper_diag_tmp = None
        self.x_rhs_tmp = None
        self.y_upper_diag_tmp = None
        self.y_rhs_tmp = None

    def evaluate_coefs(self, t: float) -> None:
        self.mu_x[:] = [[self.process.mu_x(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs]
        self.mu_y[:] = [[self.process.mu_y(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs]
        self.gamma_x_squared[:] = [[self.process.gamma_x(t, x, y)**2 for y in self.mesh.ys] for x in self.mesh.xs]
        self.r[:] = [[self.process.r(t, x) for y in self.mesh.ys] for x in self.mesh.xs]

        txx_ratio = self.t_step / (self.mesh.x_step**2)
        self.x_upper_diag[1:] = 0.25 * txx_ratio * (self.gamma_x_squared + self.mesh.x_step * self.mu_x)
        self.x_lower_diag[:-1] = 0.25 * txx_ratio * (self.gamma_x_squared - self.mesh.x_step * self.mu_x)
        self.x_diag[1:-1] = 0.5 * txx_ratio * (self.gamma_x_squared + 0.5 * self.mesh.x_step**2 * self.r)

        self.y_upper_diag[1:] = 0.25 * self.t_step / self.mesh.y_step * self.my_y
        self.y_lower_diag[:-1] = -self.y_upper_diag
        self.y_diag[1:-1] = 0.25 * self.t_step * self.r

    @staticmethod
    def solve(lower_diag, diag, upper_diag, upper_diag_tmp, rhs_tmp, rhs, sol):
        """
        solves (I + 0.5 * dt * L_x ) * sol = rhs

        implements Thomas algorithm (rus. metod progonki)
        https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        Complexity: 8*N-6, where N = Mesh.size
        """
        # Forward sweep (6*N-4 operations)
        upper_diag_tmp[0] = upper_diag[0] / diag[0]
        rhs_tmp[0] = rhs[0] / diag[0]
        for k in range(1, len(upper_diag)):
            denominator = diag[k] - lower_diag[k - 1] * upper_diag_tmp[k - 1]
            upper_diag_tmp[k] = upper_diag[k] / denominator
            rhs_tmp[k] = (rhs[k] - lower_diag[k - 1] * rhs_tmp[k - 1]) \
                              / denominator

        # Back substitution (2N-2 operations)
        sol[-1] = rhs_tmp[-1]
        for k in range(len(sol) - 2, -1, -1):
            sol[k] = rhs_tmp[k] - upper_diag_tmp[k] * sol[k + 1]

    @staticmethod
    def apply(lower_diag, diag, upper_diag, arg, out):
        # computes out = (I - L) * arg
        for i, (down, mid, up) in enumerate(zip(arg, arg[1:], arg[2:]), 1):
            out[i] = lower_diag[i] * down + (1 - diag[i]) * mid + upper_diag[i] * up

    def x_solve(self, rhs: Matrix, sol: Matrix) -> None:
        for rhs_row, sol_row in zip(rhs, sol):
            self.solve(self.x_lower_diag, self.x_diag, self.x_upper_diag,
                       self.x_upper_diag_tmp, self.x_rhs_tmp, rhs_row, sol_row)

    def y_solve(self, rhs: Matrix, sol: Matrix) -> None:
        for rhs_col, sol_col in zip(rhs.T, sol.T):
            self.solve(self.y_lower_diag, self.y_diag, self.y_upper_diag,
                       self.y_upper_diag_tmp, self.y_rhs_tmp, rhs_col, sol_col)

    def x_apply(self, arg: Matrix, out: Matrix) -> None:
        for arg_row, out_row in zip(arg, out):
            self.apply(self.x_lower_diag, self.x_diag, self.x_upper_diag, arg_row, out_row)

    def y_apply(self, arg: Matrix, out: Matrix) -> None:
        for arg_col, out_col in zip(arg.T, out.T):
            self.apply(self.y_lower_diag, self.y_diag, self.y_upper_diag, arg_col, out_col)


class CheyettePeacemanRachford(CheyetteStepping):
    # (ru. схема Писмана-Рекфорда)
    def __init__(self, mesh: Mesh2D, times: List[float], process: CheyetteProcess, product: CheyetteProduct,
                 operator: CheyetteOperator):
        self.mesh = mesh
        self.times = times
        self.process = process
        self.product = product
        self.operator = operator

        self.tmp_values = [[]]
        self.x_operator = None
        self.y_operator = None
        self.x_rhs = mesh.zeros()
        self.y_rhs = mesh.zeros()

    def do_one_step(self, mid_time, old_values, new_values,
                    x_lower_boundary_values, x_upper_boundary_values,
                    y_lower_boundary_values, y_upper_boundary_values) -> None:
        self.operator.evaluate_coefs(mid_time)

        self.operator.y_apply(arg=old_values, out=self.x_rhs)
        self.x_rhs[0, :] = x_lower_boundary_values
        self.x_rhs[-1, :] = x_upper_boundary_values
        self.operator.x_solve(rhs=self.x_rhs, sol=self.tmp_values)

        self.operator.x_apply(arg=self.tmp_values, out=self.y_rhs)
        self.y_rhs[:, 0] = y_upper_boundary_values
        self.y_rhs[:, -1] = y_lower_boundary_values
        self.operator.y_solve(rhs=self.y_rhs, sol=new_values)

# Market parameters
valuation_time = 0.0
curve = FlatCurve(0.02)
process = CheyetteVasicekProcess(mean_rev=0.1, local_vol=0.02)

# Product parameters
strike = 0.02
exercise_time = 10
underlying_times = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
product = CheyetteSwaption(strike, exercise_time, underlying_times)

# Grid parameters
xlim = (-5, 5)
ylim = (-5, 5)
xfreq = 50
yfreq = 10
mesh = UniformMesh2D(xlim, ylim, xfreq, yfreq)

