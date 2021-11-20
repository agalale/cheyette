from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from math import exp
import numpy as np
from bisect import bisect_left

Vector = List[float]
Matrix = List[List[float]]


def apply_tridiagonal(lower: np.array, diag: np.array, upper: np.array, arg: np.array, out: np.array):
    """
        Computes out = A * arg for tridiagonal A

        :lower: lower-diagonal part of A
        :diag: diagonal part of A
        :upper: upper-diagonal part of A
        :arg: argument
        :out: output
    """
    out[0] = diag[0] * arg[0] + upper[0] * arg[1]
    out[-1] = lower[-1] * arg[-2] + diag[-1] * arg[-1]
    for i, (up, mid, down) in enumerate(zip(arg, arg[1:], arg[2:]), 1):
        out[i] = lower[i-1] * up + diag[i] * mid + upper[i] * down


def solve_tridiagonal(lower, diag, upper, upper_tmp, rhs_tmp, rhs, sol):
    """
    solves A * sol = rhs with tridiagonal A

    :lower: (n-1) lower-diagonal part of A
    :diag: (n) diagonal part of A
    :upper: (n-1) upper-diagonal part of A
    :rhs: (n) right-hand-side
    :sol: (n) solution
    :rhs_tmp: (n) temporary buffer of the same size as rhs
    :upper_tmp: (n-1) temporary buffer of the same size as upper

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


def bilin_interp(xs: Vector, ys: Vector, values: Matrix, x: float, y: float) -> float:
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


class Curve(ABC):
    @abstractmethod
    def df(self, t: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def fwd(self, t1: float, t2: float) -> float:
        raise NotImplementedError


class FlatCurve(Curve):
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def df(self, t: float) -> float:
        return exp(- t * self.rate)

    def fwd(self, t1: float, t2: float) -> float:
        return self.rate


class Mesh2D(ABC):
    def __init__(self) -> None:
        self.xs: np.array = []
        self.ys: np.array = []

    def zeros(self) -> List[List[float]]:
        return [[0.0 for _ in self.xs] for _ in self.ys]

    def eval(self, f: Callable[[float, float], float], output: List[List[float]]) -> None:
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                output[i][j] = f(x, y)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.xs), len(self.ys)

    def interpolate(self, values: np.array, x: float, y: float) -> float:
        return bilin_interp(self.xs, self.ys, values, x, y)


class UniformMesh2D(Mesh2D):
    def __init__(self, xlim: Tuple[float, float], ylim: Tuple[float, float], x_freq: int, y_freq: int) -> None:
        super().__init__()
        self.xs = np.linspace(xlim[0], xlim[1], x_freq)
        self.ys = np.linspace(ylim[0], ylim[1], y_freq)

        self.x_step = self.xs[1] - self.xs[0]
        self.y_step = self.ys[1] - self.ys[0]

    def __repr__(self):
        return f'UniformMesh(xs=[{self.xs[0]:.2f},{self.xs[1]:.2f}..{self.xs[-1]:.2f}], ys=[{self.ys[0]:.2f},{self.ys[1]:.2f}..{self.ys[-1]:.2f}])'


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
        return self.curve.df(t) * exp(-self.G(t, T) * x - 0.5 * self.G(t, T) ** 2 * y)

    def annuity(self, t: float, x: float, y: float, underlying_times: List[float]) -> float:
        return sum((t2 - t1 ) * self.df(t, t2, x, y)
                    for t1, t2 in zip(underlying_times, underlying_times[1:]))

    def swap_value(self, t: float, x: float, y: float, strike: float, underlying_times: List[float]) -> float:
        return (self.df(t, underlying_times[0], x, y) - self.df(t, underlying_times[-1], x, y)
                - strike * self.annuity(t, x, y, underlying_times))


class VasicekProcess(CheyetteProcess):
    def __init__(self, curve: Curve, mean_rev: float, local_vol: float):
        super().__init__(curve)
        self.mean_rev = mean_rev
        self.local_vol = local_vol  # Normal local vol

    def mu_x(self, t: float, x: float, y: float) -> float:
        return y - self.mean_rev * x

    def gamma_x(self, t: float, x: float, y:float) -> float:
        return self.local_vol

    def mu_y(self, t: float, x: float, y: float) -> float:
        return self.local_vol ** 2 - 2 * self.mean_rev * y

    def G(self, t: float, T: float):
        return (1 - np.exp(-self.mean_rev * (T - t))) / self.mean_rev

    def __repr__(self):
        return f'dx[t] = (y[t] - kappa*x[t])dt + sigma*dW[t]\ndy[t] = (sigma**2 - 2*kappa*y[t])dt\nkappa={self.mean_rev}, sigma={self.local_vol}'


class CheyetteProduct(ABC):
    @abstractmethod
    def inner_value(self, t: float, x: float, y: float) -> float:
        raise NotImplementedError


class PayerSwaption(CheyetteProduct):
    def __init__(self, process: CheyetteProcess, strike: float, exercise_time: float,
                 underlying_times: List[float]) -> None:
        self.process = process
        self.strike = strike
        self.exercise_time = exercise_time
        self.underlying_times = underlying_times

    def inner_value(self, t: float, x: float, y: float) -> float:
        return max(self.process.swap_value(t, x, y, self.strike, self.underlying_times), 0)

    def __repr__(self):
        return f'PayerSwaption(strike={strike:.2f}, exercise_time={exercise_time:.2f}, underlying_times={self.underlying_times})'


class CheyetteOperator:
    def __init__(self, process: CheyetteProcess, mesh: UniformMesh2D, t_step: float):
        self.t_step = t_step
        self.process = process
        self.mesh = mesh
        self.shape = mesh.shape

        self.mu_x = np.zeros(mesh.shape)
        self.mu_y = np.zeros(mesh.shape)
        self.gamma_x_squared = np.zeros(mesh.shape)
        self.r = np.zeros(mesh.shape)

        # Initialization of operators L_x and L_y
        self.x_lower_diag = np.array([[0.0 for _ in mesh.ys] for _ in mesh.xs[1:]])
        self.x_diag = np.zeros(mesh.shape)
        self.x_upper_diag = np.array([[0.0 for _ in mesh.ys] for _ in mesh.xs[1:]])

        self.y_lower_diag = np.array([[0.0 for _ in mesh.ys[1:]] for _ in mesh.xs])
        self.y_diag = np.zeros(mesh.shape)
        self.y_upper_diag = np.array([[0.0 for _ in mesh.ys[:-1]] for _ in mesh.xs])

        # allocating memory for the solver
        self.x_upper_diag_tmp = np.zeros((mesh.shape[0]-1, mesh.shape[1]))
        self.x_rhs_tmp = np.zeros(mesh.shape)
        self.y_upper_diag_tmp = np.zeros((mesh.shape[0], mesh.shape[1]-1))
        self.y_rhs_tmp = np.zeros(mesh.shape)

    def evaluate_coefs(self, t: float) -> None:
        """
            Evaluate coefficients of the tridiagonal operators
              A_x = Id - 0.5 * dt * L_x
              A_y = Id + 0.5 * dt * L_y
            where L_x, L_y are the tridiagonal operators:
              L_x = mu_x * D_x + 0.5 * gamma**2 * D_x**2
              L_y = my_u * D_y
        """
        self.mu_x[:] = np.array([[self.process.mu_x(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.mu_y[:] = np.array([[self.process.mu_y(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.gamma_x_squared[:] = np.array([[self.process.gamma_x(t, x, y)**2 for y in self.mesh.ys] for x in self.mesh.xs])
        self.r[:] = np.array([[self.process.r(t, x) for _ in self.mesh.ys] for x in self.mesh.xs])

        # Operator A_x = I - 0.5 * dt * L_x
        txx_ratio = self.t_step / (self.mesh.x_step**2)
        self.x_upper_diag[:] = - 0.25 * txx_ratio * (self.gamma_x_squared[:-1] + self.mesh.x_step * self.mu_x[:-1])
        self.x_lower_diag[:] = - 0.25 * txx_ratio * (self.gamma_x_squared[1:] - self.mesh.x_step * self.mu_x[1:])
        self.x_diag[:] = 1 - 0.5 * txx_ratio * (self.gamma_x_squared + 0.5 * self.mesh.x_step**2 * self.r)

        # Operator A_y = I + 0.5 * dt * L_y
        self.y_upper_diag[:] = 0.25 * self.t_step / self.mesh.y_step * self.mu_y.T[:-1].T
        self.y_lower_diag[:] = - 0.25 * self.t_step / self.mesh.y_step * self.mu_y.T[1:].T
        self.y_diag[:] = 1 + 0.25 * self.t_step * self.r

        # Dirichlet boundary condition for x
        self.x_diag[0, :] = np.ones(self.mesh.shape[1])
        self.x_diag[-1, :] = np.ones(self.mesh.shape[1])
        self.x_upper_diag[0, :] = np.zeros(self.mesh.shape[1])
        self.x_lower_diag[-1, :] = np.zeros(self.mesh.shape[1])

        # Dirichlet boundary condition for y
        self.y_diag[:, 0] = np.ones(self.mesh.shape[0])
        self.y_diag[:, -1] = np.ones(self.mesh.shape[0])
        self.y_upper_diag[:, 0] = np.zeros(self.mesh.shape[0])
        self.y_lower_diag[:, -1] = np.zeros(self.mesh.shape[0])

    def x_solve(self, rhs: np.array, sol: np.array) -> None:
        for j, _ in enumerate(rhs.T):
            solve_tridiagonal(self.x_lower_diag[:, j], self.x_diag[:, j], self.x_upper_diag[:, j],
                       self.x_upper_diag_tmp[:, j], self.x_rhs_tmp[:, j], rhs[:, j], sol[:, j])

    def y_solve(self, rhs: np.array, sol: np.array) -> None:
        for i, _ in enumerate(rhs):
            solve_tridiagonal(self.y_lower_diag[i, :], self.y_diag[i, :], self.y_upper_diag[i, :],
                       self.y_upper_diag_tmp[i, :], self.y_rhs_tmp[i, :], rhs[i, :], sol[i, :])

    def x_apply(self, arg: np.array, out: np.array) -> None:
        for j, _ in enumerate(arg.T):
            apply_tridiagonal(self.x_lower_diag[:, j], self.x_diag[:, j], self.x_upper_diag[:, j], arg[:, j], out[:, j])

    def y_apply(self, arg: np.array, out: np.array) -> None:
        for i, _ in enumerate(arg):
            apply_tridiagonal(self.y_lower_diag[i, :], self.y_diag[i, :], self.y_upper_diag[i, :], arg[i, :], out[i, :])

    def __repr__(self):
        return f'A_x = Id - 0.5 * t_step * L_x\nA_y = Id - 0.5 * t_step * L_y\nL_x = (y-kappa * x) * D_x + 0.5*sigma**2 * D_x**2 - (x+f(0,t))\nL_y = (sigma**2 - 2*kappa*y)D_y'


class CheyetteStepping(ABC):
    def __init__(self, times: np.array, process: CheyetteProcess, product: CheyetteProduct,
                 operator: CheyetteOperator):
        self.times = times
        self.process = process
        self.product = product
        self.operator = operator

        self.tmp_values = np.zeros(self.operator.mesh.shape)
        self.x_rhs = np.zeros(self.operator.mesh.shape)
        self.y_rhs = np.zeros(self.operator.mesh.shape)

    @abstractmethod
    def do_one_step(self, mid_time, old_values, new_values,
                    x_lower_boundary_values, x_upper_boundary_values,
                    y_lower_boundary_values, y_upper_boundary_values) -> None:
        raise NotImplementedError


class PeacemanRachford(CheyetteStepping):
    def do_one_step(self, mid_time, old_values, new_values,
                    x_lower_boundary_values, x_upper_boundary_values,
                    y_lower_boundary_values, y_upper_boundary_values) -> None:
        self.operator.evaluate_coefs(mid_time)

        self.operator.y_apply(arg=old_values, out=self.x_rhs)
        self.x_rhs[0, :] = x_lower_boundary_values    # lower Dirichlet condition for x
        self.x_rhs[-1, :] = x_upper_boundary_values   # upper Dirichlet condition for x
        self.operator.x_solve(rhs=self.x_rhs, sol=self.tmp_values)

        self.operator.x_apply(arg=self.tmp_values, out=self.y_rhs)
        self.y_rhs[:, 0] = y_upper_boundary_values    # upper Dirichlet condition for y
        self.y_rhs[:, -1] = y_lower_boundary_values   # lower Dirichlet condition for y
        self.operator.y_solve(rhs=self.y_rhs, sol=new_values)


class CheyetteDirichletBC:
    def __init__(self, evolution_times: np.array, mesh: Mesh2D, product: CheyetteProduct):
        self.evolution_times = evolution_times
        self.mesh = mesh
        self.product = product

        self.x_lower_boundary_values = np.zeros((len(evolution_times)-1, mesh.shape[1]))
        self.x_upper_boundary_values = np.zeros((len(evolution_times)-1, mesh.shape[1]))
        self.y_lower_boundary_values = np.zeros((len(evolution_times)-1, mesh.shape[0]))
        self.y_upper_boundary_values = np.zeros((len(evolution_times)-1, mesh.shape[0]))

        self.precompute_x_boundary_values()
        self.precompute_y_boundary_values()

    def precompute_x_boundary_values(self) -> None:
        # this is more easy for Douglas-Rachford, but evolved for Peaceman-Rachford
        # we use an approximate boundary value which is exact if the coefs are time-independent
        # the approximate boundary condition assumes that U ~ V((t[i] + t[i+1]) / 2)
        x_first = self.mesh.xs[0]
        x_last = self.mesh.xs[-1]
        for i, (t1, t2) in enumerate(zip(self.evolution_times, self.evolution_times[1:])):
            for j, y in enumerate(self.mesh.ys):
                mid_time = 0.5 * (t1 + t2)
                self.x_lower_boundary_values[i][j] = self.product.inner_value(mid_time, x_first, y)
                self.x_upper_boundary_values[i][j] = self.product.inner_value(mid_time, x_last, y)

    def precompute_y_boundary_values(self) -> None:
        y_first = self.mesh.ys[0]
        y_last = self.mesh.ys[-1]
        for i, t in enumerate(self.evolution_times[1:]):
            for j, x in enumerate(self.mesh.xs):
                self.y_lower_boundary_values[i][j] = self.product.inner_value(t, x, y_first)
                self.y_upper_boundary_values[i][j] = self.product.inner_value(t, x, y_last)

    def __repr__(self):
        return 'x lower: Dirichlet\nx upper: Dirichlet\ny lower: Dirichlet\ny upper: Dirichlet'


class CheyetteEngine:

    def __init__(self, valuation_time: float, t_step: float, end_time: float, mesh: Mesh2D, product: CheyetteProduct,
                 stepping_method: CheyetteStepping):
        self.t_step = t_step
        self.evolution_times = np.arange(end_time, -t_step, valuation_time-t_step)
        self.stepping_method = stepping_method
        self.mesh = mesh
        self.product = product
        self.values = np.array([[product.inner_value(end_time, x, y) for y in mesh.ys] for x in mesh.xs])
        self.tmp_values = np.array([[0.0 for y in mesh.ys] for x in mesh.xs])
        self.bc = CheyetteDirichletBC(self.evolution_times, mesh, product)

    def solve(self):
        for i, (this_time, next_time) in enumerate(zip(self.evolution_times, self.evolution_times[1:])):
            mid_time = 0.5 * (this_time + next_time)
            self.stepping_method.do_one_step(mid_time, self.values, self.tmp_values,
                                             self.bc.x_lower_boundary_values[i], self.bc.x_upper_boundary_values[i],
                                             self.bc.y_lower_boundary_values[i], self.bc.y_upper_boundary_values[i])
            self.values[:] = self.tmp_values
        results = dict()
        results['PV'] = self.mesh.interpolate(self.values, 0.0, 0.0)
        return results


# Market parameters
valuation_time = 0.0
curve = FlatCurve(0.02)
process = VasicekProcess(curve=curve, mean_rev=0.1, local_vol=0.02)

# Product parameters
strike = 0.02
exercise_time = 10
underlying_times = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
product = PayerSwaption(process, strike, exercise_time, underlying_times)

# Space grid
xlim = (-5, 5)
ylim = (-5, 5)
xfreq = 50
yfreq = 10
mesh = UniformMesh2D(xlim, ylim, xfreq, yfreq)

# Time grid
t_step = 0.1
times = np.arange(valuation_time, exercise_time, t_step)

operator = CheyetteOperator(process, mesh, t_step)
stepping_method = PeacemanRachford(times, process, product, operator)
engine = CheyetteEngine(valuation_time, t_step, exercise_time, mesh, product, stepping_method)

results = engine.solve()
print('Pricing results:', results)

# apply_tridiagonal on diagonal matrix
diag = np.array([1, 2, 3])
upper = np.array([0, 0])
lower = np.array([0, 0])
x = np.array([10, 0.1, 20])
res = np.array([0.0, 0.0, 0.0])
true_res = np.array([10, 0.2, 60])
apply_tridiagonal(lower, diag, upper, x, res)
print('Testing apply_tridiagonal() on diagonal matrix:', 'passed' if np.allclose(res, true_res) else 'failed')

# apply_tridiagonal on general tridiagonal matrix
diag = np.array([1, 1, 0])
upper = np.array([1, 1])
lower = np.array([0, 2])
x = [1, 2, 3]
res = np.array([0.0, 0.0, 0.0])
apply_tridiagonal(lower, diag, upper, x, res)
true_res = [3.0, 5.0, 4.0]
print('Testing apply_tridiagonal() on tridiagonal matrix:', 'passed' if np.allclose(res, true_res) else 'failed')

# solve_tridiagonal on diagonal_matrix
diag = np.array([1.0, 2, 3])
upper = np.array([0.0, 0.0])
upper_tmp = np.array([0.0, 0.0])
lower = np.array([0.0, 0])
true_x = np.array([10.0, 0.1, 20])
y = np.array([10.0, 0.2, 60.0])
y_tmp = np.array([0.0, 0.0, 0.0])

solve_tridiagonal(lower, diag, upper, upper_tmp, y_tmp, y, x)
print('Testing solve_tridiagonal() on diagonal matrix:', 'passed' if np.allclose(x, true_x) else 'failed')

# solve_tridiagonal on general tridiagonal matrix
diag = np.array([1, 1, 0])
upper = np.array([1, 1])
lower = np.array([0, 2])
true_x = [1.0, 2.0, 3.0]
x = np.array([0.0, 0, 0])
y = np.array([3.0, 5.0, 4.0])
y_tmp = np.array([0.0, 0, 0])
upper_tmp = np.array([0.0, 0])
solve_tridiagonal(lower, diag, upper, upper_tmp, y_tmp, y, x)
print('Testing solve_tridiagonal() on tridiagonal matrix:', 'passed' if np.allclose(x, true_x) else 'failed')
