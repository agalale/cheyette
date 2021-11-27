from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from math import exp
import numpy as np
from scipy.stats import norm
from bisect import bisect_left


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

    def __repr__(self):
        return f'FlatCurve({self.rate})'


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
    def __init__(self, x_grid_stddevs, y_grid_stddevs, x_grid_center, y_grid_center,
                 x_grid_stddev, y_grid_stddev, x_freq, y_freq) -> None:
        super().__init__()
        x_lim = (x_grid_center - x_grid_stddevs * x_grid_stddev, x_grid_center + x_grid_stddevs * x_grid_stddev)
        y_lim = (y_grid_center - y_grid_stddevs * y_grid_stddev, y_grid_center + y_grid_stddevs * y_grid_stddev)
        self.x_step = x_grid_stddev / x_freq
        self.y_step = y_grid_stddev / y_freq
        self.xs = np.arange(x_lim[0], x_lim[1] + self.x_step, self.x_step)
        self.ys = np.arange(y_lim[0], y_lim[1] + self.y_step, self.y_step)

    def __repr__(self):
        return f'UniformMesh(xs=[{self.xs[0]:.2f},{self.xs[1]:.2f}..{self.xs[-1]:.2f}], ys=[{self.ys[0]:.2f},{self.ys[1]:.2f}..{self.ys[-1]:.2f}])'


class CheyetteProcess(ABC):
    """
    dot x = mu_x dt + gamma_x dW[t]
    dot y = mu_y dt

    L = L_x + L_y
    L_x = mu_x * d_x + 0.5 * gamma_x**2 d_x**2 - 0.5 * r
    L_y = mu_y * d_y - 0.5 * r
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
        return self.curve.df(T) / self.curve.df(t) * exp(-self.G(t, T) * x - 0.5 * self.G(t, T) ** 2 * y)

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

    def gamma_x(self, t: float, x: float, y: float) -> float:
        return self.local_vol

    def mu_y(self, t: float, x: float, y: float) -> float:
        return self.local_vol ** 2 - 2 * self.mean_rev * y

    def G(self, t: float, T: float):
        return (1 - np.exp(-self.mean_rev * (T - t))) / self.mean_rev

    def __repr__(self):
        return f'dx[t] = (y[t] - kappa*x[t])dt + sigma*dW[t]\n' \
               f'dy[t] = (sigma**2 - 2*kappa*y[t])dt\n' \
               f'kappa={self.mean_rev}, sigma={self.local_vol}'


class CheyetteProduct(ABC):
    def __init__(self, process: CheyetteProcess):
        self.process = process

    @abstractmethod
    def intrinsic_value(self, t: float, x: float, y: float) -> float:
        raise NotImplementedError


class ZCB(CheyetteProduct):
    def __init__(self, process: CheyetteProcess, expiry: float):
        super().__init__(process)
        self.expiry = expiry

    def intrinsic_value(self, t: float, x: float, y: float) -> float:
        return self.process.df(t, self.expiry, x, y)

    def __repr__(self):
        return f'ZCB(expiry={self.expiry:.2f})'


class ZCBCall(CheyetteProduct):
    def __init__(self, process: CheyetteProcess, strike: float, exercise_time: float, bond_expiry: float) -> None:
        super().__init__(process)
        self.strike = strike
        self.exercise_time = exercise_time
        self.bond_expiry = bond_expiry

    def intrinsic_value(self, t: float, x: float, y: float):
        return max(self.process.df(t, self.bond_expiry, x, y)
                   - self.process.df(t, self.exercise_time, x, y) * self.strike, 0.0)

    def __repr__(self):
        return f'ZCBCall(strike={self.strike}, exercise_time={self.exercise_time}, bond_expiry={self.bond_expiry})'


class PayerSwaption(CheyetteProduct):
    def __init__(self, process: CheyetteProcess, strike: float, exercise_time: float,
                 underlying_times: List[float]) -> None:
        super().__init__(process)
        self.strike = strike
        self.exercise_time = exercise_time
        self.underlying_times = underlying_times

    def intrinsic_value(self, t: float, x: float, y: float) -> float:
        return max(self.process.swap_value(t, x, y, self.strike, self.underlying_times), 0.0)

    def __repr__(self):
        return f'PayerSwaption(strike={self.strike:.2f}, exercise_time={self.exercise_time:.2f}, underlying_times={self.underlying_times})'


class CheyetteOperator:
    def __init__(self, process: CheyetteProcess, mesh: UniformMesh2D):
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

    def evaluate_coefs(self, t: float, t_step: float, x_mult: float, y_mult: float) -> None:
        """
            Evaluate coefficients of the tridiagonal operators
              A_x = Id + x_mult * dt * L_x
              A_y = Id + y_mult * dt * L_y
            where L_x, L_y are the tridiagonal operators:
              L_x = mu_x * D_x + 0.5 * gamma**2 * D_x**2 - 0.5 r
              L_y = my_u * D_y - 0.5 r

            Central difference is used for D_x, D_y, D_x**2, D_y**2
        """
        self.mu_x[:] = np.array([[self.process.mu_x(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.mu_y[:] = np.array([[self.process.mu_y(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.gamma_x_squared[:] = np.array([[self.process.gamma_x(t, x, y)**2 for y in self.mesh.ys] for x in self.mesh.xs])
        self.r[:] = np.array([[self.process.r(t, x) for _ in self.mesh.ys] for x in self.mesh.xs])

        # Operator A_x = I - 0.5 * dt * L_x
        txx_ratio = t_step / (self.mesh.x_step**2)
        self.x_upper_diag[:] = x_mult * 0.5 * txx_ratio * (self.gamma_x_squared[:-1] + self.mesh.x_step * self.mu_x[:-1])
        self.x_lower_diag[:] = x_mult * 0.5 * txx_ratio * (self.gamma_x_squared[1:] - self.mesh.x_step * self.mu_x[1:])
        self.x_diag[:] = 1 - x_mult * txx_ratio * (self.gamma_x_squared + 0.5 * self.mesh.x_step**2 * self.r)

        # Operator A_y = I + 0.5 * dt * L_y
        self.y_upper_diag[:] = y_mult * 0.5 * t_step / self.mesh.y_step * self.mu_y.T[:-1].T
        self.y_lower_diag[:] = - y_mult * 0.5 * t_step / self.mesh.y_step * self.mu_y.T[1:].T
        self.y_diag[:] = 1 - y_mult * 0.5 * t_step * self.r

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
        return f'A_x = Id - 0.5 * t_step * L_x\n' \
               f'A_y = Id - 0.5 * t_step * L_y\n' \
               f'L_x = (y-kappa * x) * D_x + 0.5*sigma**2 * D_x**2 - 0.5(x+f(0,t))\n' \
               f'L_y = (sigma**2 - 2*kappa*y)D_y - 0.5*(x+f(0,t))'


class CheyetteStepping(ABC):
    def __init__(self, operator: CheyetteOperator):
        self.operator = operator

        self.tmp_values = np.zeros(self.operator.mesh.shape)
        self.x_rhs = np.zeros(self.operator.mesh.shape)
        self.y_rhs = np.zeros(self.operator.mesh.shape)

    @abstractmethod
    def do_one_step(self, mid_time, t_step, old_values, new_values,
                    x_lower_boundary_values, x_upper_boundary_values,
                    y_lower_boundary_values, y_upper_boundary_values) -> None:
        raise NotImplementedError


class PeacemanRachford(CheyetteStepping):
    def do_one_step(self, mid_time, t_step, old_values, new_values,
                    x_lower_boundary_values, x_upper_boundary_values,
                    y_lower_boundary_values, y_upper_boundary_values) -> None:

        self.operator.evaluate_coefs(mid_time, t_step, x_mult=-0.5, y_mult=0.5)
        self.operator.y_apply(arg=old_values, out=self.x_rhs)
        self.x_rhs[0, :] = x_lower_boundary_values    # lower Dirichlet condition for x
        self.x_rhs[-1, :] = x_upper_boundary_values   # upper Dirichlet condition for x
        self.operator.x_solve(rhs=self.x_rhs, sol=self.tmp_values)

        self.operator.evaluate_coefs(mid_time, t_step, x_mult=0.5, y_mult=-0.5)
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
                self.x_lower_boundary_values[i][j] = self.product.intrinsic_value(mid_time, x_first, y)
                self.x_upper_boundary_values[i][j] = self.product.intrinsic_value(mid_time, x_last, y)

    def precompute_y_boundary_values(self) -> None:
        y_first = self.mesh.ys[0]
        y_last = self.mesh.ys[-1]
        for i, t in enumerate(self.evolution_times[1:]):
            for j, x in enumerate(self.mesh.xs):
                self.y_lower_boundary_values[i][j] = self.product.intrinsic_value(t, x, y_first)
                self.y_upper_boundary_values[i][j] = self.product.intrinsic_value(t, x, y_last)

    def __repr__(self):
        return 'x lower: Dirichlet\nx upper: Dirichlet\ny lower: Dirichlet\ny upper: Dirichlet'


class CheyetteEngine:
    @abstractmethod
    def price(self):
        raise NotImplementedError


class CheyettePDEEngine(CheyetteEngine):

    def __init__(self, valuation_time: float, t_step: float, end_time: float, product: CheyetteProduct,
                 stepping_method: CheyetteStepping):
        self.t_step = t_step
        self.evolution_times = np.arange(end_time, -t_step, valuation_time-t_step)
        self.stepping_method = stepping_method
        self.mesh = stepping_method.operator.mesh
        self.product = product
        self.values = np.array([[product.intrinsic_value(end_time, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.tmp_values = np.array([[0.0 for y in self.mesh.ys] for x in self.mesh.xs])
        self.bc = CheyetteDirichletBC(self.evolution_times, self.mesh, product)

    def price(self):
        full_solution = np.zeros((len(self.evolution_times), *self.mesh.shape))
        full_solution[0, ...] = self.values
        for i, (this_time, next_time) in enumerate(zip(self.evolution_times, self.evolution_times[1:])):
            mid_time = 0.5 * (this_time + next_time)
            self.stepping_method.do_one_step(mid_time, self.t_step, self.values, self.tmp_values,
                                             self.bc.x_lower_boundary_values[i], self.bc.x_upper_boundary_values[i],
                                             self.bc.y_lower_boundary_values[i], self.bc.y_upper_boundary_values[i])
            self.values[:] = self.tmp_values
            full_solution[i+1, ...] = self.values
        results = dict()
        results['PV'] = self.mesh.interpolate(self.values, 0.0, 0.0)
        results['EvolutionTimes'] = self.evolution_times
        results['FullSolution'] = full_solution
        return results


class CheyetteAnalyticEngine(CheyetteEngine):
    def __init__(self, valuation_time: float, product: CheyetteProduct):
        self.valuation_time = valuation_time
        self.product = product

    def price(self):
        results = dict()
        if isinstance(self.product, ZCBCall):
            if isinstance(self.product.process, VasicekProcess):
                k = self.product.process.mean_rev
                s = self.product.process.local_vol
                S = self.product.exercise_time
                T = self.product.bond_expiry
                K = self.product.strike
                B_T = self.product.process.curve.df(T)
                B_S = self.product.process.curve.df(S)
                nu = s**2 / (2*k**3) * (1 - np.exp(-k*(T-S)))**2 * (1 - np.exp(-2*k*S))
                d_plus = (np.log(B_T / (K*B_S)) + 0.5 * nu)/np.sqrt(nu)
                d_minus = d_plus - np.sqrt(nu)
                results['PV'] = B_T * norm.cdf(d_plus) - K * B_S * norm.cdf(d_minus)

            else:
                raise Exception(f'Analytic engine can\'t be used to price ZCB when '
                                + f'the underlying process is {self.product.process.__class__}')

        else:
            raise Exception(f'Product {self.product.__class__} is not supported by the analytic engine')

        return results
