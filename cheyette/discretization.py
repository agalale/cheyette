from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from cheyette.curves import Curve
from cheyette.utils import apply_tridiagonal, solve_tridiagonal, bilin_interp
from cheyette.processes import CheyetteProcess
from cheyette.boundary_conditions import CheyetteBC
from cheyette.products import CheyetteProduct


class Mesh2D:
    def __init__(self, x_grid_stddevs: int, y_grid_stddevs: int,
                 x_freq: int, y_freq: int,
                 x_grid_center: float, y_grid_center: float,
                 x_grid_stddev: float, y_grid_stddev: float) -> None:

        self.x_grid_stddevs = x_grid_stddevs
        self.y_grid_stddevs = y_grid_stddevs
        self.x_grid_center = x_grid_center
        self.y_grid_center = y_grid_center
        self.x_grid_stddev = x_grid_stddev
        self.y_grid_stddev = y_grid_stddev
        self.x_freq = x_freq
        self.y_freq = y_freq

        x_lim = (self.x_grid_center - self.x_grid_stddevs * self.x_grid_stddev,
                 self.x_grid_center + self.x_grid_stddevs * self.x_grid_stddev)
        y_lim = (self.y_grid_center - self.y_grid_stddevs * self.y_grid_stddev,
                 self.y_grid_center + self.y_grid_stddevs * self.y_grid_stddev)
        self.x_step = self.x_grid_stddev / self.x_freq
        self.y_step = self.y_grid_stddev / self.y_freq
        self.xs = np.arange(x_lim[0], x_lim[1] + self.x_step, self.x_step)
        self.ys = np.arange(y_lim[0], y_lim[1] + self.y_step, self.y_step)

    @property
    def shape(self) -> Tuple:
        return len(self.xs), len(self.ys)

    def interpolate(self, values: np.array, x: float, y: float) -> float:
        return bilin_interp(self.xs, self.ys, values, x, y)

    def __repr__(self):
        return f'Mesh(x_grid_stddevs={self.x_grid_stddevs}, y_grid_stddevs={self.y_grid_stddevs}, ' \
               f'x_grid_center={self.x_grid_center}, y_grid_center={self.y_grid_center}, ' \
               f'x_grid_stddev={self.x_grid_stddev}, y_grid_stddev={self.y_grid_stddev})'


class CheyetteOperator:
    """
        Tridiagonal operators
          A_x = Id +/- 0.5 * dt * L_x
          A_y = Id -/+ 0.5 * dt * L_y
        where L_x, L_y are the tridiagonal operators:
          L_x = mu_x * D_x + 0.5 * gamma**2 * D_x**2 - 0.5 r
          L_y = my_u * D_y - 0.5 r

        Central difference is used for D_x, D_y, D_x**2, D_y**2

        Coefficients are evaluated at each iteration of stepping, no observation of observables needed
    """
    def __init__(self, mesh: Mesh2D, process: CheyetteProcess, t_step: float,
                 x_upper_bc: CheyetteBC, x_lower_bc: CheyetteBC, y_upper_bc: CheyetteBC, y_lower_bc: CheyetteBC):
        self.mesh = mesh
        self.shape = mesh.shape
        self.t_step = t_step

        self.txx_ratio = t_step / (self.mesh.x_step ** 2)
        self.mu_x = np.array([[process.mu_x(x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.mu_y = np.array([[process.mu_y(x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.gamma_x_squared = np.array([[process.gamma_x(x, y) ** 2 for y in self.mesh.ys] for x in self.mesh.xs])

        self.r = np.zeros(mesh.shape)

        # Initialization of operators A_x and A_y
        self.x_upper_minus = -0.5 * 0.5 * self.txx_ratio * (
                self.gamma_x_squared[:-1] + self.mesh.x_step * self.mu_x[:-1])
        self.x_lower_minus = -0.5 * 0.5 * self.txx_ratio * (
                self.gamma_x_squared[1:] - self.mesh.x_step * self.mu_x[1:])
        self.x_diag = 1 - (-0.5) * self.txx_ratio * self.gamma_x_squared
        self.x_upper_plus = -self.x_upper_minus
        self.x_lower_plus = -self.x_lower_minus
        self.x_upper_tmp = np.zeros_like(self.x_upper_plus)

        self.y_upper_plus = (0.5) * 0.5 * t_step / self.mesh.y_step * self.mu_y.T[:-1].T
        self.y_lower_plus = - (0.5) * 0.5 * t_step / self.mesh.y_step * self.mu_y.T[1:].T
        self.y_upper_minus = -self.y_upper_plus
        self.y_lower_minus = -self.y_lower_plus
        self.y_upper_tmp = np.zeros_like(self.y_upper_plus)

        # allocating memory for the solver
        self.x_diag_plus = np.zeros_like(self.x_diag)
        self.x_diag_minus = np.zeros_like(self.x_diag)
        self.x_rhs_tmp = np.zeros(mesh.shape)

        self.y_diag_plus = np.zeros(mesh.shape)
        self.y_diag_minus = np.zeros_like(self.y_diag_plus)
        self.y_rhs_tmp = np.zeros(mesh.shape)

        # Boundary conditions for x
        x_lower_bc.adjust_matrix_diag(self.x_diag_plus[0, :])
        x_lower_bc.adjust_matrix_diag(self.x_diag_minus[0, :])
        x_lower_bc.adjust_matrix_next(self.x_upper_plus[0, :])
        x_lower_bc.adjust_matrix_next(self.x_upper_minus[0, :])
        x_upper_bc.adjust_matrix_diag(self.x_diag_plus[-1, :])
        x_upper_bc.adjust_matrix_diag(self.x_diag_minus[-1, :])
        x_upper_bc.adjust_matrix_next(self.x_lower_plus[-1, :])
        x_upper_bc.adjust_matrix_next(self.x_lower_minus[-1, :])

        # Boundary conditions for y
        y_lower_bc.adjust_matrix_diag(self.y_diag_plus[:, 0])
        y_lower_bc.adjust_matrix_diag(self.y_diag_minus[:, 0])
        y_lower_bc.adjust_matrix_next(self.y_upper_plus[:, 0])
        y_lower_bc.adjust_matrix_next(self.y_upper_minus[:, 0])
        y_upper_bc.adjust_matrix_diag(self.y_diag_plus[:, -1])
        y_upper_bc.adjust_matrix_diag(self.y_diag_minus[:, -1])
        y_upper_bc.adjust_matrix_next(self.y_lower_plus[:, -1])
        y_upper_bc.adjust_matrix_next(self.y_lower_minus[:, -1])

    def evaluate_coefs(self, t: float, curve: Curve) -> None:
        self.r[:] = np.array([[curve.fwd(0, t) + x for _ in self.mesh.ys] for x in self.mesh.xs])
        self.x_diag_minus[1:-1, :] = self.x_diag[1:-1, :] - (-0.5) * self.txx_ratio * 0.5 * self.mesh.x_step ** 2 * self.r[1:-1, :]
        self.x_diag_plus[1:-1, :] = 2 - self.x_diag_minus[1:-1, :]
        self.y_diag_plus[:, 1:-1] = 1 - (0.5) * 0.5 * self.t_step * self.r[:, 1:-1]
        self.y_diag_minus[:, 1:-1 ] = 2 - self.y_diag_plus[:, 1:-1]

    def x_solve(self, rhs: np.array, sol: np.array, x_lower: np. array, x_diag: np.array, x_upper: np.array):
        for j, _ in enumerate(rhs.T):
            solve_tridiagonal(x_lower[:, j], x_diag[:, j], x_upper[:, j],
                              self.x_upper_tmp[:, j], self.x_rhs_tmp[:, j], rhs[:, j], sol[:, j])

    def y_solve(self, rhs: np.array, sol: np.array, y_lower: np.array, y_diag: np.array, y_upper: np.array):
        for i, _ in enumerate(rhs):
            solve_tridiagonal(y_lower[i, :], y_diag[i, :], y_upper[i, :],
                              self.y_upper_tmp[i, :], self.y_rhs_tmp[i, :], rhs[i, :], sol[i, :])

    @staticmethod
    def x_apply(arg: np.array, out: np.array, x_lower: np.array, x_diag: np.array, x_upper: np.array) -> None:
        for j, _ in enumerate(arg.T):
            apply_tridiagonal(x_lower[:, j], x_diag[:, j], x_upper[:, j], arg[:, j], out[:, j])

    @staticmethod
    def y_apply(arg: np.array, out: np.array, y_lower: np.array, y_diag: np.array, y_upper: np.array) -> None:
        for i, _ in enumerate(arg):
            apply_tridiagonal(y_lower[i, :], y_diag[i, :], y_upper[i, :], arg[i, :], out[i, :])

    def __repr__(self):
        return f'A_x = Id +/- 0.5 * t_step * L_x\n' \
               f'A_y = Id -/+ 0.5 * t_step * L_y\n' \
               f'L_x = (y - kappa * x) * D_x + 0.5*sigma**2 * D_x**2 - 0.5 * (x + f(0,t))\n' \
               f'L_y = (sigma**2 - 2 * kappa * y) * D_y - 0.5 * (x + f(0,t))'


class CheyetteStepping(ABC):
    def initialize(self, mesh: Mesh2D):
        self.mesh = mesh
        self.tmp_values = np.zeros(mesh.shape)
        self.x_rhs = np.zeros(mesh.shape)
        self.y_rhs = np.zeros(mesh.shape)

    @abstractmethod
    def do_one_step(self, operator: CheyetteOperator,
                    t_this: float, t_next: float,
                    old_values: np.array, new_values: np.array,
                    x_lower_bc: CheyetteBC, x_upper_bc: CheyetteBC,
                    y_lower_bc: CheyetteBC, y_upper_bc: CheyetteBC,
                    curve: Curve, process: CheyetteProcess, product: CheyetteProduct) -> None:
        raise NotImplementedError


class PeacemanRachford(CheyetteStepping):
    def do_one_step(self, operator: CheyetteOperator,
                    t_this: float, t_next: float,
                    old_values: np.array, new_values: np.array,
                    x_lower_bc: CheyetteBC, x_upper_bc: CheyetteBC,
                    y_lower_bc: CheyetteBC, y_upper_bc: CheyetteBC,
                    curve: Curve, process: CheyetteProcess, product: CheyetteProduct) -> None:

        t_mid = 0.5*(t_this + t_next)

        operator.evaluate_coefs(t_mid, curve)

        operator.y_apply(old_values, self.x_rhs, operator.y_lower_plus, operator.y_diag_plus, operator.y_upper_plus)
        x_lower_bc.adjust_x_rhs(t_mid, self.mesh.xs[0], self.mesh.ys, self.x_rhs[0, :], curve, process, product)
        x_upper_bc.adjust_x_rhs(t_mid, self.mesh.xs[-1], self.mesh.ys, self.x_rhs[-1, :], curve, process, product)
        operator.x_solve(self.x_rhs, self.tmp_values, operator.x_lower_minus, operator.x_diag_minus, operator.x_upper_minus)

        operator.x_apply(self.tmp_values, self.y_rhs, operator.x_lower_plus, operator.x_diag_plus, operator.x_upper_plus)
        y_lower_bc.adjust_y_rhs(t_next, self.mesh.xs, self.mesh.ys[0], self.y_rhs[:, 0], curve, process, product)
        y_upper_bc.adjust_y_rhs(t_next, self.mesh.xs, self.mesh.ys[-1], self.y_rhs[:, -1], curve, process, product)
        operator.y_solve(self.y_rhs, new_values, operator.y_lower_minus, operator.y_diag_minus, operator.y_upper_minus)

    def __repr__(self):
        return 'PeacemanRachford()'

