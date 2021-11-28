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
    def __init__(self, mesh: Mesh2D):
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

    def evaluate_coefs(self, t: float, t_step: float, x_mult: float, y_mult: float,
                       x_lower_bc: CheyetteBC, x_upper_bc: CheyetteBC,
                       y_lower_bc: CheyetteBC, y_upper_bc: CheyetteBC,
                       curve: Curve, process: CheyetteProcess) -> None:
        """
            Evaluate coefficients of the tridiagonal operators
              A_x = Id + x_mult * dt * L_x
              A_y = Id + y_mult * dt * L_y
            where L_x, L_y are the tridiagonal operators:
              L_x = mu_x * D_x + 0.5 * gamma**2 * D_x**2 - 0.5 r
              L_y = my_u * D_y - 0.5 r

            Central difference is used for D_x, D_y, D_x**2, D_y**2

            Coefficients are evaluated at each iteration of stepping, no observation of observables needed
        """
        self.mu_x[:] = np.array([[process.mu_x(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.mu_y[:] = np.array([[process.mu_y(t, x, y) for y in self.mesh.ys] for x in self.mesh.xs])
        self.gamma_x_squared[:] = np.array([[process.gamma_x(t, x, y)**2 for y in self.mesh.ys] for x in self.mesh.xs])
        self.r[:] = np.array([[process.r(curve, t, x) for _ in self.mesh.ys] for x in self.mesh.xs])

        # Operator A_x = I - 0.5 * dt * L_x
        txx_ratio = t_step / (self.mesh.x_step**2)
        self.x_upper_diag[:] = x_mult * 0.5 * txx_ratio * (self.gamma_x_squared[:-1] + self.mesh.x_step * self.mu_x[:-1])
        self.x_lower_diag[:] = x_mult * 0.5 * txx_ratio * (self.gamma_x_squared[1:] - self.mesh.x_step * self.mu_x[1:])
        self.x_diag[:] = 1 - x_mult * txx_ratio * (self.gamma_x_squared + 0.5 * self.mesh.x_step**2 * self.r)

        # Operator A_y = I + 0.5 * dt * L_y
        self.y_upper_diag[:] = y_mult * 0.5 * t_step / self.mesh.y_step * self.mu_y.T[:-1].T
        self.y_lower_diag[:] = - y_mult * 0.5 * t_step / self.mesh.y_step * self.mu_y.T[1:].T
        self.y_diag[:] = 1 - y_mult * 0.5 * t_step * self.r

        # Boundary conditions for x
        x_lower_bc.adjust_matrix_diag(self.x_diag[0, :])
        x_lower_bc.adjust_matrix_next(self.x_upper_diag[0, :])
        x_upper_bc.adjust_matrix_diag(self.x_diag[-1, :])
        x_upper_bc.adjust_matrix_next(self.x_lower_diag[-1, :])

        # Boundary conditions for y
        y_lower_bc.adjust_matrix_diag(self.y_diag[:, 0])
        y_lower_bc.adjust_matrix_next(self.y_upper_diag[:, 0])
        y_upper_bc.adjust_matrix_diag(self.y_diag[:, -1])
        y_upper_bc.adjust_matrix_next(self.y_lower_diag[:, -1])

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
        t_step = t_this - t_next

        operator.evaluate_coefs(t_mid, t_step, x_mult=-0.5, y_mult=0.5,
                                x_lower_bc=x_lower_bc, x_upper_bc=x_upper_bc,
                                y_lower_bc=y_lower_bc, y_upper_bc=y_upper_bc,
                                curve=curve, process=process)
        operator.y_apply(arg=old_values, out=self.x_rhs)
        x_lower_bc.adjust_x_rhs(t_mid, self.mesh.xs[0], self.mesh.ys, self.x_rhs[0, :], curve, process, product)
        x_upper_bc.adjust_x_rhs(t_mid, self.mesh.xs[-1], self.mesh.ys, self.x_rhs[-1, :], curve, process, product)
        operator.x_solve(rhs=self.x_rhs, sol=self.tmp_values)

        operator.evaluate_coefs(t_mid, t_step, x_mult=0.5, y_mult=-0.5,
                                x_lower_bc=x_lower_bc, x_upper_bc=x_upper_bc,
                                y_lower_bc=y_lower_bc, y_upper_bc=y_upper_bc,
                                curve=curve, process=process)
        operator.x_apply(arg=self.tmp_values, out=self.y_rhs)
        y_lower_bc.adjust_y_rhs(t_next, self.mesh.xs, self.mesh.ys[0], self.y_rhs[:, 0], curve, process, product)
        y_upper_bc.adjust_y_rhs(t_next, self.mesh.xs, self.mesh.ys[-1], self.y_rhs[:, -1], curve, process, product)
        operator.y_solve(rhs=self.y_rhs, sol=new_values)

    def __repr__(self):
        return 'PeacemanRachford()'

