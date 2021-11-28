from abc import ABC, abstractmethod
import numpy as np
from cheyette.curves import Curve
from cheyette.products import CheyetteProduct
from cheyette.processes import CheyetteProcess


class CheyetteBC(ABC):
    @abstractmethod
    def adjust_matrix_diag(self, us: np.array):
        raise NotImplementedError

    @abstractmethod
    def adjust_matrix_next(self, us: np.array):
        raise NotImplementedError

    @abstractmethod
    def adjust_x_rhs(self, t: float, x: float, ys: np.array, us: np.array,
                     curve: Curve, process: CheyetteProcess, product: CheyetteProduct):
        raise NotImplementedError

    @abstractmethod
    def adjust_y_rhs(self, t: float, xs: np.array, y: float, us: np.array,
                     curve: Curve, process: CheyetteProcess, product: CheyetteProduct):
        raise NotImplementedError


class DirichletIntrinsicBC(CheyetteBC):
    def adjust_matrix_diag(self, us: np.array):
        us[:] = np.ones_like(us)

    def adjust_matrix_next(self, us: np.array):
        us[:] = np.zeros_like(us)

    def adjust_x_rhs(self, t: float, x: float, ys: np.array, us: np.array,
                     curve: Curve, process: CheyetteProcess, product: CheyetteProduct):
        us[:] = [product.intrinsic_value(curve, process, t, x, y) for y in ys]

    def adjust_y_rhs(self, t: float, xs: np.array, y: float, us: np.array,
                     curve: Curve, process: CheyetteProcess, product: CheyetteProduct):
        us[:] = [product.intrinsic_value(curve, process, t, x, y) for x in xs]

    def __repr__(self):
        return 'DirichletIntrinsicBC()'
