from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from cheyette.processes import CheyetteProcess, VasicekProcess
from cheyette.products import CheyetteProduct, ZCBCall
from cheyette.curves import Curve
from cheyette.boundary_conditions import CheyetteBC
from cheyette.discretization import Mesh2D, CheyetteStepping, CheyetteOperator


class CheyetteModel(ABC):
    @abstractmethod
    def price(self, curve: Curve, process: CheyetteProcess, product: CheyetteProduct, valuation_time: float):
        raise NotImplementedError


class CheyettePDEModel(CheyetteModel):
    def __init__(self, x_grid_stddevs: int, y_grid_stddevs: int,
                 x_freq: int, y_freq: int, t_freq: int,
                 stepping_method: CheyetteStepping,
                 x_lower_bc: CheyetteBC, x_upper_bc: CheyetteBC,
                 y_lower_bc: CheyetteBC, y_upper_bc: CheyetteBC):
        self.x_grid_stddevs = x_grid_stddevs
        self.y_grid_stddevs = y_grid_stddevs
        self.x_freq = x_freq
        self.y_freq = y_freq
        self.t_freq = t_freq
        self.stepping_method = stepping_method
        self.x_lower_bc = x_lower_bc
        self.x_upper_bc = x_upper_bc
        self.y_lower_bc = y_lower_bc
        self.y_upper_bc = y_upper_bc

    def price(self, curve: Curve, process: CheyetteProcess, product: CheyetteProduct, valuation_time: float):
        t_step = 1.0 / self.t_freq
        evolution_times = np.arange(product.expiry, -t_step, valuation_time - t_step)
        x_grid_center = process.x_mean(product.expiry)
        y_grid_center = process.y_mean(product.expiry)
        x_grid_stddev = process.x_stddev(product.expiry)
        y_grid_stddev = max(process.y_stddev(product.expiry), 0.01)
        mesh = Mesh2D(self.x_grid_stddevs, self.y_grid_stddevs, self.x_freq, self.y_freq,
                      x_grid_center, y_grid_center, x_grid_stddev, y_grid_stddev)
        values = np.array([[product.intrinsic_value(curve, process, product.expiry, x, y) for y in mesh.ys] for x in mesh.xs])
        tmp_values = np.zeros_like(values)
        self.stepping_method.initialize(mesh)
        operator = CheyetteOperator(mesh)

        for i, (t_this, t_next) in enumerate(zip(evolution_times, evolution_times[1:])):
            self.stepping_method.do_one_step(operator, t_this, t_next, values, tmp_values,
                                             self.x_lower_bc, self.x_upper_bc,
                                             self.y_lower_bc, self.y_upper_bc,
                                             curve, process, product)
            values[:] = tmp_values
        results = dict()
        results['PV'] = mesh.interpolate(values, 0.0, 0.0)
        return results

    def __repr__(self):
        return f'CheyettePDEModel(x_grid_stddevs={self.x_grid_stddevs}, ' \
               f'y_grid_stddevs={self.y_grid_stddevs}, ' \
               f'x_freq={self.x_freq}, y_freq={self.y_freq}, t_freq={self.t_freq}, ' \
               f'stepping_method={self.stepping_method}, ' \
               f'x_lower_bc={self.x_lower_bc}, x_upper_bc={self.x_upper_bc}, ' \
               f'y_lower_bc={self.y_lower_bc}, x_upper_bc={self.y_upper_bc})'

    def set(self, key, value):
        if key == 'x_freq':
            self.x_freq = value
            return self
        elif key == 'y_freq':
            self.y_freq = value
            return self
        elif key == 'x_grid_stddevs':
            self.x_grid_stddevs = value
        elif key == 'y_grid_stddevs':
            self.y_grid_stddevs = value
        else:
            raise Exception(f'Attribute {key} not supported by {self.__class__}')

        return self

class CheyetteAnalyticModel(CheyetteModel):
    """
    Better reimplement using visitor pattern
    """
    def price(self, curve: Curve, process: CheyetteProcess, product: CheyetteProduct, valuation_time: float):
        results = dict()
        if isinstance(product, ZCBCall):
            if isinstance(process, VasicekProcess):
                k = process.mean_rev
                s = process.local_vol
                S = product.expiry
                T = product.bond_expiry
                K = product.strike
                B_T = curve.df(T)
                B_S = curve.df(S)
                nu = s ** 2 / (2 * k ** 3) * (1 - np.exp(-k * (T - S))) ** 2 * (1 - np.exp(-2 * k * S))
                d_plus = (np.log(B_T / (K * B_S)) + 0.5 * nu) / np.sqrt(nu)
                d_minus = d_plus - np.sqrt(nu)
                results['PV'] = B_T * norm.cdf(d_plus) - K * B_S * norm.cdf(d_minus)

            else:
                raise Exception(f'Analytic engine can\'t be used to price ZCB when '
                                + f'the underlying process is {process.__class__}')

        else:
            raise Exception(f'Product {product.__class__} is not supported by the analytic engine')

        return results

    def __repr__(self):
        return 'CheyetteAnalyticModel()'
