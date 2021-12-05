from abc import ABC, abstractmethod
import numpy as np
from cheyette.processes import CheyetteProcess, ConstMeanRevProcess, VasicekProcess, QuadraticAnnuityProcess
from cheyette.curves import Curve
from cheyette.utils import NewtonSolver
from enum import Enum, IntEnum
from functools import lru_cache


class Frequency(IntEnum):
    Annually = 1
    Semi = 2
    Quarterly = 4


default_notional = 10000


class CheyetteProduct(ABC):
    def __init__(self, expiry: float, notional: float = default_notional):
        self.expiry = expiry
        self.notional = notional

    def initialize(self, curve: Curve, process: CheyetteProcess):
        pass

    @abstractmethod
    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float) -> float:
        raise NotImplementedError


class ZCB(CheyetteProduct):
    def __init__(self, expiry: float):
        CheyetteProduct.__init__(self, expiry)

    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float) -> float:
        return process.df(curve, t, self.expiry, x, y)

    def __repr__(self):
        return f'ZCB(expiry={self.expiry:.2f})'


class OptionType(Enum):
    Call = 0
    Put = 1


class BondOpt(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, bond_expiry: float, option_type: OptionType,
                 notional=default_notional) -> None:
        CheyetteProduct.__init__(self, expiry, notional)
        self.strike = strike
        self.bond_expiry = bond_expiry
        self.option_type = option_type

    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float):
        if self.option_type == OptionType.call:
            return max(process.df(curve, t, self.bond_expiry, x, y)
                       - process.df(curve, t, self.expiry, x, y) * self.strike, 0.0)
        if self.option_type == OptionType.put:
            return max(process.df(curve, t, self.expiry, x, y) * self.strike
                       - process.df(curve, t, self.bond_expiry, x, y), 0.0)

    def __repr__(self):
        return f'BondOpt(strike={self.strike}, ' \
               f'expiry={self.expiry}, ' \
               f'bond_expiry={self.bond_expiry}, )'\
               f'option_type={self.option_type})'


class PayerSwaption(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, tenor: float, frequency: Frequency):
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.tenor = tenor
        self.frequency = frequency

    def initialize(self, curve: Curve, process: CheyetteProcess):
        self.underlying_times = np.arange(self.expiry, self.expiry + self.tenor + 0.001, 1.0 / float(self.frequency))
        self.underlying_dfs = np.array([curve.df(T) for T in self.underlying_times])

    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float) -> float:
        return max(process.swap_value(curve, t, x, y, self.strike, self.underlying_times, self.underlying_dfs), 0.0)

    def __repr__(self):
        return f'PayerSwaption(strike={self.strike:.2f}, ' \
               f'expiry={self.expiry:.2f}, ' \
               f'underlying_times={self.underlying_times})'


class PayerSwaptionAnnuity(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, tenor: float, frequency: Frequency, number_of_updates: int = 0):
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.tenor = tenor
        self.frequency = frequency
        self.number_of_updates = number_of_updates

    def initialize(self, curve: Curve, process: QuadraticAnnuityProcess):
        self.underlying_times = np.arange(self.expiry, self.expiry + self.tenor + 0.001, 1.0 / float(self.frequency))
        self.taus = [T2 - T1 for T1, T2 in zip(self.underlying_times, self.underlying_times[1:])]
        self.dfs = [curve.df(t) for t in self.underlying_times]
        self.annuity = curve.annuity(self.underlying_times)
        self.alphas = self.__compute_alphas(process)
        self.weights = [tau * df / self.annuity for tau, df in zip(self.taus, self.dfs[1:])]
        self.solver = NewtonSolver(lambda x: self.swap_rate(process, self.expiry, x, self.y_bar(process, self.expiry)) - self.strike,
                                   lambda x: self.swap_rate_derivative(process, self.expiry, x, self.y_bar(process, self.expiry)))

        self.alpha_factor_first, self.alpha_factor_last = self.__compute_alpha_factors(process)

        for _ in range(self.number_of_updates):
            self.__update_weights(process)

    def swap_rate(self, process: QuadraticAnnuityProcess, t: float, x: float, y: float):
        multiplier = np.exp(process.mean_rev * t)
        alpha_first = self.alpha_factor_first * multiplier
        alpha_last = self.alpha_factor_last * multiplier
        return self.dfs[0] / self.annuity * np.exp(alpha_first * x - 0.5 * alpha_first ** 2 * y) \
               - self.dfs[-1] / self.annuity * np.exp(-alpha_last * x - 0.5 * alpha_last ** 2 * y)

    def swap_rate_derivative(self, process: QuadraticAnnuityProcess, t: float, x: float, y: float):
        multiplier = np.exp(process.mean_rev * t)
        alpha_first = self.alpha_factor_first * multiplier
        alpha_last = self.alpha_factor_last * multiplier
        return alpha_first * self.dfs[0] / self.annuity * np.exp(alpha_first * x - 0.5 * alpha_first ** 2 * y)\
               + alpha_last * self.dfs[-1] / self.annuity * np.exp(-alpha_last * x - 0.5 * alpha_last ** 2 * y)

    def __compute_alphas(self, process: QuadraticAnnuityProcess):
        return [[process.G(t, T) * np.exp(-process.mean_rev * (t - self.expiry))
                 for t in self.underlying_times[1:]] for T in self.underlying_times]

    def __compute_alpha_factors(self, process: QuadraticAnnuityProcess):
        alpha_first_factor = sum(w * process.G(self.underlying_times[0], t) for w, t
                                      in zip(self.weights, self.underlying_times[1:])) \
                                  * np.exp(-process.mean_rev * self.underlying_times[0])
        alpha_last_factor = sum(w * process.G(t, self.underlying_times[-1]) * np.exp(-process.mean_rev * t)
                                     for w, t in zip(self.weights, self.underlying_times[1:]))
        return alpha_first_factor, alpha_last_factor

    def __update_weights(self, process: QuadraticAnnuityProcess):
        x_star = self.x_star()
        y_bar = self.y_bar(process, self.expiry)
        tmp_weights = [0.0] * len(self.weights)
        for j, (T, df) in enumerate(zip(self.underlying_times[1:], self.dfs[1:])):
            this_sum = sum(w * alpha for w, alpha in zip(self.weights, self.alphas[j + 1]))
            w_star = self.taus[j] * df / self.annuity * np.exp(- this_sum * x_star - 0.5 * this_sum ** 2 * y_bar)
            tmp_weights[j] = self.weights[j] + w_star
        self.weights[:] = [w / sum(tmp_weights) for w in tmp_weights]
        self.alpha_factor_first, self.alpha_factor_last = self.__compute_alpha_factors(process)

    @lru_cache(maxsize=None)
    def y_bar(self, process: QuadraticAnnuityProcess, t: float):
        return process.local_vol ** 2 * (1 - np.exp(-2 * process.mean_rev * t)) / (2 * process.mean_rev)

    def x_star(self):
        return self.solver.solve(x=0.0, tolerance=1e-6)

    def intrinsic_value(self, curve: Curve, process: QuadraticAnnuityProcess, t: float, x: float, y: float) -> float:
        return self.annuity * max(self.swap_rate(process, t, x, y) - self.strike, 0.0)

    def __repr__(self):
        return f'PayerSwaptionAnnuity(strike={self.strike:.2f}, expiry={self.expiry:.2f})'


class ZCBCallAnnuity(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, bond_expiry: float, curve: Curve, process: ConstMeanRevProcess):
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.expiry = expiry
        self.bond_expiry = bond_expiry

        self.underlying_times = [expiry, bond_expiry]
        self.annuity = curve.annuity(self.underlying_times)
        ws = [(t2 - t1) * curve.df(t2) / self.annuity for t1, t2 in zip(self.underlying_times, self.underlying_times[1:])]
        psi_factors_first = [process.G(self.underlying_times[0], T) * np.exp(-process.mean_rev * self.underlying_times[0])
                             for T in self.underlying_times[1:]]
        psi_factors_last = [process.G(T, self.underlying_times[-1]) * np.exp(-process.mean_rev * T) for T in
                            self.underlying_times[1:]]
        self.alpha_factor_first = sum(w * psi for w, psi in zip(ws, psi_factors_first))
        self.alpha_factor_last = sum(w * psi for w, psi in zip(ws, psi_factors_last))
        self.df_first = curve.df(self.underlying_times[0]) / self.annuity
        self.df_last = curve.df(self.underlying_times[-1]) / self.annuity

    def intrinsic_value(self, curve: Curve, process: ConstMeanRevProcess, t: float, x: float, y: float) -> float:
        multiplier = np.exp(process.mean_rev * t)
        alpha_first = self.alpha_factor_first * multiplier
        alpha_last = self.alpha_factor_last * multiplier
        return self.annuity * max(self.df_last * np.exp(-alpha_last * x - 0.5 * alpha_last ** 2 * y)
                                  - self.strike * self.df_first * np.exp(alpha_first * x - 0.5 * alpha_first ** 2 * y), 0.0)

    def __repr__(self):
        return f'ZCBCallAnnuity(strike={self.strike}, expiry={self.expiry}, bond_expiry={self.bond_expiry})'
