from abc import ABC, abstractmethod
import numpy as np
from cheyette.processes import CheyetteProcess, ConstMeanRevProcess
from cheyette.curves import Curve
from enum import Enum


class CheyetteProduct(ABC):
    def __init__(self, expiry: float):
        self.expiry = expiry

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
    call = 0
    put = 1


class BondOpt(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, bond_expiry: float, option_type: OptionType) -> None:
        CheyetteProduct.__init__(self, expiry)
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
    def __init__(self, strike: float, expiry: float, underlying_times: np.array, curve: Curve):
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.underlying_times = underlying_times

        self.underlying_dfs = np.array([curve.df(T) for T in self.underlying_times])

    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float) -> float:
        return max(process.swap_value(curve, t, x, y, self.strike, self.underlying_times, self.underlying_dfs), 0.0)

    def __repr__(self):
        return f'PayerSwaption(strike={self.strike:.2f}, ' \
               f'expiry={self.expiry:.2f}, ' \
               f'underlying_times={self.underlying_times})'


class PayerSwaptionAnnuity(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, underlying_times: np.array,
                 curve: Curve, process: ConstMeanRevProcess):
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.underlying_times = underlying_times

        self.annuity = curve.annuity(self.underlying_times)
        ws = [(t2 - t1) * curve.df(t2) / self.annuity for t1, t2 in zip(underlying_times, underlying_times[1:])]
        psi_factors_first = [process.G(self.underlying_times[0], T) * np.exp(-process.mean_rev * self.underlying_times[0])
                             for T in self.underlying_times[1:]]
        psi_factors_last = [process.G(T, self.underlying_times[-1]) * np.exp(-process.mean_rev * T) for T in
                            underlying_times[1:]]
        self.alpha_factor_first = sum(w * psi for w, psi in zip(ws, psi_factors_first))
        self.alpha_factor_last = sum(w * psi for w, psi in zip(ws, psi_factors_last))
        self.df_first = curve.df(self.underlying_times[0]) / self.annuity
        self.df_last = curve.df(self.underlying_times[-1]) / self.annuity

    def intrinsic_value(self, curve: Curve, process: ConstMeanRevProcess, t: float, x: float, y: float) -> float:
        multiplier = np.exp(process.mean_rev * t)
        alpha_first = self.alpha_factor_first * multiplier
        alpha_last = self.alpha_factor_last * multiplier
        return self.annuity * max(self.df_first * np.exp(alpha_first * x - 0.5 * alpha_first ** 2 * y)
                                  - self.df_last * np.exp(-alpha_last * x - 0.5 * alpha_last ** 2 * y)
                                  - self.strike, 0.0)

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
