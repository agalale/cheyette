from abc import ABC, abstractmethod
import numpy as np
from cheyette.processes import CheyetteProcess
from cheyette.curves import Curve


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


class ZCBCall(CheyetteProduct):
    def __init__(self, strike: float, expiry: float, bond_expiry: float) -> None:
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.bond_expiry = bond_expiry

    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float):
        return max(process.df(curve, t, self.bond_expiry, x, y)
                   - process.df(curve, t, self.expiry, x, y) * self.strike, 0.0)

    def __repr__(self):
        return f'ZCBCall(strike={self.strike}, ' \
               f'expiry={self.expiry}, ' \
               f'bond_expiry={self.bond_expiry})'


class PayerSwaption(CheyetteProduct):
    def __init__(self, strike: float, expiry: float,
                 underlying_times: np.array) -> None:
        CheyetteProduct.__init__(self, expiry)
        self.strike = strike
        self.underlying_times = underlying_times

    def intrinsic_value(self, curve: Curve, process: CheyetteProcess, t: float, x: float, y: float) -> float:
        return max(process.swap_value(curve, t, x, y, self.strike, self.underlying_times), 0.0)

    def __repr__(self):
        return f'PayerSwaption(strike={self.strike:.2f}, ' \
               f'expiry={self.expiry:.2f}, ' \
               f'underlying_times={self.underlying_times})'
