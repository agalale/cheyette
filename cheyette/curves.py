from abc import ABC, abstractmethod
import numpy as np


class Curve(ABC):
    @abstractmethod
    def df(self, t: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def fwd(self, t1: float, t2: float) -> float:
        raise NotImplementedError

    def annuity(self, underlying_times: np.array):
        return sum((t2 - t1) * self.df(t2) for t1, t2 in zip(underlying_times, underlying_times[1:]))

    def swap_rate(self, underlying_times: np.array):
        return (self.df(underlying_times[0]) - self.df(underlying_times[-1])) / self.annuity(underlying_times)


class FlatCurve(Curve):
    def __init__(self, short_rate: float) -> None:
        self.short_rate = short_rate

    def df(self, t: float) -> float:
        return np.exp(- t * self.short_rate)

    def fwd(self, t1: float, t2: float) -> float:
        return self.short_rate

    def __repr__(self):
        return f'FlatCurve({self.short_rate:.4f})'

    def set(self, key, value):
        if key == 'short_rate':
            self.short_rate = value
            return self
        raise Exception(f'Wrong attribute {key} for {self.__class__}')
