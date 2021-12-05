from abc import ABC, abstractmethod
import numpy as np
from cheyette.curves import Curve
from functools import lru_cache
from numba import jit


class CheyetteProcess(ABC):
    """
    dot x = mu_x dt + gamma_x dW[t]
    dot y = mu_y dt

    L = L_x + L_y
    L_x = mu_x * d_x + 0.5 * gamma_x**2 d_x**2 - 0.5 * r
    L_y = mu_y * d_y - 0.5 * r
    """
    @abstractmethod
    def mu_x(self, x: float, y: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def gamma_x(self, x: float, y:float) -> float:
        raise NotImplementedError

    @abstractmethod
    def mu_y(self, x: float, y: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def G(self, t: float, T: float):
        raise NotImplementedError

    @lru_cache(maxsize=None)
    def G_squared_halved(self, t: float, T: float):
        return - 0.5 * self.G(t, T) ** 2

    @lru_cache(maxsize=None)
    def df_x(self, t: float, T: float, x: float):
        return np.exp(- self.G(t, T) * x)

    @lru_cache(maxsize=None)
    def df_y(self, t: float, T: float, y: float):
        return np.exp(self.G_squared_halved(t, T) * y)

    def df(self, curve: Curve, t: float, T: float, x: float, y: float) -> float:
        return curve.df(T) / curve.df(t) * self.df_x(t, T, x) * self.df_y(t, T, y)

    def annuity(self, curve: Curve, t: float, x: float, y: float, underlying_times: np.array, dfs: np.array) -> float:
        return sum((t2 - t1) * curve.df(t2) * self.df_x(t, t2, x) * self.df_y(t, t2, y)
                    for t1, t2 in zip(underlying_times, underlying_times[1:])) / curve.df(t)

    def swap_value(self, curve: Curve, t: float, x: float, y: float, strike: float, underlying_times: np.array, dfs: np.array) -> float:
        return (self.df(curve, t, underlying_times[0], x, y) - self.df(curve, t, underlying_times[-1], x, y)
                - strike * self.annuity(curve, t, x, y, underlying_times, dfs))

    @abstractmethod
    def x_mean(self, t: float):
        raise NotImplementedError

    @abstractmethod
    def y_mean(self, t: float):
        raise NotImplementedError

    @abstractmethod
    def x_stddev(self, t: float):
        raise NotImplementedError

    @abstractmethod
    def y_stddev(self, t: float):
        raise NotImplementedError

    @abstractmethod
    def r(self, curve: Curve, t: float, x: float):
        raise NotImplementedError


class ConstMeanRevProcess(CheyetteProcess):
    def __init__(self, mean_rev: float):
        self.mean_rev = mean_rev

    @lru_cache(maxsize=None)
    def G(self, t: float, T: float):
        return (1 - np.exp(-self.mean_rev * (T - t))) / self.mean_rev

    def mu_y(self, x: float, y: float) -> float:
        return self.gamma_x(x, y) ** 2 - 2 * self.mean_rev * y


@jit(nopython=True)
def swap_value_raw(K, Ts, dfs, t, df_t, x, y, mean_rev):
    G = (1 - np.exp(-mean_rev * (Ts[0] - t))) / mean_rev
    res = dfs[0] * np.exp(-G * x - 0.5 * G**2 * y)
    for T1, T2, df in zip(Ts, Ts[1:], dfs[1:]):
        G = (1 - np.exp(-mean_rev * (T2 - t))) / mean_rev
        this_factor = df * np.exp(-G * x - 0.5 * G**2 * y)
        res -= K * (T2 - T1) * this_factor
    res -= this_factor
    return res / df_t


@jit(nopython=True)
def annuity_raw(Ts, dfs, t, df_t, x, y, mean_rev):
    res = 0
    for T1, T2, df in zip(Ts, Ts[1:], dfs[1:]):
        G = (1 - np.exp(-mean_rev * (T2 - t))) / mean_rev
        res += (T2 - T1) * df * np.exp(-G * x - 0.5 * G**2 * y)
    return res / df_t


class VasicekProcess(ConstMeanRevProcess):
    """
        dx[t] = (y[t] - mean_rev * y[t])*dt + local_vol * dW[t]
        dy[t] = (local_vol**2 - 2 * mean-rev * y[t]) * dt
    """
    def __init__(self, mean_rev: float, local_vol: float):
        ConstMeanRevProcess.__init__(self, mean_rev)
        self.local_vol = local_vol
        self.local_vol_squared = local_vol ** 2

    def mu_x(self, x: float, y: float) -> float:
        return y - self.mean_rev * x

    def mu_y(self, x: float, y: float) -> float:
        return self.local_vol_squared - 2 * self.mean_rev * y

    def gamma_x(self, x: float, y: float) -> float:
        return self.local_vol

    def x_mean(self, t: float):
        return self.local_vol ** 2 / (2 * self.mean_rev ** 2) * (1 - np.exp(-self.mean_rev * t)) ** 2

    def y_mean(self, t: float):
        return self.local_vol ** 2 / (2 * self.mean_rev) * (1 - np.exp(-2 * self.mean_rev * t))

    def x_stddev(self, t: float):
        return np.sqrt(self.local_vol ** 2 / (2 * self.mean_rev) * (1 - np.exp(-2 * self.mean_rev * t)))

    def y_stddev(self, t: float):
        return 0.0

    def swap_value(self, curve: Curve, t: float, x: float, y: float, strike: float, underlying_times: np.array, dfs: np.array) -> float:
        return swap_value_raw(strike, underlying_times, dfs, t, curve.df(t), x, y, self.mean_rev)

    def annuity(self, curve: Curve, t: float, x: float, y: float, underlying_times: np.array, dfs: np.array) -> float:
        return annuity_raw(underlying_times, dfs, t, curve.df(t), x, y, self.mean_rev)

    @classmethod
    def r(cls, curve: Curve, t: float, x: float):
        return curve.fwd(0, t) + x

    def __repr__(self):
        return f'Stationary Hull-White process' \
               f'dx[t] = (y[t] - kappa*x[t])dt + sigma*dW[t]\n' \
               f'dy[t] = (sigma**2 - 2*kappa*y[t])dt\n' \
               f'kappa={self.mean_rev}, sigma={self.local_vol}'

    def set(self, key, value):
        if key == 'local_vol':
            self.local_vol = value
            return self
        elif key == 'mean_rev':
            self.mean_rev = value
            return self
        raise Exception(f'Attribute {key} does not exist for {self.__class__}')


class QuadraticProcess(ConstMeanRevProcess):
        """
            dx[t] = (y[t] - mean_rev * y[t])*dt + sigma(x[t]) * dW[t]
            dy[t] = (sigma(x[t])**2 - 2 * mean-rev * y[t]) * dt
            sigma(x) = a + b * x + c * x**2
        """

        def __init__(self, mean_rev: float, a: float, b: float, c: float):
            ConstMeanRevProcess.__init__(self, mean_rev)
            self.a = a
            self.b = b
            self.c = c

        def mu_x(self, x: float, y: float) -> float:
            return y - self.mean_rev * x

        def gamma_x(self, x: float, y: float) -> float:
            return self.a + self.b * x + self.c * x**2

        def x_mean(self, t: float):
            return self.a ** 2 / (2 * self.mean_rev ** 2) * (1 - np.exp(-self.mean_rev * t))**2

        def y_mean(self, t: float):
            return self.a**2 / (2 * self.mean_rev) * (1 - np.exp(-2 * self.mean_rev * t))

        def x_stddev(self, t: float):
            return np.sqrt(self.a**2 / (2 * self.mean_rev) * (1 - np.exp(-2 * self.mean_rev * t)))

        def y_stddev(self, t: float):
            return 0.0

        @classmethod
        def r(cls, curve: Curve, t: float, x: float):
            return curve.fwd(0, t) + x

        def __repr__(self):
            return f'dx[t] = (y[t] - kappa*x[t])dt + sigma(x[t])*dW[t]\n' \
                   f'dy[t] = (sigma(x[t])**2 - 2*kappa*y[t])dt\n' \
                   f'sigma(x)=a + b * x + c * x**2'


class QuadraticAnnuityProcess(ConstMeanRevProcess):
    """
    dx[t] = mu(t, x, y) dt + sigma(t, x, y) dW[t]
    dy[t] = (sigma(t, x, y) ** 2 - 2 * k * y[t]) * dt
    sigma(t, x, y) = a + b * x + c * x ^ 2
    """
    def __init__(self, mean_rev: float, local_vol: float, skew: float, smile: float):
        ConstMeanRevProcess.__init__(self, mean_rev)
        self.local_vol = local_vol
        self.skew = skew
        self.smile = smile

    def mu_x(self, x: float, y: float) -> float:
        return - self.mean_rev * x

    def gamma_x(self, x: float, y: float) -> float:
        return self.local_vol + self.skew * x + self.smile * x ** 2

    @classmethod
    def r(cls, curve: Curve, t: float, x: float):
        return 0.0

    def x_mean(self, t: float):
        return self.local_vol ** 2 / (2 * self.mean_rev ** 2) * (1 - np.exp(-self.mean_rev * t)) ** 2

    def y_mean(self, t: float):
        return self.local_vol ** 2 / (2 * self.mean_rev) * (1 - np.exp(-2 * self.mean_rev * t))

    def x_stddev(self, t: float):
        return np.sqrt(self.local_vol ** 2 / (2 * self.mean_rev) * (1 - np.exp(-2 * self.mean_rev * t)))

    def y_stddev(self, t: float):
        return 0.0

    def __repr__(self):
        return 'Quadratic annuity process\n' \
               'dx[t]=-k*x[t]*dt + sigma(x[t])*dW[t]\n' \
               'dy[t]=(sigma(x[t])**2 - 2*k*x[t])*dt\n' \
               f'sigma(x)={self.local_vol:.2f} + {self.skew:.2f} * x + {self.smile:.2f} * x**2'
