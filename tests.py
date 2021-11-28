import numpy as np
from cheyette.utils import apply_tridiagonal, solve_tridiagonal
from cheyette.curves import FlatCurve
from cheyette.processes import VasicekProcess
from cheyette.products import ZCB
from cheyette.discretization import PeacemanRachford
from cheyette.models import CheyettePDEModel
from cheyette.boundary_conditions import DirichletIntrinsicBC
from cheyette.pricers import CheyettePricer

# apply_tridiagonal on diagonal matrix
diag = np.array([1, 2, 3])
upper = np.array([0, 0])
lower = np.array([0, 0])
x = np.array([10, 0.1, 20])
res = np.array([0.0, 0.0, 0.0])
true_res = np.array([10, 0.2, 60])
apply_tridiagonal(lower, diag, upper, x, res)
print('Testing apply_tridiagonal() on diagonal matrix:', 'passed' if np.allclose(res, true_res) else 'failed')

# apply_tridiagonal on general tridiagonal matrix
diag = np.array([1, 1, 0])
upper = np.array([1, 1])
lower = np.array([0, 2])
x = [1, 2, 3]
res = np.array([0.0, 0.0, 0.0])
apply_tridiagonal(lower, diag, upper, x, res)
true_res = [3.0, 5.0, 4.0]
print('Testing apply_tridiagonal() on tridiagonal matrix:', 'passed' if np.allclose(res, true_res) else 'failed')

# solve_tridiagonal on diagonal_matrix
diag = np.array([1.0, 2, 3])
upper = np.array([0.0, 0.0])
upper_tmp = np.array([0.0, 0.0])
lower = np.array([0.0, 0])
true_x = np.array([10.0, 0.1, 20])
y = np.array([10.0, 0.2, 60.0])
y_tmp = np.array([0.0, 0.0, 0.0])

solve_tridiagonal(lower, diag, upper, upper_tmp, y_tmp, y, x)
print('Testing solve_tridiagonal() on diagonal matrix:', 'passed' if np.allclose(x, true_x) else 'failed')

# solve_tridiagonal on general tridiagonal matrix
diag = np.array([1, 1, 0])
upper = np.array([1, 1])
lower = np.array([0, 2])
true_x = [1.0, 2.0, 3.0]
x = np.array([0.0, 0, 0])
y = np.array([3.0, 5.0, 4.0])
y_tmp = np.array([0.0, 0, 0])
upper_tmp = np.array([0.0, 0])
solve_tridiagonal(lower, diag, upper, upper_tmp, y_tmp, y, x)
print('Testing solve_tridiagonal() on tridiagonal matrix:', 'passed' if np.allclose(x, true_x) else 'failed')

# Pricing a zero coupon bond
curve = FlatCurve(0.02)
process = VasicekProcess(mean_rev=0.10, local_vol=0.02)
model_pde = CheyettePDEModel(x_grid_stddevs=1, y_grid_stddevs=1,
                             x_freq=10, y_freq=10, t_freq=10,
                             stepping_method=PeacemanRachford(),
                             x_lower_bc=DirichletIntrinsicBC(),
                             x_upper_bc=DirichletIntrinsicBC(),
                             y_lower_bc=DirichletIntrinsicBC(),
                             y_upper_bc=DirichletIntrinsicBC())
product = ZCB(expiry=1.0)
pricer_pde = CheyettePricer(model_pde, curve, process, product, valuation_time=0.0)

results = pricer_pde.price()
print('Testing pricing of a zero coupon bond ZCB')
print('    PDE PV:', results['PV'])
print('    Analytic PV:', curve.df(product.expiry))
print(f"    Abs. Error: {results['PV'] - curve.df(product.expiry):.2e}")
print(f"    Rel. Error', {(results['PV'] - curve.df(product.expiry)) / results['PV']:.2e}")
