import numpy as np
from main import apply_tridiagonal, solve_tridiagonal
from main import FlatCurve, VasicekProcess, UniformMesh2D, ZCB, CheyetteOperator, PeacemanRachford, CheyettePDEEngine
from main import ZCBCall, CheyetteAnalyticEngine

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
valuation_time = 0.0
curve = FlatCurve(0.1)
local_vol = 0.02
expiry = 1.0
process = VasicekProcess(curve=curve, mean_rev=0.1, local_vol=local_vol)

# Space grid
x_grid_stddevs = 1
y_grid_stddevs = 1
x_grid_center = 0
y_grid_center = 0
x_grid_stddev = 1
y_grid_stddev = 1
x_freq = 100
y_freq = 50

mesh = UniformMesh2D(x_grid_stddevs, y_grid_stddevs,
                       x_grid_center, y_grid_center,
                       x_grid_stddev, y_grid_stddev,
                       x_freq, y_freq)

# Time grid
t_step = 0.1

# Product
product = ZCB(process, expiry)
times = np.arange(valuation_time, expiry, t_step)

operator = CheyetteOperator(process, mesh)
stepping_method = PeacemanRachford(operator)
pde_engine = CheyettePDEEngine(valuation_time, t_step, expiry, product, stepping_method)

results = pde_engine.price()
print('Testing pricing of a zero coupon bond ZCB')
print('    PDE PV:', results['PV'])
print('    Analytic PV:', curve.df(expiry))
print(f"    Abs. Error: {results['PV'] - curve.df(expiry)}")
print(f"    Rel. Error', {(results['PV'] - curve.df(expiry))/results['PV']:.2e}")
