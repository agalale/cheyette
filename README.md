# Finite differences implementation of the Cheyette short-rate model.

## Recent tests

Testing apply_tridiagonal() on diagonal matrix: passed
Testing apply_tridiagonal() on tridiagonal matrix: passed
Testing solve_tridiagonal() on diagonal matrix: passed
Testing solve_tridiagonal() on tridiagonal matrix: passed
Testing pricing of a zero coupon bond ZCB
    FDM PV: 0.9801991244899613
    Analytic PV: 0.9801986733067553
    Abs. Error 4.5118320601833517e-07
    Rel. Error 4.602974995035878e-07