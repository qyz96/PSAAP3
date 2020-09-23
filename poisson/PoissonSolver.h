#include <iostream>

double f(double y, double x) {
    return y*y*y + 3*y*y + 4 - x;
}

double df(double y, double x) {
    return 3*y*y + 6*y;
}

double SolvePoisson(double x) {
    double y_sol = 10.0;
    for (int i = 0; i < 100; i++) {
        y_sol = y_sol - f(y_sol, x) / df(y_sol, x);
    }
    return y_sol;
}

void PoissonSolver_forward(double *y, const double *x, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = SolvePoisson(x[i]);
        
    }
    return;
}

void PoissonSolver_backward(double *grad_x, const double *grad_y, const double *x, const double *y, int n) {
    for (int i = 0; i < n; i++) {
        grad_x[i] = grad_y[i] * 1.0 / df(y[i],x[i]);
    }
    return;
}