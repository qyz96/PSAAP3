// y^3 + y = x
double CubicSolver_f(double y, double x){
  return y*y*y + y - x;
}

double CubicSolver_df(double y, double x){
  return 3*y*y + 1;
}

double get_cubic_solution(double x){
  double y0 = 0.0;
  for(int i = 0; i < 10; i++){
    y0 = y0 - CubicSolver_f(y0, x)/CubicSolver_df(y0, x);
  }  
  return y0;
}

void CubicSolver_forward(double *y, const double *x, int n){
  for(int i = 0; i< n; i++)
    y[i] = get_cubic_solution(x[i]);
}

// y' = 1/(3y^2 + 1)
void CubicSolver_backward(
  double *grad_x, 
  const double *grad_y, 
  const double *y, int n
){
  for(int i = 0; i<n; i++){
    grad_x[i] = grad_y[i] * 1.0/(3*y[i]*y[i] + 1.0);
  }
}