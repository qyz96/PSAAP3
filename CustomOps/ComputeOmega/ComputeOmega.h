#include "../Common/Data.h"
#include "had.h"


// compute wk
using namespace had;
namespace had { threadDefine ADGraph* g_ADGraph = 0; }


void forward_ComputeOmega(
  double *omega,
  double *domega_dvars, // N x (N + 1)
  const double *Y, double T){
   ADGraph adGraph;


   int N = gd.N, M = gd.M;
   

   for (int i = 0; i < N; i++){

     std::vector<AReal> AY;
     for (int p = 0; p < N; p++) AY.emplace_back(Y[p]);
     AReal AT = T;
     AReal Omega = 0.0;

      for (int j = 0; j < M; j++){
        auto Kf = gd.A[j] * pow(AT, gd.beta[j]) * exp(-gd.E[j]/gd.R/AT);
        auto Kr = gd.K[j] / pow(gd.pa/gd.R/AT, gd.nu[j]) / 
                      exp( gd.dS[j]/gd.R - gd.dH[j]/gd.R/AT );
        AReal Xk1 = 1.0, Xk2 = 1.0;
        for (int k = 0; k < N; k++){
          Xk1 *= pow(gd.rho * AY[k] / gd.W[k], gd.nu1[k][j]);
          Xk2 *= pow(gd.rho * AY[k] / gd.W[k], gd.nu2[k][j]);
        }
        auto Qj = Kf * Xk1 - Kr * Xk2;
        Omega += Qj * (gd.nu1[i][j] - gd.nu2[i][j]) * gd.W[i];
      }
      SetAdjoint(Omega, 1.0);
      PropagateAdjoint();
      for (int j = 0; j < N; j++){
        *(domega_dvars++) = GetAdjoint(AY[j]);
      }
      *(domega_dvars++) = GetAdjoint(AT);
      omega[i] = Omega.val;

      adGraph.Clear();

   }
}


extern "C" void forward_ComputeOmega_Julia(
  double *omega,
  double *domega_dvars, // N x (N + 1)
  const double *Y, double T){
forward_ComputeOmega(omega, domega_dvars, Y, T);
  }

void backward_ComputeOmega(
  double *grad_Y, double *grad_T,
  const double *grad_omega,
  const double *Y, double T){
   ADGraph adGraph;
   int N = gd.N, M = gd.M;
   
   AReal loss = 0.0;
   std::vector<AReal> AY;
    for (int p = 0; p < N; p++) AY.emplace_back(Y[p]);
    AReal AT = T;
    
    
   for (int i = 0; i < N; i++){
      AReal Omega = 0.0;
      for (int j = 0; j < M; j++){
        auto Kf = gd.A[j] * pow(AT, gd.beta[j]) * exp(-gd.E[j]/gd.R/AT);
        auto Kr = gd.K[j] / pow(gd.pa/gd.R/AT, gd.nu[j]) / 
                      exp( gd.dS[j]/gd.R - gd.dH[j]/gd.R/AT );
        AReal Xk1 = 1.0, Xk2 = 1.0;
        for (int k = 0; k < N; k++){
          Xk1 *= pow(gd.rho * AY[k] / gd.W[k], gd.nu1[k][j]);
          Xk2 *= pow(gd.rho * AY[k] / gd.W[k], gd.nu2[k][j]);
        }
        auto Qj = Kf * Xk1 - Kr * Xk2;
        Omega += Qj * (gd.nu1[i][j] - gd.nu2[i][j]) * gd.W[i];
      }
      loss += Omega * grad_omega[i];

   }
   SetAdjoint(loss, 1.0);
   PropagateAdjoint();
   for (int i = 0; i < N; i++)
      grad_Y[i] = GetAdjoint(AY[i]);
   grad_T[0] = GetAdjoint(AT);
}

