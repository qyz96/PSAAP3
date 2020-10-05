#include "Data.h"

GlobalData gd;

GlobalData::GlobalData(){};
void GlobalData::setValues(){};
void GlobalData::TestSetValues(){
    N = 1;
    M = 1;
    nu1.resize(1);
    nu1[0].push_back(2);
    nu2.resize(1);
    nu2[0].push_back(2);
    rho = 1.0;
    R = 1.0;
    W.push_back(2.0);
    A.push_back(2.0);
    K.push_back(2.0);
    beta.push_back(2.0);
    E.push_back(2.0);
    dS.push_back(2.0);
    dH.push_back(2.0);
    nu.push_back(1);
    pa = 1.0;
};
