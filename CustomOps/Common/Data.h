#include <vector> 
#include <iostream> 

class GlobalData{
public:
    int N; // n species
    int M; // number of reactions
    std::vector< std::vector<int> > nu1;
    std::vector< std::vector<int> > nu2; 
    double rho; 
    double R;

    std::vector< double > W;
    std::vector< double > A;
    std::vector< double > K;
    std::vector<double> beta;
    std::vector<double> E;
    std::vector< double > dS;
    std::vector< double > dH;
    std::vector< int > nu; // length = M, sum_k nu_{kj}
    double pa;
    
    GlobalData();
    void setValues();
    void TestSetValues();

};

extern GlobalData gd;