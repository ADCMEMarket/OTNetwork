#include "EMD.h"
#include "eigen3/Eigen/Core"
// int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, int maxIter);

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main(){
    int n1 = 2, n2 = 2;
    VectorXd X = VectorXd::Ones(n1)/n1;
    VectorXd Y =  VectorXd::Ones(n2)/n2;
    MatrixXd D(n1, n2); D << 0., 1., 1., 0.;
    VectorXd alpha(n1);
    VectorXd beta(n2);
    MatrixXd G(n1, n2);
    double cost = 0.0;
    int maxIter = 100;
    int k = EMD_wrap(n1, n2, X.data(), Y.data(), D.data(), G.data(), alpha.data(), beta.data(), &cost, maxIter);
    
    std::cout << k << std::endl;
    std::cout << G << std::endl;
    std::cout << cost << std::endl;
    return 1;
};