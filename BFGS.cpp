#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <tuple>
#include "ReadInput.h"
#include <fstream>
#include <map>
#include <stdlib.h> 
#include <algorithm> // std::sort
#include <iomanip>

double CalcdDrr(int r, Eigen::MatrixXd &Z, Eigen::MatrixXd &CoeffMatrix, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals)
{
    Eigen::VectorXd rComponentOcc(OccupiedOrbitals.size());
    Eigen::VectorXd rComponentVir(VirtualOrbitals.size());
    for(int i = 0; i < OccupiedOrbitals.size(); i++)
    {
        rComponentOcc[i] = CoeffMatrix.coeffRef(r, OccupiedOrbitals[i]);
    }
    for(int a = 0; a < VirtualOrbitals.size(); a++)
    {
        rComponentVir[a] = CoeffMatrix.coeffRef(r, VirtualOrbitals[a]);
    }

    double dDrr = (rComponentOcc.transpose() * Z.transpose() * rComponentVir + rComponentVir.transpose() * Z * rComponentOcc).sum();
    return dDrr;
}

Eigen::MatrixXd CalcZMatrix(int k, int l, Eigen::MatrixXd CoeffMatrix, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, Eigen::VectorXd OrbitalEV)
{
    Eigen::MatrixXd Z(VirtualOrbitals.size(), OccupiedOrbitals.size());
    for(int a = 0; a < VirtualOrbitals.size(); a++)
    {
        for(int i = 0; i < OccupiedOrbitals.size(); i++)
        {
            Z(a, i) = CoeffMatrix.coeffRef(k, VirtualOrbitals[a]) * CoeffMatrix.coeffRef(l, OccupiedOrbitals[i]) / (OrbitalEV[VirtualOrbitals[a]] - OrbitalEV[OccupiedOrbitals[i]]);
        }
    }
    return Z;
}

void BFGS(Eigen::MatrixXd &Hessian, Eigen::VectorXd Gradient, Eigen::VectorXd GradientPrev, Eigen::VectorXd &x)
{
    Eigen::VectorXd p;
    p = Hessian.colPivHouseholderQr().solve(-1 * Gradient);
    double a = 0.1;
    Eigen::VectorXd s = a * p;
    x = x + s;
    Eigen::VectorXd y = Gradient - GradientPrev;
    Hessian = Hessian + (y * y.transpose()) / (y.dot(s)) - (Hessian * s * s.transpose() * B) / (s.transpose() * B * s);
}