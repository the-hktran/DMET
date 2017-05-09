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
#include <pair>

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

Eigen::MatrixXd CalcZMatrix(int k, int l, double ukl, Eigen::MatrixXd &CoeffMatrix, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, Eigen::VectorXd OrbitalEV)
{
    Eigen::MatrixXd Z(VirtualOrbitals.size(), OccupiedOrbitals.size());
    for(int a = 0; a < VirtualOrbitals.size(); a++)
    {
        for(int i = 0; i < OccupiedOrbitals.size(); i++)
        {
            Z(a, i) = CoeffMatrix.coeffRef(k, VirtualOrbitals[a]) * ukl * CoeffMatrix.coeffRef(l, OccupiedOrbitals[i]) / (OrbitalEV[VirtualOrbitals[a]] - OrbitalEV[OccupiedOrbitals[i]]);
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

int CalcTotalPositions(std::vector< std::vector< std::pair< int, int > > > &PotentialPositions)
{
    int TotPos = 0;
    for(int x = 0; x < PotentialPositions.size(); x++)
    {
        TotPos += PotentialPositions[x].size();
    }
    return TotPos;
}

Eigen::VectorXd CalcRRGradient(int r, std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, std::vector< std::vector< double > > PotentialElements, Eigen::MatrixXd &CoeffMatrix, Eigen::VectorXd OrbitalEV, InputObj Input)
{
    int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd Gradient(TotPos);
    int TotIndex = 0;
    for(int x = 0; x < PotentialPositions.size(); x++)
    {
        for(int i = 0; i < PotentialPositions[x].size(); i++)
        {
            Eigen::MatrixXd Z = CalcZMatrix(PotentialPositions[x][i].first, PotentialPositions[x][i].second, PotentialElements[x][i], CoeffMatrix, Input.OccupiedOrbitals[x], Input.VirtualOrbitals[x], OrbitalEV);
            double dDrr = CalcdDrr(r, Z, CoeffMatrix, Input.OccupiedOrbitals[x], Input.VirtualOrbitals[x]);
            Gradient[TotIndex] = dDrr;
            TotIndex++;
        }
    }

    return Gradient;
}

// This forms a vector of the positions of nonzero elements in the correlation potential. These are separated by fragments, which doesn't
// matter but makes the gradient calculation neater to compute.
void SetUVector(std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, InputObj &Input)
{
    for(int x = 0; x < Input.NumFragments; x++)
    {
        std::vector< std::pair< int, int > > FragmentPotentialPositions;
        for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
        {
            for(int j = i; j < Input.FragmentOrbitals[x].size(); j++)
            {
                std::pair< int, int > tmpPair = std::make_pair(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]);
                FragmentPotentialPositions.push_back(tmpPair);
            }
        }
        PotentialPositions.push_back(FragmentPotentialPositions);
    }
}

void FormDMETPotential(Eigen::MatrixXd &DMETPotential, std::vector< double > PotentialElements, std::vector< std::pair< int, int > > PotentialPositions)
{
    for(int i = 0; i < PotentialPositions.size(); i++)
    {
        DMETPotential(PotentialPositions[i].first, PotentialPositions[i].second) = PotentialElements[i];
        DMETPotential(PotentialPositions[i].second, PotentialPositions[i].first) = PotentialElements[i];
    }
}

Eigen::VectorXd CalcGradCF(InputObj &Input, std::vector< std::vector< double > > PotentialElements, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV)
{
    std::vector< std::vector< std::pair< int, int > > > PotentialPositions;
    SetUVector(PotentialPositions, Input);
    int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd GradCF = Eigen::VectorXd::Zero(TotPos);

    for(int x = 0; x < Input.NumFragments; x++)
    {
        std::vector<int> FragPos, BathPos;
        GetCASPos(Input, x, FragPos, BathPos);
        for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
        {
            Eigen::VectorXd GradDrr = CalcRRGradient(FragmentOrbitals[x][i], PotentialPositions, PotentialElements, CoeffMatrix, OrbitalEV, Input)
            CF += 2 * (FragmentDensities[x].coeffRef(FragPos[i], FragPos[i]) - FullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][i])) * 
                  * GradDrr;
        }
    }
    return CF;
}