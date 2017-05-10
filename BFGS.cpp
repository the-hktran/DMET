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

void GetCASPos(InputObj Input, int FragmentIndex, std::vector< int > &FragmentPos, std::vector< int > &BathPos);

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
            Z(a, i) = CoeffMatrix.coeffRef(k, VirtualOrbitals[a]) * 1 * CoeffMatrix.coeffRef(l, OccupiedOrbitals[i]) / (OrbitalEV[VirtualOrbitals[a]] - OrbitalEV[OccupiedOrbitals[i]]);
        }
    }
    return Z;
}

void BFGS_1(Eigen::MatrixXd &Hessian, Eigen::VectorXd &s, Eigen::VectorXd Gradient, Eigen::VectorXd &x)
{
    Eigen::VectorXd p;
    p = Hessian.colPivHouseholderQr().solve(-1 * Gradient);
    std::cout << "p\n" << p << std::endl;
    std::cout << "Gradient\n" << Gradient << std::endl;
    double a = 10000;
    s = a * p;
    x = x + s;
}
void BFGS_2(Eigen::MatrixXd &Hessian, Eigen::VectorXd &s, Eigen::VectorXd Gradient, Eigen::VectorXd GradientPrev, Eigen::VectorXd &x)
{
    Eigen::VectorXd y = Gradient - GradientPrev;
    Hessian = Hessian + (y * y.transpose()) / (y.dot(s)) - (Hessian * s * s.transpose() * Hessian) / (s.transpose() * Hessian * s);
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

// This is del D_rr, which is part of the full derivative
Eigen::VectorXd CalcRRGradient(int r, std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, std::vector< std::vector< double > > PotentialElements, Eigen::MatrixXd &CoeffMatrix, Eigen::VectorXd OrbitalEV, InputObj Input, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals)
{
    int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd Gradient(TotPos);
    int TotIndex = 0;
    for(int x = 0; x < PotentialPositions.size(); x++)
    {
        for(int i = 0; i < PotentialPositions[x].size(); i++)
        {
            Eigen::MatrixXd Z = CalcZMatrix(PotentialPositions[x][i].first, PotentialPositions[x][i].second, PotentialElements[x][i], CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, OrbitalEV);
            double dDrr = CalcdDrr(r, Z, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals);
            Gradient[TotIndex] = dDrr;
            TotIndex++;
        }
    }

    return Gradient;
}

// This forms a vector of the positions of nonzero elements in the correlation potential. These are separated by fragments, which doesn't
// matter but makes the gradient calculation neater to compute.
void SetUVector(std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, std::vector< std::vector< double > > &PotentialElements, Eigen::MatrixXd &DMETPotential, InputObj &Input)
{
    for(int x = 0; x < Input.NumFragments; x++)
    {
        std::vector< std::pair< int, int > > FragmentPotentialPositions;
        std::vector< double > FragmentPotentialElements;
        for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
        {
            for(int j = i; j < Input.FragmentOrbitals[x].size(); j++)
            {
                std::pair< int, int > tmpPair = std::make_pair(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]);
                FragmentPotentialPositions.push_back(tmpPair);
                FragmentPotentialElements.push_back(DMETPotential.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]));
            }
        }
        PotentialPositions.push_back(FragmentPotentialPositions);
        PotentialElements.push_back(FragmentPotentialElements);
    }
}

Eigen::VectorXd FragUVectorToFullUVector(std::vector< std::vector< double > > &PotentialElements, int TotPos)
{
    Eigen::VectorXd PotentialElementsVec(TotPos);
    int PosIndex = 0;
    for(int x = 0; x < PotentialElements.size(); x++)
    {
        for(int i = 0; i < PotentialElements[x].size(); i++)
        {
            PotentialElementsVec[PosIndex] = PotentialElements[x][i];
            PosIndex++;
        }
    }
    return PotentialElementsVec;
}

void FullUVectorToFragUVector(std::vector< std::vector< double > > &PotentialElements, Eigen::VectorXd PotentialElementsVec)
{
    int PosIndex = 0;
    for(int x = 0; x < PotentialElements.size(); x++)
    {
        for(int i = 0; i < PotentialElements[x].size(); i++)
        {
            PotentialElements[x][i] = PotentialElementsVec[PosIndex];
            PosIndex++;
        }
    }
}

void FormDMETPotential(Eigen::MatrixXd &DMETPotential, std::vector< std::vector< double > > PotentialElements, std::vector< std::vector< std::pair< int, int > > > PotentialPositions)
{
    for(int x = 0; x < PotentialPositions.size(); x++)
    {
        for(int i = 0; i < PotentialPositions[x].size(); i++)
        {
            DMETPotential(PotentialPositions[x][i].first, PotentialPositions[x][i].second) = PotentialElements[x][i];
            DMETPotential(PotentialPositions[x][i].second, PotentialPositions[x][i].first) = PotentialElements[x][i];
        }
    }
}

// Returns the full gradient of the cost function.
// del (sum_x sum_r (D_rr^x - D_rr)^2 = sum_x sum_r [2 * (D_rr^x - D_rr) * del D_rr]
Eigen::VectorXd CalcGradCF(InputObj &Input, std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, std::vector< std::vector< double > > &PotentialElements, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, std::vector< Eigen::MatrixXd > FragmentDensities, Eigen::MatrixXd FullDensity)
{
    int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd GradCF = Eigen::VectorXd::Zero(TotPos);

    for(int x = 0; x < Input.NumFragments; x++)
    {
        std::vector<int> FragPos, BathPos;
        GetCASPos(Input, x, FragPos, BathPos);
        for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
        {
            Eigen::VectorXd GradDrr = CalcRRGradient(Input.FragmentOrbitals[x][i], PotentialPositions, PotentialElements, CoeffMatrix, OrbitalEV, Input, OccupiedOrbitals, VirtualOrbitals);
            GradCF += 2 * (FragmentDensities[x].coeffRef(FragPos[i], FragPos[i]) - FullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][i])) * GradDrr;
        }
    }
    return GradCF;
}

void UpdatePotential(Eigen::MatrixXd &DMETPotential, InputObj &Input, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, std::vector< Eigen::MatrixXd > FragmentDensities, Eigen::MatrixXd FullDensity)
{
    // First vector holds the positions of the nonzero elements, the second holds the values at each of those posisions.
    // They are divided by fragment, to make some of the code neater.
    std::vector< std::vector< std::pair< int, int > > > PotentialPositions;
    std::vector< std::vector< double > > PotentialElements;
    SetUVector(PotentialPositions, PotentialElements, DMETPotential, Input); // This sets up the previous vectors.
    Eigen::VectorXd GradCF = CalcGradCF(Input, PotentialPositions, PotentialElements, CoeffMatrix, OrbitalEV, OccupiedOrbitals, VirtualOrbitals, FragmentDensities, FullDensity);
    double NormOfGrad = 1;
    int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd PotentialElementsVec = FragUVectorToFullUVector(PotentialElements, TotPos);
    Eigen::MatrixXd Hessian = Eigen::MatrixXd::Identity(TotPos, TotPos);
    Eigen::VectorXd PrevGrad;
    Eigen::VectorXd s;
    while(fabs(NormOfGrad) > 1E-8)
    {
        BFGS_1(Hessian, s, GradCF, PotentialElementsVec);
        PrevGrad = GradCF;
        FullUVectorToFragUVector(PotentialElements, PotentialElementsVec);
        CalcGradCF(Input, PotentialPositions, PotentialElements, CoeffMatrix, OrbitalEV, OccupiedOrbitals, VirtualOrbitals, FragmentDensities, FullDensity);
        BFGS_2(Hessian, s, GradCF, PrevGrad, PotentialElementsVec);
        NormOfGrad = (GradCF - PrevGrad).squaredNorm();
    }
    FormDMETPotential(DMETPotential, PotentialElements, PotentialPositions);
}