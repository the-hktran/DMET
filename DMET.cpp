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

void SchmidtDecomposition(Eigen::MatrixXd &DensityMatrix, Eigen::MatrixXd &RotationMatrix, const int NumAOImp, const int NumAOEnv)
{
    Eigen::MatrixXd DensityEnv = DensityMatrix.bottomRightCorner(NumAOEnv, NumAOEnv);
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > ESDensityEnv(DensityEnv);
    RotationMatrix = Eigen::MatrixXd::Zero(NumAOImp + NumAOEnv, NumAOImp + NumAOEnv);
    RotationMatrix.topLeftCorner(NumAOImp, NumAOImp) = Eigen::MatrixXd::Identity(NumAOImp, NumAOImp);
    RotationMatrix.bottomRightCorner(NumAOEnv, NumAOEnv) = ESDensityEnv.eigenvectors();
}

double OneElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int c, int d)
{
    double hcd = 0;
    for(int p = 0; p < RotationMatrix.rows(); p++)
    {
        for(int q = 0; q < RotationMatrix.rows(); q++)
        {
            hcd += RotationMatrix(p, c) * Integrals[std::to_string(c + 1) + " " + std::to_string(d + 1) + " 0 0"] * RotationMatrix(q, d);
        }
    }
    return hcd;
}

/* Equation 11 in J.S. Kretchmer and G.K-L. Chan, Preprint: https://arxiv.org/abs/1609.07678, (2016). */
double TwoElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int c, int d, int e, int f)
{
    double Vcdef = 0;
    for(int p = 0; p < RotationMatrix.rows(); p++)
    {
        for(int q = 0; q < RotationMatrix.rows(); q++)
        {
            for(int r = 0; r < RotationMatrix.rows(); r++)
            {
                for(int s = 0; s < RotationMatrix.rows(); s++)
                {
                    Vcdef += RotationMatrix(p, c) * RotationMatrix(q, d) * Integrals[std::to_string(p + 1) + " " + std::to_string(r + 1) + " " + std::to_string(q + 1) + " " + std::to_string(s + 1)] * RotationMatrix(r, e) * RotationMatrix(s, f);
                }
            }
        }
    }
    return Vcdef;
}

Eigen::MatrixXd FormImpurityHamiltonian(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int NumAOImp, int NumOcc, double &ChemicalPotential)
{
    Eigen::MatrixXd HamImp(RotationMatrix.rows(), RotationMatrix.cols());// = RotationMatrix.transpose() * HCore * RotationMatrix;
    for(int c = 0; c < HamImp.rows(); c++)
    {
        for(int d = 0; d < HamImp.cols(); d++)
        {
            double Hcd = OneElectronEmbedding(Integrals, RotationMatrix, c, d);
            for(int u = 0; u < NumAOImp - NumOcc; u++)
            {
                int uu = RotationMatrix.rows() - 1 - u; // This indexes the core orbitals, which are at the end of the rotation matrix.
                double Vcudu = TwoElectronEmbedding(Integrals, RotationMatrix, c, u, d, u);
                double Vcuud = TwoElectronEmbedding(Integrals, RotationMatrix, c, u, u, d);
                Hcd += (2 * Vcudu - Vcuud);
            }
            if(c == d)
            {
                Hcd -= ChemicalPotential;
            }
            HamImp(c, d) = Hcd;
        }
    }
    return HamImp;
}

int main(int argc, char* argv[])
{
    InputObj Input;
    if(argc == 4)
    {
        Input.SetNames(argv[1], argv[2], argv[3]);
    }
    else
    {
        Input.GetInputName();
    }
    Input.Set();

    int NumAO = 4;
    int NumAOImp = 2;
    int NumAOEnv = NumAO - NumAOImp;
    int NumOcc = 2;

    std::vector< int > OccupiedOrbitals;
    std::vector< int > EmbeddedOccupiedOrbitals;

    Eigen::MatrixXd FockMatrix(NumAO, NumAO);
    Eigen::MatrixXd DensityMatrix(NumAO, NumAO);

    for(int i = 0; i < NumAO; i++)
    {
        for(int j = 0; j < NumAO; j++)
        {
            DensityMatrix(i, j) = 1;
        }
    }
    /* Solve SCF and obtain density matrix */
    /* Obtain Schmidt Decomposition */

    /* The eigenvalues are ordered lowest to highest. So the first orbitals are the virtual orbitals, then bath, then core. */
    Eigen::MatrixXd RotationMatrix(NumAO, NumAO);
    SchmidtDecomposition(DensityMatrix, RotationMatrix, NumAOImp, NumAOEnv);
    std::cout << RotationMatrix << std::endl;

    /* Rotate Hamiltonian Matrix using Schmidt Decomposition */
    double ChemicalPotential = 0;
    Eigen::MatrixXd FockImp = FormImpurityHamiltonian(Input.Integrals, RotationMatrix, NumAOImp, NumOcc, ChemicalPotential);
    std::cout << FockImp << std::endl;
    /* Solve again */
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemFockImp(FockImp); // Eigenvectors and eigenvalues ordered from lowest to highest eigenvalues
    Eigen::MatrixXd CoeffImp = EigensystemFockImp.eigenvectors();
    Eigen::MatrixXd DensityImp(NumAO, NumAO);
    for(int i = 0; i < NumAO; i++)
    {
        for(int j = 0; j < NumAO; j++)
        {
            DensityImp(i, j) = 0;
            for(int k = 0; k < NumOcc; k++)
            {
                DensityImp(i, j) += CoeffImp(i, k) * CoeffImp(j, k);
            }
        }
    }
    /* Match density to get u */
        /* Change u */
        /* Check SCF */
    /* Repeat */
    return 0;
}