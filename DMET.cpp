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

void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &DensityMatrix, std::map<std::string, double> &Integrals, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int NumElectrons);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd &RotationMatrix, int NumAOImp, double ChemicalPotential, int FragmentIndex);

void SchmidtDecomposition(Eigen::MatrixXd &DensityMatrix, Eigen::MatrixXd &RotationMatrix, std::vector< int > FragmentOrbitals, std::vector< int > EnvironmentOrbitals)
{
    // Eigen::MatrixXd DensityEnv = DensityMatrix.bottomRightCorner(NumAOEnv, NumAOEnv);
    Eigen::MatrixXd DensityEnv(FragmentOrbitals.size(), FragmentOrbitals.size());
    for(int a = 0; a < FragmentOrbitals.size(); a++)
    {
        for(int b = 0; b < FragmentOrbitals.size(); b++)
        {
            DensityEnv(a, b) = DensityMatrix.coeffRef(FragmentOrbitals[a], FragmentOrbitals[b]);
        }
    }
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > ESDensityEnv(DensityEnv);

    int NumAOImp = FragmentOrbitals.size();
    int NumAOEnv = EnvironmentOrbitals.size();
    RotationMatrix = Eigen::MatrixXd::Zero(NumAOImp + NumAOEnv, NumAOImp + NumAOEnv);
    RotationMatrix.topLeftCorner(NumAOImp, NumAOImp) = Eigen::MatrixXd::Identity(NumAOImp, NumAOImp);
    RotationMatrix.bottomRightCorner(NumAOEnv, NumAOEnv) = ESDensityEnv.eigenvectors();
    // Note that the orbitals have been reordered so that the fragment orbitals are first
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

void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &DensityImp, Eigen::MatrixXd &RotationMatrix, InputObj &Input, double &ChemicalPotential, int FragmentIndex)
{
    for(int c = 0; c < FockMatrix.rows(); c++) // c and d go over active orbitals
    {
        for(int d = 0; d < FockMatrix.cols(); d++)
        {
            double Hcd = OneElectronEmbedding(Input.Integrals, RotationMatrix, c, d);
            for(int u = 0; u < Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size(); u++) // u goes over core orbitals
            {
                int uu = RotationMatrix.rows() - 1 - u; // This indexes the core orbitals, which are at the end of the rotation matrix.
                double Vcudu = TwoElectronEmbedding(Input.Integrals, RotationMatrix, c, uu, d, uu);
                double Vcuud = TwoElectronEmbedding(Input.Integrals, RotationMatrix, c, uu, uu, d);
                Hcd += DensityImp.coeffRef(c, d) * (2 * Vcudu - Vcuud);
            }
            if(c == d)
            {
                Hcd -= ChemicalPotential;
            }
            FockMatrix(c, d) = Hcd;
        }
    }
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

    std::ofstream Output(Input.OutputName);

    int NumAO = Input.NumAO;
    int NumOcc = Input.NumOcc;

    int NumFragments = Input.NumFragments;
    // std::vector< std::vector< int > > FragmentOrbitals(NumFragments);
    // std::vector< std::vector< int > > EnvironmentOrbitals(NumFragments);
    // FragmentOrbitals[0].push_back(0); FragmentOrbitals[0].push_back(1);
    // EnvironmentOrbitals[0].push_back(2); EnvironmentOrbitals[0].push_back(3);
    // FragmentOrbitals[1].push_back(2); FragmentOrbitals[1].push_back(3);
    // EnvironmentOrbitals[1].push_back(0); EnvironmentOrbitals[1].push_back(1);

    /*****   STEP 1 *****/
    /* Solve for full system at the RHF level of theory */
    
    // Begin by defining some variables.
    std::vector< std::tuple< Eigen::MatrixXd, double, double > > EmptyBias; // SCF is capable of metadynamics, but we won't touch this for now.
    Eigen::MatrixXd HCore(NumAO, NumAO);
    Eigen::MatrixXd DensityMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // Initialize to Zero
    BuildFockMatrix(HCore, DensityMatrix, Input.Integrals, EmptyBias, Input.NumElectrons); // Build HCore;
    for(int i = 0; i < Input.NumOcc; i++)
    {
        DensityMatrix(i, i) = 1;
    }

    /* Form S^-1/2, the orthogonalization transformation */
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemS(Input.OverlapMatrix);
    Eigen::SparseMatrix< double > LambdaSOrtho(Input.NumAO, Input.NumAO); // Holds the inverse sqrt matrix of eigenvalues of S ( Lambda^-1/2 )
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    for(int i = 0; i < Input.NumAO; i++)
    {
        tripletList.push_back(T(i, i, 1 / sqrt(EigensystemS.eigenvalues()[i])));
    }
    LambdaSOrtho.setFromTriplets(tripletList.begin(), tripletList.end());
    Eigen::MatrixXd SOrtho = EigensystemS.eigenvectors() * LambdaSOrtho * EigensystemS.eigenvectors().transpose(); // S^-1/2

    std::vector< int > OccupiedOrbitals;
    std::vector< int > VirtualOrbitals;
    for(int i = 0; i < NumOcc; i++) // Fix the lowest MOs as the occupied MO.
    {
        OccupiedOrbitals.push_back(i);
    }
    for(int i = NumOcc; i < NumAO; i++)
    {
        VirtualOrbitals.push_back(i);
    }

    std::vector< double > AllEnergies;
    Eigen::MatrixXd CoeffMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // Don't think I need this.
    int SCFCount = 0;
    double ChemicalPotential = 0;

    // Solve the full system using RHF.
    SCF(EmptyBias, 1, DensityMatrix, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF);
    AllEnergies.clear();

    // Now start cycling through each fragment.
    for(int x = 0; x < NumFragments; x++)
    {
        int NumAOImp = Input.FragmentOrbitals[x].size();
        int NumAOEnv = NumAO - NumAOImp;
        /*****   STEP 2   *****/
        /* Obtain Schmidt Decomposition */

        /* The eigenvalues are ordered lowest to highest. So the first orbitals are the virtual orbitals, then bath, then core. */
        Eigen::MatrixXd RotationMatrix(NumAO, NumAO);
        SchmidtDecomposition(DensityMatrix, RotationMatrix, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x]);
        
        Eigen::MatrixXd RotatedDensity = RotationMatrix.transpose() * DensityMatrix * RotationMatrix;
        SCF(EmptyBias, 1, RotatedDensity, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, RotationMatrix, NumAOImp, ChemicalPotential, x);
        AllEnergies.clear();
        /* Rotate Hamiltonian Matrix using Schmidt Decomposition */
    
        /* Solve again */
        
        /* Match density to get u */
            /* Change u */
            /* Check SCF */
        /* Repeat */
    }
    return 0;
}