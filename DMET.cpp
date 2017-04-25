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
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd &RotationMatrix, int NumAOImp, double ChemicalPotential, int FragmentIndex);

void SchmidtDecomposition(Eigen::MatrixXd &DensityMatrix, Eigen::MatrixXd &RotationMatrix, std::vector< int > FragmentOrbitals, std::vector< int > EnvironmentOrbitals)
{
    // Eigen::MatrixXd DensityEnv = DensityMatrix.bottomRightCorner(NumAOEnv, NumAOEnv);
    Eigen::MatrixXd DensityEnv(EnvironmentOrbitals.size(), EnvironmentOrbitals.size());
    for(int a = 0; a < EnvironmentOrbitals.size(); a++)
    {
        for(int b = 0; b < EnvironmentOrbitals.size(); b++)
        {
            DensityEnv(a, b) = DensityMatrix.coeffRef(EnvironmentOrbitals[a], EnvironmentOrbitals[b]);
        }
    }
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > ESDensityEnv(DensityEnv);

    int NumAOImp = FragmentOrbitals.size();
    int NumAOEnv = EnvironmentOrbitals.size();
    RotationMatrix.topLeftCorner(NumAOImp, NumAOImp) = Eigen::MatrixXd::Identity(NumAOImp, NumAOImp);
    RotationMatrix.bottomRightCorner(NumAOEnv, NumAOEnv) = ESDensityEnv.eigenvectors();
    // Note that the orbitals have been reordered so that the fragment orbitals are first
}

/* This function projects a matrix in the full impurity - bath space into the impurity - active bath space */
/* The matrix looks something like this
    |----------------|----------------|----------------|----------------|
    |                |                |                |                |
    |                |                |                |                |
    |    IMPURITY    |     I - BV     |     I - BA     |     I - BC     |
    |                |                |                |                |
    |                |                |                |                |
    |----------------|----------------|----------------|----------------|
    |                |                |                |                |
    |                |                |                |                |
    |                |   BATH VIRT    |    BV - BA     |    BV - BC     |
    |                |                |                |                |
    |                |                |                |                |
    |----------------|----------------|----------------|----------------|
    |                |                |                |                |
    |                |                |                |                |
    |                |                |    BATH ACT    |    BA - BC     |
    |                |                |                |                |
    |                |                |                |                |
    |----------------|----------------|----------------|----------------|
    |                |                |                |                |
    |                |                |                |                |
    |                |                |                |   BATH CORE    |
    |                |                |                |                |
    |                |                |                |                |
    |----------------|----------------|----------------|----------------|

    and we project it into
    |----------------|----------------|
    |                |                | 
    |                |                | 
    |    IMPURITY    |     I - BA     | 
    |                |                | 
    |                |                |
    |----------------|----------------|
    |                |                | 
    |                |                | 
    |                |   BATH ACT     | 
    |                |                |
    |                |                |
    |----------------|----------------|
                                                                                            */
void ProjectCAS(Eigen::MatrixXd &NewMatrix, Eigen::MatrixXd &OldMatrix, int NumAOImp, int NumElectrons, int NumOcc)
{
    for(int i = 0; i < NewMatrix.rows(); i++)
    {
        int ii;
        if(i > NumAOImp)
        {
            ii = i % NumAOImp + NumElectrons - NumAOImp - NumOcc;
        }
        else
        {
            ii = i;
        }
        for(int j = 0; j < NewMatrix.cols(); j++)
        {
            int jj;
            if(j > NumAOImp)
            {
                jj = j % NumAOImp + NumElectrons - NumAOImp - NumOcc;
            }
            else
            {
                jj = j;
            }
            NewMatrix(i, j) = OldMatrix(ii, jj);
        }
    }
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

/* FockMatrix and DensityImp have the same dimension, the number of active space orbitals, or 2 * N_imp */
void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &HCore, Eigen::MatrixXd &DensityImp, Eigen::MatrixXd &RotationMatrix, InputObj &Input, double &ChemicalPotential, int FragmentIndex)
{
    for(int c = 0; c < FockMatrix.rows(); c++) // c and d go over active orbitals. The first NumAOImp are the impurity orbitals, the next NumAOImp are bath orbitals, which sit in the middle of the basis ordering.
    {
        int cc;
        if(c > Input.FragmentOrbitals[FragmentIndex].size()) // Means c is a bath orbitals
        {
            cc = c % Input.FragmentOrbitals[FragmentIndex].size() + Input.NumElectrons - Input.FragmentOrbitals[FragmentIndex].size() - Input.NumOcc; // mod plus N_virt
        }
        else // means c is impurity orbital
        {
            cc = c;
        }
        for(int d = 0; d < FockMatrix.cols(); d++)
        {
            int dd;
            if(d > Input.FragmentOrbitals[FragmentIndex].size()) // Means d is a bath orbitals
            {
                dd = d % Input.FragmentOrbitals[FragmentIndex].size() + Input.NumElectrons - Input.FragmentOrbitals[FragmentIndex].size() - Input.NumOcc; // mod plus N_virt
            }
            else // means d is impurity orbital
            {
                dd = d;
            }
            double Hcd = OneElectronEmbedding(Input.Integrals, RotationMatrix, cc, dd);
            HCore(c, d) = Hcd;
            for(int u = 0; u < Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size(); u++) // u goes over core orbitals
            {
                int uu = RotationMatrix.rows() - 1 - u; // This indexes the core orbitals, which are at the end of the rotation matrix.
                double Vcudu = TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, uu, dd, uu);
                double Vcuud = TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, uu, uu, dd);
                Hcd += (2 * Vcudu - Vcuud);
            }
            for(int l = 0; l < FockMatrix.rows(); l++) // XC within active space.
            {
                int ll;
                if(l > Input.FragmentOrbitals[FragmentIndex].size()) // Means c is a bath orbitals
                {
                    ll = l % Input.FragmentOrbitals[FragmentIndex].size() + Input.NumElectrons - Input.FragmentOrbitals[FragmentIndex].size() - Input.NumOcc; // mod plus N_virt
                }
                else // means c is impurity orbital
                {
                    ll = l;
                }
                for(int k = 0; k < FockMatrix.cols(); k++)
                {
                    int kk;
                    if(k > Input.FragmentOrbitals[FragmentIndex].size()) // Means c is a bath orbitals
                    {
                        kk = k % Input.FragmentOrbitals[FragmentIndex].size() + Input.NumElectrons - Input.FragmentOrbitals[FragmentIndex].size() - Input.NumOcc; // mod plus N_virt
                    }
                    else // means c is impurity orbital
                    {
                        kk = k;
                    }
                    Hcd += 0.5 * DensityImp(l, k) * TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, dd, ll, kk);
                }
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
        Eigen::MatrixXd RotationMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
        // This defines the bath space for the given impurity space. The density matrix is of the full system and it does not change.
        SchmidtDecomposition(DensityMatrix, RotationMatrix, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x]);

        /* Before we continue with the SCF, we need to reduce the dimensionality of everything into the active space */
        // The meaning of the density isn't important since we optimize it with SCF anyways. 
        Eigen::MatrixXd CASDensity = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
        // Rotate the overlap matrix.
        Eigen::MatrixXd RotOverlap = RotationMatrix.transpose() * Input.OverlapMatrix * RotationMatrix;
        Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemRotS(RotOverlap);
        Eigen::SparseMatrix< double > LambdaRotSOrtho(Input.NumAO, Input.NumAO); // Holds the inverse sqrt matrix of eigenvalues of S ( Lambda^-1/2 )
        std::vector<T> tripletList;
        for(int i = 0; i < Input.NumAO; i++)
        {
            tripletList.push_back(T(i, i, 1 / sqrt(EigensystemRotS.eigenvalues()[i])));
        }
        LambdaRotSOrtho.setFromTriplets(tripletList.begin(), tripletList.end());
        Eigen::MatrixXd RotSOrtho = EigensystemRotS.eigenvectors() * LambdaRotSOrtho * EigensystemRotS.eigenvectors().transpose();
        // Then project the overlap matrix into CAS
        Eigen::MatrixXd CASSOrtho(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
        ProjectCAS(CASSOrtho, RotSOrtho, 2 * Input.FragmentOrbitals[x].size(), Input.NumElectrons, Input.NumOcc);
        
        SCF(EmptyBias, 1, CASDensity, Input, Output, CASSOrtho, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, RotationMatrix, NumAOImp, ChemicalPotential, x);
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