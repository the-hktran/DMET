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
#include <queue>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cstdio>

#include "Bootstrap.h"
#include "Functions.h"
#include "RealTime.h"
#include "FCI.h"
#include "Fragmenting.h"



// #define H2H2H2
// #define H10

void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &DensityMatrix, std::map<std::string, double> &Integrals, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int NumElectrons);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd CASOverlap, Eigen::MatrixXd &SOrtho, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd &RotationMatrix, double FragmentOcc, int NumAOImp, double ChemicalPotential, int FragmentIndex);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd DMETPotential, Eigen::VectorXd &OrbitalEV);
void UpdatePotential(Eigen::MatrixXd &DMETPotential, InputObj &Input, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV, std::vector< std::vector< int > > OccupiedOrbitals, std::vector< std::vector< int > > VirtualOrbitals, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd> &FullDensities, std::ofstream &Output, std::vector< Eigen::MatrixXd > &FragmentRotations, std::vector< int > ImpurityStates, std::vector< int > BathStates);
double CalcL(InputObj &Input, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd > &FullDensities, std::vector< Eigen::MatrixXd > FragmentRotations, std::vector< int > BathStates, int CostFunctionVariant = 2);

Eigen::MatrixXd ReadMat(std::string MatFile, int N)
{
    Eigen::MatrixXd Mat(N, N);
    std::ifstream File(MatFile);
    int i, j;
    double val;
    std::cout << "N = " << N << std::endl;
    while(!File.eof())
    {
        File >> i >> j >> val;
        Mat(i, j) = val;
        std::cout << val << std::endl;
    }
    return Mat;
}

Eigen::MatrixXd DensityFromCoeff(Eigen::MatrixXd CoeffMatrix, std::vector<int> OccupiedOrbitals)
{
    int NumAO = CoeffMatrix.rows();
    int NumOcc = OccupiedOrbitals.size();
    Eigen::MatrixXd Density(NumAO, NumAO);
    for (int i = 0; i < NumAO; i++)
    {
        for (int j = 0; j < NumAO; j++)
        {
            double Pij = 0.0;
            for (int k = 0; k < NumOcc; k++)
            {
                Pij += CoeffMatrix.coeffRef(i, OccupiedOrbitals[k]) * CoeffMatrix.coeffRef(j, OccupiedOrbitals[k]);
            }
            Density(i, j) = Pij;
        }
    }
    return Density;
}

void GetCASList(InputObj Input, int FragmentIndex, std::vector<int> &ActiveList, std::vector<int> &CoreList, std::vector<int> &VirtualList)
{
    int NumOcc = Input.NumOcc;
    
    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - NumOcc;
    int NumCore = NumOcc - Input.FragmentOrbitals[FragmentIndex].size();

    for (int i = 0; i < NumVirt; i++)
    {
        VirtualList.push_back(Input.EnvironmentOrbitals[FragmentIndex][i]);
    }
    for (int i = 0; i < Input.FragmentOrbitals[FragmentIndex].size(); i++)
    {
        ActiveList.push_back(Input.EnvironmentOrbitals[FragmentIndex][NumVirt + i]);
        ActiveList.push_back(Input.FragmentOrbitals[FragmentIndex][i]);
    }
    std::sort(ActiveList.begin(), ActiveList.end());
    for (int i = 0; i < NumCore; i++)
    {
        CoreList.push_back(Input.EnvironmentOrbitals[FragmentIndex][NumVirt + Input.FragmentOrbitals[FragmentIndex].size() + i]);
    }
}

void GetCASList(InputObj Input, int FragmentIndex, std::vector<int> &ActiveList, std::vector<int> &CoreList, std::vector<int> &VirtualList, bool Alpha)
{
    int NumOcc;
    if (Alpha)
    {
        NumOcc = Input.aNumElectrons;
    }
    else
    {
        NumOcc = Input.bNumElectrons;
    }
    
    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - NumOcc;
    int NumCore = NumOcc - Input.FragmentOrbitals[FragmentIndex].size();

    for (int i = 0; i < NumVirt; i++)
    {
        VirtualList.push_back(Input.EnvironmentOrbitals[FragmentIndex][i]);
    }
    for (int i = 0; i < Input.FragmentOrbitals[FragmentIndex].size(); i++)
    {
        ActiveList.push_back(Input.EnvironmentOrbitals[FragmentIndex][NumVirt + i]);
        ActiveList.push_back(Input.FragmentOrbitals[FragmentIndex][i]);
    }
    std::sort(ActiveList.begin(), ActiveList.end());
    for (int i = 0; i < NumCore; i++)
    {
        CoreList.push_back(Input.EnvironmentOrbitals[FragmentIndex][NumVirt + Input.FragmentOrbitals[FragmentIndex].size() + i]);
    }
}
/* 
   The purpose of this function is to distinguish between fragment or bath orbital in the active space basis.
   The vectors FragmentPos and BathPos hold the index for the impurity and bath orbitals, respectively, in the active space basis
   which is of size 2 * N_imp. This is necessary because the impurity and bath orbitals may not be cleanly separated.

   For example, consider a system with 5 orbitals: 0 1 2 3 4
   Make the impurity orbitals the two orbitals: 2 4
   And suppose we have 1 virtual orbital.
   Then we rotate the orbitals and we have: 0' 1' 2 3' 4 where the prime denotes that the bath orbitals are rotated.
   The bath orbitals are organized so that the virtual orbitals are first, then the active bath, then the core orbitals.
   So the impurity orbitals are still 2 and 4 and the bath orbitals are 1 and 3.
   The CAS ordering would look like this:
        Orbital Number:     1  2  3  4
        Index in CAS:       0  1  2  3
        Imp or Bath:        B  I  B  I
   So in this case, FragPos = [1, 3] and BathPos = [0, 2], the index of the orbitals in the CAS indexing scheme.
*/ 
void GetCASPos(InputObj Input, int FragmentIndex, std::vector< int > &FragmentPos, std::vector< int > &BathPos)
{
    int NumOcc = Input.NumOcc;

    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - NumOcc;
    int NumCore = NumOcc - Input.FragmentOrbitals[FragmentIndex].size();
    int NumBathBefore = 0;

    int NextFragPos = 0;
    int CurrentBathPos = 0;
    /* The idea of the following loop is as follows. We are going to loop through each orbital and check whether we are looking at an impurity
       or active bath orbital. Then we add the count for how many times we've been through the loop into the correct vector. */
    // We loop through each orbital until both vectors have N_imp elements.
    while(BathPos.size() < Input.FragmentOrbitals[FragmentIndex].size() || FragmentPos.size() < Input.FragmentOrbitals[FragmentIndex].size())
    {
        if(NextFragPos < Input.FragmentOrbitals[FragmentIndex].size() && CurrentBathPos < Input.FragmentOrbitals[FragmentIndex].size()) // Means neither vector is "full"
        {
            /* The first condition checks whether the next unaccounted active bath orbital is before the next unaccounted impurity
               orbital. If that is true, then that means we should mark the next index as a bath orbital. We add the number of times 
               we've been through the loop into the BathPos, and then start looking at the next bath orbital by incrementing CurrentBathPos.
               If it is false, it means the next impurity orbital is before the next bath orbital, so we do the opposite */
            if(Input.EnvironmentOrbitals[FragmentIndex][CurrentBathPos + NumVirt] < Input.FragmentOrbitals[FragmentIndex][NextFragPos])
            {
                BathPos.push_back(CurrentBathPos + NextFragPos);
                CurrentBathPos++;
            }
            else
            {
                FragmentPos.push_back(CurrentBathPos + NextFragPos);
                NextFragPos++;
            }
        }
        else // Means one of the vectors is full, so just add the next few orbitals into the not full one until the size is appropriate.
        {
            if(NextFragPos == Input.FragmentOrbitals[FragmentIndex].size())
            {
                BathPos.push_back(CurrentBathPos + NextFragPos);
                CurrentBathPos++;
            }
            else
            {
                FragmentPos.push_back(CurrentBathPos + NextFragPos);
                NextFragPos++;
            }
        }
    }
}

void GetCASPos(InputObj Input, int FragmentIndex, std::vector< int > &FragmentPos, std::vector< int > &BathPos, bool Alpha)
{
    int NumOcc;
    if (Alpha) NumOcc = Input.aNumElectrons;
    else NumOcc = Input.bNumElectrons;

    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - NumOcc;
    int NumCore = NumOcc - Input.FragmentOrbitals[FragmentIndex].size();
    int NumBathBefore = 0;

    int NextFragPos = 0;
    int CurrentBathPos = 0;
    /* The idea of the following loop is as follows. We are going to loop through each orbital and check whether we are looking at an impurity
       or active bath orbital. Then we add the count for how many times we've been through the loop into the correct vector. */
    // We loop through each orbital until both vectors have N_imp elements.
    while(BathPos.size() < Input.FragmentOrbitals[FragmentIndex].size() || FragmentPos.size() < Input.FragmentOrbitals[FragmentIndex].size())
    {
        if(NextFragPos < Input.FragmentOrbitals[FragmentIndex].size() && CurrentBathPos < Input.FragmentOrbitals[FragmentIndex].size()) // Means neither vector is "full"
        {
            /* The first condition checks whether the next unaccounted active bath orbital is before the next unaccounted impurity
               orbital. If that is true, then that means we should mark the next index as a bath orbital. We add the number of times 
               we've been through the loop into the BathPos, and then start looking at the next bath orbital by incrementing CurrentBathPos.
               If it is false, it means the next impurity orbital is before the next bath orbital, so we do the opposite */
            if(Input.EnvironmentOrbitals[FragmentIndex][CurrentBathPos + NumVirt] < Input.FragmentOrbitals[FragmentIndex][NextFragPos])
            {
                BathPos.push_back(CurrentBathPos + NextFragPos);
                CurrentBathPos++;
            }
            else
            {
                FragmentPos.push_back(CurrentBathPos + NextFragPos);
                NextFragPos++;
            }
        }
        else // Means one of the vectors is full, so just add the next few orbitals into the not full one until the size is appropriate.
        {
            if(NextFragPos == Input.FragmentOrbitals[FragmentIndex].size())
            {
                BathPos.push_back(CurrentBathPos + NextFragPos);
                CurrentBathPos++;
            }
            else
            {
                FragmentPos.push_back(CurrentBathPos + NextFragPos);
                NextFragPos++;
            }
        }
    }
}

/* 
   Takes index on the reduced space consisting of only impurity and bath orbitals and returns the actual orbital that index 
   For example, consider the example above the previous function. In that case
        Orbitals:   0 1 2 3 4
        CAS Index:  . 0 1 2 3
   where 2 and 4 are impurity orbitals. The CAS space has dimension 4 and if we are in position 3 (index 2), we want to know that
   we are in orbital 3. This function takes the index (2) and gives the orbital (3). What this does is it takes the index and checks
   whether than index is in the list of impurity or bath orbitals. Then it checks to see in what position that index is on. So if we
   are looking at index 2 in the example, we know FragPos = [1, 3] and BathPos = [0, 2]. So index 2 is in BathPos. Then we count
   which number element it is in the list, so in this case 2 is the second element in BathPos. Then we simply take the second
   active bath orbital as the orbital it is pointing to.
*/
int ReducedIndexToOrbital(int c, InputObj Input, int FragmentIndex)
{
    int NumOcc = Input.NumOcc;

    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - NumOcc;
    int Orbital;
    std::vector< int > FragPos;
    std::vector< int > BathPos;
    GetCASPos(Input, FragmentIndex, FragPos, BathPos);
    auto PosOfIndex = std::find(FragPos.begin(), FragPos.end(), c);
    if (PosOfIndex == FragPos.end()) // Means the index is in the bath orbital.
    {
        PosOfIndex = std::find(BathPos.begin(), BathPos.end(),c);
        auto IndexOnList = std::distance(BathPos.begin(), PosOfIndex);
        Orbital = Input.EnvironmentOrbitals[FragmentIndex][IndexOnList + NumVirt]; // Add NumVirt because the virtual orbitals are before the bath active orbitals.
    }
    else 
    {
        auto IndexOnList = std::distance(FragPos.begin(), PosOfIndex);
        Orbital = Input.FragmentOrbitals[FragmentIndex][IndexOnList];
    }
    return Orbital;
}

int ReducedIndexToOrbital(int c, InputObj Input, int FragmentIndex, bool Alpha)
{
    int NumOcc;
    if (Alpha) NumOcc = Input.aNumElectrons;
    else NumOcc = Input.bNumElectrons;

    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - NumOcc;
    int Orbital;
    std::vector< int > FragPos;
    std::vector< int > BathPos;
    GetCASPos(Input, FragmentIndex, FragPos, BathPos, Alpha);
    auto PosOfIndex = std::find(FragPos.begin(), FragPos.end(), c);
    if (PosOfIndex == FragPos.end()) // Means the index is in the bath orbital.
    {
        PosOfIndex = std::find(BathPos.begin(), BathPos.end(),c);
        auto IndexOnList = std::distance(BathPos.begin(), PosOfIndex);
        Orbital = Input.EnvironmentOrbitals[FragmentIndex][IndexOnList + NumVirt]; // Add NumVirt because the virtual orbitals are before the bath active orbitals.
    }
    else 
    {
        auto IndexOnList = std::distance(FragPos.begin(), PosOfIndex);
        Orbital = Input.FragmentOrbitals[FragmentIndex][IndexOnList];
    }
    return Orbital;
}

void MakeFragmentInput(InputObj &Input, InputObj &FragInput, std::map<std::string, double> &aaInt, std::map<std::string, double> &abInt, std::map<std::string, double> &baInt, std::map<std::string, double> &bbInt, int NumFragElectrons, int NumFragOrbitals)
{
    FragInput = Input;
    FragInput.NumElectrons = NumFragElectrons;
    FragInput.NumOcc = NumFragElectrons / 2;
    FragInput.aNumElectrons = FragInput.NumOcc;
    FragInput.bNumElectrons = FragInput.NumOcc;
    FragInput.NumAO = NumFragOrbitals;
    FragInput.aaIntegrals = aaInt;
    FragInput.abIntegrals = abInt;
    FragInput.baIntegrals = baInt;
    FragInput.bbIntegrals = bbInt;

    Eigen::MatrixXd ZeroMat = Eigen::MatrixXd::Zero(NumFragOrbitals, NumFragOrbitals);
    Eigen::MatrixXd aHCore = Eigen::MatrixXd::Zero(NumFragOrbitals, NumFragOrbitals);
    Eigen::MatrixXd bHCore = Eigen::MatrixXd::Zero(NumFragOrbitals, NumFragOrbitals);
    std::vector< std::tuple< Eigen::MatrixXd, double, double > > EmptyBias;
    BuildFockMatrix(aHCore, ZeroMat, aaInt, EmptyBias, NumFragElectrons);
    BuildFockMatrix(bHCore, ZeroMat, bbInt, EmptyBias, NumFragElectrons);

    FragInput.OverlapMatrix = Eigen::MatrixXd::Identity(NumFragOrbitals, NumFragOrbitals);

    FragInput.aHCore = aHCore;
    FragInput.bHCore = bHCore;
    // Wow such coding, much algorithms
}

//double CalcCostDMETPot(std::vector<Eigen::MatrixXd> FragmentDensities, Eigen::MatrixXd FullDensity, InputObj Input)
//{
//    /* This matches the diagonal of the density matrices */
//    double CF = 0;
//    for(int x = 0; x < Input.NumFragments; x++)
//    {
//        std::vector<int> FragPos, BathPos;
//        GetCASPos(Input, x, FragPos, BathPos);
//        for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
//        {
//            CF += (FragmentDensities[x].coeffRef(FragPos[i], FragPos[i]) - FullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][i])) * 
//                  (FragmentDensities[x].coeffRef(FragPos[i], FragPos[i]) - FullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][i]));
//        }
//    }
//    return CF;
//}

double CalcCostChemPot(std::vector<Eigen::MatrixXd> FragmentDensities, InputObj &Input)
{
    double CF = 0;
    for(int x = 0; x < FragmentDensities.size(); x++) // sum over fragments
    {
        std::vector< int > FragPos;
        std::vector< int > BathPos;
        GetCASPos(Input, x , FragPos, BathPos);
        for(int i = 0; i < FragPos.size(); i++) // sum over diagonal matrix elements belonging to the fragment orbitals.
        {
            CF += FragmentDensities[x](FragPos[i], FragPos[i]);
        }
    }
    CF -= Input.NumElectrons;
    CF = CF * CF;
    return CF;
}

void SchmidtDecomposition(Eigen::MatrixXd &DensityMatrix, Eigen::MatrixXd &RotationMatrix, std::vector< int > FragmentOrbitals, std::vector< int > EnvironmentOrbitals, int NumEnvVirt, std::ofstream &Output)
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

    Eigen::MatrixXd DensityImp(FragmentOrbitals.size(), FragmentOrbitals.size());
    for(int i = 0; i < FragmentOrbitals.size(); i++)
    {
        for(int j = 0; j < FragmentOrbitals.size(); j++)
        {
            DensityImp(i, j) = DensityMatrix.coeffRef(FragmentOrbitals[i], FragmentOrbitals[j]);
        }
    }
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > ESDensityEnv(DensityEnv);
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > ESDensityImp(DensityImp);

    std::cout << "DMET: Bath eigenvalues:\n" << ESDensityEnv.eigenvalues() << std::endl;
    Output << "DMET: Bath eigenvalues:\n" << ESDensityEnv.eigenvalues() << std::endl;

    int NumAOImp = FragmentOrbitals.size();
    int NumAOEnv = EnvironmentOrbitals.size();
    RotationMatrix = Eigen::MatrixXd::Zero(NumAOImp + NumAOEnv, NumAOImp + NumAOEnv);

    // First, put the identity matrix in the blocks ocrresponding to the impurity.
    for(int i = 0; i < NumAOImp; i++)
    {
        RotationMatrix(FragmentOrbitals[i], FragmentOrbitals[i]) = 1;
    }
    // for(int i = 0; i < NumAOImp; i++)
    // {
    //     for(int j = 0; j < NumAOImp; j++)
    //     {
    //         RotationMatrix(FragmentOrbitals[i], FragmentOrbitals[j]) = ESDensityImp.eigenvectors().col(j)[i];
    //     }
    // }

    // Then put the eigenvector matrix for the environment in the environment blocks.
    for(int a = 0; a < NumAOEnv; a++)
    {
        for(int b = 0; b < NumAOEnv; b++)
        {
            RotationMatrix(EnvironmentOrbitals[a], EnvironmentOrbitals[b]) = ESDensityEnv.eigenvectors().col(b)[a];
        }
    }
    // RotationMatrix.topLeftCorner(NumAOImp, NumAOImp) = Eigen::MatrixXd::Identity(NumAOImp, NumAOImp);
    // RotationMatrix.bottomRightCorner(NumAOEnv, NumAOEnv) = ESDensityEnv.eigenvectors();
    // Note that the orbitals have been reordered so that the fragment orbitals are first
    // #ifdef H2H2H2
    // RotationMatrix(EnvironmentOrbitals[0], EnvironmentOrbitals[0]) = 1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[1], EnvironmentOrbitals[0]) = -1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[2], EnvironmentOrbitals[0]) = 0;
    // RotationMatrix(EnvironmentOrbitals[3], EnvironmentOrbitals[0]) = 0;
    // RotationMatrix(EnvironmentOrbitals[0], EnvironmentOrbitals[1]) = 0;
    // RotationMatrix(EnvironmentOrbitals[1], EnvironmentOrbitals[1]) = 0;
    // RotationMatrix(EnvironmentOrbitals[2], EnvironmentOrbitals[1]) = 1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[3], EnvironmentOrbitals[1]) = -1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[0], EnvironmentOrbitals[2]) = 1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[1], EnvironmentOrbitals[2]) = 1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[2], EnvironmentOrbitals[2]) = 0;
    // RotationMatrix(EnvironmentOrbitals[3], EnvironmentOrbitals[2]) = 0;
    // RotationMatrix(EnvironmentOrbitals[0], EnvironmentOrbitals[3]) = 0;
    // RotationMatrix(EnvironmentOrbitals[1], EnvironmentOrbitals[3]) = 0;
    // RotationMatrix(EnvironmentOrbitals[2], EnvironmentOrbitals[3]) = 1 / sqrt(2);
    // RotationMatrix(EnvironmentOrbitals[3], EnvironmentOrbitals[3]) = 1 / sqrt(2);
    // #endif
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
void ProjectCAS(Eigen::MatrixXd &NewMatrix, Eigen::MatrixXd &OldMatrix, std::vector< int > FragmentOrbitals, std::vector< int > EnvironmentOrbitals, int NumAO, int NumOcc)
{
    int NumEnvVirt = NumAO - FragmentOrbitals.size() - NumOcc;
    /* First determine where the impurity block lies */
    int OrbitalsBefore;
    if(FragmentOrbitals[0] < NumEnvVirt)
    {
        OrbitalsBefore = 0;
    }
    else
    {
        OrbitalsBefore = FragmentOrbitals[0] - NumEnvVirt;
        if(OrbitalsBefore > FragmentOrbitals.size())
        {
            OrbitalsBefore = FragmentOrbitals.size();
        }
    }
    for(int i = 0; i < NewMatrix.rows(); i++)
    {
        int OldRow;
        if(i < OrbitalsBefore)
        {
            OldRow = EnvironmentOrbitals[NumEnvVirt + i];
        }
        else
        {
            if(i < FragmentOrbitals.size() + OrbitalsBefore)
            {
                OldRow = FragmentOrbitals[i - OrbitalsBefore];
            }
            else
            {
                OldRow = EnvironmentOrbitals[NumEnvVirt + i - FragmentOrbitals.size()];
            }
        }
        for(int j = 0; j < NewMatrix.cols(); j++)
        {
            int OldCol;
            if(j < OrbitalsBefore)
            {
                OldCol = EnvironmentOrbitals[NumEnvVirt + j];
            }
            else
            {
                if(j < FragmentOrbitals.size() + OrbitalsBefore)
                {
                    OldCol = FragmentOrbitals[j - OrbitalsBefore];
                }
                else
                {
                    OldCol = EnvironmentOrbitals[NumEnvVirt + j - FragmentOrbitals.size()];
                }
            }
            NewMatrix(i, j) = OldMatrix.coeffRef(OldRow, OldCol);       
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
            hcd += RotationMatrix(p, c) * Integrals[std::to_string(p + 1) + " " + std::to_string(q + 1) + " 0 0"] * RotationMatrix(q, d);
        }
    }
    return hcd;
}

std::map<std::string, double> RotateIntegrals(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix)
{
    std::map<std::string, double> EmbeddedERI;
    for (int p = 0; p < RotationMatrix.rows(); p++)
    {
        for (int q = 0; q < RotationMatrix.rows(); q++)
        {
            EmbeddedERI[std::to_string(p + 1) + " " + std::to_string(q + 1) + " 0 0"] = OneElectronEmbedding(Integrals, RotationMatrix, p, q);
            for (int r = 0; r < RotationMatrix.rows(); r++)
            {
                for (int s = 0; s < RotationMatrix.rows(); s++)
                {
                    EmbeddedERI[std::to_string(p + 1) + " " + std::to_string(q + 1) + " " + std::to_string(r + 1) + " " + std::to_string(s + 1)] = TwoElectronEmbedding(Integrals, RotationMatrix, p, q, r, s);
                }
            }
        }
    }
    return EmbeddedERI;
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

double TwoElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &aRotationMatrix, Eigen::MatrixXd &bRotationMatrix, int c, int d, int e, int f)
{
    double Vcdef = 0;
    for(int p = 0; p < aRotationMatrix.rows(); p++)
    {
        for(int q = 0; q < bRotationMatrix.rows(); q++)
        {
            for(int r = 0; r < aRotationMatrix.rows(); r++)
            {
                for(int s = 0; s < bRotationMatrix.rows(); s++)
                {
                    Vcdef += aRotationMatrix(p, c) * bRotationMatrix(q, d) * Integrals[std::to_string(p + 1) + " " + std::to_string(r + 1) + " " + std::to_string(q + 1) + " " + std::to_string(s + 1)] * aRotationMatrix(r, e) * bRotationMatrix(s, f);
                }
            }
        }
    }
    // The way I have it defined here, it returns (ij|kl) where ij are beta orbitals and kl are alpha orbitals. This works with HongZhou's implementation of U-FCI.
    return Vcdef;
}

/* This is tilde h_cd, which is the one electron component of the Hamiltonian in the embedding basis, plus XC with the core
   electrons. */
double OneElectronPlusCore (InputObj &Input, Eigen::MatrixXd &RotationMatrix, int FragmentIndex, int c, int d)
{
	double tildehcd = 0;
	tildehcd = OneElectronEmbedding(Input.Integrals, RotationMatrix, c, d);
	for (int u = 0; u < Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size(); u++) // XC with core
	{
		int uu = Input.EnvironmentOrbitals[FragmentIndex][Input.EnvironmentOrbitals[FragmentIndex].size() - 1 - u];
		double Vcudu = TwoElectronEmbedding(Input.Integrals, RotationMatrix, c, uu, d, uu);
		double Vcuud = TwoElectronEmbedding(Input.Integrals, RotationMatrix, c, uu, uu, d);
		tildehcd += (2 * Vcudu - Vcuud);
	}
	return tildehcd;
}

double OneElectronPlusCoreRotated (InputObj &Input, Eigen::MatrixXd &RotationMatrix, int FragmentIndex, int c, int d)
{
	double tildehcd = 0;
	tildehcd = Input.Integrals[std::to_string(c + 1) + " " + std::to_string(d + 1) + " 0 0"];
	for (int u = 0; u < Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size(); u++) // XC with core
	{
		int uu = Input.EnvironmentOrbitals[FragmentIndex][Input.EnvironmentOrbitals[FragmentIndex].size() - 1 - u];
		double Vcudu = Input.Integrals[std::to_string(c + 1) + " " + std::to_string(d + 1) + " " + std::to_string(uu + 1) + " " + std::to_string(uu + 1)];
		double Vcuud = Input.Integrals[std::to_string(c + 1) + " " + std::to_string(uu + 1) + " " + std::to_string(uu + 1) + " " + std::to_string(d + 1)];
		tildehcd += (2 * Vcudu - Vcuud);
	}
	return tildehcd;
}

void makeDMETFCIDUMP(InputObj &Input, Eigen::MatrixXd &RotationMatrix)
{
    std::ofstream FCIDUMP("FCIDUMP");
    for(int i = 0; i < Input.NumAO; i++)
    {
        for(int j = 0; j < Input.NumAO; j++)
        {
            for(int k = 0; k < Input.NumAO; k++)
            {
                for(int l = 0; l < Input.NumAO; l++)
                {
                    FCIDUMP << TwoElectronEmbedding(Input.Integrals, RotationMatrix, i, j, k, l) << "\t" << i + 1 << "\t" << j + 1 << "\t" << k + 1 << "\t" << l + 1 << std::endl;
                }
            }
        }
    }
    for(int i = 0; i < Input.NumAO; i++)
    {
        for(int j = 0; j < Input.NumAO; j++)
        {
            FCIDUMP << OneElectronEmbedding(Input.Integrals, RotationMatrix, i, j) << "\t" << i + 1 << "\t" << j + 1 << "\t0\t0" << std::endl;
        }
    }
    FCIDUMP << Input.Integrals["0 0 0 0"] << "\t0\t0\t0\t0" << std::endl;
}

Eigen::MatrixXd MakeOrthogonalMatrix(Eigen::MatrixXd S)
{
    int NumAO = S.rows();
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemS(S);
    Eigen::SparseMatrix< double > LambdaSOrtho(NumAO, NumAO); // Holds the inverse sqrt matrix of eigenvalues of S ( Lambda^-1/2 )
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    for(int i = 0; i < NumAO; i++)
    {
        tripletList.push_back(T(i, i, 1 / sqrt(EigensystemS.eigenvalues()[i])));
    }
    LambdaSOrtho.setFromTriplets(tripletList.begin(), tripletList.end());
    Eigen::MatrixXd SOrtho = EigensystemS.eigenvectors() * LambdaSOrtho * EigensystemS.eigenvectors().transpose(); // S^-1/2
	return SOrtho;
}

Eigen::MatrixXd ReadMatrixFromFile(std::string Filename, int Dim)
{
    Eigen::MatrixXd Mat(Dim, Dim);
    Mat = Eigen::MatrixXd::Zero(Dim, Dim);
    std::ifstream File(Filename);

    double tmpDouble;
    for (int Row = 0; Row < Dim; Row++)
    {
        for (int Col = 0; Col < Dim; Col++)
        {
            File >> tmpDouble;
            Mat(Row, Col) = tmpDouble;
        }
    }
    
    return Mat;
}

Eigen::MatrixXd ProjectMatrix(Eigen::MatrixXd Mat, std::vector<int> ProjectionList)
{
    Eigen::MatrixXd MatP(ProjectionList.size(), ProjectionList.size());
    for (int i = 0; i < ProjectionList.size(); i++)
    {
        for (int j = 0; j < ProjectionList.size(); j++)
        {
            MatP(i, j) = Mat(ProjectionList[i], ProjectionList[j]);
        }
    }
    return MatP;
}

int BestRDM(Eigen::MatrixXd RefMat, std::vector<int> FragOrb, std::vector<Eigen::MatrixXd> TestMats, std::vector<int> FragPos)
{
    int Choice = 0;
    Eigen::MatrixXd RefMatP = ProjectMatrix(RefMat, FragOrb);
    double BestDiff = 10000.0;
    for (int i = 0; i < TestMats.size(); i++)
    {
        Eigen::MatrixXd TestMatP = ProjectMatrix(TestMats[i], FragPos);
        double CurrentDiff = (RefMatP.cwiseAbs() - TestMatP.cwiseAbs()).squaredNorm();
        if (CurrentDiff < BestDiff)
        {
            BestDiff = CurrentDiff;
            Choice = i;
        }
    }
    return Choice;
}

int BestRDM(Eigen::MatrixXd RefPop, Eigen::MatrixXd RefCoh, std::vector<int> FragOrb, std::vector<Eigen::MatrixXd> TestMats, std::vector<int> FragPos)
{
    int Choice = 0;
    Eigen::MatrixXd RefMatP = ProjectMatrix(RefPop, FragOrb);
    Eigen::MatrixXd RefMatC = ProjectMatrix(RefCoh, FragOrb);
    Eigen::MatrixXd RefMat = RefMatC;
    for (int i = 0; i < RefMat.rows(); i++)
    {
        RefMat(i, i) = RefMatP(i, i);
    }
    double BestDiff = 10000.0;
    for (int i = 0; i < TestMats.size(); i++)
    {
        Eigen::MatrixXd TestMatP = ProjectMatrix(TestMats[i], FragPos);
        double CurrentDiff = (RefMat - TestMatP).squaredNorm();
        if (CurrentDiff < BestDiff)
        {
            BestDiff = CurrentDiff;
            Choice = i;
        }
    }
    return Choice;
}

int SameRDM(Eigen::MatrixXd OldP, std::vector<Eigen::MatrixXd> NewPs)
{
    int Choice = 0;
    double BestDiff = 10000.0;
    for (int i = 0; i < NewPs.size(); i++)
    {
        double CurrentDiff = (OldP.cwiseAbs() - NewPs[i].cwiseAbs()).squaredNorm();
        if (CurrentDiff < BestDiff)
        {
            BestDiff = CurrentDiff;
            Choice = i;
        }
    }
    return Choice;
}

// int main(int argc, char* argv[])
// {
//     InputObj Input;
//     if(argc == 4)
//     {
//         Input.SetNames(argv[1], argv[2], argv[3]);
//     }
//     else
//     {
//         Input.GetInputName();
//     for(int i = 0; i < FragPos.size(); i++)
//     {
//         std::cout << FragPos[i] << "\t";
//     }
//     std::cout << std::endl << "BATH" << std::endl;
//     for(int i = 0; i < BathPos.size(); i++)
//     {
//         std::cout << BathPos[i] << "\t";
//     }
//     std::cout << "\n";
//     std::cout << ReducedIndexToOrbital(0, Input, 2) << std::endl;
//     return 0;
// }    std::vector< int > BathPos;
//     GetCASPos(Input, 2, FragPos, BathPos);
//     std::cout << "FRAG" << std::endl;
//     for(int i = 0; i < FragPos.size(); i++)
//     {
//         std::cout << FragPos[i] << "\t";
//     }
//     std::cout << std::endl << "BATH" << std::endl;
//     for(int i = 0; i < BathPos.size(); i++)
//     {
//         std::cout << BathPos[i] << "\t";
//     }
//     std::cout << "\n";
//     std::cout << ReducedIndexToOrbital(0, Input, 2) << std::endl;
//     return 0;
// }

int main(int argc, char* argv[])
{
    /* Read from the input */
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

    std::vector< int > ImpurityStates(NumFragments); // Which FCI state is desired from the fragment FCI.
    std::vector< int > BathStates(NumFragments);

    ImpurityStates = Input.ImpurityStates;
    BathStates = Input.BathStates;

    int NumSCFStates = *max_element(BathStates.begin(), BathStates.end());
    NumSCFStates++;

    int NumFCIStates = *max_element(ImpurityStates.begin(), ImpurityStates.end());
    NumFCIStates++;
    Input.NumberOfEV = NumFCIStates;

    bool Unrestricted = true;
    bool DeltaSCF = false;
    bool HalfUnrestricted = false;
    bool useRefP = false;
    std::ifstream FragPFile("FragP_1");
    bool doScan = FragPFile.good();
    
    // Begin by defining some variables.
    std::vector< std::tuple< Eigen::MatrixXd, double, double > > EmptyBias; // This code is capable of metadynamics, but this isn't utilized. We will use an empty bias to do standard SCF.
    Eigen::MatrixXd HCore(NumAO, NumAO); // T + V_eN
    Eigen::MatrixXd DensityMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // Will hold the density matrix of the full system.
    std::ifstream PInit("P.txt");
    if (PInit.good())
    {
        DensityMatrix = ReadMatrixFromFile("P.txt", Input.NumAO);
    }
    Eigen::MatrixXd RefDensity = Eigen::MatrixXd::Zero(NumAO, NumAO);
    BuildFockMatrix(HCore, DensityMatrix, Input.Integrals, EmptyBias, Input.NumElectrons); // Build HCore, which is H when the density matrix is zero.
	Input.HCore = HCore;
    // for(int i = 0; i < Input.NumOcc; i++) // This initializes the density matrix to be exact in the MO basis.
    // {
    //     DensityMatrix(i, i) = 1;
    // }

    Eigen::MatrixXd tmpMat = Eigen::MatrixXd::Random(NumAO, NumAO); // Will hold the density matrix of the full system.
    Eigen::MatrixXd aDensityMatrix = tmpMat + tmpMat.transpose();
    tmpMat = Eigen::MatrixXd::Random(NumAO, NumAO); // Will hold the density matrix of the full system.
    Eigen::MatrixXd bDensityMatrix = tmpMat + tmpMat.transpose();
    if (DeltaSCF)
    {
        aDensityMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
        bDensityMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
    }
    std::ifstream aPInit("aP.txt");
    std::ifstream bPInit("bP.txt");
    if (aPInit.good() && bPInit.good())
    {
        aDensityMatrix = ReadMatrixFromFile("aP.txt", Input.NumAO);
        bDensityMatrix = ReadMatrixFromFile("bP.txt", Input.NumAO);
    }

    std::vector< Eigen::MatrixXd > FullDensities(NumSCFStates);
    std::vector< Eigen::MatrixXd > aFullDensities(NumSCFStates);
    std::vector< Eigen::MatrixXd > bFullDensities(NumSCFStates);

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
	Input.SOrtho = SOrtho;

    Eigen::MatrixXd DMETPotential = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO); // Correlation potential. Will be optimize to match density matrices.
    Eigen::MatrixXd DMETPotentialPrev = DMETPotential; // Will check self-consistency of this potential.

    // If orthogonalized orbitals are read in from the input, this will unorthogonalize them when necessary.
    std::string LocalToAOName = "ao2loc.loc";
    std::ifstream LocalToAOFile(LocalToAOName);
    Eigen::MatrixXd LocalToAO = Eigen::MatrixXd::Identity(Input.NumAO, Input.NumAO);
    if (LocalToAOFile.good()) // Otherwise, keep the transformation an identity so that nothing happens.
    {
        LocalToAO = ReadMatrixFromFile(LocalToAOName, Input.NumAO);
    }
    
    // DEBUGGING - This is the H10 u from Wouter's code
    // for (int i = 0; i < 5; i++)
    // {
    //     DMETPotential(2 * i, 2 * i) = 1.541E-13;
    //     DMETPotential(2 * i + 1, 2 * i + 1) = -1.541E-13;
    //     DMETPotential(2 * i, 2 * i + 1) = -2.334E-03;
    //     DMETPotential(2 * i + 1, 2 * i) = -2.334E-03;
    // }

	// DEBUG - Bootstrap
	//Bootstrap BE;
	//BE.debugInit(Input); 

	//// Now do SCF to get density matrix.
	//std::vector< int > OccupiedOrbitals;
	//std::vector< int > VirtualOrbitals;
	//for (int i = 0; i < NumOcc; i++)
	//{
	//	OccupiedOrbitals.push_back(i);
	//}
	//for (int i = NumOcc; i < NumAO; i++)
	//{
	//	VirtualOrbitals.push_back(i);
	//}
	//// OccupiedOrbitals[NumOcc - 1] = NumOcc;
	//// VirtualOrbitals[0] = NumOcc - 1;
	//Input.OccupiedOrbitals = OccupiedOrbitals;
	//Input.VirtualOrbitals = VirtualOrbitals;
	//double SCFEnergy = 0.0;
	//std::vector< std::tuple< Eigen::MatrixXd, double, double > > Bias;
	//std::vector< Eigen::MatrixXd > SCFMD1RDM;
	//std::vector< double > AllEnergies;
	//std::priority_queue< std::pair< double, int > > SCFMDEnergyQueue;
	//std::vector< std::vector< int > > SCFMDOccupied;
	//std::vector< std::vector< int > > SCFMDVirtual;
	//std::vector< Eigen::MatrixXd > SCFMDCoeff;
	//std::vector< Eigen::VectorXd > SCFMDOrbitalEV;

	//std::ofstream BlankOutput;

	//Eigen::VectorXd OrbitalEV;
	//int SCFCount = 0;
	//Eigen::MatrixXd CoeffMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
	//SCFEnergy = SCF(Bias, 0, DensityMatrix, Input, BlankOutput, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);

	//// Get all Schmidt Decompositions.
	//for (int i = 0; i < DensityMatrix.rows(); i++)
	//{
	//	for (int j = 0; j < DensityMatrix.cols(); j++)
	//	{
	//		if (fabs(DensityMatrix.coeffRef(i, j)) < 1E-12)
	//		{
	//			DensityMatrix(i, j) = 0;
	//		}
	//	}
	//}
	//Output << DensityMatrix << std::endl;
	//BE.doBootstrap(Input, DensityMatrix, Output);
	//BE.printDebug(Output);
	//return 0;

	//END DEBUG - Bootstrap

    // These hold the density matrix and energies of each impurity.
    std::vector< Eigen::MatrixXd > FragmentDensities(Input.NumFragments);
    std::vector< Eigen::Tensor<double, 4> > Fragment2RDM(Input.NumFragments);
    std::vector< Eigen::MatrixXd > FragmentRotations(Input.NumFragments); // Stores rotation matrix in case we wish to rotation the 1RDM, such as for certain types of density matching functions.
    std::vector< Eigen::MatrixXd > aFragmentRotations(Input.NumFragments); 
    std::vector< Eigen::MatrixXd > bFragmentRotations(Input.NumFragments); 
	std::vector< Eigen::VectorXd > FragmentEigenstates(Input.NumFragments);

	double ChemicalPotential = 0; // The value of the chemical potential. This is a diagonal element on the Hamiltonian, on the diagonal positions corresponding to impurity orbitals.

    double DMETPotentialChange = 1;
    int uOptIt = 0; // Number of iterations to optimize u
    bool OneShot = true;
    while(fabs(DMETPotentialChange) > 1E-3) // || uOptIt < 10) // Do DMET until correlation potential has converged.
    {
        uOptIt++;

        // STEP 1: Solve the full system at the RHF level of theory.
        Eigen::VectorXd OrbitalEV; // Holds the orbital EVs from the proceeding SCF calculation. Needed to take derivatives.
        Eigen::VectorXd aOrbitalEV;
        Eigen::VectorXd bOrbitalEV;
        
        std::vector< std::vector< double > > FragmentEnergies(Input.NumFragments); // Vector of double incase multiple solutions are desired from one impurity.

        /* These are the list of occupied and virtual orbitals of the full system calculation. This is necessary in case I want
           to change the choice of occupied orbitals, such as with MoM. For now, they are intialized to be the lowest energy MOs */
        std::vector< int > OccupiedOrbitals;
        std::vector< int > VirtualOrbitals;
        for(int i = 0; i < NumOcc; i++)
        {
            OccupiedOrbitals.push_back(i);
        }
        for(int i = NumOcc; i < NumAO; i++)
        {
            VirtualOrbitals.push_back(i);
        }
        Input.OccupiedOrbitals = OccupiedOrbitals;
        Input.VirtualOrbitals = VirtualOrbitals;

        std::vector< int > aOccupiedOrbitals;
        std::vector< int > aVirtualOrbitals;
        std::vector< int > bOccupiedOrbitals;
        std::vector< int > bVirtualOrbitals;
        for (int i = 0; i < Input.aNumElectrons; i++)
        {
            aOccupiedOrbitals.push_back(i);
        }
        for (int i = 0; i < Input.bNumElectrons; i++)
        {
            bOccupiedOrbitals.push_back(i);
        }
        for (int i = Input.aNumElectrons; i < NumAO; i++)
        {
            aVirtualOrbitals.push_back(i);
        }
        for (int i = Input.bNumElectrons; i < NumAO; i++)
        {
            bVirtualOrbitals.push_back(i);
        }
        if (DeltaSCF)
        {
            // OccupiedOrbitals[Input.bNumElectrons - 1] = Input.bNumElectrons;
            // VirtualOrbitals[0] = Input.bNumElectrons - 1;
            int bExciteLevel = 1;
            bOccupiedOrbitals[Input.bNumElectrons - 1] = Input.bNumElectrons + bExciteLevel - 1;
            std::sort(bOccupiedOrbitals.begin(), bOccupiedOrbitals.end(), greater<int>());
            for (int i = 0; i < bExciteLevel; i++)
            {
                bVirtualOrbitals[i] = Input.bNumElectrons - 1 + i;
            }
            int aExciteLevel = 0;
            aOccupiedOrbitals[Input.aNumElectrons - 1] = Input.aNumElectrons + aExciteLevel - 1;
            std::sort(aOccupiedOrbitals.begin(), aOccupiedOrbitals.end(), greater<int>());
            for (int i = 0; i < bExciteLevel; i++)
            {
                aVirtualOrbitals[i] = Input.aNumElectrons - 1 + i;
            }
        }

        Eigen::MatrixXd CoeffMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // Holds the coefficient matrix of the full system calculation
        Eigen::MatrixXd aCoeffMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
        Eigen::MatrixXd bCoeffMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
        int SCFCount = 0; // Counts how many SCF iterations have been done in total.

        // These are values we need to calculate the derivative of the 1RDM later in the optimization. They are separated by state.
        std::vector< std::vector< int > > OccupiedByState;
        std::vector< std::vector< int > > VirtualByState;
        std::vector< Eigen::MatrixXd > CoeffByState;
        std::vector< Eigen::VectorXd > OrbitalEVByState;

        /* Begin FCI calculation on full system */
        // FCI FullFCI(Input);
        // FullFCI.ERIMapToArray(Input.Integrals);
        // FullFCI.runFCI();
        // std::cout << "FCI = " << FullFCI.Energies[0] + FullFCI.ENuc << std::endl;
        // FullFCI.getSpecificRDM(0, true);
        // FullFCI.CalcE1E2(0);
        // std::vector<Eigen::MatrixXd> SchmidtBasis = FullFCI.HalfFilledSchmidtBasis(0);
        // Eigen::MatrixXd PH = FullFCI.ProjectMatrix(SchmidtBasis);
        // std::cout << PH << std::endl;

        /* This performs an SCF calculation, with the correlation energy added to the Hamiltonian. 
           We retreive the density matrix, coefficient matrix, and orbital EVs */
        // #ifdef H2H2H2
        //     DensityMatrix << 
        //         // 0.56, 0.11, -.07, -.27, 0.06, -.39,
        //         // 0.11, 0.56, -.27, -.07, -.39, 0.06,
        //         // -.07, -.27, 0.37, 0.28, -.07, -.27,
        //         // -.27, -.07, 0.28, 0.37, -.27, -.07,
        //         // 0.06, -.39, -.07, -.27, 0.56, 0.11,
        //         // -.39, 0.06, -.27, -.07, 0.11, 0.56;

        //         // 0.5, 0.5, 0.0, 0.0, 0.0, 0.0,
        //         // 0.5, 0.5, 0.0, 0.0, 0.0, 0.0,
        //         // 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        //         // 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
        //         // 0.0, 0.0, 0.0, 0.0, 0.5, 0.5,
        //         // 0.0, 0.0, 0.0, 0.0, 0.5, 0.5;

        //         // 0.4365959597,  0.3915640739,  0.0667023412,  0.2691279919, -0.0634040467, -0.1084359360,
        //         // 0.3915640739,  0.4365959597,  0.2691279919,  0.0667023412, -0.1084359360, -0.0634040467,
        //         // 0.0667023412,  0.2691279919,  0.6268080935, -0.2831281279,  0.0667023495,  0.2691280157,
        //         // 0.2691279919,  0.0667023412, -0.2831281279,  0.6268080935,  0.2691280157,  0.0667023495,
        //         // -0.0634040467, -0.1084359360,  0.0667023495,  0.2691280157,  0.4365959468,  0.3915640540,
        //         // -0.1084359360, -0.0634040467,  0.2691280157,  0.0667023495,  0.3915640540,  0.4365959468;

        //         // 0.5, 0.5, 0.0, 0.5, 0.0, 0.0,
        //         // 0.5, 0.5, 0.5, 0.0, 0.0, 0.0,
        //         // 0.0, 0.5, 0.5, 0.0, 0.5, 0.0,
        //         // 0.5, 0.0, 0.0, 0.5, 0.0, 0.5,
        //         // 0.0, 0.0, 0.5, 0.0, 0.5, 0.5,
        //         // 0.0, 0.0, 0.0, 0.5, 0.5, 0.5;

        //          0.5417506668,  0.0410202983,  0.0008034825, -0.0005123971,  0.0034482223, -0.4965495127,
        //          0.0410202983,  0.5417506668, -0.0005123971,  0.0008034825, -0.4965495127,  0.0034482223,
        //          0.0008034825, -0.0005123971,  0.5000016275,  0.4999981694,  0.0008289865, -0.0004849678,
        //         -0.0005123971,  0.0008034825,  0.4999981694,  0.5000016275, -0.0004849678,  0.0008289865,
        //          0.0034482223, -0.4965495127,  0.0008289865, -0.0004849678,  0.4582477058, -0.0410184678,
        //         -0.4965495127,  0.0034482223, -0.0004849678,  0.0008289865, -0.0410184678,  0.4582477058;

        //     std::cout << "Starting 1RDM:\n" << DensityMatrix << std::endl;
        //     OccupiedOrbitals[0] = 2;
        //     OccupiedOrbitals[1] = 3;
        //     OccupiedOrbitals[2] = 0;
        //     VirtualOrbitals[0] = 1;
        //     VirtualOrbitals[1] = 4;
        //     VirtualOrbitals[2] = 5;
        // #endif
        std::cout << "DMET: Running SCF calculation for DMET iteration " << uOptIt << std::endl;
        std::cout << "DMET: Currently looking for " << Input.NumSoln << " metadynamics solutions. The lowest " << NumSCFStates << " solutions will be saved." << std::endl;
        Output << "DMET: Beginning DMET Iteration Number " << uOptIt << std::endl;
        Output << "DMET: Currently looking for " << Input.NumSoln << " metadynamics solutions. The lowest " << NumSCFStates << " solutions will be saved." << std::endl;

        double SCFEnergy = 0.0;
        std::vector< std::tuple< Eigen::MatrixXd, double, double > > Bias;
        std::vector< Eigen::MatrixXd > SCFMD1RDM;
        std::vector< Eigen::MatrixXd > SCFMDa1RDM;
        std::vector< Eigen::MatrixXd > SCFMDb1RDM;
        std::vector< double > AllEnergies;
        std::priority_queue< std::pair< double, int > > SCFMDEnergyQueue;
        std::vector< std::vector< int > > SCFMDOccupied;
        std::vector< std::vector< int > > SCFMDVirtual;
        std::vector< std::vector< int > > SCFMDaOccupied;
        std::vector< std::vector< int > > SCFMDaVirtual;
        std::vector< std::vector< int > > SCFMDbOccupied;
        std::vector< std::vector< int > > SCFMDbVirtual;
        std::vector< Eigen::MatrixXd > SCFMDCoeff;
        std::vector< Eigen::MatrixXd > SCFMDaCoeff;
        std::vector< Eigen::MatrixXd > SCFMDbCoeff;
        std::vector< Eigen::VectorXd > SCFMDOrbitalEV;
        std::vector< Eigen::VectorXd > SCFMDaOrbitalEV;
        std::vector< Eigen::VectorXd > SCFMDbOrbitalEV;

        std::vector< std::tuple< Eigen::MatrixXd, double, double > > aBias;
        std::vector< std::tuple< Eigen::MatrixXd, double, double > > bBias;

        std::ofstream BlankOutput;

        // First, run SCF some number of times to find a few solutions.
        bool useDIIS = Input.Options[0];
        for (int i = 0; i < Input.NumSoln; i++)
        {
            if (i == 0) // For the ground state always use DIIS.
            {
                Input.Options[0] = true;
            }
            #ifdef H2H2H2
            if (i == 1)
            {
                //std::vector< std::tuple< Eigen::MatrixXd, double, double > > EmptyBias;
                //Bias = EmptyBias;
                //  int MiddleH2 = Input.NumAO / 2;
                //  DensityMatrix(MiddleH2 - 1, MiddleH2) = 0;
                //  DensityMatrix(MiddleH2, MiddleH2 - 1) = 0;
                // DensityMatrix <<     1.01285  , 0.759489 ,  0.604052 ,-0.0110383 , 0.0128477  ,-0.240511,
                //                     0.759489 ,   1.01285 ,-0.0110383 ,  0.604052 , -0.240511 , 0.0128477,
                //                     0.604052, -0.0110383  , 0.974305 , -0.518977  , 0.604052, -0.0110383,
                //                     -0.0110383  , 0.604052 , -0.518977 ,  0.974305 ,-0.0110383 ,  0.604052,
                //                     0.0128477 , -0.240511  , 0.604052, -0.0110383 ,   1.01285  , 0.759489,
                //                     -0.240511 , 0.0128477, -0.0110383 ,  0.604052 ,  0.759489  ,  1.01285; // solution at d = 1.80
                // DensityMatrix = 0.5 * DensityMatrix;
            }
            #endif
            // This redirects the std::cout buffer, so we don't  have massive amounts of terminal output.
            std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
            //std::cout.rdbuf(NULL); // sets to null
            if (Unrestricted)
            {
                SCFEnergy = SCF(aBias, bBias, i + 1, aDensityMatrix, bDensityMatrix, Input, BlankOutput, SOrtho, HCore, AllEnergies, aCoeffMatrix, bCoeffMatrix, aOccupiedOrbitals, bOccupiedOrbitals, aVirtualOrbitals, bVirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, aOrbitalEV, bOrbitalEV);
                DensityMatrix = aDensityMatrix + bDensityMatrix;
                // std::cout.rdbuf(orig_buf); // restore buffer
                std::cout << "DMET: Solution " << i + 1 << " found with energy " << SCFEnergy << "." << std::endl;
                Output << "DMET: Solution " << i + 1 << " found with energy " << SCFEnergy << "." << std::endl;
                std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(aDensityMatrix, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
                aBias.push_back(tmpTuple);
                tmpTuple = std::make_tuple(bDensityMatrix, Input.StartNorm, Input.StartLambda);
                bBias.push_back(tmpTuple);
                SCFEnergy *= -1;
                SCFMDEnergyQueue.push(std::pair<double, int>(SCFEnergy, i));
                SCFMD1RDM.push_back(DensityMatrix);
                SCFMDa1RDM.push_back(aDensityMatrix);
                SCFMDb1RDM.push_back(bDensityMatrix);
                SCFMDaOccupied.push_back(aOccupiedOrbitals);
                SCFMDaVirtual.push_back(aVirtualOrbitals);
                SCFMDbOccupied.push_back(bOccupiedOrbitals);
                SCFMDbVirtual.push_back(bVirtualOrbitals);
                SCFMDCoeff.push_back(aCoeffMatrix);
                SCFMDCoeff.push_back(bCoeffMatrix);
                SCFMDaOrbitalEV.push_back(aOrbitalEV);
                SCFMDbOrbitalEV.push_back(bOrbitalEV);
            }
            else // Does RHF
            {
                SCFEnergy = SCF(Bias, i + 1, DensityMatrix, Input, BlankOutput, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
                std::cout.rdbuf(orig_buf); // restore buffer
                std::cout << "DMET: Solution " << i + 1 << " found with energy " << SCFEnergy << "." << std::endl;
                Output << "DMET: Solution " << i + 1 << " found with energy " << SCFEnergy << "." << std::endl;
                std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(DensityMatrix, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
                Bias.push_back(tmpTuple);
                SCFEnergy *= -1;
                SCFMDEnergyQueue.push(std::pair<double, int>(SCFEnergy, i));
                SCFMD1RDM.push_back(DensityMatrix);
                SCFMDOccupied.push_back(OccupiedOrbitals);
                SCFMDVirtual.push_back(VirtualOrbitals);
                SCFMDCoeff.push_back(CoeffMatrix);
                SCFMDOrbitalEV.push_back(OrbitalEV);
            } 
            
            if (i == 0) // Reset to original DIIS vs no DIIS option.
            {
                Input.Options[0] = useDIIS;
            }
        }
        for (int i = 0; i < NumSCFStates; i++)
        {
            if (SCFMDEnergyQueue.size() == 0) // If we removed everything and theres nothing to check against
            {
                // Run metadynamics again.
                //std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
                //std::cout.rdbuf(NULL); // sets to null
                SCFEnergy = SCF(Bias, i + 1, DensityMatrix, Input, BlankOutput, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
                //std::cout.rdbuf(orig_buf); // restore buffer
                std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(DensityMatrix, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
                Bias.push_back(tmpTuple);
                SCFEnergy *= -1;
                SCFMDEnergyQueue.push(std::pair<double, int>(SCFEnergy, i));
                SCFMD1RDM.push_back(DensityMatrix);
                SCFMDOccupied.push_back(OccupiedOrbitals);
                SCFMDVirtual.push_back(VirtualOrbitals);
                SCFMDCoeff.push_back(CoeffMatrix);
                SCFMDOrbitalEV.push_back(OrbitalEV);
            }
            int NextIndex = SCFMDEnergyQueue.top().second;
            // Run SCF again, because sometimes the minimum change slightly and this is what we do to take the derivative.
            // The occupied and virtual orbitals will be locked in.
            DensityMatrix = SCFMD1RDM[NextIndex]; // Start from correct matrix.
            std::vector< double > EmptyAllEnergies;
            if (Unrestricted)
            {
                aDensityMatrix = SCFMDa1RDM[NextIndex];
                bDensityMatrix = SCFMDb1RDM[NextIndex];
                SCFEnergy = SCF(EmptyBias, EmptyBias, i + 1, aDensityMatrix, bDensityMatrix, Input, Output, SOrtho, HCore, EmptyAllEnergies, aCoeffMatrix, bCoeffMatrix, SCFMDaOccupied[NextIndex], SCFMDbOccupied[NextIndex], SCFMDaVirtual[NextIndex], SCFMDbVirtual[NextIndex], SCFCount, Input.MaxSCF, DMETPotential, aOrbitalEV, bOrbitalEV);
                DensityMatrix = aDensityMatrix + bDensityMatrix;
            }
            else
            {
                SCFEnergy = SCF(EmptyBias, i + 1, DensityMatrix, Input, Output, SOrtho, HCore, EmptyAllEnergies, CoeffMatrix, SCFMDOccupied[NextIndex], SCFMDVirtual[NextIndex], SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
            }
            if (fabs(fabs(SCFEnergy) - fabs(SCFMDEnergyQueue.top().first)) > 1E-2 || (DensityMatrix - SCFMD1RDM[NextIndex]).squaredNorm() > 1E-3) // Not the same solution, for some reason...
            {
                // Remove this solution from the list and go on to the next one.
                std::cout << "DMET: SCFMD solution was not a minimum. Trying different SCFMD solution." << std::endl;
                Output << "DMET: SCFMD solution was not a minimum. Trying different SCFMD solution." << std::endl;
                SCFMDEnergyQueue.pop();
                i--;
                continue;
            }
            #ifdef H10
            if ((fabs(DensityMatrix.coeffRef(0, 0) - DensityMatrix.coeffRef(1, 1)) > 1E-3 || fabs(DensityMatrix.coeffRef(0, 0) - DensityMatrix.coeffRef(2, 2)) > 1E-3) && uOptIt == 1)
            {
                std::cout << "DMET: SCFMD solution is not translationally symmetric. Trying different SCFMD solution." << std::endl;
                Output << "DMET: SCFMD solution is not translationally symmetric. Trying different SCFMD solution." << std::endl;
                SCFMDEnergyQueue.pop();
                i--;
                continue;
            }
            #endif
            #ifdef H2H2H2
            if ((fabs(DensityMatrix.coeffRef(0, 0) - DensityMatrix.coeffRef(DensityMatrix.rows() - 2, DensityMatrix.rows() - 2)) > 1E-3) && uOptIt == 1)
            {
                std::cout << "DMET: SCFMD solution is not translationally symmetric. Trying different SCFMD solution." << std::endl;
                Output << "DMET: SCFMD solution is not translationally symmetric. Trying different SCFMD solution." << std::endl;
                SCFMDEnergyQueue.pop();
                i--;
                continue;
            }
            #endif
            int PFactor = 2.0;
            if (Unrestricted) 
            {
                PFactor = 1.0;
                std::cout << "DMET: aP\n" << aDensityMatrix << "\nbP\n" << bDensityMatrix << std::endl;
                Output << "DMET: aP\n" << aDensityMatrix << "\nbP\n" << bDensityMatrix << std::endl;
            }
            std::cout << "DMET: SCF solution for state " << i + 1 << " has an energy of " << SCFEnergy << std::endl;
            std::cout << "DMET: and 1RDM of \n " << PFactor * DensityMatrix << std::endl;
            Output << "DMET: SCF solution for state " << i + 1 << " has an energy of " << SCFEnergy << std::endl;
            Output << "DMET: and 1RDM of \n " << PFactor * DensityMatrix << std::endl;
            if (LocalToAOFile.good())
            {
                // Rotate molecular orbitals into AO basis
                // First put MOs into a matrix
                Eigen::MatrixXd MO(Input.NumAO, SCFMDOccupied[NextIndex].size());
                for (int ao = 0; ao < MO.rows(); ao++)
                {
                    for (int mo = 0; mo < MO.cols(); mo++)
                    {
                        MO(ao, mo) = CoeffMatrix(ao, SCFMDOccupied[NextIndex][mo]);
                    }
                }
                // Then rotate
                Output << "DMET: and MOs in AO basis:\n" << LocalToAO * MO << std::endl;
            }

            // std::cout << "DMET: SCF solution for state " << i + 1 << " has an energy of " << -1 * SCFMDEnergyQueue.top().first << std::endl;
            // std::cout << "DMET: and 1RDM of \n " << 2 * SCFMD1RDM[NextIndex] << std::endl;
            // Output << "DMET: SCF solution for state " << i + 1 << " has an energy of " << -1 * SCFMDEnergyQueue.top().first << std::endl;
            // Output << "DMET: and 1RDM of \n " << 2 * SCFMD1RDM[NextIndex] << std::endl;

            FullDensities[i] = PFactor * DensityMatrix; // SCFMD1RDM[NextIndex];
            if (Unrestricted) 
            {
                aFullDensities[i] = aDensityMatrix;
                bFullDensities[i] = bDensityMatrix;
            }

            // Collect information needed for derivative calculations later.
            if (!Unrestricted)
            {   
                OccupiedByState.push_back(SCFMDOccupied[NextIndex]);
                VirtualByState.push_back(SCFMDVirtual[NextIndex]);
                CoeffByState.push_back(CoeffMatrix); // (SCFMDCoeff[NextIndex]);
                OrbitalEVByState.push_back(OrbitalEV); // (SCFMDOrbitalEV[NextIndex]);
            }

            SCFMDEnergyQueue.pop();
        }

        if (useRefP)
        {
            Input.Options[0] = true;
            SCFEnergy = SCF(aBias, bBias, 1, aDensityMatrix, bDensityMatrix, Input, BlankOutput, SOrtho, HCore, AllEnergies, aCoeffMatrix, bCoeffMatrix, aOccupiedOrbitals, bOccupiedOrbitals, aVirtualOrbitals, bVirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, aOrbitalEV, bOrbitalEV);
            RefDensity = aDensityMatrix + bDensityMatrix;
            std::ifstream RefPFile("RefP.txt");
            if (RefPFile.good())
            {
                RefDensity = ReadMatrixFromFile("RefP.txt", Input.NumAO);
            }
            std::cout << "DMET: Reference Density:\n" << RefDensity << std::endl;
            Output << "Reference Density:\n" << RefDensity << std::endl;
            Output << "Reference Energy = " << SCFEnergy << std::endl;
        }

        if (PInit.good())
        {
            std::remove("P.txt");
            std::ofstream POut("P.txt");
            POut << DensityMatrix;
        }

        if (aPInit.good() && bPInit.good())
        {
            std::remove("aP.txt");
            std::remove("bP.txt");
            std::ofstream aPOut("aP.txt");
            std::ofstream bPOut("bP.txt");
            aPOut << aDensityMatrix;
            bPOut << bDensityMatrix;
        }
        // break; // Skips to the end to initiate BE or otherwise.
        // ***** OLD LOCKED ORBITALS METHOD
        //for (int i = 0; i < NumSCFStates; i++)
        //{
        //    std::vector< double > EmptyAllEnergies;
        //    SCFEnergy = SCF(Bias, 1, DensityMatrix, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
        //    // Run SCF again, with the occupied orbitals locked in, but no bias. The occupied orbitals will not change in the unbiased SCF.
        //    if (i > 0)
        //    {
        //        for(int j = 0; j < NumOcc; j++)
        //        {
        //            OccupiedOrbitals[j] = j;
        //        }
        //        for(int j = NumOcc; j < NumAO; j++)
        //        {
        //            VirtualOrbitals[j - NumOcc] = j;
        //        }
        //        for (int j = 0; j < i; j++)
        //        {
        //            OccupiedOrbitals[NumOcc - 1 - j] = NumOcc + j;
        //            VirtualOrbitals[j] = NumOcc - 1 - j;
        //        }
        //    }
        //    SCFEnergy = SCF(EmptyBias, 1, DensityMatrix, Input, Output, SOrtho, HCore, EmptyAllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
        //    std::cout << "DMET: SCF calculation has converged with an energy of " << SCFEnergy << std::endl;
        //    std::cout << "DMET: and 1RDM of \n" << 2 * DensityMatrix << std::endl;
        //    Output << "SCF calculation has converged with an energy of " << SCFEnergy << std::endl;
        //    Output << "and 1RDM of \n" << 2 * DensityMatrix << std::endl;
        //    std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(DensityMatrix, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
        //    Bias.push_back(tmpTuple);

        //    FullDensities[i] = 2 * DensityMatrix;

        //    // Collect information needed for derivative calculations later.
        //    OccupiedByState.push_back(OccupiedOrbitals);
        //    VirtualByState.push_back(VirtualOrbitals);
        //    CoeffByState.push_back(CoeffMatrix);
        //    OrbitalEVByState.push_back(OrbitalEV);
        //}
        // ***** END OLD LOCKED ORBITALS METHOD

        // SCFEnergy = SCF(EmptyBias, 1, DensityMatrix, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
        // DensityMatrix = FullDensities[0];
        // std::cout << "DMET: SCF calculation has converged with an energy of " << SCFEnergy << std::endl;
        // std::cout << DensityMatrix << std::endl;

        Fragmenting FragObj(Input.NumFragments);
        Bootstrap BE;
        BE.InitFromFragmenting(FragObj, Output);
        if (!Unrestricted)
        {
            for (int i = 0; i < SCFMD1RDM.size(); i++)
            {
                std::cout << i << "/" << SCFMD1RDM.size() << std::endl;
                SCFMDa1RDM.push_back(SCFMD1RDM[i]);
                SCFMDb1RDM.push_back(SCFMD1RDM[i]);
            }
        }
        BE.doBootstrap(Input, SCFMDa1RDM, SCFMDb1RDM, Output);
        return 0;

        // These are definitions for the global chemical potential, which ensures that the number of electrons stays as it should.
        double CostMu = 1; // Cost function of mu, the sum of squares of difference in diagonal density matrix elements corresponding to impurity orbitals.
        double CostMuPrev = 0;
        double StepSizeMu = 1E-4; // How much to change chemical potential by each iteration. No good reason to choosing this number.
        int MuIteration = 0;
        while(fabs(CostMu) > 1E-2) // While the derivative of the cost function is nonzero, keep changing mu and redoing all fragment calculations.
        {
            std::cout << "DMET: -- Running impurity FCI calculations with a chemical potential of " << ChemicalPotential << std::endl;
            Output << "\nDMET: -- Running impurity FCI calculations with a chemical potential of " << ChemicalPotential << ".\n";
            for(int x = 0; x < NumFragments; x++) // Loop over all fragments.
            {
                // Use the correct density matrix for this fragment.
                DensityMatrix = FullDensities[BathStates[x]];
                if (Unrestricted)
                {
                    aDensityMatrix = aFullDensities[BathStates[x]];
                    bDensityMatrix = bFullDensities[BathStates[x]];
                }

                // The densityDensity matrix of the full system defines the orbitals for each impurity. We begin with some definitions. These numbers depend on which impurity we are looking at.
                int NumAOImp = Input.FragmentOrbitals[x].size(); // Number of impurity states.
                int NumAOEnv = NumAO - NumAOImp; // The rest of the states.

                Eigen::MatrixXd RotationMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // The matrix of our orbital rotation.
                Eigen::MatrixXd aRotationMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
                Eigen::MatrixXd bRotationMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
                int NumEnvVirt = NumAO - NumAOImp - NumOcc; // Number of virtual (environment) orbitals in the bath.
                int aNumEnvVirt = NumAO - NumAOImp - Input.aNumElectrons;
                int bNumEnvVirt = NumAO - NumAOImp - Input.bNumElectrons;

                Eigen::MatrixXd ActiveRotation(NumAO, 2 * NumAOImp);
                Eigen::MatrixXd aActiveRotation(NumAO, 2 * NumAOImp);
                Eigen::MatrixXd bActiveRotation(NumAO, 2 * NumAOImp);
                std::vector<int> aFragPos, bFragPos;
                std::vector<int> aBathPos, bBathPos;
                GetCASPos(Input, x, aFragPos, aBathPos, true);
                GetCASPos(Input, x, bFragPos, bBathPos, false);
                
                // STEP 2: Do Schmidt Decomposition to get impurity and bath states.
                /* Do the Schmidt-Decomposition on the full system hamiltonian. Which sub matrix is taken to be the impurity density and which to be the bath density
                   is what differs between impurities. From this, the matrix of eigenvectors of the bath density is put into the rotation matrix. */
                if (Unrestricted && !HalfUnrestricted)
                {
                    SchmidtDecomposition(aDensityMatrix, aRotationMatrix, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], aNumEnvVirt, Output);
                    SchmidtDecomposition(bDensityMatrix, bRotationMatrix, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], bNumEnvVirt, Output);
                    aFragmentRotations[x] = aRotationMatrix;
                    bFragmentRotations[x] = bRotationMatrix;
                    for (int i = 0; i < NumAOImp; i++)
                    {
                        aActiveRotation.col(aFragPos[i]) = aRotationMatrix.col(Input.FragmentOrbitals[x][i]);
                        bActiveRotation.col(bFragPos[i]) = bRotationMatrix.col(Input.FragmentOrbitals[x][i]);
                    }
                    for (int i = 0; i < NumAOImp; i++)
                    {
                        int aActBathOrb = ReducedIndexToOrbital(aBathPos[i], Input, x, true);
                        int bActBathOrb = ReducedIndexToOrbital(bBathPos[i], Input, x, false);
                        aActiveRotation.col(aBathPos[i]) = aRotationMatrix.col(aActBathOrb);
                        bActiveRotation.col(bBathPos[i]) = bRotationMatrix.col(bActBathOrb);
                    }
                }
                else
                {
                    SchmidtDecomposition(DensityMatrix, RotationMatrix, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], NumEnvVirt, Output);
                    FragmentRotations[x] = RotationMatrix;
                    for (int i = 0; i < NumAOImp; i++)
                    {
                        ActiveRotation.col(aFragPos[i]) = RotationMatrix.col(Input.FragmentOrbitals[x][i]);
                    }
                    for (int i = 0; i < NumAOImp; i++)
                    {
                        int ActBathOrb = ReducedIndexToOrbital(aBathPos[i], Input, x, true);
                        ActiveRotation.col(aBathPos[i]) = RotationMatrix.col(ActBathOrb);
                    }
                }

                // ----- I think this is all part of the SCF impurity solver, which is no longer used. 
                // /* Before we continue with the SCF, we need to reduce the dimensionality of everything into the active space */
                
                // // First we rotate the density. Not needed but it should put us closer to the true answer.
                // Eigen::MatrixXd CASDensity = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size()); 
                // Eigen::MatrixXd RDR = RotationMatrix.transpose() * DensityMatrix * RotationMatrix;
                // ProjectCAS(CASDensity, RDR , Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], Input.NumAO, Input.NumOcc);

                // // Rotate the overlap matrix. Very necessary to do SCF inside the impurity. Also the reason we need to keep the ordering of orbitals consistent */
                // Eigen::MatrixXd RotOverlap = RotationMatrix.transpose() * Input.OverlapMatrix * RotationMatrix; // R^T S R = Rotated S
                // Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemRotS(RotOverlap); // Project out rows and columns not belonging to active impurity and bath space.
                // // Now use rotated S to get S^-1/2 in the rotated basis.
                // Eigen::SparseMatrix< double > LambdaRotSOrtho(Input.NumAO, Input.NumAO);
                // std::vector<T> tripletList;
                // for(int i = 0; i < Input.NumAO; i++)
                // {
                //     tripletList.push_back(T(i, i, 1 / sqrt(EigensystemRotS.eigenvalues()[i])));
                // }
                // LambdaRotSOrtho.setFromTriplets(tripletList.begin(), tripletList.end());
                // Eigen::MatrixXd RotSOrtho = EigensystemRotS.eigenvectors() * LambdaRotSOrtho * EigensystemRotS.eigenvectors().transpose();

                // // Finally, project the rotated S and S^-1/2 into the CAS.
                // Eigen::MatrixXd CASSOrtho(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
                // Eigen::MatrixXd CASOverlap(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
                // ProjectCAS(CASSOrtho, RotSOrtho, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], Input.NumAO, Input.NumOcc);
                // ProjectCAS(CASOverlap, RotOverlap, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], Input.NumAO, Input.NumOcc);

                // // Some definitions I don't really need but I need it for the SCF function.
                // std::vector<double> tmpVec;
                // FragmentEnergies[x].clear();
                // Eigen::MatrixXd FragmentCoeff;
                // -----

                // Now, solve the impurity. - This is using HenryFCI
                // std::vector< double > FCIEnergies;
                // Eigen::MatrixXd Fragment1RDM = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size()); // Will hold OneRDM
                // FCIEnergies = ImpurityFCI(Fragment1RDM, Input, x, RotationMatrix, ChemicalPotential, ImpurityStates[x], Fragment2RDM[x], FragmentEigenstates[x]);
                // FragmentEnergies[x] = FCIEnergies;

                // ***** The following solves the impurity using code from TroyFCI
                std::vector<int> aActiveList, aVirtualList, aCoreList, bActiveList, bVirtualList, bCoreList;
                GetCASList(Input, x, aActiveList, aCoreList, aVirtualList, true);
                GetCASList(Input, x, bActiveList, bCoreList, bVirtualList, false);

                int aElectronImp = (Input.FragmentOrbitals[x].size() < Input.aNumElectrons ? Input.FragmentOrbitals[x].size() : Input.aNumElectrons);
                int bElectronImp = (Input.FragmentOrbitals[x].size() < Input.bNumElectrons ? Input.FragmentOrbitals[x].size() : Input.bNumElectrons);

                FCI myFCI(Input, Input.FragmentOrbitals[x].size(), Input.FragmentOrbitals[x].size(), aCoreList, aActiveList, aVirtualList, bCoreList, bActiveList, bVirtualList);
                if (Unrestricted && !HalfUnrestricted) 
                {
                    myFCI.ERIMapToArray(Input.Integrals, aRotationMatrix, bRotationMatrix, aActiveList, bActiveList);
                }
                else 
                {
                    myFCI.ERIMapToArray(Input.Integrals, RotationMatrix, aActiveList);
                }
                // myFCI.PrintERI(true);
                std::cout << "aR\n" << aRotationMatrix << std::endl;
                std::cout << "bR\n" << bRotationMatrix << std::endl;

                // Begin SCF-IN-SCF Embedding Impurity Calculation
                // std::map<std::string, double> aaInt, abInt, baInt, bbInt;
                // myFCI.ERIArrayToMap(aaInt, abInt, baInt, bbInt);
                // InputObj FragInput;
                // MakeFragmentInput(Input, FragInput, aaInt, abInt, baInt, bbInt, NumAOImp * 2, 2 * Input.FragmentOrbitals[x].size());

                // std::vector< std::tuple< Eigen::MatrixXd, double, double> > EmptyBias;

                // Eigen::MatrixXd xtmp = Eigen::MatrixXd::Random(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
                // Eigen::MatrixXd aP = xtmp + xtmp.transpose();
                // xtmp = Eigen::MatrixXd::Random(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
                // Eigen::MatrixXd bP = xtmp + xtmp.transpose();
                // Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());

                // Eigen::MatrixXd RaR = aRotationMatrix.transpose() * aDensityMatrix * aRotationMatrix;
                // Eigen::MatrixXd RbR = bRotationMatrix.transpose() * bDensityMatrix * bRotationMatrix;
                // for (int i = 0; i < aActiveList.size(); i++)
                // {
                //     for (int j = 0; j < aActiveList.size(); j++)
                //     {
                //         aP(i, j) = RaR.coeffRef(aActiveList[i], aActiveList[j]);
                //     }
                // }
                // for (int i = 0; i < bActiveList.size(); i++)
                // {
                //     for (int j = 0; j < bActiveList.size(); j++)
                //     {
                //         bP(i, j) = RbR.coeffRef(bActiveList[i], bActiveList[j]);
                //     }
                // }

                // std::cout << aP << std::endl;
                // std::cout << bP << std::endl;

                // std::vector<double> AllFragE;
                // Eigen::MatrixXd aC, bC;
                // aC = bC = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
                // std::vector<int> aOcc, bOcc, aVirt, bVirt;
                // Eigen::VectorXd aOrbEV, bOrbEV;
                // for (int ii = 0; ii < Input.FragmentOrbitals[x].size(); ii++)
                // {
                //     aOcc.push_back(ii);
                //     bOcc.push_back(ii);
                // }
                // for (int ii = Input.FragmentOrbitals[x].size(); ii < 2 * Input.FragmentOrbitals[x].size(); ii++)
                // {
                //     aVirt.push_back(ii);
                //     bVirt.push_back(ii);
                // }
                // int xcount = 0;
                // Eigen::MatrixXd ZeroMat = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());

                // double xE = SCF(EmptyBias, EmptyBias, 1, aP, bP, FragInput, Output, I, FragInput.aHCore, FragInput.bHCore, AllFragE, aC, bC, aOcc, bOcc, aVirt, bVirt, xcount, -1, ZeroMat, aOrbEV, bOrbEV);
                // myFCI.aOneRDMs[ImpurityStates[x]] = aP;
                // myFCI.bOneRDMs[ImpurityStates[x]] = bP;
                // Eigen::MatrixXd Fragment1RDM = aP + bP;
                // xE = CalcSCFImpurityEnergy(aP, bP, aFragPos, bFragPos, myFCI.aOEI, myFCI.bOEI, myFCI.aOEIPlusCore, myFCI.bOEIPlusCore, myFCI.aaTEI, myFCI.abTEI, myFCI.bbTEI);
                // std::vector<double> tmpDVec;
                // tmpDVec.push_back(xE);
                // FragmentEnergies[x] = tmpDVec;

                // End SCF-in-SCF impurity calculation


                myFCI.AddChemicalPotentialGKLC(aFragPos, bFragPos, ChemicalPotential);
                myFCI.runFCI();
                // myFCI.DirectFCI();
                if (useRefP)
                {
                    myFCI.getRDM(true);
                    int ChosenImpState = BestRDM(RefDensity, Input.FragmentOrbitals[x], myFCI.OneRDMs, aFragPos);
                    ImpurityStates[x] = ChosenImpState;
                    std::cout << "DMET: Selected " << ChosenImpState << std::endl;
                    Output << "Selected " << ChosenImpState << std::endl;
                }
                else if (doScan)
                {
                    myFCI.getRDM(true);
                    std::string OldPName = "FragP_" + std::to_string(x + 1);
                    Eigen::MatrixXd OldP = ReadMatrixFromFile(OldPName, Input.FragmentOrbitals[x].size() * 2);
                    int ChosenImpState = SameRDM(OldP, myFCI.OneRDMs);
                    std::remove(OldPName.c_str());
                    ImpurityStates[x] = ChosenImpState;
                    std::ofstream PFile(OldPName);
                    PFile << myFCI.OneRDMs[ImpurityStates[x]];
                }
                else
                {
                    myFCI.getSpecificRDM(ImpurityStates[x], true);
                }
                
                // myFCI.dbgMyShitUp(Input.Integrals, aRotationMatrix, bRotationMatrix);

                // myFCI.DirectFCI();
                // // myFCI.getSpecificRDM(ImpurityStates[x], true);

                // std::vector<Eigen::MatrixXd> ProjBasis = FullFCI.DirectProjection(aDensityMatrix, bDensityMatrix, aRotationMatrix, bRotationMatrix, Input.FragmentOrbitals[x][0]);
                // Eigen::MatrixXd ProjH = FullFCI.ProjectMatrix(ProjBasis);
                // std::cout << "ProjH\n" << ProjH << std::endl;
                // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ProjEV(ProjH);
                // std::cout << "E1 = " << myFCI.Energies[0] << std::endl;
                // std::cout << "E2 = " << ProjEV.eigenvalues()[0] << std::endl;

                // myFCI.getRDM(true);
                // Determine the best density matrix:
                // int ChosenImpState = BestRDM(DensityMatrix, Input.FragmentOrbitals[x], myFCI.OneRDMs, FragPos);
                // ImpurityStates[x] = ChosenImpState;
                
                Eigen::MatrixXd Fragment1RDM = myFCI.OneRDMs[ImpurityStates[x]];
                std::vector<double> tmpDVec;
                tmpDVec.push_back(myFCI.calcImpurityEnergy(ImpurityStates[x], aFragPos, bFragPos));
                FragmentEnergies[x] = tmpDVec;

                // ***** END IMPURITY CALCULATION WITH TROYFCI

                // // ***** An impurity solver using HenryFCI for sigma FCI
                // FCI myFCI(Input, Input.FragmentOrbitals[x].size(), Input.FragmentOrbitals[x].size(), CoreList, ActiveList, VirtualList);
                // myFCI.GenerateHamiltonian(x, RotationMatrix, ChemicalPotential, 0);
                // if (x == 0 && MuIteration == 0)
                // {
                //     std::ofstream w_scan("w_scan8_4site.out");
                //     for (double w = -4.50; w < 6.00; w += 0.01)
                //     {
                //         myFCI.doSigmaFCI(w);
                //         std::vector<double> dbgDVec;
                //         Eigen::MatrixXd dbgFragment1RDM = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size());
                //         double Ex = myFCI.RDMFromHenryFCI(myFCI.SigmaFCIVector, x, RotationMatrix, dbgFragment1RDM);
                //         w_scan << w << "\t" << Ex * 2.0 << std::endl;
                //     }
                // }
                // myFCI.doSigmaFCI(-6.00);
                // std::vector<double> tmpDVec;
                // Eigen::MatrixXd Fragment1RDM = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size()); // Will hold OneRDM
                // tmpDVec.push_back(myFCI.RDMFromHenryFCI(myFCI.SigmaFCIVector, x, RotationMatrix, Fragment1RDM));
                // FragmentEnergies[x] = tmpDVec;
                // // ***** END

                // SCF(EmptyBias, 1, CASDensity, Input, Output, CASOverlap, CASSOrtho, FragmentEnergies[x], FragmentCoeff, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, RotationMatrix, FragmentOcc, NumAOImp, ChemicalPotential, x);
                FragmentDensities[x] = Fragment1RDM; // Save the density matrix after SCF calculation has converged.
                std::cout << "DMET: -- Fragment " << x + 1 << " complete with energy " << FragmentEnergies[x][0] << std::endl;
                Output << "DMET: -- Fragment " << x + 1 << " complete with energy " << FragmentEnergies[x][0] << std::endl;
                Output << "R:\n" << RotationMatrix << "\nD:\n" << Fragment1RDM << std::endl;
                if (Unrestricted)
                {
                    Output << "aP\n" << myFCI.aOneRDMs[ImpurityStates[x]] << "\nbP\n" << myFCI.bOneRDMs[ImpurityStates[x]] << std::endl;
                }
                Eigen::MatrixXd Unrotated1RDM = ActiveRotation * Fragment1RDM * ActiveRotation.transpose();
                // Eigen::MatrixXd AO1RDM = LocalToAO.transpose().inverse() * Unrotated1RDM * LocalToAO.inverse();
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> OccAndNO(Unrotated1RDM);
                if(LocalToAOFile.good())
                {
                    Output << "In the AO Basis:" << std::endl;
                    Output << "Bath Orbitals:\n" << LocalToAO * RotationMatrix << std::endl;
                    Output << "Occupation numbers:\n" << OccAndNO.eigenvalues() << std::endl;
                    Output << "Natural orbitals:\n" << LocalToAO * OccAndNO.eigenvectors() << std::endl;
                }
                std::cout << "Frag Density\n" << Fragment1RDM << std::endl;
            }
            // Start checking if chemical potential is converged.
            CostMuPrev = CostMu;
            CostMu = CalcCostChemPot(FragmentDensities, Input);

            std::cout << "DMET: All impurity calculations complete with a chemical potential of " << ChemicalPotential << " and cost function of " << CostMu << std::endl;
            Output << "DMET: All impurity calculations complete with a chemical potential of " << ChemicalPotential << " and cost function of " << CostMu << std::endl;
            if (OneShot)
            {
                break;
            }
            /* Change mu somehow */
            if(MuIteration % 2 == 0 && MuIteration > 0)
            {
                // Do Newton's Method
                double dLdmu = (CostMu - CostMuPrev) / StepSizeMu;
                ChemicalPotential = ChemicalPotential - CostMu / dLdmu;
            }
            else
            {
                ChemicalPotential += StepSizeMu; // Change chemical potential.
            }
            MuIteration++;
            if (MuIteration > 10)
            {
                break;
            }

            // double DMETEnergy = 0;
            // for(int x = 0; x < Input.NumFragments; x++)
            // {
            //     DMETEnergy += FragmentEnergies[x][0];
            // }
            // DMETEnergy += Input.Integrals["0 0 0 0"];
            // Output << "DMET: Energy = " << DMETEnergy << std::endl;
            // std::cout << "DMET: Energy = " << DMETEnergy << std::endl;
        }
        // Now the number of electrons are converged and each fragment is calculated.
        // Optimize the correlation potential to match the density matrix.
        DMETPotentialPrev = DMETPotential;
        // The following does BFGS to optimize the match and change the correlation potential.
        for(int i = 0; i < NumOcc; i++) // Reinitialize
        {
            OccupiedOrbitals.push_back(i);
        }
        for(int i = NumOcc; i < NumAO; i++)
        {
            VirtualOrbitals.push_back(i);
        }

        std::cout << "DMET: Beginning DMET potential optimization." << std::endl;
        Output << "DMET: Beginning DMET potential optimization." << std::endl;
        if(!OneShot)
        {
            UpdatePotential(DMETPotential, Input, CoeffMatrix, OrbitalEV, OccupiedByState, VirtualByState, FragmentDensities, FullDensities, Output, FragmentRotations, ImpurityStates, BathStates);
        }
        DMETPotentialChange = (DMETPotential - DMETPotentialPrev).squaredNorm(); // Square of elements as error measure.
        double CostU = CalcL(Input, FragmentDensities, FullDensities, FragmentRotations, BathStates);
        std::cout << "DMET: The cost function of this iteration is " << CostU << std::endl;
        Output << "DMET: The cost function of this iteration is " << CostU << std::endl;
        // Calculate full systme energy from each fragment energy.
        double DMETEnergy = 0;
        for(int x = 0; x < Input.NumFragments; x++)
        {
            DMETEnergy += FragmentEnergies[x][0];
        }
        DMETEnergy += Input.Integrals["0 0 0 0"];
        std::cout << "DMET: Energy = " << DMETEnergy << std::endl;
        Output << "DMET: Energy = " << DMETEnergy << std::endl;
        std::cout << "DMET: u = \n" << DMETPotential << std::endl;
        Output << "DMET: u = \n" << DMETPotential << std::endl;
        Output << "**********************************************" << std::endl;

        // std::string tmpstring;
        // std::getline(std::cin, tmpstring);
        if (uOptIt > 20)
        {
            std::cout << "DMET: Maximum number of interations reached." << std::endl;
            Output << "DMET: Maximum number of interations reached." << std::endl;
            break;
        }
        if (CostU < 1E-3)
        {
            break;
        }
    }
    std::cout << "DMET: DMET has converged." << std::endl;
    Output << "DMET: DMET has converged." << std::endl;

    // Bootstrap BE;
    // std::cout << "Init" << std::endl;
    // BE.debugInit(Input, Output);
    // std::cout << "schmidt" << std::endl;
    // BE.CollectSchmidt(FullDensities, Output);
    // std::cout << "run" << std::endl;
    // BE.runDebug();

    return 0;
}
