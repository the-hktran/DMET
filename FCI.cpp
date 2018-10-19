#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <map>
#include <stdlib.h> 
#include <algorithm> // std::sort
#include <iomanip>
#include <queue>
#include "ReadInput.h"
#include "Functions.h"
#include "FCI.h"

using namespace Eigen;
using namespace std;
typedef vector<double> dv1;
typedef vector<int> iv1;
typedef vector<iv1> iv2;
typedef vector<iv2> iv3;
typedef vector<long unsigned int> luiv1;
typedef vector<MatrixXd> vMatrixXd;

int cpind(const int, const int);
int cpind4(const int, const int, const int, const int);
long int nchoosek(const int, const int);

int __s1, __s2, __s3;
#define ind2(i,j) i*__s1+j
#define ind4(i,j,k,l) i*__s3+j*__s2+k*__s1+l

/* 
   This is the initalization function when only the input object is given. In this case, we assume that the active space
   is the complete space 
*/
FCI::FCI(InputObj &Input)
{
    Inp = Input;
    aElectrons = Input.NumOcc;
    bElectrons = Input.NumOcc;
    aElectronsActive = aElectrons;
    bElectronsActive = bElectrons;
    aActive = Input.NumAO;
    bActive = Input.NumAO;
    aCore = 0;
    bCore = 0;
    aVirtual = 0;
    bVirtual = 0;

    aDim = nchoosek(aElectrons, aActive);
    bDim = nchoosek(bElectrons, bActive);
    Dim = aDim * bDim;

    NumberOfEV = Input.NumberOfEV;
    L = NumberOfEV + 50;

    OneRDMs.resize(NumberOfEV);
    TwoRDMs.resize(NumberOfEV);

    // for (int i = 0; i < aDim; i++)
    // {
    //     std::vector<bool> DetString;
    //     GetOrbitalString(i, aElectrons, aActive, DetString);
    //     for (int j = 0; j < aVirtual; j++)
    //     {
    //         DetString.push_back(false);
    //     }
    //     for (int j = 0; j < aCore; j++)
    //     {
    //         DetString.insert(DetString.begin(), true);
    //     }
    //     aStrings.push_back(DetString);
    // }

    // for (int i = 0; i < bDim; i++)
    // {
    //     std::vector<bool> DetString;
    //     GetOrbitalString(i, bElectrons, bActive, DetString);
    //     for (int j = 0; j < bVirtual; j++)
    //     {
    //         DetString.push_back(false);
    //     }
    //     for (int j = 0; j < bCore; j++)
    //     {
    //         DetString.insert(DetString.begin(), true);
    //     }
    //     bStrings.push_back(DetString);
    // }
}

FCI::FCI(InputObj &Input, int aElectronsAct, int bElectronsAct, std::vector<int> CoreList, std::vector<int> ActiveList, std::vector<int> VirtualList)
{
    Inp = Input;

    aElectrons = Input.NumOcc;
    bElectrons = Input.NumOcc;
    aElectronsActive = aElectronsAct;
    bElectronsActive = bElectronsAct;
    aActive = ActiveList.size();
    bActive = ActiveList.size();
    aActiveList = ActiveList;
    bActiveList = ActiveList;
    aCore = CoreList.size();
    bCore = CoreList.size();
    aCoreList = CoreList;
    bCoreList = CoreList;
    aVirtual = VirtualList.size();
    bVirtual = VirtualList.size();
    aVirtualList = VirtualList;
    bVirtualList = VirtualList;
    aOrbitals = Input.NumAO;
    bOrbitals = Input.NumAO;

    aDim = nchoosek(aElectronsActive, aActive);
    bDim = nchoosek(bElectronsActive, bActive);
    Dim = aDim * bDim;

    NumberOfEV = Input.NumberOfEV;
    L = NumberOfEV + 50;

    OneRDMs.resize(NumberOfEV);
    TwoRDMs.resize(NumberOfEV);

    /* Generate the strings. We have to insert the virtual and core orbitals according to their order in the list. */
    std::vector<int> CoreAndVirtualList = CoreList;
    CoreAndVirtualList.insert(CoreAndVirtualList.end(), VirtualList.begin(), VirtualList.end());
    std::sort(CoreAndVirtualList.begin(), CoreAndVirtualList.end());

    // for (int i = 0; i < aDim; i++)
    // {
    //     std::vector<bool> DetString;
    //     GetOrbitalString(i, aElectronsActive, aActive, DetString);
    //     for (int j = 0; j < CoreAndVirtualList.size(); j++)
    //     {
    //         DetString.insert(DetString.begin() + CoreAndVirtualList[j], std::find(CoreList.begin(), CoreList.end(), CoreAndVirtualList[j]) != CoreList.end());
    //     }
    //     aStrings.push_back(DetString);
    // }

    // for (int i = 0; i < bDim; i++)
    // {
    //     std::vector<bool> DetString;
    //     GetOrbitalString(i, bElectronsActive, bActive, DetString);
    //     for (int j = 0; j < CoreAndVirtualList.size(); j++)
    //     {
    //         DetString.insert(DetString.begin() + CoreAndVirtualList[j], std::find(CoreList.begin(), CoreList.end(), CoreAndVirtualList[j]) != CoreList.end());
    //     }
    //     bStrings.push_back(DetString);
    // }
}

FCI::FCI(InputObj &Input, int aElectronsActive, int bElectronsActive, std::vector<int> aCorList, std::vector<int> aActList, std::vector<int> aVirList, std::vector<int> bCorList, std::vector<int> bActList, std::vector<int> bVirList)
{
    Inp = Input;
    aElectrons = Input.NumOcc;
    bElectrons = Input.NumOcc;
    aActiveList = aActList;
    bActiveList = bActList;
    aActive = aActiveList.size();
    bActive = bActiveList.size();
    aCoreList = aCorList;
    bCoreList = bCorList;
    aCore = aCoreList.size();
    bCore = bCoreList.size();
    aVirtualList = aVirList;
    bVirtualList = bVirList;
    aVirtual = aVirtualList.size();
    bVirtual = bVirtualList.size();
    aOrbitals = Input.NumAO;
    bOrbitals = Input.NumAO;

    aDim = nchoosek(aElectronsActive, aActive);
    bDim = nchoosek(bElectronsActive, bActive);
    Dim = aDim * bDim;

    NumberOfEV = Input.NumberOfEV;
    L = NumberOfEV + 50;

    OneRDMs.resize(NumberOfEV);
    TwoRDMs.resize(NumberOfEV);

    /* Generate the strings. We have to insert the virtual and core orbitals according to their order in the list. */
    std::vector<int> aCoreAndVirtualList = aCoreList;
    aCoreAndVirtualList.insert(aCoreAndVirtualList.end(), aVirtualList.begin(), aVirtualList.end());
    std::sort(aCoreAndVirtualList.begin(), aCoreAndVirtualList.end());

    std::vector<int> bCoreAndVirtualList = bCoreList;
    aCoreAndVirtualList.insert(bCoreAndVirtualList.end(), bVirtualList.begin(), bVirtualList.end());
    std::sort(bCoreAndVirtualList.begin(), bCoreAndVirtualList.end());

    // for (int i = 0; i < aDim; i++)
    // {
    //     std::vector<bool> DetString;
    //     GetOrbitalString(i, aElectronsActive, aActive, DetString);
    //     for (int j = 0; j < aCoreAndVirtualList.size(); j++)
    //     {
    //         DetString.insert(DetString.begin() + aCoreAndVirtualList[j], std::find(aCoreList.begin(), aCoreList.end(), aCoreAndVirtualList[j]) != aCoreList.end());
    //     }
    //     aStrings.push_back(DetString);
    // }

    // for (int i = 0; i < bDim; i++)
    // {
    //     std::vector<bool> DetString;
    //     GetOrbitalString(i, bElectronsActive, bActive, DetString);
    //     for (int j = 0; j < bCoreAndVirtualList.size(); j++)
    //     {
    //         DetString.insert(DetString.begin() + bCoreAndVirtualList[j], std::find(bCoreList.begin(), bCoreList.end(), bCoreAndVirtualList[j]) != bCoreList.end());
    //     }
    //     bStrings.push_back(DetString);
    // }
}

void FCI::ERIMapToArray(std::map<std::string, double> &ERIMap)
{
    aOEI = new double[aActive * aActive];
    aaTEI = new double[aActive * aActive * aActive * aActive];
    for (int i = 0; i < aActive; i++)
    {
        for (int j = 0; j < aActive; j++)
        {
            aOEI[i * aActive + j] = ERIMap[std::to_string(i + 1) + " " + std::to_string(j + 1) + " 0 0"];
            for (int k = 0; k < aActive; k++)
            {
                for (int l = 0; l < aActive; l++)
                {
                    aaTEI[i * aActive * aActive * aActive + j * aActive * aActive + k * aActive + l] = ERIMap[std::to_string(i + 1) + " " + std::to_string(j + 1) + " " + std::to_string(k + 1) + " " + std::to_string(l + 1)];
                }
            }
        }
    }

    bOEI = aOEI;
    bbTEI = aaTEI;
    abTEI = aaTEI;

    ENuc = ERIMap["0 0 0 0"];
}

void FCI::ERIMapToArray(std::map<std::string, double> &ERIMap, Eigen::MatrixXd RotationMatrix, std::vector<int> ActiveOrbitals)
{
    int N = ActiveOrbitals.size();
    aOEI = new double[N * N];
    aaTEI = new double[N * N * N * N];
    aOEIPlusCore = new double [N * N];
    double Vcudu2MinusVcuud = 0.0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            aOEI[i * N + j] = OneElectronEmbedding(ERIMap, RotationMatrix, ActiveOrbitals[i], ActiveOrbitals[j]);
            for (int k = 0; k < N; k++)
            {
                for (int l = 0; l < N; l++)
                {
                    aaTEI[i * N * N * N + j * N * N + k * N + l] = TwoElectronEmbedding(ERIMap, RotationMatrix, ActiveOrbitals[i], ActiveOrbitals[k], ActiveOrbitals[j], ActiveOrbitals[l]);
                }
            }
            Vcudu2MinusVcuud = 0.0;
            for (int u = 0; u < aCoreList.size(); u++)
            {
                Vcudu2MinusVcuud += 2 * TwoElectronEmbedding(ERIMap, RotationMatrix, ActiveOrbitals[i], aCoreList[u], ActiveOrbitals[j], aCoreList[u]) - TwoElectronEmbedding(ERIMap, RotationMatrix, ActiveOrbitals[i], aCoreList[u], aCoreList[u], ActiveOrbitals[j]);
            }
            aOEIPlusCore[i * N + j] = aOEI[i * N + j] + Vcudu2MinusVcuud;
        }
    }

    bOEI = aOEI;
    bbTEI = aaTEI;
    abTEI = aaTEI;
    bOEIPlusCore = aOEIPlusCore;

    ENuc = ERIMap["0 0 0 0"];
}

void FCI::ERIMapToArray(std::map<std::string, double> &aERIMap, std::map<std::string, double> &bERIMap, std::map<std::string, double> &abERIMap, Eigen::MatrixXd aRotationMatrix, Eigen::MatrixXd bRotationMatrix, std::vector<int> aActiveList, std::vector<int> bActiveList)
{
    int aN = aActiveList.size();
    aOEI = new double[aN * aN];
    aaTEI = new double[aN * aN * aN * aN];
    aOEIPlusCore = new double[aN * aN];
    double Vcudu2MinusVcuud = 0.0;
    for (int i = 0; i < aN; i++)
    {
        for (int j = 0; j < aN; j++)
        {
            aOEI[i * aN + j] = OneElectronEmbedding(aERIMap, aRotationMatrix, aActiveList[i], aActiveList[j]);
            for (int k = 0; k < aN; k++)
            {
                for (int l = 0; l < aN; l++)
                {
                    aaTEI[i * aN * aN * aN + j * aN * aN + k * aN + l] = TwoElectronEmbedding(aERIMap, aRotationMatrix, aActiveList[i], aActiveList[k], aActiveList[j], aActiveList[l]);
                }
            }
            Vcudu2MinusVcuud = 0.0;
            for (int u = 0; u < aCoreList.size(); u++)
            {
                Vcudu2MinusVcuud += 2 * TwoElectronEmbedding(aERIMap, aRotationMatrix, aActiveList[i], aCoreList[u], aActiveList[j], aCoreList[u]) - TwoElectronEmbedding(aERIMap, aRotationMatrix, aActiveList[i], aCoreList[u], aCoreList[u], aActiveList[j]);
            }
            aOEIPlusCore[i * aN + j] = aOEI[i * aN + j] + Vcudu2MinusVcuud;
        }
    }

    int bN = bActiveList.size();
    bOEI = new double[bN * bN];
    bbTEI = new double[bN * bN * bN * bN];
    bOEIPlusCore = new double[bN * bN];
    for (int i = 0; i < bN; i++)
    {
        for (int j = 0; j < bN; j++)
        {
            bOEI[i * bN + j] = OneElectronEmbedding(bERIMap, bRotationMatrix, bActiveList[i], bActiveList[j]);
            for (int k = 0; k < bN; k++)
            {
                for (int l = 0; l < bN; l++)
                {
                    bbTEI[i * bN * bN * bN + j * bN * bN + k * bN + l] = TwoElectronEmbedding(bERIMap, bRotationMatrix, bActiveList[i], bActiveList[k], bActiveList[j], bActiveList[l]);
                }
            }
            Vcudu2MinusVcuud = 0.0;
            for (int u = 0; u < aCoreList.size(); u++)
            {
                Vcudu2MinusVcuud += 2 * TwoElectronEmbedding(bERIMap, bRotationMatrix, bActiveList[i], bCoreList[u], bActiveList[j], bCoreList[u]) - TwoElectronEmbedding(bERIMap, bRotationMatrix, bActiveList[i], bCoreList[u], bCoreList[u], bActiveList[j]);
            }
            bOEIPlusCore[i * bN + j] = bOEI[i * bN + j] + Vcudu2MinusVcuud;
        }
    }

    abTEI = new double[aN * bN * aN * bN];
    for (int i = 0; i < aN; i++)
    {
        for (int j = 0; j < bN; j++)
        {
            for (int k = 0; k < aN; k++)
            {
                for (int l = 0; l < bN; l++)
                {
                    abTEI[i * bN * aN * bN + j * aN * bN + k * bN + l] = TwoElectronEmbedding(abERIMap, aRotationMatrix, bRotationMatrix, bActiveList[i], bActiveList[k], aActiveList[j], bActiveList[l]);
                }
            }
        }
    }

    ENuc = aERIMap["0 0 0 0"];
}

void FCI::AddChemicalPotentialGKLC(std::vector<int> FragPos, double Mu)
{
    int aN = aActiveList.size();
    int bN = bActiveList.size();
    for (int i = 0; i < FragPos.size(); i++)
    {
        int iIdx = FragPos[i];
        aOEI[iIdx * aN + iIdx] -= Mu;
        bOEI[iIdx * bN + iIdx] -= Mu;
    }
}

bool SortEigenpairs(std::tuple<double, Eigen::MatrixXd, double> &Tuple1 , std::tuple<double, Eigen::MatrixXd, double> &Tuple2)
{
    return std::get<0>(Tuple1) < std::get<0>(Tuple2);
}

void FCI::runFCI()
{
    Eigenvectors.resize(NumberOfEV);
    Energies.resize(NumberOfEV);
    Symmetries.resize(NumberOfEV);
    FCIErrors.resize(NumberOfEV);
    int NumStrings = nchoosek(aActive, aElectronsActive);
    int it;

    FCIman(aActive, aElectronsActive, NumStrings, Conditioner, NumberOfEV, aOEI, aaTEI, Eigenvectors, Energies, Symmetries, FCIErrors, it, MaxIteration, 1E-12, false);

    // Now we sort the eigenpairs based on eigenenergies.
    std::vector< std::tuple<double, Eigen::MatrixXd, double> > EigenPairs;
    for (int i = 0; i < Energies.size(); i++)
    {
        EigenPairs.push_back(std::make_tuple(Energies[i], Eigenvectors[i], Symmetries[i]));
    }
    std::sort(EigenPairs.begin(), EigenPairs.end(), SortEigenpairs);
    for (int i = 0; i < Energies.size(); i++)
    {
        Energies[i] = std::get<0>(EigenPairs[i]);
        Eigenvectors[i] = std::get<1>(EigenPairs[i]);
        Symmetries[i] = std::get<2>(EigenPairs[i]);
    }
}

/* This imposes an order onto the binary strings. The function takes an index and returns the corresponding binary string.
   We order the strings as such:
   0: 11000
   1: 10100
   2: 10010
   3: 10001
   4: 01100
   5: 01010
   6: 01001
   7: 00110
   8: 00101
   9: 00011
   We find the binary string recursively. We first ask, is the first digit 0 or 1. The first digit turns into a zero when
   all of the remaining electrons (n_electrons - 1) have permuted through all of the orbitals excluding the lowest one 
   (n_orbitals - 1). So if the index is above (n_orb - 1) choose (n_e - 1), then the first digit is zero. In this case, we
   ask ourselves the same problem for the remaining digits, but now we have one less orbital and we should subtract the number
   of permutations from the index. If the index is below, then the first digit is one. We fill that in and ask ourselves the 
   same question with the remaining digits. It is the same problem but with one less orbital AND one less electron, but the 
   index should not be changed. */
void FCI::GetOrbitalString(int Index, int NumElectrons, int NumOrbitals, std::vector<bool> &OrbitalString)
{
    if(NumOrbitals > 0) // Stop when we have chosen a digit for all orbitals. We take off an orbital each time we fill it.
    {
        int PreviousComb = nchoosek(NumOrbitals - 1, NumElectrons - 1); // Number of ways for the higher electrons to permute in the higher orbitals.
        if (NumElectrons < 1) // If we don't have any electrons left, then all the remaining orbitals have to be empty.
        {
            OrbitalString.push_back(false);
            GetOrbitalString(Index, NumElectrons, NumOrbitals - 1, OrbitalString); // Move onto next orbital, remove one orbital from the list.
        }
        else if(Index < PreviousComb) // Means we have not finished all permutations and there is still an electron in the lowest orbital.
        {
            OrbitalString.push_back(true); // Put this electron there.
            GetOrbitalString(Index, NumElectrons - 1, NumOrbitals - 1, OrbitalString); // Consider the same problem, but with one less orbital and one less electron.
        }
        else // Means we have finished all those permuations and the electron in the first orbital should have moved.
        {
            Index -= PreviousComb; // Truncate the index, since we are considering a reduced problem.
            OrbitalString.push_back(false); // Empty orbital.
            GetOrbitalString(Index, NumElectrons, NumOrbitals - 1, OrbitalString); // Consider the same problem, but with one less orbital and a truncated index.
        }
    }
}

// This converts the vector storing the 2RDM into a tensor.
// Eigen::Tensor<double, 4> FCI::Make2RDMTensor(std::vector<double> GVector, int Dim)
// {
//     int N = Dim;
//     __s1 = N;
// 	__s2 = N * N;
// 	__s3 = N * N * N;
//     Eigen::Tensor<double, 4> TwoRDM(Dim, Dim, Dim, Dim);
//     for (int i = 0; i < Dim; i++)
//     {
//         for (int j = 0; j < Dim; j++)
//         {
//             for (int k = 0; k < Dim; k++)
//             {
//                 for (int l = 0; l < Dim; l++)
//                 {
//                     TwoRDM(i, j, k, l) = GVector[ind4(i, j, k, l)];
//                 }
//             }
//         }
//     }
//     return TwoRDM;
// }

void FCI::getSpecificRDM(int State, bool calc2RDM)
{
    Eigen::MatrixXd OneRDM;
    std::vector<double> TwoRDM;
    // Eigen::Tensor<double, 4> TwoRDM;

    Dim = nchoosek(aActive, aElectronsActive);
    RDM12(aActive, aElectronsActive, Dim, Eigenvectors[State], OneRDM, TwoRDM, calc2RDM);
    // if (calc2RDM)
    // {
    //     TwoRDM = Make2RDMTensor(TwoRDMVector, Dim);
    // }

    OneRDMs[State] = 2.0 * OneRDM;
    for (int i = 0; i < TwoRDM.size(); i++)
    {
        TwoRDM[i] = 2.0 * TwoRDM[i];
    }
    TwoRDMs[State] = TwoRDM;
}

double FCI::calcImpurityEnergy(int ImpState, std::vector<int> FragPos)
{
    double ImpEnergy = 0.0;

    int N = aActive;
    __s1 = N;
	__s2 = N * N;
	__s3 = N * N * N;
    for (int i = 0; i < FragPos.size(); i++)
    {
        int iIdx = FragPos[i];
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                for (int l = 0; l < N; l++)
                {
                    ImpEnergy += 0.5 * TwoRDMs[ImpState][ind4(iIdx, j, k, l)] * aaTEI[ind4(iIdx, j, k, l)];
                }
            }
            ImpEnergy += 0.5 * OneRDMs[ImpState](FragPos[i], j) * (aOEI[ind2(iIdx, j)] + aOEIPlusCore[ind2(iIdx, j)]);
        }
    }

    return ImpEnergy;
}

Eigen::MatrixXd FCI::GenerateHamiltonian(int FragmentIndex, Eigen::MatrixXd &RotationMatrix, double ChemicalPotential, int State)
{
    int NumAOImp = Inp.FragmentOrbitals[FragmentIndex].size();
    int NumVirt = Inp.NumAO - NumAOImp - Inp.NumOcc;
    int NumCore = Inp.NumOcc - NumAOImp;
    int NumEnv = Inp.EnvironmentOrbitals[FragmentIndex].size();

    /* There doesn't seem to be any method that chooses number of electrons */
    int aElectrons = Inp.NumOcc; // 2 * NumAOImp + NumCore;
    int bElectrons = Inp.NumOcc; // 2 * NumAOImp + NumCore;
    int aElectronsCAS = NumAOImp; // N_occ - N_core?????????????
    int bElectronsCAS = NumAOImp;
    int aOrbitals = Inp.NumAO;
    int bOrbitals = Inp.NumAO;
    int aCAS = 2 * NumAOImp;// + NumVirt;
    int bCAS = 2 * NumAOImp;// + NumVirt;
    int NumberOfEV = 1; // Input.NumberOfEV; // Number of eigenvalues desired from Davidson Diagonalization
    // int aDim = BinomialCoeff(aOrbitals, aElectrons);
    // int bDim = BinomialCoeff(bOrbitals, bElectrons);
    int aDim = BinomialCoeff(aCAS, aElectronsCAS);
    int bDim = BinomialCoeff(bCAS, aElectronsCAS);
    int Dim = aDim * bDim;
    int L = NumberOfEV + 50; // Dimension of starting subspace in Davidson Diagonalization
    while(L > Dim)
    {
        L /= 10;
    }

    double MatTol = 1E-12; // Zeros elements below this threshold, significiantly reduces storage requirements.

    // std::vector< std::vector<bool> > aStrings;
    // std::vector< std::vector<bool> > bStrings;

    // std::ofstream Output(Input.OutputName);
    // Output << "FCI Calculation\n\nInput File: " << Input.InputName << "\n\nNumber of Alpha Electrons: " << aElectrons << 
    // "\nNumber of Alpha Orbitals: " << aOrbitals << "\nNumber of Beta Electrons: " << bElectrons << "\nNumber of Beta Orbitals: "
    // << bOrbitals << "\nDimension of Space: " << aDim << " x " << bDim << " = " << Dim << "\n\nLooking for " << NumberOfEV << 
    // " solutions.\n" << std::endl;

    // clock_t Start = clock();
    double Start = omp_get_wtime();

    std::cout << "FCI: Generating all determinant binary representations and enumerating determinants with differences... ";
    for(int i = 0; i < aDim; i++)
    {
        std::vector<bool> tmpVec;
        GetOrbitalString(i, aElectronsCAS, aCAS, tmpVec); // Get binary string for active space orbitals.
        for(int a = 0; a < NumVirt; a++) // Insert frozen virtual orbitals into string
        {
            tmpVec.insert(tmpVec.begin() + Inp.EnvironmentOrbitals[FragmentIndex][a], false);
        }
        for(int c = 0; c < NumCore; c++) // Insert the core orbitals into the string.
        {
            tmpVec.insert(tmpVec.begin() + Inp.EnvironmentOrbitals[FragmentIndex][NumVirt + NumAOImp + c], true);
        }
        aStrings.push_back(tmpVec);
    }
    for(int i = 0; i < bDim; i++)
    {
        std::vector<bool> tmpVec;
        GetOrbitalString(i, bElectronsCAS, bCAS, tmpVec); // Get binary string for active space orbitals.
        for(int a = 0; a < NumVirt; a++) // Insert frozen virtual orbitals into string
        {
            tmpVec.insert(tmpVec.begin() + Inp.EnvironmentOrbitals[FragmentIndex][a], false);
        }
        for(int c = 0; c < NumCore; c++) // Insert the core orbitals into the string.
        {
            tmpVec.insert(tmpVec.begin() + Inp.EnvironmentOrbitals[FragmentIndex][NumVirt + NumAOImp + c], true);
        }
        bStrings.push_back(tmpVec);
    }

    std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > aSingleDifference; // i index, j index, sign, list of different orbitals.
    std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > aDoubleDifference;
    unsigned short int tmpInt;
    std::tuple<unsigned short int, unsigned short int, short int, std::vector<unsigned short int>> tmpTuple;

    for(unsigned int i = 0; i < aDim; i++)
    {
        for(unsigned int j = i + 1; j < aDim; j++)
        {
            tmpInt = CountDifferences(aStrings[i], aStrings[j]);
            if(tmpInt == 1)
            {
                short int tmpInt2 = FindSign(aStrings[i], aStrings[j]);
                std::vector<unsigned short int> tmpVec = ListDifference(aStrings[i], aStrings[j]);
                tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
                aSingleDifference.push_back(tmpTuple);
            }
            if(tmpInt == 2)
            {
                short int tmpInt2 = FindSign(aStrings[i], aStrings[j]);
                std::vector<unsigned short int> tmpVec = ListDifference(aStrings[i], aStrings[j]);
                tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
                aDoubleDifference.push_back(tmpTuple);
            }
        }
    }

    std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > bSingleDifference;
    std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > bDoubleDifference;
    for(unsigned int i = 0; i < bDim; i++)
    {
        for(unsigned int j = i + 1; j < bDim; j++)
        {
            tmpInt = CountDifferences(bStrings[i], bStrings[j]);
            if(tmpInt == 1)
            {
                short int tmpInt2 = FindSign(bStrings[i], bStrings[j]);
                std::vector<unsigned short int> tmpVec = ListDifference(bStrings[i], bStrings[j]);
                tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
                bSingleDifference.push_back(tmpTuple);
            }
            if(tmpInt == 2)
            {
                short int tmpInt2 = FindSign(bStrings[i], bStrings[j]);
                std::vector<unsigned short int> tmpVec = ListDifference(bStrings[i], bStrings[j]);
                tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
                bDoubleDifference.push_back(tmpTuple);
            }
        }
    }

    unsigned int NonzeroElements = Dim + aSingleDifference.size() * bDim * 2 + bSingleDifference.size() * aDim * 2 + aDoubleDifference.size() * bDim * 2
    + bDoubleDifference.size() * aDim * 2 + aSingleDifference.size() * bSingleDifference.size() * 4;
    //Output << "Number of alpha singles: " << aSingleDifference.size() << "\nNumber of beta singles: " << bSingleDifference.size() 
    //<< "\nNumber of alpha doubles: " << aDoubleDifference.size() << "\nNumber of beta doubles: " << bDoubleDifference.size() 
    //<< "\nChecking " << NonzeroElements << " elements.\n" << std::endl;

    std::cout << "done.\nFCI: Commencing with matrix initialization... " << std::endl;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Ham(Dim, Dim);
    // Ham.reserve(Eigen::VectorXi::Constant(Dim,NonzeroElements));
    // clock_t Timer = clock();
    double Timer = omp_get_wtime();

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    // std::vector< std::vector<T> > tripletList_Private(NumThreads);

    tripletList.reserve(NonzeroElements);

    /* The basis of the matrix is ordered by reverse lexicographic ordering (A,B) where A is the A'th  alpha orbital
       and B is the B'th beta orbital. Essentially, this means we have beta blocks and inside each block is a matrix for
       the alpha elements. */

    /* Diagonal Elements */
    /* Since I order the orbitals the same way in the bra and ket, there should be no sign change. There is a one electron
       component (single particle Hamiltonian), two electron component (coulombic repulsion), and zero electron component
       (nuclear repulsion). The nuclear repulsion term only appears in the diagonal. */
    double NuclearEnergy = Inp.Integrals["0 0 0 0"]; // Nuclear repulsion, will shift total energy and needs to be added to diagonal.

    // Isolated Core-Core interaction. This is excluded in the DMET impurity solver.
    double CoreInteraction = 0;
    for(int c = 0; c < 2 * NumCore; c++)
    {
        int cc = c % NumCore; // Loops back to handle both spins.
        int CoreOrbital1 = Inp.EnvironmentOrbitals[FragmentIndex][NumEnv - 1 - cc] + 1; // Count from 1
		bool c_isAlpha = true;
		if (c > NumCore - 1)
		{
			c_isAlpha = false;
		}
        for(int d = c + 1; d < 2 * NumCore; d++)
        {
            int dd = d % NumCore;
            int CoreOrbital2 = Inp.EnvironmentOrbitals[FragmentIndex][NumEnv - 1 - dd] + 1;
			bool d_isAlpha = true;
			if (d > NumCore - 1)
			{
				d_isAlpha = false;
			}
			CoreInteraction += (double)TwoElectronIntegral(CoreOrbital1, CoreOrbital2, CoreOrbital1, CoreOrbital2, c_isAlpha, d_isAlpha, c_isAlpha, d_isAlpha, Inp.Integrals, RotationMatrix);
        }
        CoreInteraction += (double)OneElectronEmbedding(Inp.Integrals, RotationMatrix, CoreOrbital1 - 1, CoreOrbital1 - 1);
    }

    std::vector< std::vector<unsigned short int> > aOrbitalList; // [Determinant Number][Occupied Orbital]
    std::vector< std::vector<unsigned short int> > bOrbitalList;
    for(unsigned short int i = 0; i < aDim; i++)
    {
        aOrbitalList.push_back(ListOrbitals(aStrings[i]));
    }
    for(unsigned short int j = 0; j < bDim; j++)
    {
        bOrbitalList.push_back(ListOrbitals(bStrings[j]));
    }
    #pragma omp parallel for
    for(int i = 0; i < aDim; i++) // Loop through every matrix element
    {
        // int Thread = omp_get_thread_num();
        std::vector<T> tripletList_Private;
        for(unsigned int j = 0; j < bDim; j++) // See above comment.
        {
            double tmpDoubleD = 0;
            /* Zero electron operator */
            // tmpDoubleD += NuclearEnergy; // Nuclear potential.
            /* One electron operator */
            for(int ii = 0; ii < aOrbitalList[i].size(); ii++)
            {
                tmpDoubleD += (double)OneElectronEmbedding(Inp.Integrals, RotationMatrix, aOrbitalList[i][ii] - 1, aOrbitalList[i][ii] - 1);
            }
            for(int jj = 0; jj < bOrbitalList[j].size(); jj++)
            {
                tmpDoubleD += (double)OneElectronEmbedding(Inp.Integrals, RotationMatrix, bOrbitalList[j][jj] - 1, bOrbitalList[j][jj] - 1);
            }

            /* Two electron operator in the notation <mn||mn> */
            std::vector<unsigned short int> abOrbitalList = aOrbitalList[i]; // List of all orbitals, starting with alpha.
            abOrbitalList.insert(abOrbitalList.end(), bOrbitalList[j].begin(), bOrbitalList[j].end());
            for(unsigned short int n = 0; n < aElectrons + bElectrons; n++) // Sum over occupied orbitals n
            {
                for(unsigned short int m = n + 1; m < aElectrons + bElectrons; m++) // Sum over m > n
                {
                    bool n_isAlpha = true;
                    bool m_isAlpha = true;
                    if(n > aElectrons - 1) n_isAlpha = false; // Means we have looped through the alpha orbitals and are now looking at a beta orbital
                    if(m > aElectrons - 1) m_isAlpha = false;
                    tmpDoubleD += (double)TwoElectronIntegral(abOrbitalList[m], abOrbitalList[n], abOrbitalList[m], abOrbitalList[n], m_isAlpha, n_isAlpha, m_isAlpha, n_isAlpha, Inp.Integrals, RotationMatrix);
                }
            }

            tmpDoubleD -= CoreInteraction; // Removes the core-core interaction.

			int NumSameImp = CountSameImpurity(aStrings[i], aStrings[i], Inp.FragmentOrbitals[FragmentIndex]) + CountSameImpurity(bStrings[j], bStrings[j], Inp.FragmentOrbitals[FragmentIndex]); // This totals the number of impurity orbitals in the alpha and beta lists.
            tmpDoubleD -= ChemicalPotential * (double)NumSameImp; // Form of chemical potential matrix element.

            // tripletList_Private[Thread].push_back(T(i + j * aDim, i + j * aDim, tmpDoubleD));
            tripletList_Private.push_back(T(i + j * aDim, i + j * aDim, tmpDoubleD));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }
    std::cout << "FCI: ...diagonal elements completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
    // Output << "Diagonal elements generated in " << (omp_get_wtime() - Timer)  << " seconds." << std::endl;
    // Timer = clock();
    Timer = omp_get_wtime();

    /* 
       Now we begin setting the nonzero off-diagonal elements. We separate this into three groups.
          1) Elements differing by one spin-orbital (one orbital in alpha and beta electrons)
             This is achieved by looping through single differences of alpha (beta) and choosing
             to keep beta (alpha) diagonal so that there is only one difference.
          2) Elements differing by two spin-orbitals of the same spin.
             This is achieved by looping through double differences of alpha (beta) and choosing
             to keep beta (alpha) diagonal so that there is only two differences.
          3) Elements differing by two spin-orbitals of different spin.
             This is achieved by looping through single differences of both alpha and beta spaces
             and gives us two differences composed of a one different alpha and one different beta
             orbital.
    */

    /* 
       We start with Group 1. The matrix of single differences in alpha is block diagonal in beta states,
       but off diagonal in alpha states within that block.
               |      *   * |            |            |
               |        *   |            |            |
               |          * |            |            |
               |            |            |            |
               |____________|____________|____________|
               |            |      *   * |            |
               |            |        *   |            |
               |            |          * |            |
               |            |            |            |
               |____________|____________|____________|
               |            |            |     *   *  |
               |            |            |       *    |
               |            |            |         *  |
               |            |            |            |
               |____________|____________|____________|
      We denote the bra <...mn...|
            and the ket |...pn...>

      To find these elements, we are going to calculation <m|h|p> using the list of differences. Then we construct
      an orbital list for this basis function and loop through all shared orbitals, meaning without m and p, and 
      calculate the two electron operator contribution.

      Matrix Elements: <m|h|p> + sum_n <mn||pn>
    */
    #pragma omp parallel for
    for(int i = 0; i < aSingleDifference.size(); i++)
    {
        // int Thread = omp_get_thread_num();
        std::vector<T> tripletList_Private;
        unsigned int Index1, Index2;
        double tmpDouble1 = 0;
        // First, add the one electron contribution.
        tmpDouble1 += (double)OneElectronEmbedding(Inp.Integrals, RotationMatrix, std::get<3>(aSingleDifference[i])[0] - 1, std::get<3>(aSingleDifference[i])[1] - 1); // Input.Integrals[std::to_string(std::get<3>(aSingleDifference[i])[0]) + " " + std::to_string(std::get<3>(aSingleDifference[i])[1]) + " 0 0"];

        // Now, two electron contribution
        for(unsigned int j = 0; j < bDim; j++)
        {
            double tmpDouble2 = 0;
            std::vector<unsigned short int> KetOrbitalList = aOrbitalList[std::get<1>(aSingleDifference[i])]; // To make the orbital list of the bra, we take the list of the current alpha determinant...
            KetOrbitalList.insert(KetOrbitalList.end(), bOrbitalList[j].begin(), bOrbitalList[j].end()); // ...and append the beta determinant, which is the same for bra and ket.
            int pos_m = CountOrbitalPosition(std::get<3>(aSingleDifference[i])[1], true, KetOrbitalList, aElectrons); // Position in ket of orbital missing in bra.
            for(unsigned short int n = 0; n < KetOrbitalList.size(); n++) // Sum over electrons in the Ket.
            {
                bool n_isAlpha = true; // Checks if we are looking at an alpha electron.
                if(n > aElectrons - 1) n_isAlpha = false; // There are aElectrons alpha orbitals at the front of the list. After this, we are done looping over alpha orbitals.
                if(n + 1 == pos_m) continue; // n shouldn't loop over different orbitals. We're looping over ket orbitals, so ignore the m.
                tmpDouble2 += (double)TwoElectronIntegral(std::get<3>(aSingleDifference[i])[0], KetOrbitalList[n], std::get<3>(aSingleDifference[i])[1], KetOrbitalList[n], true, n_isAlpha, true, n_isAlpha, Inp.Integrals, RotationMatrix);
                // For this case, we know that m and p orbitals are alpha. n may or may not be alpha depending on the index of the sum.
            }

            if(fabs(tmpDouble1 + tmpDouble2) < MatTol) continue;

            Index1 = std::get<0>(aSingleDifference[i]) + j * aDim; // Diagonal in beta states. Hop to other beta blocks.
            Index2 = std::get<1>(aSingleDifference[i]) + j * aDim;

            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
            tripletList_Private.push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
            tripletList_Private.push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }

    /* 
       The matrix of single differences in beta is not block diagonal, but is diagonal within the
       beta blocks that are off diagonal.
               |            |            |            |
               |            |  *         |            |
               |            |     *      |            |
               |            |        *   |            |
               |____________|____________|____________|
               |            |            |            |
               |            |            |  *         |
               |            |            |     *      |
               |            |            |        *   |
               |____________|____________|____________|
               |            |            |            |
               |            |            |            |
               |            |            |            |
               |            |            |            |
               |____________|____________|____________|
       
       The matrix elements for bra and ket 
       <...mn...|
       |...pn...>
       are: <m|h|p> + sum_n <mn||pn>
    */
    #pragma omp parallel for
    for(int i = 0; i < bSingleDifference.size(); i++)
    {
        // int Thread = omp_get_thread_num();
        std::vector<T> tripletList_Private;
        unsigned int Index1, Index2;
        double tmpDouble1 = 0;
        // First, add the one electron contribution.
        tmpDouble1 += (double)OneElectronEmbedding(Inp.Integrals, RotationMatrix, std::get<3>(bSingleDifference[i])[0] - 1, std::get<3>(bSingleDifference[i])[1] - 1); // Input.Integrals[std::to_string(std::get<3>(bSingleDifference[i])[0]) + " " + std::to_string(std::get<3>(bSingleDifference[i])[1]) + " 0 0"];

        // Now, two electron contribution
        for(unsigned int j = 0; j < aDim; j++)
        {
            double tmpDouble2 = 0;
            std::vector<unsigned short int> KetOrbitalList = aOrbitalList[j]; // Same as before, but now we set the alpha orbital in front and then append the beta orbitals.
            KetOrbitalList.insert(KetOrbitalList.end(), bOrbitalList[std::get<1>(bSingleDifference[i])].begin(), bOrbitalList[std::get<1>(bSingleDifference[i])].end());
            int pos_m = CountOrbitalPosition(std::get<3>(bSingleDifference[i])[1], false, KetOrbitalList, aElectrons); // Position in ket of orbital missing in bra.
            for(unsigned short int n = 0; n < KetOrbitalList.size(); n++) // Sum over orbitals in Ket.
            {
                bool n_isAlpha = true; // Checks if we are looking at an alpha electron.
                if(n > aElectrons - 1) n_isAlpha = false; // Finished looping over all alpha electrons.
                if(n + 1 == pos_m) continue; // n shouldn't loop over different orbitals. We're looping over ket orbitals, so ignore the m.
                tmpDouble2 += (double)TwoElectronIntegral(std::get<3>(bSingleDifference[i])[0], KetOrbitalList[n], std::get<3>(bSingleDifference[i])[1], KetOrbitalList[n], false, n_isAlpha, false, n_isAlpha, Inp.Integrals, RotationMatrix);
                // In this case, both the unique orbitals are beta orbitals.
            }

            if(fabs(tmpDouble1 + tmpDouble2) < MatTol) continue;

            Index1 = std::get<0>(bSingleDifference[i]) * aDim + j; // Loop through each same alpha state in each beta block.
            Index2 = std::get<1>(bSingleDifference[i]) * aDim + j;

            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
            tripletList_Private.push_back(T(Index1, Index2 , (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
            tripletList_Private.push_back(T(Index2, Index1 , (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }

    std::cout << "FCI: ...elements differing by one spin-orbital completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
    // Output << "Elements differing by one spin-orbital generated in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;

    // Timer = clock();
    Timer = omp_get_wtime();

    /* Now Group 2. The elements of the matrix for two differences, exclusively alpha or beta spin-orbitals, has the same
       matrix form as before. We have to loop through the other spins having no differences. 
       The notation used to denote the bra and ket is
       <...mn...|
       |...pq...>
       and the matrix element is <mn||pq> */
    #pragma omp parallel for
    for(int i = 0; i < aDoubleDifference.size(); i++)
    {
        // int Thread = omp_get_thread_num();
        std::vector<T> tripletList_Private;
        unsigned int Index1, Index2;
        for(unsigned int j = 0; j < bDim; j++)
        {
            /* This case is easier than the previous cases in that we do not need to obtain the list of similar orbitals,
               we only need to calculate the two electron integral involving the two differing orbitals and we know that
               both of these orbitals hold alpha electrons. */
            double tmpDouble;
            tmpDouble = (double)TwoElectronIntegral(std::get<3>(aDoubleDifference[i])[0], std::get<3>(aDoubleDifference[i])[1], std::get<3>(aDoubleDifference[i])[2], std::get<3>(aDoubleDifference[i])[3], true, true, true, true, Inp.Integrals, RotationMatrix);
            // The four electron differences, all of them alpha electrons.

            if(fabs(tmpDouble) < MatTol) continue;
            
            Index1 = std::get<0>(aDoubleDifference[i]) + j * aDim;
            Index2 = std::get<1>(aDoubleDifference[i]) + j * aDim;

            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
            tripletList_Private.push_back(T(Index1, Index2 , (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
            tripletList_Private.push_back(T(Index2, Index1 , (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }

    #pragma omp parallel for
    for(int i = 0; i < bDoubleDifference.size(); i++)
    {
        // int Thread = omp_get_thread_num();
        std::vector<T> tripletList_Private;
        unsigned int Index1, Index2;

        for(unsigned int j = 0; j < aDim; j++)
        {
            double tmpDouble;
            tmpDouble = (double)TwoElectronIntegral(std::get<3>(bDoubleDifference[i])[0], std::get<3>(bDoubleDifference[i])[1], std::get<3>(bDoubleDifference[i])[2], std::get<3>(bDoubleDifference[i])[3], false, false, false, false, Inp.Integrals, RotationMatrix);
            // The four electron differences, all of them beta electrons.

            if(fabs(tmpDouble) < MatTol) continue;

            Index1 = std::get<0>(bDoubleDifference[i]) * aDim + j; // Loop through each same alpha state in each beta block.
            Index2 = std::get<1>(bDoubleDifference[i]) * aDim + j;

            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
            tripletList_Private.push_back(T(Index1, Index2 , (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
            tripletList_Private.push_back(T(Index2, Index1 , (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }

    /* Now Group 3. Unlike before, we don't have to loop over alpha or beta having no differences. We simply loop
       over both alpha and beta having one difference. */
    int H2OMemoryWorkAround = 0;
    if(aSingleDifference.size() > 45000) // This is a workaround for the case of H2O. Cut down memory costs
    {
        // MatTol = 1E-12;
        H2OMemoryWorkAround = 17000; // This is how many differences I exclude from each alpha and beta string.
        // For H2O, the memory requirements are just too large. My work around is to increase the tolerance,
        // and remove the highest excitations, which shouldn't contribute a great deal to the ground state.
        // This is not systematic and I am not considering specific excitaitons.
    }
    #pragma omp parallel for
    for(int i = 0; i < aSingleDifference.size() - H2OMemoryWorkAround; i++)
    {
        // int Thread = omp_get_thread_num();
        std::vector<T> tripletList_Private;
        unsigned int Index1, Index2;
        for(unsigned int j = 0; j < bSingleDifference.size() - H2OMemoryWorkAround; j++)
        {
            double tmpDouble;
            tmpDouble = (double)TwoElectronIntegral(std::get<3>(aSingleDifference[i])[0], std::get<3>(bSingleDifference[j])[0], std::get<3>(aSingleDifference[i])[1], std::get<3>(bSingleDifference[j])[1], true, false, true, false, Inp.Integrals, RotationMatrix);
            /* There is one alpha and one beta orbital mismatched between the bra and ket. According to the formula, we put the bra unique orbitals in first,
               which are the first two arguments, the first one alpha and the second one beta. Then the next two arguments are the unique orbitals
               of the ket. We know whether these electrons are alpha or beta. */

            if(fabs(tmpDouble) < MatTol) continue;
            Index1 = std::get<0>(aSingleDifference[i]) + aDim * std::get<0>(bSingleDifference[j]);
            Index2 = std::get<1>(aSingleDifference[i]) + aDim * std::get<1>(bSingleDifference[j]);
            // Note that the sign is the product of the signs of the alpha and beta strings. This is because we can permute them independently.
            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            tripletList_Private.push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            tripletList_Private.push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
			
            /* We have to be a little more careful in this case. We want the upper triangle, but this only gives us half 
               of the upper triangle. In particular, the upper half of each beta block in upper triangle of the full matrix
               are the only nonzero elements. We want the whole beta block in the upper triangle of the full matrix to be
               nonzero where needed.
               |            |      *   * |      *   * |
               |            |        *   |        *   |
               |            |          * |          * |
               |            |            |            |
               |____________|____________|____________|
               |            |            |      *   * |
               |            |            |        *   |
               |            |            |          * |
               |            |            |            |
               |____________|____________|____________|
               |            |            |            |
               |            |            |            |
               |            |            |            |
               |            |            |            |
               |____________|____________|____________|
               So we have to include some transposed elements too. It is enough to transpose the alpha indices. this
               transposes each block above, and we end up with a fully upper triangular matrix. */
            Index1 = std::get<1>(aSingleDifference[i]) + aDim * std::get<0>(bSingleDifference[j]);
            Index2 = std::get<0>(aSingleDifference[i]) + aDim * std::get<1>(bSingleDifference[j]); // Note that first and second are switched for alpha here.

            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            /* IDK why but this is the culprit to the memory issue */
		    tripletList_Private.push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            tripletList_Private.push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }

    std::cout << "FCI: ...elements differing by two spin-orbitals completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;

    Ham.setFromTriplets(tripletList.begin(), tripletList.end());
    std::vector<T>().swap(tripletList); // This clears the memory, but I'm unsure if it does or doesn't increase the memory usage to do so

    std::cout << "FCI: Hamiltonian initialization took " << (omp_get_wtime() - Start) << " seconds." << std::endl;

    /* This section is for direct diagonalization. Uncomment if desired. */
    Timer = omp_get_wtime();

    Hamiltonian = Ham;
    return Hamiltonian;
//     Eigen::SelfAdjointEigenSolver< Eigen::MatrixXf > HamEV;
//     HamEV.compute(HamDense);

//     std::cout << " done" << std::endl;
//     std::cout << "FCI: Direct Diagonalization took " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
//  //   Output << "\nDirect Diagonalization took " << (omp_get_wtime() - Timer) << " seconds.\nThe eigenvalues are" << std::endl;
//     // std::cout << "FCI: The eigenvaues are";
//     std::vector< double > FCIEnergies;
//     for(int k = 0; k < State + 1; k++)
//     {
//         FCIEnergies.push_back(HamEV.eigenvalues()[k]);
//         // std::cout << "\n" << HamEV.eigenvalues()[k];
// //        Output << "\n" << HamEV.eigenvalues()[k];
//     }
// 	Eigenstate = HamEV.eigenvectors().col(State).cast<double>();

//     DensityMatrix = Form1RDM(Input, FragmentIndex, HamEV.eigenvectors().col(State), aStrings, bStrings);
// 	TwoRDM = Form2RDM(Input, FragmentIndex, HamEV.eigenvectors().col(State), aStrings, bStrings, DensityMatrix);

// 	/* Now we calculate the fragment energy */
// 	double Energy = 0;
// 	std::vector<int> FragPos;
// 	std::vector<int> BathPos;
// 	GetCASPos(Input, FragmentIndex, FragPos, BathPos);

// 	for (int i = 0; i < FragPos.size(); i++) // sum over impurity orbitals only
// 	{
// 		int iOrbital = ReducedIndexToOrbital(FragPos[i], Input, FragmentIndex);
// 		for (int j = 0; j < DensityMatrix.rows(); j++) // sum over all imp and bath orbitals
// 		{
// 			int jOrbital = ReducedIndexToOrbital(j, Input, FragmentIndex);
// 			for (int k = 0; k < DensityMatrix.rows(); k++)
// 			{
// 				int kOrbital = ReducedIndexToOrbital(k, Input, FragmentIndex);
// 				for (int l = 0; l < DensityMatrix.rows(); l++)
// 				{
// 					int lOrbital = ReducedIndexToOrbital(l, Input, FragmentIndex);
// 					Energy += 0.5 * TwoRDM(FragPos[i], k, l, j) * TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrbital, kOrbital, jOrbital, lOrbital);
// 				}
// 			}
// 			Energy += 0.5 * DensityMatrix(FragPos[i], j) * (OneElectronEmbedding(Input.Integrals, RotationMatrix, iOrbital, jOrbital) + OneElectronPlusCore(Input, RotationMatrix, FragmentIndex, iOrbital, jOrbital));
// 		}
// 	}

//     FCIEnergies[0] = Energy; // I don't see a need to save all lower states, so let's just put it in the bottom of the vector.
    
//     std::cout << "\nFCI: Total running time: " << (omp_get_wtime() - Start) << " seconds." << std::endl;
}

void FCI::doSigmaFCI(double w)
{
    Eigen::MatrixXd dSigma = Hamiltonian * Hamiltonian - 2.0 * w * Hamiltonian;
    // Diagonalize once.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> HamEV;
    HamEV.compute(dSigma);
    Eigen::VectorXd EVecOld = Eigen::VectorXd::Zero(HamEV.eigenvectors().rows());
    while ((EVecOld - HamEV.eigenvectors().col(0)).squaredNorm() > 1E-3 && (EVecOld + HamEV.eigenvectors().col(0)).squaredNorm() > 1E-3)
    {
        EVecOld = HamEV.eigenvectors().col(0);
        // Calculate the new Energy Expectation.
        double ExpE = HamEV.eigenvectors().col(0).transpose() * Hamiltonian * HamEV.eigenvectors().col(0);
        dSigma = Hamiltonian * Hamiltonian - 2.0 * ExpE * Hamiltonian;
        HamEV.compute(dSigma);
    }
    SigmaFCIVector = HamEV.eigenvectors().col(0);
}

/* This calculates the elements < a_i,alpha^\dagger a_j,alpha + a_i,beta^\dagger a_j,beta > */
Eigen::MatrixXd Form1RDM(InputObj &Input, int FragmentIndex, Eigen::VectorXd Eigenvector, std::vector< std::vector< bool > > aStrings, std::vector< std::vector< bool > > bStrings)
{
    std::vector<int> FragPos;
    std::vector<int> BathPos;
    GetCASPos(Input, FragmentIndex, FragPos, BathPos);

    int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();

    Eigen::MatrixXd DensityMatrix(2 * NumAOImp, 2 * NumAOImp);
    for(int i = 0; i < 2 * NumAOImp; i++)
    {
        for(int j = 0; j < 2 * NumAOImp; j++)
        {
            double DijA = 0;
            /* We formulate the alpha and beta density matrices separately, and add them together by element. */
            // Determine which orbital we are looking at, since i and j loop over the CAS index and not the real orbitals.
            int iOrbital = ReducedIndexToOrbital(i, Input, FragmentIndex);
            int jOrbital = ReducedIndexToOrbital(j, Input, FragmentIndex);

            // First, make the alpha density matrix.
            for(int ai = 0; ai < aStrings.size(); ai++)
            {
                for(int bi = 0; bi < bStrings.size(); bi++)
                {
                    for(int aj = 0; aj < aStrings.size(); aj++)
                    {
                        for(int bj = 0; bj < bStrings.size(); bj++)
                        {
                            std::vector< bool > BraAnnil = aStrings[ai];
                            std::vector< bool > KetAnnil = aStrings[aj];
							short int BraSign = 1;
							short int KetSign = 1;
                            if(aStrings[ai][iOrbital] && aStrings[aj][jOrbital]) // If bra contains i and ket contains j, then we annihilate the orbitals.
                            {
								BraSign = AnnihilationParity(BraAnnil, iOrbital);
								KetSign = AnnihilationParity(KetAnnil, jOrbital);
                                BraAnnil[iOrbital] = false; // bra with i annihilated
                                KetAnnil[jOrbital] = false; // ket with j annhiliated
                            }
                            else // Annihilated the bra or ket to zero
                            {
                                continue; // zero contribution
                            }
                            
                            // Count the differences in the bra and ket, which is the sum of differences in the alpha and beta
                            // components : <alpha|alpha><beta|beta>
                            int Diff = CountDifferences(BraAnnil, KetAnnil) + CountDifferences(bStrings[bi], bStrings[bj]);
                            if(Diff > 0)
                            {
                                continue;
                            }
                            DijA += BraSign * KetSign * Eigenvector[ai + bi * aStrings.size()] * Eigenvector[aj + bj * aStrings.size()];
                        }
                    }
                }
            }
            DensityMatrix(i, j) = DijA;

            // Now, do the same to make the beta density matrix. See above comments for explanation.
            double DijB = 0;
            for(int ai = 0; ai < aStrings.size(); ai++)
            {
                for(int bi = 0; bi < bStrings.size(); bi++)
                {
                    for(int aj = 0; aj < aStrings.size(); aj++)
                    {
                        for(int bj = 0; bj < bStrings.size(); bj++)
                        {
                            std::vector< bool > BraAnnil = bStrings[bi];
                            std::vector< bool > KetAnnil = bStrings[bj];
							short int BraSign = 1;
							short int KetSign = 1;
                            if(bStrings[bi][iOrbital] && bStrings[bj][jOrbital])
                            {
								BraSign = AnnihilationParity(BraAnnil, iOrbital);
								KetSign = AnnihilationParity(KetAnnil, jOrbital);
                                BraAnnil[iOrbital] = false;
                                KetAnnil[jOrbital] = false;
                            }
                            else // Annihilated the bra or ket to zero
                            {
                                continue;
                            }
                            
                            int Diff = CountDifferences(BraAnnil, KetAnnil) + CountDifferences(aStrings[ai], aStrings[aj]);
                            if(Diff > 0)
                            {
                                continue;
                            }
                            DijB += BraSign * KetSign * Eigenvector[ai + bi * aStrings.size()] * Eigenvector[aj + bj * aStrings.size()];
                        }
                    }
                }
            }
            DensityMatrix(i, j) += DijB;
        } // end loop over j
    } // end loop over i
    return DensityMatrix;
}

/* Calculates the 2RDM with elements P_ijkl = <a^dagger_i a^dagger_j a_k a_l>
However, note that given the definition of P_{ij|kl} = < a_j^dagger a_l^dagger a_i a_k > this means that the element TwoRDM(i,j,k,l) actually corresponds to P_{ki|lj} */
Eigen::Tensor<double, 4> Form2RDM(InputObj &Input, int FragmentIndex, Eigen::VectorXd Eigenvector, std::vector< std::vector< bool > > aStrings, std::vector< std::vector< bool > > bStrings, Eigen::MatrixXd &OneRDM)
{
	std::vector<int> FragPos;
	std::vector<int> BathPos;
	GetCASPos(Input, FragmentIndex, FragPos, BathPos);

	int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();

    /* Note that in order to convert to spacial orbitals, the actual element we are calculating is
        < a^t_i,alpha a^t_j,alpha a_k,alpha a_l,alpha + a^t_i,alpha a^t_j,beta a_k,alpha a_l,beta
          a^t_i,beta a^t_j,alpha a_k,beta a_l,alpha + a^t_i,beta a^t_j,beta a_k,beta a_l,beta >
       We will refer to these as elements 1, 2, 3, and 4 respectively. */
	Eigen::Tensor < double, 4> TwoRDM(2 * NumAOImp, 2 * NumAOImp, 2 * NumAOImp, 2 * NumAOImp);
    #pragma omp parallel for
	for (int i = 0; i < 2 * NumAOImp; i++)
	{
		// This is the orbital associated with the index, in the CAS index.
		int iOrbital = ReducedIndexToOrbital(i, Input, FragmentIndex);
		for (int j = 0; j < 2 * NumAOImp; j++)
		{
			int jOrbital = ReducedIndexToOrbital(j, Input, FragmentIndex);
			for (int k = 0; k < 2 * NumAOImp; k++)
			{
				int kOrbital = ReducedIndexToOrbital(k, Input, FragmentIndex);
				for (int l = 0; l < 2 * NumAOImp; l++)
				{
                    int lOrbital = ReducedIndexToOrbital(l, Input, FragmentIndex);
                    double Pijkl = 0;
                    // Element 1 (alpha, alpha)
                    for(int aBra = 0; aBra < aStrings.size(); aBra++)
                    {
                        for(int bBra = 0; bBra < bStrings.size(); bBra++)
                        {
                            for(int aKet = 0; aKet < aStrings.size(); aKet++)
                            {
                                for(int bKet = 0; bKet < bStrings.size(); bKet++)
                                {
                                    // First annihilate one orbital in the bra and ket.
                                    std::vector< bool > aBraAnnihilate = aStrings[aBra];
                                    std::vector< bool > bBraAnnihilate = bStrings[bBra];
                                    std::vector< bool > aKetAnnihilate = aStrings[aKet];
                                    std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

                                    if(aBraAnnihilate[iOrbital] && aKetAnnihilate[lOrbital])
                                    {
										tmpInt1 = AnnihilationParity(aBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
                                        aBraAnnihilate[iOrbital] = false;
                                        aKetAnnihilate[lOrbital] = false;
                                    }
                                    else
                                    {
                                        continue;
                                    }
                                    // Now annihilate another orbital, if it was already annihilated, the term is annihilated to zero and we move on.
                                    if(aBraAnnihilate[jOrbital] && aKetAnnihilate[kOrbital])
                                    {
										tmpInt1 = AnnihilationParity(aBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
                                        aBraAnnihilate[jOrbital] = false;
                                        aKetAnnihilate[kOrbital] = false;
                                    }
                                    else
                                    {
                                        continue;
                                    }

                                    int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
                                    if(Diff > 0)
                                    {
                                        continue;
                                    }
                                    Pijkl += BraSign * KetSign * Eigenvector[aBra + bBra * aStrings.size()] * Eigenvector[aKet + bKet * aStrings.size()];
                                }
                            }
                        }
                    } // end aBra loop.

                    // Element 2 (alpha, beta)
                    for(int aBra = 0; aBra < aStrings.size(); aBra++)
                    {
                        for(int bBra = 0; bBra < bStrings.size(); bBra++)
                        {
                            for(int aKet = 0; aKet < aStrings.size(); aKet++)
                            {
                                for(int bKet = 0; bKet < bStrings.size(); bKet++)
                                {
                                    // Annihilate orbital i, j, k, and l in respective strings. There's no possibility for overlap so we can just do it all at once.
                                    std::vector< bool > aBraAnnihilate = aStrings[aBra];
                                    std::vector< bool > bBraAnnihilate = bStrings[bBra];
                                    std::vector< bool > aKetAnnihilate = aStrings[aKet];
                                    std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

                                    if(aBraAnnihilate[iOrbital] && bBraAnnihilate[jOrbital] && aKetAnnihilate[lOrbital] && bKetAnnihilate[kOrbital])
                                    {
										tmpInt1 = AnnihilationParity(aBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
                                        aBraAnnihilate[iOrbital] = false;
                                        aKetAnnihilate[lOrbital] = false;

										tmpInt1 = AnnihilationParity(bBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										bBraAnnihilate[jOrbital] = false;
                                        bKetAnnihilate[kOrbital] = false;
                                    }
                                    else
                                    {
                                        continue;
                                    }

                                    int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
                                    if(Diff > 0)
                                    {
                                        continue;
                                    }
                                    Pijkl += BraSign * KetSign * Eigenvector[aBra + bBra * aStrings.size()] * Eigenvector[aKet + bKet * aStrings.size()];
                                }
                            }
                        }
                    } // end aBra loop.

                    // Element 3 (beta, alpha)
                    for(int aBra = 0; aBra < aStrings.size(); aBra++)
                    {
                        for(int bBra = 0; bBra < bStrings.size(); bBra++)
                        {
                            for(int aKet = 0; aKet < aStrings.size(); aKet++)
                            {
                                for(int bKet = 0; bKet < bStrings.size(); bKet++)
                                {
                                    // Annihilate orbital i, j, k, and l in respective strings. There's no possibility for overlap so we can just do it all at once.
                                    std::vector< bool > aBraAnnihilate = aStrings[aBra];
                                    std::vector< bool > bBraAnnihilate = bStrings[bBra];
                                    std::vector< bool > aKetAnnihilate = aStrings[aKet];
                                    std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

                                    if(bBraAnnihilate[iOrbital] && aBraAnnihilate[jOrbital] && bKetAnnihilate[lOrbital] && aKetAnnihilate[kOrbital])
                                    {
										tmpInt1 = AnnihilationParity(bBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
                                        bBraAnnihilate[iOrbital] = false;
                                        bKetAnnihilate[lOrbital] = false;

										tmpInt1 = AnnihilationParity(aBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										aBraAnnihilate[jOrbital] = false;
                                        aKetAnnihilate[kOrbital] = false;
                                    }
                                    else
                                    {
                                        continue;
                                    }

                                    int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
                                    if(Diff > 0)
                                    {
                                        continue;
                                    }
                                    Pijkl += BraSign * KetSign * Eigenvector[aBra + bBra * aStrings.size()] * Eigenvector[aKet + bKet * aStrings.size()];
                                }
                            }
                        }
                    } // end aBra loop.

                    // Element 4 (beta, beta)
                    for(int aBra = 0; aBra < aStrings.size(); aBra++)
                    {
                        for(int bBra = 0; bBra < bStrings.size(); bBra++)
                        {
                            for(int aKet = 0; aKet < aStrings.size(); aKet++)
                            {
                                for(int bKet = 0; bKet < bStrings.size(); bKet++)
                                {
                                    std::vector< bool > aBraAnnihilate = aStrings[aBra];
                                    std::vector< bool > bBraAnnihilate = bStrings[bBra];
                                    std::vector< bool > aKetAnnihilate = aStrings[aKet];
                                    std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

                                    if(bBraAnnihilate[iOrbital] && bKetAnnihilate[lOrbital])
                                    {
										tmpInt1 = AnnihilationParity(bBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
                                        bBraAnnihilate[iOrbital] = false;
                                        bKetAnnihilate[lOrbital] = false;
                                    }
                                    else
                                    {
                                        continue;
                                    }

                                    if(bBraAnnihilate[jOrbital] && bKetAnnihilate[kOrbital])
                                    {
										tmpInt1 = AnnihilationParity(bBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
                                        bBraAnnihilate[jOrbital] = false;
                                        bKetAnnihilate[kOrbital] = false;
                                    }
                                    else
                                    {
                                        continue;
                                    }

                                    int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
                                    if(Diff > 0)
                                    {
                                        continue;
                                    }
                                    Pijkl += BraSign * KetSign * Eigenvector[aBra + bBra * aStrings.size()] * Eigenvector[aKet + bKet * aStrings.size()];
                                }
                            }
                        }
                    } // end aBra loop.

                     // - delta_jk D_il
                     //if (j == k)
                     //{
                     //    Pijkl -= OneRDM(i, l);
                     //}

					TwoRDM(i, j, k, l) = Pijkl;
				} // l
			} // k
		} // j
	} // i
	return TwoRDM;
}

double FCI::RDMFromHenryFCI(Eigen::VectorXd Eigenstate, int FragmentIndex, Eigen::MatrixXd RotationMatrix, Eigen::MatrixXd &DensityMatrix)
{
    DensityMatrix = Form1RDM(Inp, FragmentIndex, Eigenstate, aStrings, bStrings);
	Eigen::Tensor<double, 4> TwoRDM = Form2RDM(Inp, FragmentIndex, Eigenstate, aStrings, bStrings, DensityMatrix);

	/* Now we calculate the fragment energy */
	double Energy = 0;
	std::vector<int> FragPos;
	std::vector<int> BathPos;
	GetCASPos(Inp, FragmentIndex, FragPos, BathPos);

	for (int i = 0; i < FragPos.size(); i++) // sum over impurity orbitals only
	{
		int iOrbital = ReducedIndexToOrbital(FragPos[i], Inp, FragmentIndex);
		for (int j = 0; j < DensityMatrix.rows(); j++) // sum over all imp and bath orbitals
		{
			int jOrbital = ReducedIndexToOrbital(j, Inp, FragmentIndex);
			for (int k = 0; k < DensityMatrix.rows(); k++)
			{
				int kOrbital = ReducedIndexToOrbital(k, Inp, FragmentIndex);
				for (int l = 0; l < DensityMatrix.rows(); l++)
				{
					int lOrbital = ReducedIndexToOrbital(l, Inp, FragmentIndex);
					Energy += 0.5 * TwoRDM(FragPos[i], k, l, j) * (double)TwoElectronEmbedding(Inp.Integrals, RotationMatrix, iOrbital, kOrbital, jOrbital, lOrbital);
				}
			}
			Energy += 0.5 * DensityMatrix(FragPos[i], j) * ((double)OneElectronEmbedding(Inp.Integrals, RotationMatrix, iOrbital, jOrbital) + (double)OneElectronPlusCore(Inp, RotationMatrix, FragmentIndex, iOrbital, jOrbital));
		}
	}

    return Energy;
}

// The following code is from troyfci.cpp

void eigh(const MatrixXd& A, MatrixXd& U, VectorXd& D)
{
	SelfAdjointEigenSolver<MatrixXd> es;
	es.compute (A);
	D = es.eigenvalues ();
	U = es.eigenvectors ();
}


void eigh2(const Matrix2d& A, Matrix2d& U, Vector2d& D)
{
	SelfAdjointEigenSolver<Matrix2d> es;
	es.compute (A);
	D = es.eigenvalues ();
	U = es.eigenvectors ();
}


long int nchoosek(const int n, const int k)
{
	double prod = 1.;
	if (k == 0)	return 1;
	else if (n < 0 || k < 0)	return 0;
	for (int i = 1; i <= k; i++)
		prod *= (double) (n - k + i) / i;
	return (long int) prod;
}


double MDOT(const MatrixXd& A, const MatrixXd& B)
{
    return (A.cwiseProduct (B)).sum ();
}

void GS(MatrixXd& X, const vMatrixXd Xi, const int iS)
{
    double dum;

//  Gramm-Schmidt Once
    for (int i = 0; i < iS; i++)
    {
        dum = MDOT(X, Xi[i]);
        X -= dum * Xi[i];
    }
    X.normalize (); // normalize w.r.t. its squared sum

//  Gramm-Schmidt Twice (for Stability)
    for (int i = 0; i < iS; i++)
    {
        dum = MDOT(X, Xi[i]);
        X -= dum * Xi[i];
    }
    X.normalize (); // normalize w.r.t. its squared sum
}


void Isort(const int N, iv1& X, int *Isign)
//  Sorts an Integer arrary X by straight insertion
//  Isign is the sign of the Permutation needed to bring it in order
{
    int i, j, tmp;
    bool flag;

    *Isign = 1;
    for (i = 1; i < N; i++)
    {
        flag = true;
        tmp = X[i];
        for (j = i - 1; j >= 0; j--)
        {
            if (X[j] <= tmp)    {flag = false;   break;}
            else    {X[j + 1] = X[j];   *Isign = -*Isign;}
        }
        if (flag)  j = -1;
        X[j + 1] = tmp;
    }
}


void RecurEx1(const int N, const int No, const int i, const int a,
    const int Max1, iv1& Iocc, iv3& Ex1, const int IRecur,
    int *m, int *Ind)
{
    int ii,jj,IsiO,IsaO,aind,iind,Isign,iimin;
    iv1 Isubst(No);

    if (!IRecur)    iimin = 0;
    else    iimin = Iocc[IRecur - 1] + 1;

    for (ii = iimin; ii < N; ii++)
    {
        Iocc[IRecur] = ii;

        if (IRecur == No - 1)
//      End Recursion Condition
        {
            IsiO = 0; IsaO = 0;
            for (jj = 0; jj < No; jj++)
                if (Iocc[jj] == i)  IsiO = 1;
                else if (Iocc[jj] == a) IsaO = 1;
            if (IsiO == 1 && IsaO == 0)
            {
                //_copy_array_ (Iocc, Isubst, No);
                Isubst = Iocc;
                for (jj = 0; jj < No; jj++)
                    if (i == Isubst[jj])    Isubst[jj] = a;
                Isort(No, Isubst, &Isign);

                Ex1[*m][a][i] = Isign * (*Ind);
                *m += 1;
            }
            *Ind += 1;
        }
        else
            RecurEx1(N, No, i, a, Max1, Iocc, Ex1,
                IRecur+1, m, Ind);
    }
    return;
}


void RecurEx2(const int N, const int No, const int N2,
    const int i, const int j, const int a, const int b,
    const int ij, const int ab, const int Max2,
    iv1& Iocc, iv3& Ex2, const int IRecur, int *m, int *Ind)
{
    int ii,jj;
    bool IsiO,IsjO,IsaO,IsbO;
    int iind,jind,aind,bind,Isign,iimin;
    iv1 Isubst(No);

    if (!IRecur)    iimin = 0;
    else    iimin = Iocc[IRecur - 1] + 1;

    for (ii = iimin; ii < N; ii++)
    {
        Iocc[IRecur] = ii;

        if (IRecur == No - 1)
//      End Recursion Condition
        {
            IsiO = IsjO = IsaO = IsbO = false;

//          Is abj*i* |Iocc> Non-Zero?
            for (jj = 0; jj < No; jj++)
                if (Iocc[jj] == i)  IsiO = true;
                else if (Iocc[jj] == j)  IsjO = true;
                else if (Iocc[jj] == a)  IsaO = true;
                else if (Iocc[jj] == b)  IsbO = true;
            if (IsiO && IsjO && !IsaO && !IsbO)
            {
                //_copy_array_ (Iocc, Isubst, No);
                Isubst = Iocc;
                for (jj = 0; jj < No; jj++)
                    if (Isubst[jj] == i)    Isubst[jj] = a;
                    else if (Isubst[jj] == j)    Isubst[jj] = b;

                Isort(No, Isubst, &Isign);

                Ex2[*m][ab][ij] = Isign * (*Ind);
                *m += 1;
            }
            *Ind += 1;
        }
        else
            RecurEx2(N, No, N2, i, j, a, b, ij, ab,
                Max2, Iocc, Ex2, IRecur+1, m, Ind);
    }
    return;
}


void IString(const int N, const int No, const int N2,
    const int Max1, const int Max2, iv3& Ex1, iv3& Ex2)
{
    int i, j, k, a, b, ij, ab, IRecur, m, Ind;
    iv1 Iocc(No);

//  Find Strings that differ by a Single Excitation
    for(i = 0; i < N; i++)
        for(a = 0; a < N; a++)
        {
            m = 0; Ind = 1; /* !!!!! INDEX STARTS FROM 1 !!!!! */
            RecurEx1(N, No, i, a, Max1, Iocc, Ex1, 0, &m, &Ind);
        }
//  Find Strings that differ by a Double Excitation
    ij = -1;
    for(i = 0; i < N; i++)
        for(j = i + 1; j < N; j++)
        {
            ij++; ab = -1;
            for(a = 0; a < N; a++)
                for(b = a + 1; b < N; b++)
                {
                    ab++;  m = 0;
                    Ind = 1;    /* !!!!! INDEX STARTS FROM 1 !!!!! */
                    RecurEx2(N, No, N2, i, j, a, b, ij, ab,
                        Max2, Iocc, Ex2, 0, &m, &Ind);
                }
        }
    return;
}


void GetIstr(const int N, const int No, const int Nstr, const int N0,
    MatrixXd& Hd, iv1& Istr)
{
    int itmp, a, b, i;
    double min;

    for (a = 0; a < N0; a++)
    {
        min = 500.; itmp = 0;
        for (i = 0; i < Nstr; i++)
            if (Hd(i, i) < min) {min = Hd(i, i);    itmp = i;}

        Istr[a] = itmp;
        Hd(Istr[a], Istr[a]) += 1000.;
    }
    for (a = 0; a < N0; a++)
        Hd(Istr[a], Istr[a]) -= 1000.;
}


void FCI_init(const int N, const int No, const int N2,
	const int Max1, const int Max2,
    iv2& Zindex, iv3& Ex1, iv3& Ex2)
{
//  Build indexing array for future use
    Zindex = iv2(N, iv1(N, 0));
    for (int k = 0; k < No; k++)
    for (int l = k; l <= N - No + k; l++)
        if (k == No - 1)
            Zindex[k][l] = l - k;
        else
            for (int m = N - l; m < N - k; m++)
                Zindex[k][l] += nchoosek(m , No - k - 1) -
                    nchoosek(m - 1, No - k - 2);
//  Determine connections among strings
    Ex1 = iv3(Max1, iv2(N, iv1(N, 0)));
    Ex2 = iv3(Max2, iv2(N2, iv1(N2, 0)));
    IString(N, No, N2, Max1, Max2, Ex1, Ex2);
}


int Index_(const int No, const iv1& Iocc, const iv2& Zindex)
{
    int i, Isign;
    int index = 0;  /* !!!!! INDEX STARTS FROM 1 !!!!! */

    for (i = 0; i < No; i++)    index += Zindex[i][Iocc[i]];
    return index;
}


void GetHd(const int N, const int No, const int Nstr, const iv2& Zindex,
    const double *h, const double *V,
    iv1& Iocca, iv1& Ioccb, MatrixXd& Hd, const int IRecur)
{
    int i,j,imin,jmin,k,ka,kb,l,la,lb,Isigna,Isignb;
    int kka, kkb, lla, llb, kla, klb;
    double tmp;

    if (IRecur == 0)    imin = jmin = 0;
    else
    {
        imin = Iocca[IRecur - 1] + 1;
        jmin = Ioccb[IRecur - 1] + 1;
    }

    for (i = imin; i < N; i++)
    {
        Iocca[IRecur] = i;
        for (j = jmin; j < N; j++)
        {
            Ioccb[IRecur] = j;
            if (IRecur == No - 1)
            {
                tmp = 0.;
                for (k = 0; k < No; k++)
                {
                    ka = Iocca[k];  kb = Ioccb[k];
					kka = ind2(ka,ka);	kkb = ind2(kb,kb);
//  Spin Contaminated Elements
                    tmp += h[kka] + h[kkb];
                    for (l = 0; l < No; l++)
                    {
                        la = Iocca[l];  lb = Ioccb[l];
                        tmp += 0.5 * (
                            V[ind4(ka,ka,la,la)] - V[ind4(ka,la,ka,la)] +
                            V[ind4(ka,ka,lb,lb)] + V[ind4(kb,kb,la,la)] +
                            V[ind4(kb,kb,lb,lb)] - V[ind4(kb,lb,kb,lb)]);
                    }
                }
                Hd(Index_(No, Iocca, Zindex), Index_(No, Ioccb, Zindex))
					= tmp;
            }
            else
                GetHd(N, No, Nstr, Zindex, h, V, Iocca, Ioccb, Hd, IRecur + 1);
        }
    }
    return;
}


void GetH0(const int N, const int No, const int N0, const int N2,
    const int Max1, const int Max2, const int Nstr,
    const iv3& Ex1, const iv3& Ex2, const iv1& Istr,
    const double *h, const double *V, MatrixXd& H0)
{
    iv1 RIstr(Nstr), IY;   IY.assign (N0, -1);
    iv2 ifzero(N, iv1(N));
    iv3 AEx1(Max1, iv2(N, iv1(N))), AEx2(Max2, iv2(N2, iv1(N2)));
    int i,j,k,l,ij,kl,ii,jj,I1,I2,iiMax,J1,J2,jjMax,ik,jl,m,If0,mmax;
    double Vtmp,VS,VSS,htmp,hS;
    dv1 stmp(N0);
    MatrixXd H0tmp;
    bool flag;
    const double zero = 0., one = 1., two = 2., three = 3., four = 4.;

    for (i = 0;i < Max1;i++)  for (j = 0;j < N;j++) for (k = 0;k < N;k++)
        AEx1[i][j][k] = abs (Ex1[i][j][k]);
    for (i = 0;i < Max2;i++)  for (j = 0;j < N2;j++) for (k = 0;k < N2;k++)
        AEx2[i][j][k] = abs (Ex2[i][j][k]);
//  Reverse String Ordering
    for (m = 0; m < N0; m++)    RIstr[Istr[m]] = m;

//  Remove terms that are not in H0
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (ii = 0; ii < Max1; ii++)
            {
                If0 = false;
                for (m = 0; m < N0; m++)
                    if (Istr[m] + 1 == AEx1[ii][i][j])   If0 = true;
                if (!If0)   AEx1[ii][i][j] = 0;
            }

    for (i = 0; i < N2; i++)
        for (j = 0; j < N2; j++)
            for (ii = 0; ii < Max2; ii++)
            {
                If0 = false;
                for (m = 0; m < N0; m++)
                    if (Istr[m] + 1 == AEx2[ii][i][j])   If0 = true;
                if (!If0)   AEx2[ii][i][j] = 0;
            }

//  Check for Zero blocks in V
    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++)
        {
			flag = true;
            for (j = 0; j < N && flag; j++)
                for (l = 0; l < N && flag; l++)
                    if (fabs(V[ind4(i,k,j,l)]) > 1.E-10) flag = false;
            if (flag)   ifzero[i][k] = 1;
        }

//  One Electron Part
    for (i = 0; i < N; i++) for (j = 0; j < N; j++)
    {
        htmp = h[ind2(i,j)];
        if (fabs (htmp) < 1.E-10)   continue;
        if (i == j) iiMax = nchoosek(N - 1, No - 1);
        else    iiMax = nchoosek(N - 2, No - 1);
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][j];    I2 = AEx1[ii][j][i];
            if (I1 == 0 || I2 == 0) continue;
            hS = (I1 == Ex1[ii][i][j]) ? (htmp) : (-htmp);
            for (m = 0; m < N0; m++)
                H0(m + RIstr[I2 - 1] * N0, m + RIstr[I1 - 1] * N0) += hS;
        }
    }
/*  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*   In Fortran, M(N*N, N*N) can also be viewd as Ms(N,N,N,N).
*   When viewd in this way, the conversion follows rule
*       Ms(a,b,c,d) --> M(a+b*N, c+d*N)
*   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

//  Same Spin Two-Electron Part
    int IK, IL, JK, JL;
    ij = -1;
    for (i = 0; i < N; i++) for (j = i + 1; j < N; j++)
    {
        ij++;   kl = -1;
        for (k = 0; k < N; k++) for (l = k + 1; l < N; l++)
        {
            kl++;
            Vtmp = (V[ind4(i,k,j,l)] - V[ind4(i,l,j,k)] -
                V[ind4(j,k,i,l)] + V[ind4(j,l,i,k)]) / two;
            if (fabs (Vtmp) > 1.E-10)
            {
                if (i == k && j == l)   iiMax = nchoosek(N - 2, No - 2);
                else if (i == k)    iiMax = nchoosek(N - 3, No - 2);
                else if (i == l)    iiMax = nchoosek(N - 3, No - 2);
                else if (j == k)    iiMax = nchoosek(N - 3, No - 2);
                else if (j == l)    iiMax = nchoosek(N - 3, No - 2);
                else    iiMax = nchoosek(N - 4, No - 2);
                for (ii = 0; ii < iiMax; ii++)
                {
                    I1 = AEx2[ii][ij][kl];  I2 = AEx2[ii][kl][ij];
                    if (I1 == 0 || I2 == 0) continue;
                    VS = (I1 == Ex2[ii][ij][kl]) ? (Vtmp) : (-Vtmp);
                    for (m = 0; m < N0; m++)
                        H0(m + RIstr[I2 - 1] * N0, m + RIstr[I1 - 1] * N0)
                            += VS;
                }
            }
        }
    }

//  Opposite Spin Two-Electron Part
    ik = -1;
    for (i = 0; i < N; i++) for (k = 0; k < N; k++)
    {
        ik++;   jl = -1;
        if (ifzero[i][k])   continue;
        if (i == k) iiMax = nchoosek(N - 1, No - 1);
        else    iiMax = nchoosek(N - 2, No - 1);

//  Gather together phases in Stmp
        stmp.assign (N0, 0);    IY.assign (N0, -1);
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][k];   I2 = AEx1[ii][k][i];
            if (I1 == 0 || I2 == 0) continue;
            VS = one;
            if (I1 != Ex1[ii][i][k])    VS = -VS;
            IY[RIstr[I1 - 1]] = RIstr[I2 - 1];
            stmp[RIstr[I1 - 1]] = VS;
        }
        mmax = 0;
        for (m = 0; m < N0; m++)
            if (IY[m] < 0) IY[m] = 0;
            else    mmax = m;

//  Collect Elements of H0
        for (j = 0; j < N; j++) for (l = 0; l < N; l++)
        {
            jl++;
            if (ik < jl)    continue;
            if (ik == jl)   Vtmp = V[ind4(i,k,j,l)] / two;
            if (ik > jl)    Vtmp = V[ind4(i,k,j,l)];
            if (fabs (Vtmp) < 1.E-10)   continue;

            if (j == l) jjMax = nchoosek(N - 1, No - 1);
            else    jjMax = nchoosek(N - 2, No - 1);

            for (jj = 0; jj < jjMax; jj++)
            {
                J1 = AEx1[jj][j][l];    J2 = AEx1[jj][l][j];
                if (J1 == 0 || J2 == 0) continue;
                VS = Vtmp;
                if (J1 != Ex1[jj][j][l])    VS = -VS;
                for (m = 0; m <= mmax; m++)
                {
                    H0(IY[m] + RIstr[J2 - 1] * N0, m + RIstr[J1 - 1] * N0)
                        += VS * stmp[m];
                }
            }
        }
    }

    H0tmp.setZero (N0 * N0, N0 * N0);
    for (i = 0; i < N0; i++) for (j = 0; j < N0; j++)
        for (k = 0; k < N0; k++) for (l = 0; l < N0; l++)
        {
            H0tmp(i + j * N0, k + l * N0)
                = H0(i + j * N0, k + l * N0) +
                H0(j + i * N0, l + k * N0);
        }
    H0 = H0tmp;
}


void HX(const int N, const int No, const int N2,
    const int Max1, const int Max2, const int Nstr,
    const iv3& Ex1, const iv3& Ex2, const MatrixXd& X, const double *h,
    const double *V, MatrixXd& Y)
{
    int i,j,k,l,ij,kl,ii,jj,I1,I2,iiMax,J1,J2,jjMax,ik,jl;
    iv2 ifzero(N, iv1(N));  iv3 AEx1(Max1, iv2(N, iv1(N)));
    int IfSym;
    double Vtmp,VS,VSS,htmp,hS,Tmp,Spin;
    MatrixXd Xtmp, Ytmp;
    bool flag;
    const double zero = 0., one = 1., two = 2., three = 3., four = 4.;

    Y.setZero (Nstr, Nstr);
    for (i = 0;i < Max1;i++)  for (j = 0;j < N;j++) for (k = 0;k < N;k++)
        AEx1[i][j][k] = abs (Ex1[i][j][k]);

//  Check Spin Symmetry of X
    Spin = one; Y = X;  Y -= X.transpose ();    Tmp = Y.squaredNorm ();
    if (Tmp > 1.E-1)    Spin = -one;
    Y.setZero ();

//  Check for Zero Blocks in V
    for (i = 0; i < N; i++) for (k = 0; k < N; k++)
    {
		flag = true;
        for (j = 0; j < N && flag; j++) for (l = 0; l < N && flag; l++)
            if (fabs (V[ind4(i,k,j,l)]) > 1.E-10)    flag = false;
        if (flag)   ifzero[i][k] = 1;
    }

//  Check Symmetry of V
// Don't need this in current version, as we always assume
//      <ij|kl> == <ji|lk>, i.e. e1 and e2 are exchangeable
/*    IfSym = 1;  flag = true;
    for (i = 0; i < N && flag; i++) for (k = 0; k < N && flag; k++)
    {
        ik = cpind(i,k);
        for (j = 0; j < N && flag; j++) for (l = 0; l < N && flag; l++)
        {
            jl = cpind(j,l);
            if (fabs (V[cpind(ik,jl)] - V[cpind(jl,ik)]) > 1.E-10)
            {   flag = false;   IfSym = 0;  }
        }
    }
*/

// One Electron Part
    for (i = 0; i < N; i++) for (j = 0; j < N; j++)
    {
        htmp = h[ind2(i,j)];
        if (fabs (htmp) < 1.E-10)   continue;
        if (i == j) iiMax = nchoosek(N - 1, No - 1);
        else    iiMax = nchoosek(N - 2, No - 1);
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][j];    I2 = AEx1[ii][j][i];
            hS = (I1 == Ex1[ii][i][j]) ? (htmp) : (-htmp);
            Y.col (I2 - 1) += X.col (I1 - 1) * hS;
        }
    }

//  Same Spin Two-Electron Part
    int IK, IL, JK, JL;
    ij = -1;
    for (i = 0; i < N; i++) for (j = i + 1; j < N; j++)
    {
        ij++;   kl = -1;
        for (k = 0; k < N; k++) for (l = k + 1; l < N; l++)
        {
            kl++;
            Vtmp = (V[ind4(i,k,j,l)] - V[ind4(i,l,j,k)] -
                V[ind4(j,k,i,l)] + V[ind4(j,l,i,k)]) / two;
            if (fabs (Vtmp) > 1.E-10)
            {
                if (i == k && j == l)   iiMax = nchoosek(N - 2, No - 2);
                else if (i == k)    iiMax = nchoosek(N - 3, No - 2);
                else if (i == l)    iiMax = nchoosek(N - 3, No - 2);
                else if (j == k)    iiMax = nchoosek(N - 3, No - 2);
                else if (j == l)    iiMax = nchoosek(N - 3, No - 2);
                else    iiMax = nchoosek(N - 4, No - 2);
                for (ii = 0; ii < iiMax; ii++)
                {
                    I1 = abs (Ex2[ii][ij][kl]);  I2 = abs (Ex2[ii][kl][ij]);
                    VS = (I1 == Ex2[ii][ij][kl]) ? (Vtmp) : (-Vtmp);
                    Y.col (I2 - 1) += VS * X.col (I1 - 1);
                }
            }
        }
    }

//  Opposite Spin Two-Electron Part
    ik = -1;
    Xtmp.setZero (Max1, Nstr);
    for (i = 0; i < N; i++) for (k = 0; k < N; k++)
    {
        ik++;   jl = -1;
        if (ifzero[i][k])   continue;
        if (i == k) iiMax = nchoosek(N - 1, No - 1);
        else    iiMax = nchoosek(N - 2, No - 1);

//  Gather together phases in Xtmp
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][k];
            VS = (I1 == Ex1[ii][i][k]) ? (one) : (-one);
            for (jj = 0; jj < Nstr; jj++)
                Xtmp(ii, jj) = X(I1 - 1, jj) * VS;
        }

//  Collect Elements of Ytmp
        Ytmp.setZero (Max1, Nstr);
        for (j = 0; j < N; j++) for (l = 0; l < N; l++)
        {
            jl++;
            //if (IfSym)    // As mentioned above, sym is not needed
            if (ik < jl)    continue;
            if (ik == jl)   Vtmp = V[ind4(i,k,j,l)] / two;
            if (ik > jl)    Vtmp = V[ind4(i,k,j,l)];
            if (fabs (Vtmp) < 1.E-10)   continue;

            if (j == l) jjMax = nchoosek(N - 1, No - 1);
            else    jjMax = nchoosek(N - 2, No - 1);

            for (jj = 0; jj < jjMax; jj++)
            {
                J1 = AEx1[jj][j][l];    J2 = AEx1[jj][l][j];
                VS = (J1 == Ex1[jj][j][l]) ? (Vtmp) : (-Vtmp);
                Ytmp.col (J2 - 1) += Xtmp.col (J1 - 1) * VS;
            }
        }

//  Scatter Elements of Y
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][k][i];
            for (jj = 0; jj < Nstr; jj++)
                Y(I1 - 1, jj) += Ytmp(ii, jj);
        }
    }
//  Enforce MS = 0
    MatrixXd Ytemp = Spin * Y.transpose ();
    Y += Ytemp;
}

void UHX (const int N, const int No, const int N2,
	const int Max1, const int Max2, const int Nstr,
    const iv3& Ex1, const iv3& Ex2, MatrixXd& X, const double *ha,
    const double *hb, const double *Vaa, const double *Vab, const double *Vbb,
    MatrixXd& Y)
//  Do Not Enfore Spin Symmetry
{
    int i,j,k,l,ij,kl,ii,jj,I1,I2,iiMax,J1,J2,jjMax,ik,jl;
    iv2 ifzero(N, iv1(N)), ifzero2(N, iv1(N));  iv3 AEx1(Max1, iv2(N, iv1(N)));
    int IfSym;
    double Vtmp,Vatmp,Vbtmp,VS,VSS,hatmp,hbtmp,hS,Tmp,Spin;
    MatrixXd Xtmp, Ytmp;
    bool flag;
    const double zero = 0., one = 1., two = 2., three = 3., four = 4.;

    Y.setZero (Nstr, Nstr);
    for (i = 0;i < Max1;i++)  for (j = 0;j < N;j++) for (k = 0;k < N;k++)
        AEx1[i][j][k] = abs (Ex1[i][j][k]);

//  Check for Zero Blocks in V
    for (i = 0; i < N; i++) for (k = 0; k < N; k++)
    {
		// for Vab
		flag = true;
        for (j = 0; j < N && flag; j++) for (l = 0; l < N && flag; l++)
            if (fabs (Vab[ind4(i,k,j,l)]) > 1.E-10)    flag = false;
        if (flag)   ifzero[i][k] = 1;
		// for Vba
		flag = true;
        for (j = 0; j < N && flag; j++) for (l = 0; l < N && flag; l++)
            if (fabs (Vab[ind4(j,l,i,k)]) > 1.E-10)    flag = false;
        if (flag)   ifzero2[i][k] = 1;
    }

//  Check Symmetry of V
// Don't need this in current version, as we always assume
//      <ij|kl> == <ji|lk>, i.e. e1 and e2 are exchangeable
/*    IfSym = 1;  flag = true;
    for (i = 0; i < N && flag; i++) for (k = 0; k < N && flag; k++)
    {
        ik = cpind(i,k);
        for (j = 0; j < N && flag; j++) for (l = 0; l < N && flag; l++)
        {
            jl = cpind(j,l);
            if (fabs (Vab[cpind(ik,jl)] - Vab[cpind(jl,ik)]) > 1.E-10)
            {   flag = false;   IfSym = 0;  }
        }
    }
*/

// One Electron Part
    for (i = 0; i < N; i++) for (j = 0; j < N; j++)
    {
        hatmp = ha[ind2(i,j)];  hbtmp = hb[ind2(i,j)];
        if (fabs (hatmp) + fabs (hbtmp) < 1.E-10)   continue;
        if (i == j) iiMax = nchoosek (N - 1, No - 1);
        else    iiMax = nchoosek (N - 2, No - 1);
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][j];    I2 = AEx1[ii][j][i];
            hS = (I1 == Ex1[ii][i][j]) ? (hbtmp) : (-hbtmp);
            Y.col (I2 - 1) += X.col (I1 - 1) * hS;
            hS = (I1 == Ex1[ii][i][j]) ? (hatmp) : (-hatmp);
            Y.row (I2 - 1) += X.row (I1 - 1) * hS;
        }
    }

//  Same Spin Two-Electron Part
    int IK, IL, JK, JL;
    ij = -1;
    for (i = 0; i < N; i++) for (j = i + 1; j < N; j++)
    {
        ij++;   kl = -1;
        for (k = 0; k < N; k++) for (l = k + 1; l < N; l++)
        {
            kl++;
            Vatmp = (Vaa[ind4(i,k,j,l)] - Vaa[ind4(i,l,j,k)] -
                Vaa[ind4(j,k,i,l)] + Vaa[ind4(j,l,i,k)]) / two;
            Vbtmp = (Vbb[ind4(i,k,j,l)] - Vbb[ind4(i,l,j,k)] -
                Vbb[ind4(j,k,i,l)] + Vbb[ind4(j,l,i,k)]) / two;
            if (fabs (Vatmp) + fabs (Vbtmp) > 1.E-10)
            {
                if (i == k && j == l)   iiMax = nchoosek (N - 2, No - 2);
                else if (i == k)    iiMax = nchoosek (N - 3, No - 2);
                else if (i == l)    iiMax = nchoosek (N - 3, No - 2);
                else if (j == k)    iiMax = nchoosek (N - 3, No - 2);
                else if (j == l)    iiMax = nchoosek (N - 3, No - 2);
                else    iiMax = nchoosek (N - 4, No - 2);
                for (ii = 0; ii < iiMax; ii++)
                {
                    I1 = abs (Ex2[ii][ij][kl]); I2 = abs (Ex2[ii][kl][ij]);
                    VS = (I1 == Ex2[ii][ij][kl]) ? (Vatmp) : (-Vatmp);
                    Y.col (I2 - 1) += VS * X.col (I1 - 1);
                    VS = (I1 == Ex2[ii][ij][kl]) ? (Vbtmp) : (-Vbtmp);
                    Y.row (I2 - 1) += VS * X.row (I1 - 1);
                }
            }
        }
    }

//  Opposite Spin Two-Electron Part
    ik = -1;    // X.transposeInPlace ();
    Xtmp.setZero (Max1, Nstr);
    for (i = 0; i < N; i++) for (k = 0; k < N; k++)
    {
        ik++;   jl = -1;
        if (ifzero[i][k])   continue;
        if (i == k) iiMax = nchoosek (N - 1, No - 1);
        else    iiMax = nchoosek (N - 2, No - 1);

//  Gather together phases in Xtmp
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][k];
            VS = (I1 == Ex1[ii][i][k]) ? (one) : (-one);
			for (jj = 0; jj < Nstr; jj++)
				Xtmp(ii, jj) = X(jj, I1-1) * VS;
			// Xtmp.row(ii) = X.row(I1-1) * VS;
        }

//  Collect Elements of Ytmp
        Ytmp.setZero (Max1, Nstr);
        for (j = 0; j < N; j++) for (l = 0; l < N; l++)
        {
            jl++;
            //if (IfSym)    // As mentioned above, sym is not needed
            if (ik < jl)    continue;
			if (ik == jl)   Vtmp = Vab[ind4(i,k,j,l)] / two;
            if (ik > jl)    Vtmp = Vab[ind4(i,k,j,l)];
            if (fabs (Vtmp) < 1.E-10)   continue;

            if (j == l) jjMax = nchoosek (N - 1, No - 1);
            else    jjMax = nchoosek (N - 2, No - 1);

            for (jj = 0; jj < jjMax; jj++)
            {
                J1 = AEx1[jj][j][l];    J2 = AEx1[jj][l][j];
                VS = (J1 == Ex1[jj][j][l]) ? (Vtmp) : (-Vtmp);
                Ytmp.col (J2 - 1) += Xtmp.col (J1 - 1) * VS;
            }
        }

//  Scatter Elements of Y
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][k][i];
            for (jj = 0; jj < Nstr; jj++)
                Y(jj, I1 - 1) += Ytmp(ii, jj);
        }
    }

//  Opposite Spin Two-Electron Part
    ik = -1;    // X.transposeInPlace ();
    Xtmp.setZero (Max1, Nstr);
    for (i = 0; i < N; i++) for (k = 0; k < N; k++)
    {
        ik++;   jl = -1;
        if (ifzero2[i][k])   continue;
        if (i == k) iiMax = nchoosek (N - 1, No - 1);
        else    iiMax = nchoosek (N - 2, No - 1);

//  Gather together phases in Xtmp
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][i][k];
            VS = (I1 == Ex1[ii][i][k]) ? (one) : (-one);
            for (jj = 0; jj < Nstr; jj++)
                Xtmp(ii, jj) = X(I1 - 1, jj) * VS;
        }

//  Collect Elements of Ytmp
        Ytmp.setZero (Max1, Nstr);
        for (j = 0; j < N; j++) for (l = 0; l < N; l++)
        {
            jl++;
            //if (IfSym)    // As mentioned above, sym is not needed
            if (ik < jl)    continue;
			if (ik == jl)   Vtmp = Vab[ind4(j,l,i,k)] / two;
            if (ik > jl)    Vtmp = Vab[ind4(j,l,i,k)];
            if (fabs (Vtmp) < 1.E-10)   continue;

            if (j == l) jjMax = nchoosek (N - 1, No - 1);
            else    jjMax = nchoosek (N - 2, No - 1);

            for (jj = 0; jj < jjMax; jj++)
            {
                J1 = AEx1[jj][j][l];    J2 = AEx1[jj][l][j];
                VS = (J1 == Ex1[jj][j][l]) ? (Vtmp) : (-Vtmp);
                Ytmp.col (J2 - 1) += Xtmp.col (J1 - 1) * VS;
            }
        }

//  Scatter Elements of Y
        for (ii = 0; ii < iiMax; ii++)
        {
            I1 = AEx1[ii][k][i];
			Y.row(I1-1) += Ytmp.row(ii);
        }
    }
}


void FCI::HX_(const int N, const int No, const int Nstr,
	const MatrixXd& X, const double *h, const double *V, MatrixXd& XH)
// A wrapper for HX
{
	__s1 = N;
	__s2 = N * N;
	__s3 = N * N * N;
	int N2 = nchoosek(N, 2),
		Max1 = nchoosek(N - 1, No - 1),
		Max2 = nchoosek(N - 2, No - 2);
	iv3 Ex1(Max1, iv2(N, iv1(N, 0))), Ex2(Max2, iv2(N2, iv1(N2, 0)));
	IString(N, No, N2, Max1, Max2, Ex1, Ex2);
	HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X, h, V, XH);
}


void FCI::UHX_(const int N, const int No, const int Nstr,
	MatrixXd& X, const double *ha, const double *hb,
	const double *Vaa, const double *Vab, const double *Vbb,
	MatrixXd& XH)
// A wrapper for UHX
{
	__s1 = N;
	__s2 = N * N;
	__s3 = N * N * N;
	int N2 = nchoosek(N, 2),
		Max1 = nchoosek(N - 1, No - 1),
		Max2 = nchoosek(N - 2, No - 2);
	iv3 Ex1(Max1, iv2(N, iv1(N, 0))), Ex2(Max2, iv2(N2, iv1(N2, 0)));
	IString(N, No, N2, Max1, Max2, Ex1, Ex2);
	UHX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X,
		ha, hb, Vaa, Vab, Vbb, XH);
}


bool FCI::FCIman(const int N, const int No, const int Nstr,
	const int N0, const int NS,
    //const double *ha, const double *hb, const double *Vaa, const double *Vbb, const double *Vab,
    const double *h, const double *V,
    vMatrixXd& Xi, dv1& Ei, dv1& Sym, dv1& Uncertainty, int& iter,
	const int MAXITER, const double THRESH, const bool iprint)
{
    int i,j,k,l,m,a,b,ij,kl,ab,ii,jj,kk,ll,ierr,Info;
    int iX,iS,iSpin,iSym;
    iv1 Iocc(No), Isubst(No), Istr(N0);
    double Energy, DE, fac, norm, eps, offset;
    MatrixXd Hd, H0, U0, X, X1, XH, X1H, Xtmp;
    VectorXd E0;    Matrix2d Hm, U; Vector2d Eig;
    const double zero = 0., one = 1., two = 2., three = 3., four = 4.;

	__s1 = N;
	__s2 = N * N;
	__s3 = N * N * N;

	const int N2 = nchoosek(N, 2),
		  Max1 = nchoosek(N - 1, No - 1),
		  Max2 = nchoosek(N - 2, No - 2);

	iv2 Zindex;	iv3 Ex1, Ex2;
	FCI_init(N, No, N2, Max1, Max2, Zindex, Ex1, Ex2);

    Energy = zero; fac = one;
//  Build Diagonal part of H
    Hd.setZero(Nstr, Nstr);
    GetHd(N, No, Nstr, Zindex, h, V, Iocc, Isubst, Hd, 0);

//  Get N0 lowest strings
    GetIstr(N, No, Nstr, N0, Hd, Istr);
    Isort(N0, Istr, &Info);

//  Build + Diagonalize H0
    H0.setZero(N0 * N0, N0 * N0);
    GetH0(N, No, N0, N2, Max1, Max2, Nstr, Ex1, Ex2, Istr, h, V, H0);
    eigh(H0, U0, E0);

//  Big loop over states
    iX = -1;
    for (iS = 0; iS < NS; iS++)
//  Initial vector for this state (ensure it is a singlet for now)
//  This restarts from the input Xi vector if Xi is nonzero
    {
        X = Xi[iS]; iSpin = 1;  norm = X.squaredNorm ();
        if (norm < 1.E-2)
        {
            iSpin = -1;
            while (iSpin == -1)
            {
                iX++;
                X.setZero (Nstr, Nstr); ij = -1;
                for (i = 0; i < N0; i++)    for (j = 0; j < N0; j++)
                {
//  Initialize X with eigenvectors of H0
                    ij++;   X(Istr[j], Istr[i]) = U0(ij, iX);
                }
                X1 = X; X1 -= X.transpose ();
                norm = X1.squaredNorm () / X.squaredNorm ();
                if (norm < 1.E-2)   iSpin = 1;
            }
        }

//  Check if the state has even (+1) or odd (-1) S
//  Even is singlet, quintet...  | Odd is triplet, sestet...
        X1 = X - X.transpose ();
        norm = X1.squaredNorm ();
        Sym[iS] = iSym = (norm<=0.1) ? (1) : (-1);

//  Fix up broken spin symmetry (if it exists)
        GS(X, Xi, iS);

//  Compute Initial guess energy
        XH.setZero (Nstr, Nstr);
        HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X, h, V, XH);
        Energy = MDOT(X, XH);

		if (iprint)
		{
			cout << "\t----------------------------------------------" << endl;
			cout << "\titer      total energy       error     rel var" << endl;
			cout << "\t----------------------------------------------" << endl;
		}
//  Iteratively determine eigenvalue #iS of the Hamiltonian
        fac = 1.;  iter = 0;
        while (fabs (fac) > THRESH && iter < MAXITER)
        {
            iter++; DE = Energy; offset = (iter < 2) ? (1.E-10) : (0);
            MatrixXd mat_offset = MatrixXd::Constant (Nstr, Nstr, offset);
            MatrixXd mat_E = MatrixXd::Constant (Nstr, Nstr, Energy);
//  Make the (orthogonal component of the) Davidson Update
            X1 = -(XH - Energy * X).cwiseQuotient (Hd - mat_E - mat_offset);
//  Build (H0-Energy)^-1 (excluding eigenvalues that might blow up)
            H0.setZero ();
            for (k = 0; k < N0 * N0; k++)
            {
                fac = E0(k) - Energy;
                if (fabs (fac) > 1.E-2)  fac = 1. / fac;
                else    fac = 0.;
                for (i = 0; i < N0 * N0; i++) for (j = 0; j < N0 * N0; j++)
                    H0(i, j) += U0(i, k) * U0(j, k) * fac;
            }

//  Build the Davidson Update using H0
            ij = -1;
            for (i = 0; i < N0; i++)    for (j = 0; j < N0; j++)
            {
                ij++;   kl = -1;
                X1(Istr[j], Istr[i]) = zero;
                for (k = 0; k < N0; k++)    for (l = 0; l < N0; l++)
                {
                    kl++;
                    X1(Istr[j], Istr[i]) -= H0(ij, kl) *
                        (XH(Istr[l], Istr[k]) - Energy * X(Istr[l], Istr[k])
                        + mat_offset(Istr[l], Istr[k]));
                }
            }

//  Apply Spin Symmetry and Gramm-Schmidt to Update
            Xtmp = iSym * X1.transpose ();  X1 = 0.5 * (X1 + Xtmp);
            X1.normalize ();
            norm = MDOT(X1, X);  X1 -= norm * X; X1.normalize ();
            norm = MDOT(X1, X);  X1 -= norm * X; X1.normalize ();

//  Correct if Davidson has given us a bad vector.
//  X1 should not be orthogonal to (H-E)X
//  If it is (nearly) orthogonal, add a bit of (H-E)X to X1
//  If we don't do this, it occasionally gives false convergence
//  This should happen rarely and eps controls how often
//  this correction is invoked.
//  A more elegant fix might be to just have a better
//  preconditioner.
            X1H = XH - Energy * X + mat_offset;
            GS(X1H, Xi, iS); X1H.normalize ();
            eps = 1.E-1;    fac = fabs (MDOT(X1, X1H));
            if (fac < eps)
            {
                X1 += 2. * eps * X1H;
                norm = MDOT(X1, X);  X1 -= norm * X; X1.normalize ();
                norm = MDOT(X1, X);  X1 -= norm * X; X1.normalize ();
            }

//  Make X1 orthogonal to lower eigenvectors
            GS(X1, Xi, iS);

//  Act H on Davidson Vector
            HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X1, h, V, X1H);

//  Build Hm
            Hm(0, 0) = MDOT(X, XH);
            Hm(1, 1) = MDOT(X1, X1H);
            Hm(0, 1) = Hm(1, 0) = MDOT(X, X1H);

//  Diagonalize Hm
            eigh2(Hm, U, Eig);

//  Keep Lowest Eigenvector
            if (iter < 50 || iter % 10)
            {
                fac = U(1, 0);
                X = U(0, 0) * X + U(1, 0) * X1;
                XH = U(0, 0) * XH + U(1, 0) * X1H;
            }
            else
//  If convergence is slow, sometimes it is because we are
//  Swiching back and forth between two near-solutions. To
//  fix that, every once in a while, we take half the step.
//  A more elegant fix might be to use more than one Davidson
//  vectors in the space.
            {
                fac = fabs(0.5 * U(1, 0));
                norm = one / sqrt (one + fac * fac);
                X = norm * (X + fac * X1);
                XH = norm * (XH + fac * X1H);
            }
//  Normalize X and XH
            norm = X.norm ();   X /= norm;  XH /= norm;

//  Avoid accumulating roundoff error
            if (iter % 4 == 0)
                HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X, h, V, XH);

            Energy = MDOT(X, XH) / X.squaredNorm ();
			Uncertainty[iS] = MDOT(XH, XH) / MDOT(XH, X) / MDOT(XH, X) - 1.;

			//  Standard output
			if (iprint)
				printf("\t%4d   % 15.11F   % .2E   % .2E\n", iter, Energy,
					fac, Uncertainty[iS]);

//  End of the Loop for FCI iteration
        }
          //printf ("Done after %4d iterations.\n", iter);
//  Energy, Uncertainty for THIS State
        HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X, h, V, XH);
        Energy = MDOT(X,XH) / X.squaredNorm ();
        Uncertainty[iS] = MDOT(XH, XH) / MDOT(XH, X) / MDOT(XH, X) - 1.;
        Xi[iS] = X; Ei[iS] = Energy;
//  Check if the state has even (+1) or odd (-1) S
//  Even is singlet, quintet...  | Odd is triplet, sestet...
        X1 = X - X.transpose ();
        norm = X1.squaredNorm ();
        Sym[iS] = (norm<=0.1) ? (1) : (-1);
//  End of the BIG Loop over States
    }
	if (iprint)
		cout << "\t----------------------------------------------" << endl;
    return fabs(fac) < THRESH;
}


void FCI::RDM12(const int N, const int No, const int Nstr,
	const MatrixXd& X, MatrixXd& D, dv1& G, const bool compute_rdm2)
// compute 1e and 2e density matrices
{
	__s1 = N;
	__s2 = N * N;
	__s3 = N * N * N;
    size_t i, j, k, l, ij, kl, ijkl,
		N2 = nchoosek(N, 2),
		Max1 = nchoosek(N - 1, No - 1),
		Max2 = nchoosek(N - 2, No - 2),
		s3 = N * N * N,
		s2 = N * N,
		s1 = N;
	iv3 Ex1(Max1, iv2(N, iv1(N, 0))), Ex2(Max2, iv2(N2, iv1(N2, 0)));
	IString(N, No, N2, Max1, Max2, Ex1, Ex2);
    MatrixXd XH(Nstr, Nstr);
    double T1[__s2] = {0}, T2[__s3 * N] = {0};
    double temp;

//  One Particle Density Matrix (1PDM) for spin alpha
    D.setZero(N, N);
    for (i = 0; i < N; i++) for (j = i; j < N; j++)
    {
		T1[ind2(i, j)] = T1[ind2(j, i)] = (i == j) ? (0.5) : (0.25);
        HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X, T1, T2, XH);
        D(i, j) = MDOT(X, XH);	D(j, i) = D(i, j);
        T1[ind2(i, j)] = T1[ind2(j, i)] = 0.;
    }
//  Two Particle Density Matrix (2PDM)
	if (compute_rdm2)
	{
		G = dv1(__s3 * N, 0.);
	    for (i = 0, ij = 0; i < N; i++) for (j = 0; j <= i; j++, ij++)
	    for (k = 0, kl = 0; k < N; k++) for (l = 0; l <= k; l++, kl++)
	    {
	        if (ij < kl)    continue;
	        temp = 1.;
	        if (i != j)     temp *= 0.5;
	        if (k != l)     temp *= 0.5;
	        if (ij != kl)   temp *= 0.5;
	        T2[ind4(i,j,k,l)] = T2[ind4(i,j,l,k)] =
			T2[ind4(j,i,k,l)] = T2[ind4(j,i,l,k)] =
			T2[ind4(k,l,i,j)] = T2[ind4(k,l,j,i)] =
			T2[ind4(l,k,i,j)] = T2[ind4(l,k,j,i)] = temp;
	        HX(N, No, N2, Max1, Max2, Nstr, Ex1, Ex2, X, T1, T2, XH);
	        G[i * s3 + j * s2 + k * s1 + l]			// ijkl
				= G[j * s3 + i * s2 + k * s1 + l]	// jikl
				= G[i * s3 + j * s2 + l * s1 + k]	// ijlk
				= G[j * s3 + i * s2 + l * s1 + k]	// jilk
				= G[k * s3 + l * s2 + i * s1 + j]	// klij
				= G[l * s3 + k * s2 + i * s1 + j]	// lkij
				= G[k * s3 + l * s2 + j * s1 + i]	// klji
				= G[l * s3 + k * s2 + j * s1 + i]	// lkji
				= MDOT(X, XH);
			T2[ind4(i,j,k,l)] = T2[ind4(i,j,l,k)] =
			T2[ind4(j,i,k,l)] = T2[ind4(j,i,l,k)] =
			T2[ind4(k,l,i,j)] = T2[ind4(k,l,j,i)] =
			T2[ind4(l,k,i,j)] = T2[ind4(l,k,j,i)] = 0.;
	    }
	}
}