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

// #define H2H2H2
#define H10

void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &DensityMatrix, std::map<std::string, double> &Integrals, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int NumElectrons);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd CASOverlap, Eigen::MatrixXd &SOrtho, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd &RotationMatrix, double FragmentOcc, int NumAOImp, double ChemicalPotential, int FragmentIndex);
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd DMETPotential, Eigen::VectorXd &OrbitalEV);
void UpdatePotential(Eigen::MatrixXd &DMETPotential, InputObj &Input, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd> &FullDensities, std::ofstream &Output, std::vector< Eigen::MatrixXd > &FragmentRotations, std::vector< int > ImpurityStates, std::vector< int > BathStates);
std::vector< double > ImpurityFCI(Eigen::MatrixXd &DensityMatrix, InputObj &Input, int FragmentIndex, Eigen::MatrixXd &RotationMatrix, double ChemicalPotential, int State);
double CalcL(InputObj &Input, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd > &FullDensities, std::vector< Eigen::MatrixXd > FragmentRotations, std::vector< int > BathStates, int CostFunctionVariant = 2);


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
    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - Input.NumOcc;
    int NumCore = Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size();
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
    int NumVirt = Input.NumAO - Input.FragmentOrbitals[FragmentIndex].size() - Input.NumOcc;
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

double CalcCostChemPot(std::vector<Eigen::MatrixXd> FragmentDensities, InputObj Input)
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

void SchmidtDecomposition(Eigen::MatrixXd &DensityMatrix, Eigen::MatrixXd &RotationMatrix, std::vector< int > FragmentOrbitals, std::vector< int > EnvironmentOrbitals, int NumEnvVirt)
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

/* FockMatrix and DensityImp have the same dimension, the number of active space orbitals, or 2 * N_imp */
void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &HCore, Eigen::MatrixXd &DensityImp, Eigen::MatrixXd &RotationMatrix, InputObj &Input, double &ChemicalPotential, int FragmentIndex)
{
    for(int c = 0; c < FockMatrix.rows(); c++)
    {
        int cc = ReducedIndexToOrbital(c, Input, FragmentIndex);
        for(int d = c; d < FockMatrix.cols(); d++)
        {
            int dd = ReducedIndexToOrbital(d, Input, FragmentIndex);
            double Hcd = OneElectronEmbedding(Input.Integrals, RotationMatrix, cc, dd);
            HCore(c, d) = Hcd;
            for(int u = 0; u < Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size(); u++) // XC with core
            {
                int uu = Input.EnvironmentOrbitals[FragmentIndex][Input.EnvironmentOrbitals[FragmentIndex].size() - 1 - u];
                double Vcudu = TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, uu, dd, uu);
                double Vcuud = TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, uu, uu, dd);
                Hcd += (2 * Vcudu - Vcuud);
            }
            for(int l = 0; l < FockMatrix.rows(); l++) // XC within active space.
            {
                int ll = ReducedIndexToOrbital(l, Input, FragmentIndex);
                for(int k = 0; k < FockMatrix.cols(); k++)
                {
                    int kk = ReducedIndexToOrbital(k, Input, FragmentIndex);
                    Hcd += DensityImp(l, k) * (2 * TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, ll, dd, kk) - TwoElectronEmbedding(Input.Integrals, RotationMatrix, cc, kk, ll, dd));
                }
            } // end loop over active orbital XC
            std::vector< int > FragPos;
            std::vector< int > BathPos;
            GetCASPos(Input, FragmentIndex, FragPos, BathPos);
            if(c == d && (std::find(FragPos.begin(), FragPos.end(), c) != FragPos.end()))
            {
                Hcd -= ChemicalPotential;
            }
            FockMatrix(c, d) = Hcd;
            FockMatrix(d, c) = Hcd;
        }
    }
    // std::cout << "\nDENSITY\n" << DensityImp << std::endl;
    // std::cout << "\nFOCK\n" << FockMatrix << std::endl;
    // std::string tmpstring;
    // std::getline(std::cin, tmpstring);
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
    for (int i = 0; i < ImpurityStates.size(); i++)
    {
        ImpurityStates[i] = 0;
    }
    std::vector< int > BathStates(NumFragments);
    for (int i = 0; i < BathStates.size(); i++)
    {
        BathStates[i] = 0;
    }

    #ifdef H2H2H2
        ImpurityStates[1] = 1;
        // BathStates[0] = 1;
        // BathStates[2] = 1;
    #endif // H2H2H2
    #ifdef H10
        ImpurityStates[0] = 1;
        BathStates[0] = 1;
        BathStates[1] = 1;
        BathStates[2] = 1;
        BathStates[3] = 1;
        BathStates[4] = 1;
    #endif

    int NumSCFStates = *max_element(BathStates.begin(), BathStates.end());
    NumSCFStates++;
    
    // Begin by defining some variables.
    std::vector< std::tuple< Eigen::MatrixXd, double, double > > EmptyBias; // This code is capable of metadynamics, but this isn't utilized. We will use an empty bias to do standard SCF.
    Eigen::MatrixXd HCore(NumAO, NumAO); // T + V_eN
    Eigen::MatrixXd DensityMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // Will hold the density matrix of the full system.
    BuildFockMatrix(HCore, DensityMatrix, Input.Integrals, EmptyBias, Input.NumElectrons); // Build HCore, which is H when the density matrix is zero.
	Input.HCore = HCore;
    // for(int i = 0; i < Input.NumOcc; i++) // This initializes the density matrix to be exact in the MO basis.
    // {
    //     DensityMatrix(i, i) = 1;
    // }

    std::vector< Eigen::MatrixXd > FullDensities(NumSCFStates);

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
    
    // DEBUGGING - This is the H10 u from Wouter's code
    // for (int i = 0; i < 5; i++)
    // {
    //     DMETPotential(2 * i, 2 * i) = 1.541E-13;
    //     DMETPotential(2 * i + 1, 2 * i + 1) = -1.541E-13;
    //     DMETPotential(2 * i, 2 * i + 1) = -2.334E-03;
    //     DMETPotential(2 * i + 1, 2 * i) = -2.334E-03;
    // }

    double DMETPotentialChange = 1;
    int uOptIt = 0; // Number of iterations to optimize u
    while(fabs(DMETPotentialChange) > 1E-10) // || uOptIt < 5) // Do DMET until correlation potential has converged.
    {
        uOptIt++;

        // STEP 1: Solve the full system at the RHF level of theory.
        Eigen::VectorXd OrbitalEV; // Holds the orbital EVs from the proceeding SCF calculation. Needed to take derivatives.
        // These hold the density matrix and energies of each impurity.
        std::vector< Eigen::MatrixXd > FragmentDensities(Input.NumFragments);
        std::vector< std::vector< double > > FragmentEnergies(Input.NumFragments); // Vector of double incase multiple solutions are desired from one impurity.
        std::vector< double > AllEnergies;
        std::vector< Eigen::MatrixXd > FragmentRotations(Input.NumFragments); // Stores rotation matrix in case we wish to rotation the 1RDM, such as for certain types of density matching functions.


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
        // OccupiedOrbitals[NumOcc - 1] = NumOcc;
        // VirtualOrbitals[0] = NumOcc - 1;
        Input.OccupiedOrbitals = OccupiedOrbitals;
        Input.VirtualOrbitals = VirtualOrbitals;

        Eigen::MatrixXd CoeffMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // Holds the coefficient matrix of the full system calculation
        int SCFCount = 0; // Counts how many SCF iterations have been done in total.

        // These are values we need to calculate the derivative of the 1RDM later in the optimization. They are separated by state.
        std::vector< std::vector< int > > OccupiedByState;
        std::vector< std::vector< int > > VirtualByState;
        std::vector< Eigen::MatrixXd > CoeffByState;
        std::vector< Eigen::VectorXd > OrbitalEVByState;

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
        Output << "DMET: Beginning DMET Iteration Number " << uOptIt << std::endl;
        double SCFEnergy = 0.0;
        std::vector< std::tuple< Eigen::MatrixXd, double, double > > Bias;
        for (int i = 0; i < NumSCFStates; i++)
        {
            // if (i == 1) // Change occupation for excited state.
            // {
            //     int Orb1 = NumOcc - 1;
            //     int Orb2 = NumOcc;
            //     OccupiedOrbitals[Orb1] = Orb2;
            //     VirtualOrbitals[Orb2 - NumOcc] = Orb1;
            // }
            SCFEnergy = SCF(Bias, 1, DensityMatrix, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
            std::cout << "DMET: SCF calculation has converged with an energy of " << SCFEnergy << std::endl;
            std::cout << "DMET: and 1RDM of \n" << 2 * DensityMatrix << std::endl;
            Output << "SCF calculation has converged with an energy of " << SCFEnergy << std::endl;
            Output << "and 1RDM of \n" << 2 * DensityMatrix << std::endl;
            std::cout << Input.StartNorm << std::endl;
            std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(DensityMatrix, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
            Bias.push_back(tmpTuple);
            // std::cout << DensityMatrix << std::endl;
            // for (int j = 0; j < OccupiedOrbitals.size(); j++)
            // {
            //     std::cout << OccupiedOrbitals[j] << "\t";
            // }
            // std::cout << std::endl;
            // for (int j = 0; j < VirtualOrbitals.size(); j++)
            // {
            //     std::cout << VirtualOrbitals[j] << "\t";
            // }
            // std::cout << std::endl;
            FullDensities[i] = 2 * DensityMatrix;

            // Collect information needed for derivative calculations later.
            OccupiedByState.push_back(OccupiedOrbitals);
            VirtualByState.push_back(VirtualOrbitals);
            CoeffByState.push_back(CoeffMatrix);
            OrbitalEVByState.push_back(OrbitalEV);
        }
        // SCFEnergy = SCF(EmptyBias, 1, DensityMatrix, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, DMETPotential, OrbitalEV);
        // DensityMatrix = FullDensities[0];
        // std::cout << "DMET: SCF calculation has converged with an energy of " << SCFEnergy << std::endl;
        // std::cout << DensityMatrix << std::endl;

        // These are definitions for the global chemical potential, which ensures that the number of electrons stays as it should.
        double ChemicalPotential = 0; // The value of the chemical potential. This is a diagonal element on the Hamiltonian, on the diagonal positions corresponding to impurity orbitals.
        double CostMu = 100; // Cost function of mu, the sum of squares of difference in diagonal density matrix elements corresponding to impurity orbitals.
        double CostMuPrev = 0;
        double StepSizeMu = 1E-4; // How much to change chemical potential by each iteration. No good reason to choosing this number.
        int MuIteration = 0;
        while(fabs(CostMu) > 1E-3) // While the derivative of the cost function is nonzero, keep changing mu and redoing all fragment calculations.
        {
            std::cout << "DMET: -- Running impurity FCI calculations with a chemical potential of " << ChemicalPotential << std::endl;
            Output << "\nDMET: -- Running impurity FCI calculations with a chemical potential of " << ChemicalPotential << ".\n";
            for(int x = 0; x < NumFragments; x++) // Loop over all fragments.
            {
                // Use the correct density matrix for this fragment.
                DensityMatrix = FullDensities[BathStates[x]];

                // The density matrix of the full system defines the orbitals for each impurity. We begin with some definitions. These numbers depend on which impurity we are looking at.
                int NumAOImp = Input.FragmentOrbitals[x].size(); // Number of impurity states.
                int NumAOEnv = NumAO - NumAOImp; // The rest of the states.

                Eigen::MatrixXd RotationMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO); // The matrix of our orbital rotation.
                int NumEnvVirt = NumAO - NumAOImp - NumOcc; // Number of virtual (environment) orbitals in the bath.
                
                // STEP 2: Do Schmidt Decomposition to get impurity and bath states.
                /* Do the Schmidt-Decomposition on the full system hamiltonian. Which sub matrix is taken to be the impurity density and which to be the bath density
                   is what differs between impurities. From this, the matrix of eigenvectors of the bath density is put into the rotation matrix. */
                SchmidtDecomposition(DensityMatrix, RotationMatrix, Input.FragmentOrbitals[x], Input.EnvironmentOrbitals[x], NumEnvVirt);
                FragmentRotations[x] = RotationMatrix;

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

                // Now, solve the impurity.
                std::vector< double > FCIEnergies;
                Eigen::MatrixXd Fragment1RDM = Eigen::MatrixXd::Zero(2 * Input.FragmentOrbitals[x].size(), 2 * Input.FragmentOrbitals[x].size()); // Will hold OneRDM
                FCIEnergies = ImpurityFCI(Fragment1RDM, Input, x, RotationMatrix, ChemicalPotential, ImpurityStates[x]);
                FragmentEnergies[x] = FCIEnergies;

                // SCF(EmptyBias, 1, CASDensity, Input, Output, CASOverlap, CASSOrtho, FragmentEnergies[x], FragmentCoeff, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF, RotationMatrix, FragmentOcc, NumAOImp, ChemicalPotential, x);
                FragmentDensities[x] = Fragment1RDM; // Save the density matrix after SCF calculation has converged.
                std::cout << "DMET: -- Fragment " << x + 1 << " complete with energy " << FragmentEnergies[x][0] << std::endl;
                Output << "DMET: -- Fragment " << x + 1 << " complete with energy " << FragmentEnergies[x][0] << std::endl;
                Output << "R:\n" << RotationMatrix << "\nD:\n" << Fragment1RDM << std::endl;
                std::cout << "Frag Density\n" << Fragment1RDM << std::endl;
            }
            // Start checking if chemical potential is converged.
            CostMuPrev = CostMu;
            CostMu = CalcCostChemPot(FragmentDensities, Input);

            std::cout << "DMET: All impurity calculations complete with a chemical potential of " << ChemicalPotential << " and cost function of " << CostMu << std::endl;
            Output << "DMET: All impurity calculations complete with a chemical potential of " << ChemicalPotential << " and cost function of " << CostMu << std::endl;
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
        UpdatePotential(DMETPotential, Input, CoeffMatrix, OrbitalEV, OccupiedOrbitals, VirtualOrbitals, FragmentDensities, FullDensities, Output, FragmentRotations, ImpurityStates, BathStates);
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
    }
    std::cout << "DMET: DMET has converged." << std::endl;
    Output << "DMET: DMET has converged." << std::endl;
    return 0;
}
