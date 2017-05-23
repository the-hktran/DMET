#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>
#include <time.h>
#include <utility> // Pair
#include <tuple>
#include <map>
#include "ReadInput.h"
#include <fstream>
#include <Eigen/SpectrA/SymEigsSolver.h>
#include <Eigen/SpectrA/MatOp/SparseGenMatProd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SpectrA/Util/SelectionRule.h>

void Davidson(Eigen::SparseMatrix<float, Eigen::RowMajor> &Ham, int Dim, int NumberOfEV, int L, std::vector<double> &DavidsonEV);
double OneElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int c, int d);
double TwoElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int c, int d, int e, int f);
void GetCASPos(InputObj Input, int FragmentIndex, std::vector< int > &FragmentPos, std::vector< int > &BathPos);
int ReducedIndexToOrbital(int c, InputObj Input, int FragmentIndex);

int BinomialCoeff(int n, int k) // n choose k
{
    int nCk = 1;
    int denom = 1;
    if (k <= 0 || k >= n)
    {
        return 1;
    }
    for(int i = 0; i < k; i++)
    {
        nCk *= (n - i);
        denom *= (i + 1);
    }
    nCk /= denom;
    return nCk;
}

/* Indexing equation given in Knowles and Handy */
int Z_ForIndex(int ElectronNumber, int OrbitalNumber, int NumElectrons, int NumOrbitals)
{
    if(ElectronNumber == NumElectrons)
    {
        return OrbitalNumber - NumElectrons;
    }
    int Z = 0;
    for(int m = NumOrbitals - OrbitalNumber + 1; m <= NumOrbitals - ElectronNumber; m++)
    {
        Z += (BinomialCoeff(m, NumElectrons - ElectronNumber) - BinomialCoeff(m - 1, NumElectrons - ElectronNumber - 1));
    }
    return Z;
}

int StringIndex(std::vector<int> Orbitals, int NumOrbitals)
{
    int idx = 0;
    for (int i = 0; i < Orbitals.size(); i++)
    {
        idx += Z_ForIndex(i + 1, Orbitals[i], Orbitals.size(), NumOrbitals);
    }
    return idx;
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
void GetOrbitalString(int Index, int NumElectrons, int NumOrbitals, std::vector<bool> &OrbitalString)
{
    if(NumOrbitals > 0) // Stop when we have chosen a digit for all orbitals. We take off an orbital each time we fill it.
    {
        int PreviousComb = BinomialCoeff(NumOrbitals - 1, NumElectrons - 1); // Number of ways for the higher electrons to permute in the higher orbitals.
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

/* This counts the number of differences between two different binary representations */
short int CountDifferences(std::vector<bool> BraString, std::vector<bool> KetString)
{
    short int NumDiff = 0;
    for(int i = 0; i < KetString.size(); i++)
    {
        /* We want to count different orbitals, but if we end up doing that brute force we will double count since
           01 and 10 are different by one orbital, but their strings mismatch in two positions. It is enough to check
           that a 1 on one of the strings mismatches with the other string. */
        if(KetString[i])
        {
            if(!BraString[i])
            {
                NumDiff++;
            }
        }
    }
    return NumDiff;
}

/* 
   This is a function that finds the sign after putting two determinants into Slater-Condon form.
   To do so, we cound the number of positions each similar element is away from each other, and then
   we total this number. If the total is even, then the sign is positive and negative otherwise. Since
   the orbitals are organized ascending orbital number, we can always permute the lowest match, then the
   second lowest, and so on until they all line up. This justifies this method.
*/
short int FindSign(std::vector<bool> BraString, std::vector<bool> KetString)
{
    short int Sign = 0; // Will be +1 or -1, the sign.
    std::vector<int> KetOrder; // Counts position of the matching element in the ket.
    std::vector<int> BraOrder; // Does the same for the bra.
    int KetCount = 0;
    for(int i = 0; i < KetString.size(); i++)
    {
        if(KetString[i]) // We loop through both binary strings until we find a match, adding one to the counter each time to count our position.
        {
            KetCount++; // Count position along the ket. It only counts 1's, so filled orbitals.
            if(BraString[i]) // We have a match, but we don't know which position it is on the bra.
            {
                int BraCount = 1; // We go through the bra to count the number of orbitals (1's) before it. Not counting the actual element itself, so ammend with plus one.
                for(int j = 0; j < i; j++) // Loop through the string.
                {
                    if(BraString[j]) // Add one to the counter whereever an orbital is occupied.
                    {
                        BraCount++;
                    }
                }
                KetOrder.push_back(KetCount); // Add these positions to the list.
                BraOrder.push_back(BraCount);
            }
        }
    }

    for(int i = 0; i < KetOrder.size(); i++)
    {
        Sign += (abs(KetOrder[i] - BraOrder[i])); // Take hte total sum of position differences and use parity to find sign.
    }

    if(Sign % 2 == 0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

/* This converts a binary string into a string of orbital numbers. For example, 01011 -> 245 */
std::vector<unsigned short int> ListOrbitals(std::vector<bool> DeterminantString)
{
    std::vector<unsigned short int> OrbitalList;
    for(unsigned short int i = 0; i < DeterminantString.size(); i++) // Loop through the binary string
    {
        if(DeterminantString[i]) // Put down the position of 1 for each 1 in the string.
        {
            OrbitalList.push_back(i + 1);
        }
    }
    return OrbitalList;
}

/* This function will take two binary strings and give the numerical number of the orbitals in which the two strings differ.
   For example, 0010110 and 0010011 differ in one orbital and can be reformulated as 356 and 367. We see that these two determinants
   disagree in orbital 5 and orbital 7. So when looking at <356|H|367>, this funciton stores 5, 7. That is, the difference from the 
   bra as the orbital number then the distant from the ket as the orbital number. For two electron differences, <1234|H|3456> is 
   listed in the vector as 1, 2, 5, 6, similar to the two electron integral denotation <ij|kl>. */ 
std::vector<unsigned short int> ListDifference(std::vector<bool> BraString, std::vector<bool> KetString) // For 2e difference, put in <ij|kl> form
{
    std::vector<unsigned short int> OrbitalList; // Holds the bra differences, we ammend the ket differences on afterwards.
    std::vector<unsigned short int> tmpVec; // Holds the ket differences.
    for(unsigned short int i = 0; i < KetString.size(); i++)
    {
        if(BraString[i] && !KetString[i]) // There is a mismatch, with the occupied orbital belonging to the bra. <..1..|..0..>
        {
            OrbitalList.push_back(i + 1);
        }
        if(!BraString[i] && KetString[i]) // Mismatch, occupied on the ket. <..0..|..1..>
        {
            tmpVec.push_back(i + 1);
        }
    }
    OrbitalList.insert(OrbitalList.end(), tmpVec.begin(), tmpVec.end()); // Stick the ket list onto the end.
    return OrbitalList;
}

/* This function calculated <mn||kl> and takes as arguments, the orbital numbers, the spin of the orbitals, and the map */
float TwoElectronIntegral(unsigned short int m, unsigned short int n, unsigned short int k, unsigned short int l, bool m_isAlpha, bool n_isAlpha, bool k_isAlpha, bool l_isAlpha, std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix)
{
    double mknl = 0; // First term. (mk|nl)
    double mlnk = 0; // Second term. (ml|nk)
    
    /* Deal with first term first */
    if((m_isAlpha != k_isAlpha) || (n_isAlpha != l_isAlpha)) // Means spin component is different.
    {
        mknl = 0;
    }
    else
    {
        mknl = TwoElectronEmbedding(Integrals, RotationMatrix, m - 1, n - 1, k - 1, l - 1);
        // Integrals[std::to_string(m) + " " + std::to_string(k) + " " + std::to_string(n) + " " + std::to_string(l)];
    }
    /* Now, the second term */
    if((m_isAlpha != l_isAlpha) || (n_isAlpha != k_isAlpha))
    {
        mlnk = 0;
    }
    else
    {
        mlnk = TwoElectronEmbedding(Integrals, RotationMatrix, m - 1, n - 1, l - 1, k - 1);
        // Integrals[std::to_string(m) + " " + std::to_string(l) + " " + std::to_string(n) + " " + std::to_string(k)];
    }
    return mknl - mlnk;
}

short int CountOrbitalPosition(unsigned short int Orbital, bool isAlpha, std::vector<unsigned short int> OrbitalList, int aElectrons)
{
    int Count = 0;
    for(int i = 0; i < OrbitalList.size(); i++)
    {
        Count++;
        if(!isAlpha && i < aElectrons) // If we are currently looking for a beta orbital, we should skip the alpha orbitals.
        {
            continue;
        }
        if(Orbital == OrbitalList[i])
        {
            break;
        }
    }
    return Count;
}

short int CountSameImpurity(std::vector<bool> BraString, std::vector<bool> KetString, std::vector<int> FragmentOrbitals)
{
    short int Count = 0;
    for(int i = 0; i < FragmentOrbitals.size(); i++)
    {
        if(BraString[FragmentOrbitals[i]])
        {
            if(KetString[FragmentOrbitals[i]])
            {
                Count++;
            }
        }
    }
    return Count;
}

double Calc1RDMElement(int i, int j, std::vector<bool> aStrings, std::vector<bool> bStrings)
{
    for(int p = 0; p < aStrings.size(); p++)
    
}

int AnnihilateAndCount(std::vector<bool> BraBefore, int BraRemoved, std::vector<bool> KetBefore, int KetRemoved)
{
    std::vector<bool> BraAfter, KetAfter;
    for(int i = 0; i < BraBefore.size(); i++)
    {
        if(i == BraRemoved)
        {
            continue;
        }
        else
        {
            BraAfter.push_back(BraBefore[i]);
        }
    }
    for(int i = 0; i < KetBefore.size(); i++)
    {
        if(i == KetRemoved)
        {
            continue;
        }
        else
        {
            KetAfter.push_back(KetBefore[i]);
        }
    }
    int Diff = CountDifferences(BraAfter, KetAfter);
    return Diff;
}

Eigen::MatrixXd Form1RDM(InputObj &Input, int FragmentIndex, Eigen::VectorXd &Eigenvector, std::vector< std::vector< bool > > aStrings, std::vector< std::vector< bool > > bStrings)
{
    std::vector<int> FragPos;
    std::vector<int> BathPos;
    GetCASPos(Input, FragmentIndex, FragPos, BathPos);

    int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();
    int NumVirt = Input.NumAO - NumAOImp - Input.NumOcc;
    Eigen::MatrixXd DensityMatrix(2 * NumAOImp, 2 * NumAOImp);
    for(int i = 0; i < 2 * NumAOImp; i++)
    {
        for(int j = 0; j < 2 * NumAOImp; j++)
        {
            double Dij = 0;
            /* We are looking at element < a_i^\dagger a_j > so this kills any element in the bra without orbital i and any element in the
               ket with orbital j. */
            // Determine which orbital we are looking at, since i and j loop over the CAS index and not the real orbitals.
            int iOrbital = ReducedIndexToOrbital(i, Input, FragmentIndex);
            int jOrbital = ReducedIndexToOrbital(j, Input, FragmentIndex);
            // Now check which determinants have this orbital.
            std::vector< std::pair< int, int > > iContainingDets;
            std::vector< std::pair< int, int > > jContainingDets;
            for(int m = 0; m < aStrings.size(); m++)
            {
                for(int n = 0; n < bStrings.size(); n++)
                {
                    std::pair tmpPair;
                    if(aStrings[m][iOrbital] && bStrings[n][iOrbital])
                    {
                        tmpPair = std::make_pair(m, n);
                        iContainingDets.push_back(tmpPair);
                    }
                    if(aStrings[m][jOrbital] && bStrings[n][jOrbital])
                    {
                        tmpPair = std::make_pair(m, n);
                        jContainingDets.push_back(tmpPair);
                    }
                } // end loop over b strings
            } // end loop over a strings
            for(int l = 0; l < iContainingDets.size(); l++)
            {
                for(int k = 0; k < jContainingDets; k++)
                {
                    int Diff = AnnihilateAndCount(aStrings[iContainingDets[l].first], iOrbital, aStrings[jContainingDets[k].first], jOrbital)
                             + AnnihilateAndCount(bStrings[iContainingDets[l].second], jOrbital, bStrings[jContainingDets[k].second], jOrbital);
                    if(Diff > 0)
                    {
                        continue;
                    }
                    Dij += Eigenvector[iContainingDets[l].first + iContainingDets[l].second * aStrings.size(0)] * Eigenvector[jContainingDets[k].first + jContainingDets[k].second * aStrings.size(0)];
                }
            }
            DensityMatrix(i, j) = Dij;
        } // end loop over j
    } // end loop over i
    return DensityMatrix;
}

// This is here for debugging purposes.
void PrintBinaryStrings(std::vector< std::vector< bool > > Strings)
{
    for(int i = 0; i < Strings.size(); i++)
    {
        for(int j = 0; j < Strings[i].size(); j++)
        {
            std::cout << Strings[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void PauseHere()
{
    std::string tmpstring;
    std::getline(std::cin, tmpstring);
}

void ImpurityFCI(InputObj &Input, int FragmentIndex, Eigen::MatrixXd &RotationMatrix, double ChemicalPotential)
{
    int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();
    int NumVirt = Input.NumAO - NumAOImp - Input.NumOcc;
    int NumCore = Input.NumOcc - NumAOImp;

    /* There doesn't seem to be any method that chooses number of electrons */
    int aElectrons = Input.NumOcc; // 2 * NumAOImp + NumCore;
    int bElectrons = Input.NumOcc; // 2 * NumAOImp + NumCore;
    int aElectronsCAS = NumAOImp; // N_occ - N_core?????????????
    int bElectronsCAS = NumAOImp;
    int aOrbitals = Input.NumAO;
    int bOrbitals = Input.NumAO;
    int aCAS = 2 * NumAOImp + NumVirt;
    int bCAS = 2 * NumAOImp + NumVirt;
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

    std::vector< std::vector<bool> > aStrings;
    std::vector< std::vector<bool> > bStrings;

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
        for(int c = 0; c < NumCore; c++) // Insert the core orbitals into the string.
        {
            tmpVec.insert(tmpVec.begin() + Input.EnvironmentOrbitals[FragmentIndex][NumVirt + NumAOImp + c], true);
        }
        aStrings.push_back(tmpVec);
    }
    for(int i = 0; i < bDim; i++)
    {
        std::vector<bool> tmpVec;
        GetOrbitalString(i, bElectronsCAS, bCAS, tmpVec); // Get binary string for active space orbitals.
        for(int c = 0; c < NumCore; c++) // Insert the core orbitals into the string.
        {
            tmpVec.insert(tmpVec.begin() + Input.EnvironmentOrbitals[FragmentIndex][NumVirt + NumAOImp + c], true);
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
    Eigen::SparseMatrix<float, Eigen::RowMajor> Ham(Dim, Dim);
    // Ham.reserve(Eigen::VectorXi::Constant(Dim,NonzeroElements));
    // clock_t Timer = clock();
    double Timer = omp_get_wtime();

    typedef Eigen::Triplet<float> T;
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
    double NuclearEnergy = Input.Integrals["0 0 0 0"]; // Nuclear repulsion, will shift total energy and needs to be added to diagonal.
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
            tmpDoubleD += NuclearEnergy; // Nuclear potential.
            /* One electron operator */
            for(int ii = 0; ii < aOrbitalList[i].size(); ii++)
            {
                tmpDoubleD += OneElectronEmbedding(Input.Integrals, RotationMatrix, aOrbitalList[i][ii] - 1, aOrbitalList[i][ii] - 1);
            }
            for(int jj = 0; jj < bOrbitalList[j].size(); jj++)
            {
                tmpDoubleD += OneElectronEmbedding(Input.Integrals, RotationMatrix, bOrbitalList[j][jj] - 1, bOrbitalList[j][jj] - 1);
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
                    tmpDoubleD += TwoElectronIntegral(abOrbitalList[m], abOrbitalList[n], abOrbitalList[m], abOrbitalList[n], m_isAlpha, n_isAlpha, m_isAlpha, n_isAlpha, Input.Integrals, RotationMatrix);
                }
            }
            int NumSameImp = CountSameImpurity(aStrings[i], aStrings[i], Input.FragmentOrbitals[FragmentIndex]);
            NumSameImp = NumSameImp + CountSameImpurity(bStrings[j], bStrings[j], Input.FragmentOrbitals[FragmentIndex]); // This totals the number of impurity orbitals in the alpha and beta lists.
            tmpDoubleD = tmpDoubleD - ChemicalPotential * (double)NumSameImp; // Form of chemical potential matrix element.
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
        tmpDouble1 += OneElectronEmbedding(Input.Integrals, RotationMatrix, std::get<3>(aSingleDifference[i])[0] - 1, std::get<3>(aSingleDifference[i])[1] - 1); // Input.Integrals[std::to_string(std::get<3>(aSingleDifference[i])[0]) + " " + std::to_string(std::get<3>(aSingleDifference[i])[1]) + " 0 0"];
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
                tmpDouble2 += TwoElectronIntegral(std::get<3>(aSingleDifference[i])[0], KetOrbitalList[n], std::get<3>(aSingleDifference[i])[1], KetOrbitalList[n], true, n_isAlpha, true, n_isAlpha, Input.Integrals, RotationMatrix);
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
        tmpDouble1 += OneElectronEmbedding(Input.Integrals, RotationMatrix, std::get<3>(bSingleDifference[i])[0] - 1, std::get<3>(bSingleDifference[i])[1] - 1); // Input.Integrals[std::to_string(std::get<3>(bSingleDifference[i])[0]) + " " + std::to_string(std::get<3>(bSingleDifference[i])[1]) + " 0 0"];
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
                tmpDouble2 += TwoElectronIntegral(std::get<3>(bSingleDifference[i])[0], KetOrbitalList[n], std::get<3>(bSingleDifference[i])[1], KetOrbitalList[n], false, n_isAlpha, false, n_isAlpha, Input.Integrals, RotationMatrix);
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
            tmpDouble = TwoElectronIntegral(std::get<3>(aDoubleDifference[i])[0], std::get<3>(aDoubleDifference[i])[1], std::get<3>(aDoubleDifference[i])[2], std::get<3>(aDoubleDifference[i])[3], true, true, true, true, Input.Integrals, RotationMatrix);
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
            tmpDouble = TwoElectronIntegral(std::get<3>(bDoubleDifference[i])[0], std::get<3>(bDoubleDifference[i])[1], std::get<3>(bDoubleDifference[i])[2], std::get<3>(bDoubleDifference[i])[3], false, false, false, false, Input.Integrals, RotationMatrix);
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
            tmpDouble = TwoElectronIntegral(std::get<3>(aSingleDifference[i])[0], std::get<3>(bSingleDifference[j])[0], std::get<3>(aSingleDifference[i])[1], std::get<3>(bSingleDifference[j])[1], true, false, true, false, Input.Integrals, RotationMatrix);
            /* There is one alpha and one beta orbital mismatched between the bra and ket. According to the formula, we put the bra unique orbitals in first,
               which are the first two arguments, the first one alpha and the second one beta. Then the next two arguments are the unique orbitals
               of the ket. We know whether these electrons are alpha or beta. */

            if(fabs(tmpDouble) < MatTol) continue;
            Index1 = std::get<0>(aSingleDifference[i]) + aDim * std::get<0>(bSingleDifference[j]);
            Index2 = std::get<1>(aSingleDifference[i]) + aDim * std::get<1>(bSingleDifference[j]);
            // Note that the sign is the product of the signs of the alpha and beta strings. This is because we can permute them independently.
            // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
            tripletList_Private.push_back(T(Index1, Index2 , (float)std::get<2>(aSingleDifference[i]) * (float)std::get<2>(bSingleDifference[j]) * tmpDouble));
            tripletList_Private.push_back(T(Index2, Index1 , (float)std::get<2>(aSingleDifference[i]) * (float)std::get<2>(bSingleDifference[j]) * tmpDouble));
			
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
		    tripletList_Private.push_back(T(Index1, Index2 , (float)std::get<2>(aSingleDifference[i]) * (float)std::get<2>(bSingleDifference[j]) * tmpDouble));
            tripletList_Private.push_back(T(Index2, Index1 , (float)std::get<2>(aSingleDifference[i]) * (float)std::get<2>(bSingleDifference[j]) * tmpDouble));
        }
        #pragma omp critical
        tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
    }


    std::cout << "FCI: ...elements differing by two spin-orbitals completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
    // Output << "Elements differing by two spin-orbitals generated in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;

    // for(int Thread = 0; Thread < NumThreads; Thread++)
    // {
    //     tripletList.insert(tripletList.end(), tripletList_Private[Thread].begin(), tripletList_Private[Thread].end());
    //     std::vector<T>().swap(tripletList_Private[Thread]); // Free up memory.
    // }

    Ham.setFromTriplets(tripletList.begin(), tripletList.end());
    std::vector<T>().swap(tripletList); // This clears the memory, but I'm unsure if it does or doesn't increase the memory usage to do so

    std::cout << "FCI: Hamiltonian initialization took " << (omp_get_wtime() - Start) << " seconds." << std::endl;
    // Output << "\nHamiltonian initialization took " << (omp_get_wtime() - Start) << " seconds." << std::endl;

    /* Prints matrix, used for error checking on my part */
    // Eigen::MatrixXf HD = Ham;
    // std::ofstream PrintHam("printham.out");
    // PrintHam << HD << std::endl;
    // return 0;
    
    Timer = omp_get_wtime();
    std::cout << "FCI: Beginning Davidson Diagonalization... " << std::endl;
    std::vector< double > DavidsonEV;
    Davidson(Ham, Dim, NumberOfEV, L, DavidsonEV);
    std::cout << "FCI: ...done" << std::endl;
    std::cout << "FCI: Davidson Diagonalization took " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
    // Output << "\nDavidson Diagonalization took " << (omp_get_wtime() - Timer) << " seconds.\nThe eigenvalues are" << std::endl;
    for(int k = 0; k < NumberOfEV; k++)
    {
        // Output << "\n" << DavidsonEV[k];
    }

    /* This section is for direct diagonalization. Uncomment if desired. */
    Timer = omp_get_wtime();
    std::cout << "FCI: Beginning Direct Diagonalization... ";
    Eigen::MatrixXf HamDense = Ham;
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXf > HamEV;
    HamEV.compute(HamDense);

    std::cout << " done" << std::endl;
    std::cout << "FCI: Direct Diagonalization took " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
 //   Output << "\nDirect Diagonalization took " << (omp_get_wtime() - Timer) << " seconds.\nThe eigenvalues are" << std::endl;
    std::cout << "FCI: The eigenvaues are";
    for(int k = 0; k < NumberOfEV; k++)
    {
        std::cout << "\n" << HamEV.eigenvalues()[k];
//        Output << "\n" << HamEV.eigenvalues()[k];
    }

    /* This part is not needed */
    // Spectra::SparseGenMatProd<float> op(Ham);
    // Spectra::SymEigsSolver< float, Spectra::LARGEST_MAGN, Spectra::SparseGenMatProd<float> > HamEV(&op, NumberOfEV, Dim / 10);
    // HamEV.init();
    // int nconv = HamEV.compute();
    // std::cout << " done" << std::endl;
    // std::cout << "FCI: The eigenvalues are\n" << HamEV.eigenvalues() << std::endl;
    // std::cout << "FCI: Direct Diagonalization took " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
    // Output << "\nDirect Diagonalization took " << (omp_get_wtime() - Timer) << " seconds.\nThe eigenvalues are" << std::endl;
    // Output << HamEV.eigenvalues() << std::endl;
    
    std::cout << "\nFCI: Total running time: " << (omp_get_wtime() - Start) << " seconds." << std::endl;
    // Output << "\nTotal running time: " << (omp_get_wtime() - Start) << " seconds." << std::endl;

    // std::ofstream OutputHamiltonian(Input.OutputName + ".ham");
    // OutputHamiltonian << HamDense << std::endl;

    // return 0;
}