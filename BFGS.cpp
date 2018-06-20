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

#define H10

void GetCASPos(InputObj Input, int FragmentIndex, std::vector< int > &FragmentPos, std::vector< int > &BathPos);
// This is the full system SCF
double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF, Eigen::MatrixXd DMETPotential, Eigen::VectorXd &OrbitalEV);

/* Calculates the derivative of the low level density matrix with respect to potential matrix element u_rs */
// double CalcdDrs(int r, int s, Eigen::MatrixXd &Z, Eigen::MatrixXd &CoeffMatrix, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals)
// {
//     Eigen::VectorXd rComponentOcc(OccupiedOrbitals.size());
//     Eigen::VectorXd rComponentVir(VirtualOrbitals.size());
//     Eigen::VectorXd sComponentOcc(OccupiedOrbitals.size());
//     Eigen::VectorXd sComponentVir(VirtualOrbitals.size());
//     for(int i = 0; i < OccupiedOrbitals.size(); i++)
//     {
//         rComponentOcc[i] = CoeffMatrix.coeffRef(r, OccupiedOrbitals[i]);
//         sComponentOcc[i] = CoeffMatrix.coeffRef(s, OccupiedOrbitals[i]);
//     }
//     for(int a = 0; a < VirtualOrbitals.size(); a++)
//     {
//         rComponentVir[a] = CoeffMatrix.coeffRef(r, VirtualOrbitals[a]);
//         sComponentVir[a] = CoeffMatrix.coeffRef(s, VirtualOrbitals[a]);
//     }

//     double dDrs = (rComponentOcc.transpose() * Z.transpose() * sComponentVir + rComponentVir.transpose() * Z * sComponentOcc).sum();
//     return dDrs;
// }

// Assuming H1 has one nonzero element at u_kl and the rest are zero, the resulting matrix element of Z is the dot product of the
// kth row and the lth row of the coefficient matrix, divided by the difference in orbital eigenvalues.
// Eigen::MatrixXd CalcZMatrix(int k, int l, double ukl, Eigen::MatrixXd &CoeffMatrix, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, Eigen::VectorXd OrbitalEV)
// {
//     Eigen::MatrixXd Z(VirtualOrbitals.size(), OccupiedOrbitals.size());
//     if(fabs(ukl) < 1E-12) // Work-around because at zero potential, the gradient is strictly zero and no trial direction is generated.
//     {
//         ukl = 1E-6;
//     }
//     for(int a = 0; a < VirtualOrbitals.size(); a++)
//     {
//         for(int i = 0; i < OccupiedOrbitals.size(); i++)
//         {
//             Z(a, i) = CoeffMatrix.coeffRef(k, VirtualOrbitals[a]) * ukl * CoeffMatrix.coeffRef(l, OccupiedOrbitals[i]) / (OrbitalEV[VirtualOrbitals[a]] - OrbitalEV[OccupiedOrbitals[i]]);
//         }
//     }
//     return Z;
// }

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
// Eigen::VectorXd CalcRSGradient(int r, int s, std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, std::vector< std::vector< double > > PotentialElements, Eigen::MatrixXd &CoeffMatrix, Eigen::VectorXd OrbitalEV, InputObj Input, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals)
// {
//     int TotPos = CalcTotalPositions(PotentialPositions);
//     Eigen::VectorXd Gradient(TotPos);
//     int TotIndex = 0;
//     for(int x = 0; x < PotentialPositions.size(); x++)
//     {
//         for(int i = 0; i < PotentialPositions[x].size(); i++)
//         {
//             Eigen::MatrixXd Z = CalcZMatrix(PotentialPositions[x][i].first, PotentialPositions[x][i].second, PotentialElements[x][i], CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, OrbitalEV);
//             double dDrr = CalcdDrs(r, s, Z, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals);
//             Gradient[TotIndex] = dDrr;
//             TotIndex++;
//         }
//     }

//     return Gradient;
// }

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
        #ifdef H10
        if (x > 0)
        {
            for (int i = 0; i < PotentialElements[0].size(); i++)
            {
                PotentialElementsVec[PosIndex] = PotentialElements[0][i];
                if (i == PotentialElements[0].size() - 1)
                {
                    PotentialElementsVec[PosIndex] = PotentialElements[0][0];
                    PotentialElementsVec[PotentialElements[0].size() - 1] = PotentialElements[0][0];
                }
                PosIndex++;
            
            }
            continue;
        }
        #endif

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
        #ifdef H10
        if (x > 0)
        {
            for (int i = 0; i < PotentialElements[0].size(); i++)
            {
                PotentialElements[x][i] = PotentialElementsVec[PosIndex % PotentialElements[0].size()];
                if (i == PotentialElements[0].size() - 1)
                {
                    PotentialElements[x][i] = PotentialElementsVec[0];
                    PotentialElements[0][i] = PotentialElementsVec[0];
                }
                PosIndex++;
            }
            continue;
        }
        #endif
        for(int i = 0; i < PotentialElements[x].size(); i++)
        {
            PotentialElements[x][i] = PotentialElementsVec[PosIndex];
            PosIndex++;
        }
    }
}

void FormDMETPotential(Eigen::MatrixXd &DMETPotential, std::vector< std::vector< double > > PotentialElements, std::vector< std::vector< std::pair< int, int > > > PotentialPositions)
{
    DMETPotential = Eigen::MatrixXd::Zero(DMETPotential.rows(), DMETPotential.cols()); // Initialize to zero.
    for(int x = 0; x < PotentialPositions.size(); x++)
    {
        for(int i = 0; i < PotentialPositions[x].size(); i++)
        {
            DMETPotential(PotentialPositions[x][i].first, PotentialPositions[x][i].second) = PotentialElements[x][i];
            DMETPotential(PotentialPositions[x][i].second, PotentialPositions[x][i].first) = PotentialElements[x][i];
        }
    }
}

/* This is the numerical derivative of the matrix element, as opposed to the analytical
   linear response theory result that is coded above.
   This function increments the density matrix by one element of u and determines the change in each element
   Dij of the density matrix. This is stored into 
*/
// std::vector< std::vector < Eigen::VectorXd > > CalcGradD(InputObj &Input, std::vector< std::vector< double > > &PotentialElements, std::vector< std::vector< std::pair<int, int> > > &PotentialPositions, Eigen::MatrixXd &InitialDensity)
// {
// 	int TotPos = CalcTotalPositions(PotentialPositions);
// 	double du = 0.1; // to calculate [D(u + du) - D(u)] / du

// 	std::vector< std::vector < Eigen::VectorXd > > GradD(Input.NumAO); // Gives dDij/du_k by GradD[i][j][k]
// 	Eigen::VectorXd ZeroVec = Eigen::VectorXd::Zero(TotPos);
// 	for (int i = 0; i < Input.NumAO; i++) // Initialize GradD
// 	{
// 		for (int j = 0; j < Input.NumAO; j++)
// 		{
// 			GradD[i].push_back(ZeroVec);
// 		}
// 	}

// 	std::vector< std::vector< double > > PotElemPlusDU;
// 	int uComponent = 0; // We are taking the derivative with respect to this component of u
// 	for (int x = 0; x < Input.NumFragments; x++)
// 	{
// 		for (int i = 0; i < PotentialElements[x].size(); i++)
// 		{
// 			PotElemPlusDU = PotentialElements;
// 			PotElemPlusDU[x][i] += du; // add du to the element under consideration, symmetry is enforced in the below function.
// 			Eigen::MatrixXd DMETPotPlusDU = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
// 			FormDMETPotential(DMETPotPlusDU, PotElemPlusDU, PotentialPositions); // Make new u + du matrix.
// 			Eigen::MatrixXd DensityPlusDU = InitialDensity; // Will hold esulting D(u + du)
			
// 			// Now we do the full system SCF with the u + du potential. Some fillers need to be defined.
// 			std::vector< std::tuple < Eigen::MatrixXd, double, double > > EmptyBias;
// 			std::ofstream BlankOutput;
// 			std::vector< double > AllEnergies;
// 			Eigen::MatrixXd CoeffMatrix;
// 			int SCFCount = 0;
// 			Eigen::VectorXd OrbitalEV;
// 			// This redirects the std::cout buffer, so we don't have massive amounts of terminal output.
// 			std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
// 			std::cout.rdbuf(NULL); // sets to null
// 			SCF(EmptyBias, 1, DensityPlusDU, Input, BlankOutput, Input.SOrtho, Input.HCore, AllEnergies, CoeffMatrix, Input.OccupiedOrbitals, Input.VirtualOrbitals, SCFCount, -1, DMETPotPlusDU, OrbitalEV);
// 			std::cout.rdbuf(orig_buf); // restore buffer

// 			// Now we have D(u + du)
// 			// Calculate [D(u + du) - D(u)] / du
// 			Eigen::MatrixXd dDdu = DensityPlusDU - InitialDensity;
// 			dDdu = dDdu / du; // Element i, j of this matrix is the derivative of Dij with respect to u_k, k being the step in this loop we are on.

// 			// Now store it.
// 			for (int ii = 0; ii < dDdu.rows(); ii++)
// 			{
// 				for (int jj = 0; jj < dDdu.cols(); jj++)
// 				{
// 					GradD[ii][jj][uComponent] = dDdu.coeffRef(ii, jj);
// 				}
// 			}
//                         uComponent++; // Increment this index, since we are looping through the elements by fragment, but are storing it as "lined out" or full as I have been calling it.
// 		} // end loop over fragment orbitals
// 	} // end loop over fragments
	
// 	return GradD;
// }

double CalcL(InputObj &Input, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd > &FullDensities, std::vector< Eigen::MatrixXd > FragmentRotations, std::vector< int > BathStates, int CostFunctionVariant = 2)
{
    double L = 0.0;
    if (CostFunctionVariant == 1) // Match all DIAGONAL impurity elements.
    {
        for(int x = 0; x < Input.NumFragments; x++)
        {
            std::vector<int> FragPos, BathPos;
            GetCASPos(Input, x, FragPos, BathPos);
            for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
            {
                L += (FragmentDensities[x].coeffRef(FragPos[i], FragPos[i]) - FullDensities[BathStates[x]].coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][i]))
                   * (FragmentDensities[x].coeffRef(FragPos[i], FragPos[i]) - FullDensities[BathStates[x]].coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][i]));
            }
        }
    }
    if (CostFunctionVariant == 2) // Match all impurity elements.
    {
        for(int x = 0; x < Input.NumFragments; x++)
        {
            std::vector<int> FragPos, BathPos;
            GetCASPos(Input, x, FragPos, BathPos);
            for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
            {
                for(int j = 0; j < Input.FragmentOrbitals[x].size(); j++)
                {
                    // For each fragment, we match the impurity density with the bath density that gave us that impurity density.
                    L += (FragmentDensities[x].coeffRef(FragPos[i], FragPos[j]) - FullDensities[BathStates[x]].coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]))
                       * (FragmentDensities[x].coeffRef(FragPos[i], FragPos[j]) - FullDensities[BathStates[x]].coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]));
                }
            }
            //std::cout << "CalcL\nFrag\n" << FragmentDensities[x] << "\nFull\n" << FullDensities[BathStates[x]] << "\nL = " << L << std::endl;
        }
    }
    if (CostFunctionVariant == 3)
    {
        for(int x = 0; x < Input.NumFragments; x++)
        {
            std::vector<int> FragPos, BathPos;
            GetCASPos(Input, x, FragPos, BathPos);
            Eigen::MatrixXd RotatedFullDensity = FragmentRotations[x].transpose() * FullDensities[BathStates[x]] * FragmentRotations[x]; // We need to rotate the bath portion of the 1RDM.
            int NumVirt = Input.NumAO - Input.FragmentOrbitals[x].size() - Input.NumOcc;
            for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
            {
                for(int j = 0; j < Input.FragmentOrbitals[x].size(); j++)
                {
                    L += (FragmentDensities[x].coeffRef(FragPos[i], FragPos[j]) - RotatedFullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]))
                       * (FragmentDensities[x].coeffRef(FragPos[i], FragPos[j]) - RotatedFullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j]));
                    L += (FragmentDensities[x].coeffRef(BathPos[i], BathPos[j]) - RotatedFullDensity.coeffRef(Input.EnvironmentOrbitals[x][NumVirt + i], Input.EnvironmentOrbitals[x][NumVirt + j]))
                       * (FragmentDensities[x].coeffRef(BathPos[i], BathPos[j]) - RotatedFullDensity.coeffRef(Input.EnvironmentOrbitals[x][NumVirt + i], Input.EnvironmentOrbitals[x][NumVirt + j]));
                    L += 2 * (FragmentDensities[x].coeffRef(FragPos[i], BathPos[j]) - RotatedFullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.EnvironmentOrbitals[x][NumVirt + j]))
                           * (FragmentDensities[x].coeffRef(FragPos[i], BathPos[j]) - RotatedFullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.EnvironmentOrbitals[x][NumVirt + j])); 
                }
            }
        }
    }
    return L;
}

Eigen::VectorXd CalcGradL(InputObj &Input, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd > InitialDensities, std::vector< std::vector< double > > &PotentialElements, std::vector< std::vector< std::pair<int, int> > > &PotentialPositions, std::vector< Eigen::MatrixXd > &FragmentRotations, std::vector< int > BathStates, std::vector< std::vector< int > > OccupiedByState, std::vector< std::vector< int > > VirtualByState)
{
	int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd GradL = Eigen::VectorXd::Zero(TotPos);

    int NumSCFStates = *max_element(BathStates.begin(), BathStates.end());
    NumSCFStates++;

	double du = 1E-3; // to calculate [L(u + du) - L(u)] / du
    double L_Initial = CalcL(Input, FragmentDensities, InitialDensities, FragmentRotations, BathStates);

    std::vector< std::vector< double > > PotElemPlusDU;
    int uComponent = 0; // We are taking the derivative with respect to this component of u
    for (int x = 0; x < Input.NumFragments; x++)
	{
        #ifdef H10
        if (x > 0)
        {
            for (int i = 0; i < PotentialElements[0].size(); i++)
            {
                GradL(uComponent) = GradL(uComponent % PotentialElements[0].size());
                if (i == PotentialElements[0].size() - 1)
                {
                    GradL(uComponent) = GradL(0);
                    GradL(PotentialElements[0].size() - 1) = GradL(0);
                }
                uComponent++;
            }
            continue;
        }
        #endif
		for (int i = 0; i < PotentialElements[x].size(); i++)
		{
			PotElemPlusDU = PotentialElements;
			PotElemPlusDU[x][i] += du; // add du to the element under consideration, symmetry is enforced in the below function.
			Eigen::MatrixXd DMETPotPlusDU = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
			FormDMETPotential(DMETPotPlusDU, PotElemPlusDU, PotentialPositions); // Make new u + du matrix.
			std::vector< Eigen::MatrixXd > DensityPlusDU = InitialDensities; // Will hold resulting D(u + du), holds multiple states
			
			// Now we do the full system SCF with the u + du potential. Some fillers need to be defined.
			std::vector< std::tuple < Eigen::MatrixXd, double, double > > EmptyBias;
			std::ofstream BlankOutput;
			std::vector< double > AllEnergies;
			Eigen::MatrixXd CoeffMatrix;
			int SCFCount = 0;
			Eigen::VectorXd OrbitalEV;
            std::vector< int > OccupiedOrbitals;
            std::vector< int > VirtualOrbitals;
            for(int i = 0; i < Input.NumOcc; i++)
            {
                OccupiedOrbitals.push_back(i);
            }
            for(int i = Input.NumOcc; i < Input.NumAO; i++)
            {
                VirtualOrbitals.push_back(i);
            }
			// This redirects the std::cout buffer, so we don't  have massive amounts of terminal output.
			// std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
			// std::cout.rdbuf(NULL); // sets to null
            // Eigen::MatrixXd tmpDensity = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
            for (int a = 0; a < NumSCFStates; a++) // Find RHF solutons after du step for all SCF states desired. We pick which ones to match later.
            {
                Eigen::MatrixXd tmpDensity = 0.5 * InitialDensities[a];
                std::vector< double > EmptyAllEnergies;
                std::cout << "Dinit\n" << InitialDensities[a] << std::endl;
                std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
			    std::cout.rdbuf(NULL); // sets to null
			    double SCFEnergy = SCF(EmptyBias, 1, tmpDensity, Input, BlankOutput, Input.SOrtho, Input.HCore, EmptyAllEnergies, CoeffMatrix, OccupiedByState[a], VirtualByState[a], SCFCount, -1, DMETPotPlusDU, OrbitalEV);
                // std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(tmpDensity, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
                // Bias.push_back(tmpTuple);
                // std::cout << tmpDensity << std::endl;
                std::cout.rdbuf(orig_buf); // restore buffer
                DensityPlusDU[a] = 2 * tmpDensity;
                std::cout << "E = " << SCFEnergy << std::endl;
                std::cout << "D:\n" <<  2 * tmpDensity << std::endl;
            }
            // std::cout.rdbuf(orig_buf); // restore buffer
            // std::string tmpstring;
            // std::getline(std::cin, tmpstring);

			// Now we have D(u + du)
			// Calculate [L(u + du) - L(u)] / du
			double dL = CalcL(Input, FragmentDensities, DensityPlusDU, FragmentRotations, BathStates);
            std::cout << "L(u+du) = " << dL << "\nL(u) = " << L_Initial << std::endl;
            dL = (dL - L_Initial) / du;
            std::cout << "dLdu = " << dL << std::endl;

			// Now store it.
            GradL(uComponent) = dL;
            uComponent++; // Increment this index, since we are looping through the elements by fragment, but are storing it as "lined out" or full as I have been calling it.
		} // end loop over fragment orbitals
	} // end loop over fragments

    return GradL;
}

// Eigen::MatrixXd CalcHessL(InputObj &Input, std::vector< Eigen::MatrixXd > FragmentDensities, Eigen::MatrixXd InitialDensity, std::vector< std::vector< double > > &PotentialElements, std::vector< std::vector< std::pair<int, int> > > &PotentialPositions, std::vector< Eigen::MatrixXd > &FragmentRotations)
// {
// 	int TotPos = CalcTotalPositions(PotentialPositions);
//     Eigen::MatrixXd HessL = Eigen::MatrixXd::Zero(TotPos, TotPos);

// 	double du = 1E-3;
//     double L_Initial = CalcL(Input, FragmentDensities, InitialDensity, FragmentRotations);

//     std::vector< std::vector< double > > PotElemPlusDU1;
//     std::vector< std::vector< double > > PotElemPlusDU2;
//     std::vector< std::vector< double > > PotElemPlusDU1DU2;
//     int uComponent1 = 0; // We are taking the derivative with respect to this component of u
    
//     for (int x1 = 0; x1 < Input.NumFragments; x1++)
// 	{
// 		for (int i1 = 0; i1 < PotentialElements[x1].size(); i1++)
// 		{
//             PotElemPlusDU1 = PotentialElements;
//             PotElemPlusDU1[x1][i1] += du;
//             Eigen::MatrixXd DMETPotPlusDU1 = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);

//             FormDMETPotential(DMETPotPlusDU1, PotElemPlusDU1, PotentialPositions); // Make new u + du1 matrix.
//             Eigen::MatrixXd DensityPlusDU1 = InitialDensity; // Will hold resulting D(u + du1)

//             // Now we do the full system SCF with the u + du potential. Some fillers need to be defined.
//             std::vector< std::tuple < Eigen::MatrixXd, double, double > > EmptyBias;
//             std::ofstream BlankOutput;
//             std::vector< double > AllEnergies;
//             Eigen::MatrixXd CoeffMatrix;
//             int SCFCount = 0;
//             Eigen::VectorXd OrbitalEV;
//             // This redirects the std::cout buffer, so we don't have massive amounts of terminal output.
//             std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
//             std::cout.rdbuf(NULL); // sets to null
//             SCF(EmptyBias, 1, DensityPlusDU1, Input, BlankOutput, Input.SOrtho, Input.HCore, AllEnergies, CoeffMatrix, Input.OccupiedOrbitals, Input.VirtualOrbitals, SCFCount, -1, DMETPotPlusDU1, OrbitalEV);
//             std::cout.rdbuf(orig_buf); // restore buffer

//             DensityPlusDU1 = 2 * DensityPlusDU1;

//             int uComponent2 = 0;

//             for (int x2 = 0; x2 < Input.NumFragments; x2++)
//             {
//                 for (int i2 = 0; i2 < PotentialElements[x2].size(); i2++)
//                 {
//                     PotElemPlusDU2 = PotentialElements;
//                     PotElemPlusDU1DU2 = PotentialElements;

//                     PotElemPlusDU2[x2][i2] += du; // add du to the element under consideration, symmetry is enforced in the below function.
//                     PotElemPlusDU1DU2[x1][i1] += du;
//                     PotElemPlusDU1DU2[x2][i2] += du;

//                     Eigen::MatrixXd DMETPotPlusDU2 = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
//                     Eigen::MatrixXd DMETPotPlusDU1DU2 = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);

//                     FormDMETPotential(DMETPotPlusDU2, PotElemPlusDU2, PotentialPositions); // Make new u + du2 matrix.
//                     FormDMETPotential(DMETPotPlusDU1DU2, PotElemPlusDU1DU2, PotentialPositions); // Make new u + du1 + du2 matrix.
//                     Eigen::MatrixXd DensityPlusDU2 = InitialDensity;
//                     Eigen::MatrixXd DensityPlusDU1DU2 = InitialDensity;
                    
//                     // Now we do the full system SCF with the u + du potential. Some fillers need to be defined.
//                     std::vector< std::tuple < Eigen::MatrixXd, double, double > > EmptyBias;
//                     std::ofstream BlankOutput;
//                     std::vector< double > AllEnergies2;
//                     std::vector< double > AllEnergies3;
//                     Eigen::MatrixXd CoeffMatrix;
//                     int SCFCount = 0;
//                     Eigen::VectorXd OrbitalEV;
//                     std::cout.rdbuf(NULL); // sets to null
//                     SCF(EmptyBias, 1, DensityPlusDU2, Input, BlankOutput, Input.SOrtho, Input.HCore, AllEnergies2, CoeffMatrix, Input.OccupiedOrbitals, Input.VirtualOrbitals, SCFCount, -1, DMETPotPlusDU2, OrbitalEV);
//                     SCF(EmptyBias, 1, DensityPlusDU1DU2, Input, BlankOutput, Input.SOrtho, Input.HCore, AllEnergies3, CoeffMatrix, Input.OccupiedOrbitals, Input.VirtualOrbitals, SCFCount, -1, DMETPotPlusDU1DU2, OrbitalEV);
//                     std::cout.rdbuf(orig_buf); // restore buffer
                    
//                     DensityPlusDU2 = 2 * DensityPlusDU2;
//                     DensityPlusDU1DU2 = 2 * DensityPlusDU1DU2;

//                     // Now we have D(u + du)
//                     // Calculate [L(u + du) - L(u)] / du
//                     double ddL = (CalcL(Input, FragmentDensities, DensityPlusDU1DU2, FragmentRotations) - CalcL(Input, FragmentDensities, DensityPlusDU1, FragmentRotations)
//                                 - CalcL(Input, FragmentDensities, DensityPlusDU2, FragmentRotations) + L_Initial) / (du * du);
//                     // Now store it.
//                     HessL(uComponent1, uComponent2) = ddL;
//                     uComponent2++; // Increment this index, since we are looping through the elements by fragment, but are storing it as "lined out" or full as I have been calling it.
//                 }
//             }
//             uComponent1++;       
// 		} // end loop over fragment orbitals
// 	} // end loop over fragments

//     return HessL;
// }

// Returns the full gradient of the cost function.
// del (sum_x sum_ij (D_ij^x - D_ij)^2 = sum_x sum_ij [2 * (D_ij^x - D_ij) * del D_ij]
// Eigen::VectorXd CalcGradCF(InputObj &Input, std::vector< std::vector< std::pair< int, int > > > &PotentialPositions, std::vector< std::vector< double > > &PotentialElements, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV, std::vector< int > OccupiedOrbitals, std::vector< int > VirtualOrbitals, std::vector< Eigen::MatrixXd > FragmentDensities, Eigen::MatrixXd FullDensity)
// {
//     int TotPos = CalcTotalPositions(PotentialPositions);
//     Eigen::VectorXd GradCF = Eigen::VectorXd::Zero(TotPos);
// 	std::vector< std::vector < Eigen::VectorXd > > GradD = CalcGradD(Input, PotentialElements, PotentialPositions, FullDensity);

//     for(int x = 0; x < Input.NumFragments; x++)
//     {
//         std::vector<int> FragPos, BathPos;
//         GetCASPos(Input, x, FragPos, BathPos);
//         for(int i = 0; i < Input.FragmentOrbitals[x].size(); i++)
//         {
//             for(int j = 0; j < Input.FragmentOrbitals[x].size(); j++)
//             {
//                 // Eigen::VectorXd GradDij = CalcRSGradient(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j], PotentialPositions, PotentialElements, CoeffMatrix, OrbitalEV, Input, OccupiedOrbitals, VirtualOrbitals);
// 				GradCF += 2 * (FragmentDensities[x].coeffRef(FragPos[i], FragPos[j]) - FullDensity.coeffRef(Input.FragmentOrbitals[x][i], Input.FragmentOrbitals[x][j])) * GradD[Input.FragmentOrbitals[x][i]][Input.FragmentOrbitals[x][j]]; // *  GradDij;
//             }
//         }
//     }
//     return GradCF;
// }

double doLineSearch(InputObj &Input, std::vector< Eigen::MatrixXd > &FragmentDensities, std::vector< Eigen::MatrixXd > &FullDensities, std::vector< std::vector< double > > PotentialElements, std::vector< std::vector < std::pair< int, int > > > PotentialPositions, Eigen::VectorXd p, Eigen::MatrixXd DMETPotential, std::vector< Eigen::MatrixXd > &FragmentRotations, std::vector< int > BathStates, std::vector< std::vector< int > > OccupiedByState, std::vector< std::vector< int > > VirtualByState)
{
    double a = 0.0; // Size of line step.
    double da = 1E-1; // We will increment a by this much until a loose minimum is found

    int NumSCFStates = *max_element(BathStates.begin(), BathStates.end());
    NumSCFStates++;

    std::vector< std::vector< double > > PotElemDirection = PotentialElements;
    FullUVectorToFragUVector(PotElemDirection, p);
    Eigen::MatrixXd pMatrix = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO); // This is the BFGS step direction, in DMET potential matrix form.
    FormDMETPotential(pMatrix, PotElemDirection, PotentialPositions);

    Eigen::MatrixXd IncrementedDMETPot = DMETPotential + a * pMatrix;
    double LInit;
    double LNext;

    std::vector< Eigen::MatrixXd > DNext = FullDensities;

    // Do initial SCF
    std::vector< std::tuple < Eigen::MatrixXd, double, double > > EmptyBias;
	std::ofstream BlankOutput;
	std::vector< double > AllEnergies;
    Eigen::MatrixXd CoeffMatrix;
	int SCFCount = 0;
	Eigen::VectorXd OrbitalEV;
    std::vector< int > OccupiedOrbitals;
    std::vector< int > VirtualOrbitals;
    for(int i = 0; i < Input.NumOcc; i++)
    {
        OccupiedOrbitals.push_back(i);
    }
    for(int i = Input.NumOcc; i < Input.NumAO; i++)
    {
        VirtualOrbitals.push_back(i);
    }
	// This redirects the std::cout buffer, so we don't have massive amounts of terminal output.
	std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
	std::cout.rdbuf(NULL); // sets to null
    for (int a = 0; a < NumSCFStates; a++)
    {
        Eigen::MatrixXd tmpDensity = 0.5 * DNext[a];
        std::vector< double > EmptyAllEnergies;
        SCF(EmptyBias, 1, tmpDensity, Input, BlankOutput, Input.SOrtho, Input.HCore, EmptyAllEnergies, CoeffMatrix, OccupiedByState[a], VirtualByState[a], SCFCount, -1, IncrementedDMETPot, OrbitalEV);
        // std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(tmpDensity, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
        // Bias.push_back(tmpTuple);
        DNext[a] = 2 * tmpDensity;
    }
	std::cout.rdbuf(orig_buf); // restore buffer
    
    LInit = CalcL(Input, FragmentDensities, DNext, FragmentRotations, BathStates);
    LNext = LInit;
    std::cout << "Linesearch: " << a << "\t" << LNext << std::endl;
    do // while we're decreasing L along the step direction
    {
        LInit = LNext;
        a += da;
        IncrementedDMETPot = DMETPotential + a * pMatrix;
        // std::cout.rdbuf(NULL); // sets to null
        // std::vector< std::tuple < Eigen::MatrixXd, double, double > > Bias1;
	    std::vector< double > AllEnergies1;
        std::vector< int > OccupiedOrbitals1;
        std::vector< int > VirtualOrbitals1;
        for(int i = 0; i < Input.NumOcc; i++)
        {
            OccupiedOrbitals1.push_back(i);
        }
        for(int i = Input.NumOcc; i < Input.NumAO; i++)
        {
            VirtualOrbitals1.push_back(i);
        }
        for (int a = 0; a < NumSCFStates; a++)
        {
            Eigen::MatrixXd tmpDensity1 = 0.5 * DNext[a];
            std::vector< double > EmptyAllEnergies1;
            std::cout.rdbuf(NULL); // sets to null
            std::vector< std::tuple < Eigen::MatrixXd, double, double > > Bias1;
            SCF(EmptyBias, 1, tmpDensity1, Input, BlankOutput, Input.SOrtho, Input.HCore, EmptyAllEnergies1, CoeffMatrix, OccupiedByState[a], VirtualByState[a], SCFCount, -1, IncrementedDMETPot, OrbitalEV);
            // std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(tmpDensity1, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
            // Bias1.push_back(tmpTuple);
            std::cout.rdbuf(orig_buf);
            std::cout << "DNext\n" << tmpDensity1 << std::endl;
            DNext[a] = 2 * tmpDensity1;
        }
        // std::cout.rdbuf(orig_buf); // restore buffer
        
        LNext = CalcL(Input, FragmentDensities, DNext, FragmentRotations, BathStates);
        std::cout << "Linesearch: " << a << "\t" << LNext << std::endl;
    } while(LInit - LNext > 1E-10); // while it is decreasing rapidly

    // std::cout << "DMET: Cost function = " << LNext << std::endl;

    return a - da / 2;
}

// Follows the Wikipedia article's notation. https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
void BFGS_1(Eigen::MatrixXd &Hessian, Eigen::VectorXd &s, Eigen::VectorXd Gradient, Eigen::VectorXd &x, InputObj &Input, std::vector< Eigen::MatrixXd > &FragmentDensities, std::vector< Eigen::MatrixXd > FullDensities, std::vector< std::vector< double > > PotentialElements, std::vector< std::vector < std::pair< int, int > > > PotentialPositions, Eigen::MatrixXd DMETPotential, std::vector< Eigen::MatrixXd > &FragmentRotations, std::vector< int > BathStates, std::vector< std::vector< int > > OccupiedByState, std::vector< std::vector< int > > VirtualByState)
{
    Eigen::VectorXd p;
    p = -1 * Gradient; 
    // p = Hessian.colPivHouseholderQr().solve(-1 * Gradient);
    std::cout << "DMET: Commencing line search." << std::endl;
    double a = 0.1; // doLineSearch(Input, FragmentDensities, FullDensities, PotentialElements, PotentialPositions, p, DMETPotential, FragmentRotations, BathStates, OccupiedByState, VirtualByState);
    s = a * p;
    std::cout << "s\n" << s << std::endl;
    x = x + s;
    std::cout << "DMET: Line search complete." << std::endl;
}

void BFGS_2(Eigen::MatrixXd &Hessian, Eigen::VectorXd &s, Eigen::VectorXd Gradient, Eigen::VectorXd GradientPrev, Eigen::VectorXd &x)
{
    Eigen::VectorXd y = Gradient - GradientPrev;
    // Hessian = Hessian + (y * y.transpose()) / (y.dot(s)) - (Hessian * s * s.transpose() * Hessian) / (s.transpose() * Hessian * s);
}

void UpdatePotential(Eigen::MatrixXd &DMETPotential, InputObj &Input, Eigen::MatrixXd CoeffMatrix, Eigen::VectorXd OrbitalEV, std::vector< std::vector< int > > OccupiedByState, std::vector< std::vector< int > > VirtualByState, std::vector< Eigen::MatrixXd > FragmentDensities, std::vector< Eigen::MatrixXd> &FullDensities, std::ofstream &Output, std::vector< Eigen::MatrixXd > &FragmentRotations, std::vector< int > ImpurityStates, std::vector< int > BathStates)
{
    // First vector holds the positions of the nonzero elements, the second holds the values at each of those posisions.
    // They are divided by fragment, to make some of the code neater.
    std::vector< std::vector< std::pair< int, int > > > PotentialPositions;
    std::vector< std::vector< double > > PotentialElements;
	// Now, we turn the DMET potential into a vector, by taking the DMET potential and lining up all potentially nonzero elements
	// into the previous vectors. PotentialPositions holds the position of each element whereas PotentialElements holds the value
	// of these elements from the DMET potential.
    SetUVector(PotentialPositions, PotentialElements, DMETPotential, Input);

    int NumSCFStates = *max_element(BathStates.begin(), BathStates.end());
    NumSCFStates++;

	// DEBUGGING
    // std::cout << "\nPositions - Value\n";
    // for(int i = 0; i < PotentialPositions.size(); i++)PotentialElements
    // {
    //     for(int j = 0; j < PotentialPositions[i].size(); j++)
    //     {
    //         std::cout << PotentialPositions[i][j].first << "\t" << PotentialPositions[i][j].second << "\t" << PotentialElements[i][j] << std::endl;
    //         Output << PotentialPositions[i][j].first << "\t" << PotentialPositions[i][j].second << "\t" << PotentialElements[i][j] << std::endl;
    //     }
    // }
    // std::cout << "Coeff\n" << CoeffMatrix << std::endl;
    // std::cout << "OrbEV\n" << OrbitalEV << std::endl;

    Eigen::VectorXd GradCF = CalcGradL(Input, FragmentDensities, FullDensities, PotentialElements, PotentialPositions, FragmentRotations, BathStates, OccupiedByState, VirtualByState);
    // CalcGradCF(Input, PotentialPositions, PotentialElements, CoeffMatrix, OrbitalEV, OccupiedOrbitals, VirtualOrbitals, FragmentDensities, FullDensity);

	// NormOfGrad measures the norm of the gradient, and we finish when this is sufficiently small.
    double NormOfGrad = 100.0;
    double L = CalcL(Input, FragmentDensities, FullDensities, FragmentRotations, BathStates);
    double L_Initial;
    double dL = 100.0;
    int TotPos = CalcTotalPositions(PotentialPositions);
    Eigen::VectorXd PotentialElementsVec = FragUVectorToFullUVector(PotentialElements, TotPos); // Line up every element into one neat vector.
    // Eigen::MatrixXd Hessian = CalcHessL(Input, FragmentDensities, FullDensity, PotentialElements, PotentialPositions, FragmentRotations);
    Eigen::MatrixXd Hessian = Eigen::MatrixXd::Identity(TotPos, TotPos);
    Eigen::VectorXd PrevGrad;
    Eigen::VectorXd s;
    while(fabs(NormOfGrad) > 1E-3)// && fabs(dL) > 1E-1)
    {
        // Solves Hp = -GradL and moves along direction p.
        BFGS_1(Hessian, s, GradCF, PotentialElementsVec, Input, FragmentDensities, FullDensities, PotentialElements, PotentialPositions, DMETPotential, FragmentRotations, BathStates, OccupiedByState, VirtualByState);
        PrevGrad = GradCF;
        FullUVectorToFragUVector(PotentialElements, PotentialElementsVec);

        std::cout << "DMET Potential:\n" << PotentialElementsVec << std::endl;

        // Calculate new L and GradL at the new value of u determined above. We do this out here because it is needed to make the next Hessian.
        // First, make the new DMET potential using new u values.
        FormDMETPotential(DMETPotential, PotentialElements, PotentialPositions);
        
        // Then calculate L at the new u.
        std::vector< std::tuple < Eigen::MatrixXd, double, double > > EmptyBias;
		std::ofstream BlankOutput;
		std::vector< double > AllEnergies;
		Eigen::MatrixXd CoeffMatrix;
		int SCFCount = 0;
		Eigen::VectorXd OrbitalEV;
        std::vector< int > OccupiedOrbitals;
        std::vector< int > VirtualOrbitals;
        for(int i = 0; i < Input.NumOcc; i++)
        {
            OccupiedOrbitals.push_back(i);
        }
        for(int i = Input.NumOcc; i < Input.NumAO; i++)
        {
            VirtualOrbitals.push_back(i);
        }
		// This redirects the std::cout buffer, so we don't have massive amounts of terminal output.
		std::streambuf* orig_buf = std::cout.rdbuf(); // holds original buffer
		std::cout.rdbuf(NULL); // sets to null
        for (int a = 0; a < NumSCFStates; a++)
        {
            Eigen::MatrixXd tmpDensity = 0.5 * FullDensities[a];
            std::vector< double > EmptyAllEnergies;
		    SCF(EmptyBias, 1, tmpDensity, Input, BlankOutput, Input.SOrtho, Input.HCore, EmptyAllEnergies, CoeffMatrix, OccupiedByState[a], VirtualByState[a], SCFCount, -1, DMETPotential, OrbitalEV);
            // std::tuple< Eigen::MatrixXd, double, double > tmpTuple = std::make_tuple(tmpDensity, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
            // Bias.push_back(tmpTuple);
            FullDensities[a] = 2 * tmpDensity;
        }
		std::cout.rdbuf(orig_buf); // restore buffer

        // Then use this new density matrix to calculate GradL
        GradCF = CalcGradL(Input, FragmentDensities, FullDensities, PotentialElements, PotentialPositions, FragmentRotations, BathStates, OccupiedByState, VirtualByState);
        L_Initial = L;
        L = CalcL(Input, FragmentDensities, FullDensities, FragmentRotations, BathStates);
        dL = fabs(L - L_Initial);

        // Forms Hessian for next iteration.
        // BFGS_2(Hessian, s, GradCF, PrevGrad, PotentialElementsVec);

        NormOfGrad = GradCF.squaredNorm(); // (GradCF - PrevGrad).squaredNorm();
        std::cout << "DMET: Norm of gradient = " << NormOfGrad << std::endl;
        std::cout << "DMET: L = " << L << std::endl;
        if (fabs(L) < 1E-2)
        {
            break;
        }
    }
    FormDMETPotential(DMETPotential, PotentialElements, PotentialPositions);
}
