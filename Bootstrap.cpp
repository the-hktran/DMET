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
#include "Bootstrap.h"
#include "Functions.h"
#include "NewtonRaphson.h"
#include "FCI.h"
#include "Fragmenting.h"

// This is a debug function for the time being
//void Bootstrap::debugInit(InputObj Inp, std::ofstream &OutStream)
//{
//	Output = &OutStream;
//
//	NumAO = Inp.NumAO;
//	NumOcc = Inp.NumOcc;
//	NumFrag = Inp.FragmentOrbitals.size();
//	FragmentOrbitals = Inp.FragmentOrbitals;
//	EnvironmentOrbitals = Inp.EnvironmentOrbitals;
//	Input = Inp;
//
//	for (int x = 0; x < NumFrag; x++)
//	{
//		std::vector< std::tuple <int , int, double > > FragmentBEPotential;
//		FragmentBEPotential.push_back(std::tuple< int, int, double >((x + 1) % NumFrag, (x + 2) % NumFrag, 0.0));
//		FragmentBEPotential.push_back(std::tuple< int, int, double >((x + NumFrag - 1) % NumFrag, (x + NumFrag) % NumFrag, 0.0));
//		BEPotential.push_back(FragmentBEPotential);
//
//		std::vector< int >  CenterPos;
//		if (x == NumFrag - 2)
//		{
//			CenterPos.push_back(2);
//		}
//		else if (x == NumFrag - 1)
//		{
//			CenterPos.push_back(0);
//		}
//		else
//		{
//			CenterPos.push_back(1);
//		}
//		// CenterPos.push_back(1);
//		BECenterPosition.push_back(CenterPos);
//	}
//
//	FragState = std::vector<int>(NumFrag);
//	BathState = std::vector<int>(NumFrag);
//	// H10
//	FragState[0] = 3;
//	FragState[1] = 3;
//	FragState[2] = 3;
//	FragState[3] = 3;
//	FragState[4] = 3;
//	FragState[5] = 3;
//	FragState[6] = 3;
//	FragState[7] = 3;
//	FragState[8] = 3;
//	FragState[9] = 3;
//
//	BathState[0] = 1;
//	BathState[1] = 1;
//	BathState[2] = 1;
//	BathState[3] = 1;
//	BathState[4] = 1;
//	BathState[5] = 1;
//	BathState[6] = 1;
//	BathState[7] = 1;
//	BathState[8] = 1;
//	BathState[9] = 1;
//
//	isTS = true;
//
//	// Will be important to initialize this immediately when BE is implemented.
//	for (int x = 0; x < NumFrag; x++)
//	{
//		NumConditions += BEPotential[x].size();
//		NumFragCond.push_back(BEPotential[x].size());
//	}
//}

void Bootstrap::InitFromFragmenting(Fragmenting Frag)
{
	NumFrag = Frag.MatchingConditions.size();
	NumFragCond.clear();
	BEPotential.clear();
	NumConditions = 0;
	for (int x = 0; x < Frag.MatchingConditions.size(); x++)
	{
		NumFragCond.push_back(Frag.MatchingConditions[x].size());
		NumConditions += NumFragCond[x];
		std::vector< std::tuple<int, int, int, int, int, double, bool, bool> > tmpVec;
		for (int k = 0; k < Frag.MatchingConditions[x].size(); k++)
		{
			tmpVec.push_back(std::make_tuple(std::get<0>(Frag.MatchingConditions[x][k]), std::get<1>(Frag.MatchingConditions[x][k]), std::get<2>(Frag.MatchingConditions[x][k]), std::get<3>(Frag.MatchingConditions[x][k]),
			std::get<4>(Frag.MatchingConditions[x][k]), 0.0, std::get<5>(Frag.MatchingConditions[x][k]), std::get<6>(Frag.MatchingConditions[x][k])));
		}
		BEPotential.push_back(tmpVec);
	}

	aBECenterPosition = bBECenterPosition = Frag.CenterPosition;
}

void Bootstrap::PrintOneRDMs(std::vector< std::vector<Eigen::MatrixXd> > OneRDMs)
{
	for (int x = 0; x < OneRDMs.size(); x++)
	{
		*Output << "BE-DMET: One RDM for Fragment " << x << " is\n" << OneRDMs[x][FragState[x]] << std::endl;
	}
}

int Bootstrap::OrbitalToReducedIndex(int Orbital, int FragIndex, bool Alpha)
{
	int RedIdx;
	if (Alpha)
	{
		for (int i = 0; i < aFragPos[FragIndex].size(); i++)
		{
			int Pos = aFragPos[FragIndex][i];
			int Orb = ReducedIndexToOrbital(Pos, Input, FragIndex, Alpha);
			if (Orb == Orbital) RedIdx = Pos;
		}

	}
	else
	{
		for (int i = 0; i < bFragPos[FragIndex].size(); i++)
		{
			int Pos = bFragPos[FragIndex][i];
			int Orb = ReducedIndexToOrbital(Pos, Input, FragIndex, Alpha);
			if (Orb == Orbital) RedIdx = Pos;
		}
	}
	return RedIdx;
}

double Bootstrap::CalcCostChemPot(std::vector< std::vector<Eigen::MatrixXd> > Frag1RDMs, std::vector< std::vector< int > > BECenter, std::vector<int> FragSt, InputObj &Inp)
{
    double CF = 0;
    for(int x = 0; x < Frag1RDMs.size(); x++) // sum over fragments
    {
		if (x > 0 && isTS)
		{
			CF *= NumFrag;
			break;
		}
        std::vector< int > FragPos;
        std::vector< int > BathPos;
        GetCASPos(Inp, x , FragPos, BathPos);
        for(int i = 0; i < BECenter[x].size(); i++) // sum over diagonal matrix elements belonging to the fragment orbitals.
        {
            CF += Frag1RDMs[x][FragSt[x]](FragPos[BECenter[x][i]], FragPos[BECenter[x][i]]);
        }
    }
    CF -= Inp.NumElectrons;
    CF = CF * CF;
    return CF;
}

double Bootstrap::CalcCostChemPot(std::vector<Eigen::MatrixXd> aFrag1RDMs, std::vector<Eigen::MatrixXd> bFrag1RDMs, std::vector< std::vector< int > > aBECenter, std::vector< std::vector< int > > bBECenter, std::vector<int> FragSt)
{
    double CF = 0;
	
    for(int x = 0; x < aFrag1RDMs.size(); x++) // sum over fragments
    {
		if (x > 0 && isTS)
		{
			CF *= NumFrag;
			break;
		}

        for (int i = 0; i < aBECenter[x].size(); i++) // sum over diagonal matrix elements belonging to the fragment orbitals.
        {
			int CenterIdx = OrbitalToReducedIndex(aBECenter[x][i], x, true);
            CF += aFrag1RDMs[FragSt[x]].coeffRef(CenterIdx, CenterIdx);
        }
		for (int i = 0; i < bBECenter[x].size(); i++)
		{
			int CenterIdx = OrbitalToReducedIndex(bBECenter[x][i], x, false);
            CF += bFrag1RDMs[FragSt[x]].coeffRef(CenterIdx, CenterIdx);
		}
    }
    CF -= Input.NumElectrons;
    CF = CF * CF;
    return CF;
}

std::vector<double> Bootstrap::CalcCostLambda(std::vector<Eigen::MatrixXd> aOneRDMRef, std::vector<Eigen::MatrixXd> bOneRDMRef, std::vector< std::vector<double> > aaTwoRDMRef, std::vector< std::vector<double> > abTwoRDMRef, std::vector< std::vector<double> > bbTwoRDMRef,
                                 Eigen::MatrixXd aOneRDMIter, Eigen::MatrixXd bOneRDMIter, std::vector<double> aaTwoRDMIter, std::vector<double> abTwoRDMIter, std::vector<double> bbTwoRDMIter, int FragmentIndex)
{
	std::vector<double> Loss;
	for (int i = 0; i < BEPotential[FragmentIndex].size(); i++)
	{
		double PRef, PIter;

		bool isOEI = false;
		if (std::get<3>(BEPotential[FragmentIndex][i]) == -1) isOEI = true;
		if (isOEI)
		{
			int Ind1Ref = OrbitalToReducedIndex(std::get<1>(BEPotential[FragmentIndex][i]), std::get<0>(BEPotential[FragmentIndex][i]), std::get<6>(BEPotential[FragmentIndex][i]));
			int Ind2Ref = OrbitalToReducedIndex(std::get<2>(BEPotential[FragmentIndex][i]), std::get<0>(BEPotential[FragmentIndex][i]), std::get<6>(BEPotential[FragmentIndex][i]));

			int Ind1Iter = OrbitalToReducedIndex(std::get<1>(BEPotential[FragmentIndex][i]), FragmentIndex, std::get<6>(BEPotential[FragmentIndex][i]));
			int Ind2Iter = OrbitalToReducedIndex(std::get<2>(BEPotential[FragmentIndex][i]), FragmentIndex, std::get<6>(BEPotential[FragmentIndex][i]));

			if (std::get<6>(BEPotential[FragmentIndex][i]))
			{
				PRef = aOneRDMRef[std::get<0>(BEPotential[FragmentIndex][i])].coeffRef(Ind1Ref, Ind2Ref);
				PIter = aOneRDMIter.coeffRef(Ind1Iter, Ind2Iter);
			}
			else
			{
				PRef = bOneRDMRef[std::get<0>(BEPotential[FragmentIndex][i])].coeffRef(Ind1Ref, Ind2Ref);
				PIter = bOneRDMIter.coeffRef(Ind1Iter, Ind2Iter);
			}
		}
		else
		{
			int Ind1Ref = OrbitalToReducedIndex(std::get<1>(BEPotential[FragmentIndex][i]), std::get<0>(BEPotential[FragmentIndex][i]), std::get<6>(BEPotential[FragmentIndex][i]));
			int Ind2Ref = OrbitalToReducedIndex(std::get<2>(BEPotential[FragmentIndex][i]), std::get<0>(BEPotential[FragmentIndex][i]), std::get<6>(BEPotential[FragmentIndex][i]));
			int Ind3Ref = OrbitalToReducedIndex(std::get<3>(BEPotential[FragmentIndex][i]), std::get<0>(BEPotential[FragmentIndex][i]), std::get<7>(BEPotential[FragmentIndex][i]));
			int Ind4Ref = OrbitalToReducedIndex(std::get<4>(BEPotential[FragmentIndex][i]), std::get<0>(BEPotential[FragmentIndex][i]), std::get<7>(BEPotential[FragmentIndex][i]));

			int Ind1Iter = OrbitalToReducedIndex(std::get<1>(BEPotential[FragmentIndex][i]), FragmentIndex, std::get<6>(BEPotential[FragmentIndex][i]));
			int Ind2Iter = OrbitalToReducedIndex(std::get<2>(BEPotential[FragmentIndex][i]), FragmentIndex, std::get<6>(BEPotential[FragmentIndex][i]));
			int Ind3Iter = OrbitalToReducedIndex(std::get<3>(BEPotential[FragmentIndex][i]), FragmentIndex, std::get<7>(BEPotential[FragmentIndex][i]));
			int Ind4Iter = OrbitalToReducedIndex(std::get<4>(BEPotential[FragmentIndex][i]), FragmentIndex, std::get<7>(BEPotential[FragmentIndex][i]));

			if (std::get<6>(BEPotential[FragmentIndex][i]) && std::get<7>(BEPotential[FragmentIndex][i]))
			{
				int NRef = aOneRDMRef[std::get<0>(BEPotential[FragmentIndex][i])].rows();
				PRef = aaTwoRDMRef[std::get<0>(BEPotential[FragmentIndex][i])][Ind1Ref * NRef * NRef * NRef + Ind2Ref * NRef * NRef + Ind3Ref * NRef + Ind4Ref];
				int NIter = aOneRDMIter.rows();
				PIter = aaTwoRDMIter[Ind1Iter * NIter * NIter * NIter + Ind2Iter * NIter * NIter + Ind3Iter * NIter + Ind4Iter];
			}
			else if (!std::get<6>(BEPotential[FragmentIndex][i]) && !std::get<7>(BEPotential[FragmentIndex][i]))
			{
				int NRef = bOneRDMRef[std::get<0>(BEPotential[FragmentIndex][i])].rows();
				PRef = bbTwoRDMRef[std::get<0>(BEPotential[FragmentIndex][i])][Ind1Ref * NRef * NRef * NRef + Ind2Ref * NRef * NRef + Ind3Ref * NRef + Ind4Ref];
				int NIter = bOneRDMIter.rows();
				PIter = bbTwoRDMIter[Ind1Iter * NIter * NIter * NIter + Ind2Iter * NIter * NIter + Ind3Iter * NIter + Ind4Iter];
			}
			else
			{
				int aNRef = aOneRDMRef[std::get<0>(BEPotential[FragmentIndex][i])].rows();
				int bNRef = bOneRDMRef[std::get<0>(BEPotential[FragmentIndex][i])].rows();
				PRef = abTwoRDMRef[std::get<0>(BEPotential[FragmentIndex][i])][Ind1Ref * aNRef * bNRef * bNRef + Ind2Ref * bNRef * bNRef + Ind3Ref * bNRef + Ind4Ref];
				int aNIter = aOneRDMIter.rows();
				int bNIter = bOneRDMIter.rows();
				PIter = bbTwoRDMIter[Ind1Iter * aNIter * bNIter * bNIter + Ind2Iter * bNIter * bNIter + Ind3Iter * bNIter + Ind4Iter];
			}
		}
		Loss.push_back((PRef - PIter) * (PRef - PIter));
	}
	return Loss;
}

void Bootstrap::CollectRDM(std::vector< Eigen::MatrixXd > &aOneRDMs, std::vector< Eigen::MatrixXd > &bOneRDMs, std::vector< std::vector<double> > &aaTwoRDMs, std::vector< std::vector<double> > &abTwoRDMs, std::vector< std::vector<double> > &bbTwoRDMs,
                           std::vector< std::vector< std::tuple< int, int, int, int, int, double, bool, bool > > > BEPot, double Mu)
{
	for (int x = 0; x < NumFrag; x++)
	{
		if (x > 0 && isTS)
		{
			aOneRDMs.push_back(aOneRDMs[0]);
			bOneRDMs.push_back(bOneRDMs[0]);
			continue;
		}

		FCI xFCI = FCIsBase[x];
		xFCI.AddChemicalPotentialGKLC(aFragPos[x], bFragPos[x], Mu);
		for (int i = 0; i < BEPot[x].size(); i++)
		{
			bool OEIPotential = false;
			if (std::get<3>(BEPot[x][i]) == -1) OEIPotential = true;

			if (OEIPotential)
			{
				int Ind1 = OrbitalToReducedIndex(std::get<1>(BEPot[x][i]), x, std::get<6>(BEPot[x][i]));
				int Ind2 = OrbitalToReducedIndex(std::get<2>(BEPot[x][i]), x, std::get<6>(BEPot[x][i]));

				xFCI.AddPotential(Ind1, Ind2, std::get<5>(BEPot[x][i]), std::get<6>(BEPot[x][i]));
			}
			else
			{
				int Ind1 = OrbitalToReducedIndex(std::get<1>(BEPot[x][i]), x, std::get<6>(BEPot[x][i]));
				int Ind2 = OrbitalToReducedIndex(std::get<2>(BEPot[x][i]), x, std::get<6>(BEPot[x][i]));
				int Ind3 = OrbitalToReducedIndex(std::get<3>(BEPot[x][i]), x, std::get<7>(BEPot[x][i]));
				int Ind4 = OrbitalToReducedIndex(std::get<4>(BEPot[x][i]), x, std::get<7>(BEPot[x][i]));

				xFCI.AddPotential(Ind1, Ind2, Ind3, Ind4, std::get<5>(BEPot[x][i]), std::get<6>(BEPot[x][i]), std::get<7>(BEPot[x][i]));
			}
		}
		xFCI.runFCI();
		xFCI.getSpecificRDM(FragState[x], true);
		aOneRDMs.push_back(xFCI.aOneRDMs[FragState[x]]);
		bOneRDMs.push_back(xFCI.bOneRDMs[FragState[x]]);
		aaTwoRDMs.push_back(xFCI.aaTwoRDMs[FragState[x]]);
		abTwoRDMs.push_back(xFCI.abTwoRDMs[FragState[x]]);
		bbTwoRDMs.push_back(xFCI.bbTwoRDMs[FragState[x]]);
	}
}

void Bootstrap::UpdateFCIs()
{
	FCIs.clear();
	for (int x = 0; x < NumFrag; x++)
	{
		FCI xFCI(FCIsBase[x]); // First, reset FCI
		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			bool isOEI = (std::get<3>(BEPotential[x][i]) == -1);
			if (isOEI)
			{
				int Ind1 = OrbitalToReducedIndex(std::get<1>(BEPotential[x][i]), x, std::get<6>(BEPotential[x][i]));
				int Ind2 = OrbitalToReducedIndex(std::get<2>(BEPotential[x][i]), x, std::get<6>(BEPotential[x][i]));

				xFCI.AddPotential(Ind1, Ind2, std::get<5>(BEPotential[x][i]), std::get<6>(BEPotential[x][i]));
			}
			else
			{
				int Ind1 = OrbitalToReducedIndex(std::get<1>(BEPotential[x][i]), x, std::get<6>(BEPotential[x][i]));
				int Ind2 = OrbitalToReducedIndex(std::get<2>(BEPotential[x][i]), x, std::get<6>(BEPotential[x][i]));
				int Ind3 = OrbitalToReducedIndex(std::get<3>(BEPotential[x][i]), x, std::get<7>(BEPotential[x][i]));
				int Ind4 = OrbitalToReducedIndex(std::get<4>(BEPotential[x][i]), x, std::get<7>(BEPotential[x][i]));

				xFCI.AddPotential(Ind1, Ind2, Ind3, Ind4, std::get<5>(BEPotential[x][i]), std::get<6>(BEPotential[x][i]), std::get<7>(BEPotential[x][i]));
			}
			xFCI.AddChemicalPotentialGKLC(aFragPos[x], bFragPos[x], ChemicalPotential);
		}
		xFCI.runFCI();
		xFCI.getSpecificRDM(FragState[x], true);
		FCIs.push_back(xFCI);
	}
}

std::vector< double > Bootstrap::FragmentLoss(std::vector< std::vector<Eigen::MatrixXd> > DensityReference, std::vector<Eigen::MatrixXd> IterDensity, int FragmentIndex)
{
	std::vector< double > AllLosses;
	for (int MatchedOrbital = 0; MatchedOrbital < BEPotential[FragmentIndex].size(); MatchedOrbital++)
	{
		std::vector< int > FragPosImp, BathPosImp, FragPosBath, BathPosBath;
		GetCASPos(Input, FragmentIndex, FragPosImp, BathPosImp);
		GetCASPos(Input, std::get<0>(BEPotential[FragmentIndex][MatchedOrbital]), FragPosBath, BathPosBath);

		int PElementImp = 0;
		for (int i = 0; i < Input.FragmentOrbitals[FragmentIndex].size(); i++)
		{
			if (Input.FragmentOrbitals[FragmentIndex][i] == std::get<1>(BEPotential[FragmentIndex][MatchedOrbital]))
			{
				break;
			}
			PElementImp++;
		}
		int PElementBath = 0;
		for (int i = 0; i < Input.FragmentOrbitals[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])].size(); i++)
		{
			if (Input.FragmentOrbitals[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])][i] == std::get<1>(BEPotential[FragmentIndex][MatchedOrbital]))
			{
				break;
			}
			PElementBath++;
		}
		double Loss = DensityReference[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])][FragState[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])]].coeffRef(FragPosBath[PElementBath], FragPosBath[PElementBath]) 
			- IterDensity[FragState[FragmentIndex]].coeffRef(FragPosImp[PElementImp], FragPosImp[PElementImp]);
		Loss = Loss * Loss;
		AllLosses.push_back(Loss);
	}
	return AllLosses;
}

Eigen::MatrixXd Bootstrap::CalcJacobian(Eigen::VectorXd &f)
{
	std::vector<Eigen::MatrixXd> aOneRDMs, bOneRDMs;
	std::vector< std::vector<double> > aaTwoRDMs, abTwoRDMs, bbTwoRDMs, tmpVecVecDouble;
	for (int x = 0; x < NumFrag; x++)
	{
		aOneRDMs.push_back(FCIs[x].aOneRDMs[FragState[x]]);
		bOneRDMs.push_back(FCIs[x].bOneRDMs[FragState[x]]);
		aaTwoRDMs.push_back(FCIs[x].aaTwoRDMs[FragState[x]]);
		abTwoRDMs.push_back(FCIs[x].abTwoRDMs[FragState[x]]);
		bbTwoRDMs.push_back(FCIs[x].bbTwoRDMs[FragState[x]]);
		std::cout << "Orig 1RDM of Fragment " << x << std::endl;
		std::cout << aOneRDMs[x] << std::endl;
	}

	double LMu = CalcCostChemPot(aOneRDMs, bOneRDMs, aBECenterPosition, bBECenterPosition, FragState);
	std::vector<Eigen::MatrixXd> aOneRDMsPlusdMu, bOneRDMsPlusdMu;
	std::cout << "a" << std::endl;
	CollectRDM(aOneRDMsPlusdMu, bOneRDMsPlusdMu, tmpVecVecDouble, tmpVecVecDouble, tmpVecVecDouble, BEPotential, ChemicalPotential + dMu);
	std::cout << "b" << std::endl;
	double LMuPlus = CalcCostChemPot(aOneRDMsPlusdMu, bOneRDMsPlusdMu, aBECenterPosition, bBECenterPosition, FragState);
	// We need every density matrix to calculation LMu, but since we only calculate one fragment each loop, we will use this vector
	// to store the different contributions to LMu per fragment, and then when we add the BE potential, we only affect one fragment
	// so we'll use the difference between LMu in that fragment to figure out the new LMu.
	std::vector<double> LMuByFrag;
	for (int x = 0; x < NumFrag; x++)
	{
		std::vector<Eigen::MatrixXd> xaOneRDM, xbOneRDM;
		std::vector< std::vector< int > > aSingleCenter, bSingleCenter;
		std::vector<int> SingleFragSt;
		xaOneRDM.push_back(aOneRDMs[x]);
		xbOneRDM.push_back(bOneRDMs[x]);
		aSingleCenter.push_back(aBECenterPosition[x]);
		bSingleCenter.push_back(bBECenterPosition[x]);
		SingleFragSt.push_back(FragState[x]);
		double LMuX = CalcCostChemPot(xaOneRDM, xbOneRDM, aSingleCenter, bSingleCenter, FragState);
		LMuByFrag.push_back(LMuX);
	}

	f = Eigen::VectorXd::Zero(NumConditions + 1);
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(NumConditions + 1, NumConditions + 1);

	f[NumConditions] = LMu;
	J(NumConditions, NumConditions) = (LMuPlus - LMu) / dMu;

	// Jacobian is 
	// [ df1/dx1, ..., df1/dxn ]
	// [ ..................... ]
	// [ dfn/dx1, ..., dfn/dxn ]
	// fi are the squared losses at the different matched orbitals
	// xi are the different lambdas used to match each orbital

	int JCol = 0;
	int fCount = 0;
	for (int x = 0; x < NumFrag; x++)
	{
		if (x > 0 && isTS)
		{
			for (int xTS = 1; xTS < NumFrag; xTS++)
			{
				int JRow = 0;
				for (int j = 0; j < xTS; j++)
				{
					JRow += NumFragCond[j];
				}
				for (int iTS = 0; iTS < BEPotential[xTS].size(); iTS++)
				{
					for (int j = 0; j < BEPotential[xTS].size(); j++)
					{
						J(JRow + j, JCol) = J.coeffRef(j, JCol % BEPotential[xTS].size());
					}
					J(J.rows() - 1, JCol) = J.coeffRef(J.rows() - 1, JCol % BEPotential[xTS].size());
					JCol++;
				}
				for (int j = 0; j < BEPotential[xTS].size(); j++)
				{
					J(JRow + j, J.cols() - 1) = J.coeffRef(j, J.cols() - 1);
				}
				for (int j = 0; j < BEPotential[xTS].size(); j++)
				{
					f[fCount] = f[j];
					fCount++;
				}
			}
			break;
		}

		// auto BEMinusdLambda = BEPotential;
		// // Collect all the density matrices for this iteration.
		// Eigen::MatrixXd FragOneRDMMinusdLambda;
		// BEImpurityFCI(FragOneRDMMinusdLambda, Input, x, RotationMatrices[x], ChemicalPotential, State, BEMinusdLambda[x]);
		// std::vector<double> LossesMinus = FragmentLoss(OneRDMs, FragOneRDMMinusdLambda, x);
		std::vector<double> LossesMinus = CalcCostLambda(aOneRDMs, bOneRDMs, aaTwoRDMs, abTwoRDMs, bbTwoRDMs, aOneRDMs[x], bOneRDMs[x], aaTwoRDMs[x], abTwoRDMs[x], bbTwoRDMs[x], x);

		int JRow = 0;
		for (int j = 0; j < x; j++)
		{
			JRow += NumFragCond[j];
		}

		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			bool isOEI = (std::get<3>(BEPotential[x][i]) == -1);
			// Make the + dLambda potential for the fragment.
			auto BEPlusdLambda = BEPotential;
			std::get<5>(BEPlusdLambda[x][i]) = std::get<5>(BEPotential[x][i]) + dLambda;

			// Collect all the density matrices for this iteration.
			FCI xFCI(FCIsBase[x]);
			if (isOEI)
			{
				int Ind1 = OrbitalToReducedIndex(std::get<1>(BEPlusdLambda[x][i]), x, std::get<6>(BEPlusdLambda[x][i]));
				int Ind2 = OrbitalToReducedIndex(std::get<2>(BEPlusdLambda[x][i]), x, std::get<6>(BEPlusdLambda[x][i]));

				xFCI.AddPotential(Ind1, Ind2, std::get<5>(BEPlusdLambda[x][i]), std::get<6>(BEPlusdLambda[x][i]));
				xFCI.PrintERI(false);
				std::cout << "potential added " << Ind1 << " " << Ind2 << " " << std::get<5>(BEPlusdLambda[x][i]) << std::endl;
			}
			else
			{
				int Ind1 = OrbitalToReducedIndex(std::get<1>(BEPlusdLambda[x][i]), x, std::get<6>(BEPlusdLambda[x][i]));
				int Ind2 = OrbitalToReducedIndex(std::get<2>(BEPlusdLambda[x][i]), x, std::get<6>(BEPlusdLambda[x][i]));
				int Ind3 = OrbitalToReducedIndex(std::get<3>(BEPlusdLambda[x][i]), x, std::get<7>(BEPlusdLambda[x][i]));
				int Ind4 = OrbitalToReducedIndex(std::get<4>(BEPlusdLambda[x][i]), x, std::get<7>(BEPlusdLambda[x][i]));

				xFCI.AddPotential(Ind1, Ind2, Ind3, Ind4, std::get<5>(BEPlusdLambda[x][i]), std::get<6>(BEPlusdLambda[x][i]), std::get<7>(BEPlusdLambda[x][i]));
			}
			xFCI.runFCI();
			xFCI.getSpecificRDM(FragState[x], !isOEI);
			std::vector<double> LossesPlus;
			if (isOEI)
			{
				std::vector<double> EmptyRDM;
				LossesPlus = CalcCostLambda(aOneRDMs, bOneRDMs, aaTwoRDMs, abTwoRDMs, bbTwoRDMs, xFCI.aOneRDMs[FragState[x]], xFCI.bOneRDMs[FragState[x]], EmptyRDM, EmptyRDM, EmptyRDM, x);
				std::cout << x << " " << i << "\n" << xFCI.aOneRDMs[FragState[x]] << std::endl;
			}
			else
			{
				Eigen::MatrixXd EmptyRDM;
				LossesPlus = CalcCostLambda(aOneRDMs, bOneRDMs, aaTwoRDMs, abTwoRDMs, bbTwoRDMs, EmptyRDM, EmptyRDM, xFCI.aaTwoRDMs[FragState[x]], xFCI.abTwoRDMs[FragState[x]], xFCI.bbTwoRDMs[FragState[x]], x);
			}
			std::string tmpstring;
			std::getline(std::cin, tmpstring);

			// // Make the - dLambda potential for the fragment.
			// auto BEMinusdLambda = BEPotential;
			// std::get<2>(BEMinusdLambda[x][i]) = std::get<2>(BEPotential[x][i]) - dLambda;

			// // Collect all the density matrices for this iteration.
			// Eigen::MatrixXd FragOneRDMMinusdLambda;
			// BEImpurityFCI(FragOneRDMMinusdLambda, Input, x, RotationMatrices[x], ChemicalPotential, State, BEMinusdLambda[x]);
			// std::vector<double> LossesMinus = FragmentLoss(OneRDMs[x], FragOneRDMMinusdLambda, x);
			
			// Fill in J
			for (int j = 0; j < LossesPlus.size(); j++)
			{
				J(JRow + j, JCol) = (LossesPlus[j] - LossesMinus[j]) / (dLambda);
			}

			// Add in chemical potential portion.
			std::vector<Eigen::MatrixXd> aSingleRDMPlus, bSingleRDMPlus;
			std::vector< std::vector< int > > aSingleBECenter, bSingleBECenter;
			aSingleRDMPlus.push_back(xFCI.aOneRDMs[FragState[x]]);
			bSingleRDMPlus.push_back(xFCI.bOneRDMs[FragState[x]]);
			aSingleBECenter.push_back(aBECenterPosition[x]);
			bSingleBECenter.push_back(bBECenterPosition[x]);
			double LMuPlus = CalcCostChemPot(aSingleRDMPlus, bSingleRDMPlus, aSingleBECenter, bSingleBECenter, FragState);
			// std::cout << "Mu Loss\n" << LMuPlus << "\t" << LMuByFrag[x] << std::endl;
			J(J.rows() - 1, JCol) = (LMuPlus - LMuByFrag[x]) / dLambda;

			JCol++;
		}

		// Last column is derivative of each loss with respect to chemical potential.
		// The last element of this column is already handled.
		std::vector<double> Empty2RDM;
		std::vector<double> LossesPlusMu = CalcCostLambda(aOneRDMs, bOneRDMs, aaTwoRDMs, abTwoRDMs, bbTwoRDMs, aOneRDMsPlusdMu[x], bOneRDMsPlusdMu[x], Empty2RDM, Empty2RDM, Empty2RDM, x);
		for (int j = 0; j < LossesPlusMu.size(); j++)
		{
			J(JRow + j, NumConditions) = (LossesPlusMu[j] - LossesMinus[j]) / dMu;
		}

		// Fill in f
		// The chemical potential loss is already filled in.
		for (int j = 0; j < LossesMinus.size(); j++)
		{
			f[fCount] = LossesMinus[j];
			fCount++;
		}
	}
	return J;
}

void Bootstrap::CollectSchmidt(std::vector<Eigen::MatrixXd> MFDensity, std::ofstream &Output)
{
	for (int x = 0; x < NumFrag; x++)
	{
		Eigen::MatrixXd RotMat = Eigen::MatrixXd::Zero(NumAO, NumAO);
		int NumEnvVirt = NumAO - NumOcc - FragmentOrbitals.size();
		SchmidtDecomposition(MFDensity[BathState[x]], RotMat, FragmentOrbitals[x], EnvironmentOrbitals[x], NumEnvVirt, Output);
		RotationMatrices.push_back(RotMat);
	}
}

void Bootstrap::CollectSchmidt(std::vector<Eigen::MatrixXd> aMFDensity, std::vector<Eigen::MatrixXd> bMFDensity, std::ofstream &Output)
{
	for (int x = 0; x < NumFrag; x++)
	{
		Eigen::MatrixXd aRotMat = Eigen::MatrixXd::Zero(NumAO, NumAO);
		Eigen::MatrixXd bRotMat = Eigen::MatrixXd::Zero(NumAO, NumAO);
		int aNumEnvVirt = NumAO - Input.aNumElectrons - FragmentOrbitals.size();
		int bNumEnvVirt = NumAO - Input.bNumElectrons - FragmentOrbitals.size();
		SchmidtDecomposition(aMFDensity[BathState[x]], aRotMat, FragmentOrbitals[x], EnvironmentOrbitals[x], aNumEnvVirt, Output);
		SchmidtDecomposition(bMFDensity[BathState[x]], bRotMat, FragmentOrbitals[x], EnvironmentOrbitals[x], bNumEnvVirt, Output);
		aRotationMatrices.push_back(aRotMat);
		bRotationMatrices.push_back(bRotMat);
	}
}

void Bootstrap::CollectInputs()
{
	for (int x = 0; x < NumFrag; x++)
	{
		InputObj FragInput = Input;
		FragInput.Integrals = RotateIntegrals(Input.Integrals, RotationMatrices[x]);
		Inputs.push_back(FragInput);
	}
}

void Bootstrap::VectorToBE(Eigen::VectorXd X)
{
	int xCount = 0;
	for (int x = 0; x < NumFrag; x++)
	{
		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			std::get<5>(BEPotential[x][i]) = X[xCount];
			xCount++;
		}
	}
	ChemicalPotential = X[X.size() - 1];
}

Eigen::VectorXd Bootstrap::BEToVector()
{
	Eigen::VectorXd X = Eigen::VectorXd::Zero(NumConditions + 1);
	int xCount = 0;
	for (int x = 0; x < NumFrag; x++)
	{
		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			X[xCount] = std::get<5>(BEPotential[x][i]);
			xCount++;
		}
	}
	X[X.size() - 1] = ChemicalPotential;
	return X;
}

//void Bootstrap::OptMu()
//{
//	// We will do Newton Raphson to optimize the chemical potential.
//	// First, intialize Mu to Chemical Potential and calculate intial loss.
//	double dMu = 1E-4;
//	double Mu = ChemicalPotential;
//	std::vector< std::vector< Eigen::MatrixXd > > Densities = CollectRDM(BEPotential, Mu, State);
//	double Loss = CalcCostChemPot(Densities, BECenterPosition, Input);
//
//	std::cout << "BE: Optimizing chemical potential." << std::endl;
//	// Do Newton's method until convergence
//	while (fabs(Loss) > 1e-6)
//	{
//		// Calculate Loss at Mu + dMu
//		Densities = CollectRDM(BEPotential, Mu + dMu, State);
//		double LossPlus = CalcCostChemPot(Densities, BECenterPosition, Input);
//		double dLdMu = (LossPlus - Loss) / dMu;
//
//		Mu = Mu - Loss / dLdMu;
//
//		// Update Loss at new, updated Mu
//		Densities = CollectRDM(BEPotential, Mu, State);
//		Loss = CalcCostChemPot(Densities, BECenterPosition, Input);
//
//		std::cout << "BE: Iteration yielded mu = " << Mu << " and Loss = " << Loss << std::endl;
//	}
//	std::cout << "BE: Chemical potential converged." << std::endl;
//	// After everything is done, store Mu into ChemicalPotential for the rest of the object.
//	ChemicalPotential = Mu;
//}

void Bootstrap::NewtonRaphson()
{
	std::cout << "BE-DMET: Beginning initialization for site potential optimization." << std::endl;
	// *Output << "BE-DMET: Beginning initialization for site potential optimization." << std::endl;
	// First, vectorize Lambdas
	Eigen::VectorXd x = BEToVector();

	// Initialize J and f
	Eigen::VectorXd f;
	Eigen::MatrixXd J = CalcJacobian(f);

	std::cout << "BE-DMET: Optimizing site potential." << std::endl;
	// *Output << "BE-DMET: Optimizing site potential." << std::endl;

	int NRIteration = 1;

	while (f.squaredNorm() > 1e-8)
	{
		std::cout << "BE-DMET: -- Running Newton-Raphson iteration " << NRIteration << "." << std::endl;
		// *Output << "BE-DMET: -- Running Newton-Raphson iteration " << NRIteration << "." << std::endl; 
		x = x - J.inverse() * f;
		VectorToBE(x); // Updates the BEPotential for the J and f update next.
		UpdateFCIs(); // Inputs potentials into the FCI that varies.
		J = CalcJacobian(f); // Update here to check the loss.
		// std::cout << "BE-DMET: Site potential obtained\n" << x << "\nBE-DMET: with loss \n" << f << std::endl;
		std::cout << "BE-DMET: Site potential obtained\n" << x << "\nBE-DMET: with loss \n" << f.squaredNorm() << std::endl;
		// *Output << "BE-DMET: Site potential obtained\n" << x << "\nBE-DMET: with loss \n" << f << std::endl;
		NRIteration++;
	}
}

void Bootstrap::doBootstrap(InputObj &Input, std::vector<Eigen::MatrixXd> &MFDensity, std::ofstream &Output)
{
	FragmentOrbitals = Input.FragmentOrbitals;
	EnvironmentOrbitals = Input.EnvironmentOrbitals;
	NumAO = Input.NumAO;
	NumOcc = Input.NumOcc;
	NumFrag = Input.NumFragments;

	// This will hold all of our impurity densities, calculated from using the BE potential.
	std::vector< Eigen::MatrixXd > ImpurityDensities(NumFrag);
	std::vector< Eigen::Tensor<double, 4> > Impurity2RDM(NumFrag);
	std::vector< Eigen::VectorXd > ImpurityEigenstates(NumFrag);

	CollectSchmidt(MFDensity, Output); // Runs a function to collect all rotational matrices in corresponding to each fragment.

	// Now that we have the rotation matrices, we can iterate through each fragment and get the impurity densities.
	for (int x = 0; x < NumFrag; x++)
	{
		ImpurityFCI(ImpurityDensities[x], Input, x, RotationMatrices[x], ChemicalPotential, State, Impurity2RDM[x], ImpurityEigenstates[x]);
	}

	// Now we iterate through each unique BE potential element and solve for the Lambda potential in each case.
	// In each case, we create a BENewton object that handles the Newton Raphson optimization.
}

void Bootstrap::doBootstrap(InputObj &Inp, std::vector<Eigen::MatrixXd> &aMFDensity, std::vector<Eigen::MatrixXd> &bMFDensity, std::ofstream &Output)
{
	FragmentOrbitals = Inp.FragmentOrbitals;
	EnvironmentOrbitals = Inp.EnvironmentOrbitals;
	NumAO = Inp.NumAO;
	NumOcc = Inp.NumOcc;
	NumFrag = Inp.NumFragments;
	FragState = Inp.ImpurityStates;
	BathState = Inp.BathStates;
	BEInputName = Inp.BEInput;
	Input = Inp;

	aFragPos.clear();
	bFragPos.clear();
	aBathPos.clear();
	bBathPos.clear();

	// This will hold all of our impurity densities, calculated from using the BE potential.
	std::vector< Eigen::MatrixXd > ImpurityDensities(NumFrag);
	std::vector< Eigen::Tensor<double, 4> > Impurity2RDM(NumFrag);
	std::vector< Eigen::VectorXd > ImpurityEigenstates(NumFrag);

	CollectSchmidt(aMFDensity, bMFDensity, Output); // Runs a function to collect all rotational matrices in corresponding to each fragment.

	FCIs.clear();
	// Generate impurity FCI objects for each impurity.
	for (int x = 0; x < NumFrag; x++)
	{
		std::vector<int> xaFragPos, xaBathPos, xbFragPos, xbBathPos;
		GetCASPos(Input, x, xaFragPos, xaBathPos, true);
		GetCASPos(Input, x, xbFragPos, xbBathPos, false);
		aFragPos.push_back(xaFragPos);
		bFragPos.push_back(xbFragPos);
		aBathPos.push_back(xaBathPos);
		bBathPos.push_back(xbBathPos);

		std::vector<int> aActiveList, aVirtualList, aCoreList, bActiveList, bVirtualList, bCoreList;
        GetCASList(Input, x, aActiveList, aCoreList, aVirtualList, true);
        GetCASList(Input, x, bActiveList, bCoreList, bVirtualList, false);

		FCI xFCI(Input, Input.FragmentOrbitals[x].size(), Input.FragmentOrbitals[x].size(), aCoreList, aActiveList, aVirtualList, bCoreList, bActiveList, bVirtualList);
		xFCI.ERIMapToArray(Input.Integrals, aRotationMatrices[x], bRotationMatrices[x], aActiveList, bActiveList);
		xFCI.runFCI();
		xFCI.getSpecificRDM(FragState[x], true);
		FCIs.push_back(xFCI);
	}
	for (int x = 0; x < FCIs.size(); x++)
	{
		FCI xFCI(FCIs[x]);
		FCIsBase.push_back(xFCI);
	}

	NewtonRaphson();
	BEEnergy = CalcBEEnergy();
}

void Bootstrap::printDebug(std::ofstream &Output)
{
	for (int x = 0; x < RotationMatrices.size(); x++)
	{
		Output << RotationMatrices[x] << std::endl;
		Output << "\n";
	}
}

// This calculates the BE energy assuming chemical potential and BE potential have both been optimized.
double Bootstrap::CalcBEEnergy()
{
	double Energy = 0;
	// Loop through each fragment to calculate the energy of each.
	std::cout << "BE-DMET: Calculating DMET Energy..." << std::endl;
	// *Output << "BE-DMET: Calculating DMET Energy..." << std::endl;
	double FragEnergy;
	for (int x = 0; x < NumFrag; x++)
	{
		if (x > 0 && isTS)
		{
			Energy += FragEnergy;
			continue;
		}
		
		std::vector<Eigen::MatrixXd> OneRDM;
		FragEnergy = FCIs[x].calcImpurityEnergy(FragState[x], aBECenterPosition[x], bBECenterPosition[x]);
		Energy += FragEnergy;
		std::cout << "BE-DMET: -- Energy of Fragment " << x << " is " << FragEnergy << std::endl;
		// *Output << "BE-DMET: -- Energy of Fragment " << x << " is " << FragEnergy << std::endl;
	}
	Energy += Input.Integrals["0 0 0 0"];
	return Energy;
}

//double Bootstrap::CalcBEEnergyByFrag()
//{
//	std::cout << "BE-DMET: Calculating one shot BE energy..." << std::endl;
//	*Output << "BE-DMET: Calculating one shot BE energy..." << std::endl;
//	std::vector<double> FragEnergies;
//	// Collect fragment energies for each type of state. This depends on St, which impurity state is chosen and which bath state is used for the embedding.
//	for (int St = 0; St < MaxState; St++)
//	{
//		std::vector<Eigen::MatrixXd> OneRDM;
//		std::vector<double> FragEnergy;
//		FragEnergy = BEImpurityFCI(OneRDM, Input, 0, RotationMatrices[0], ChemicalPotential, St, BEPotential[0], MaxState, BECenterPosition[0]);
//		FragEnergies.push_back(FragEnergy[0]);
//	}
//	double Energy = 0;
//	for (int x = 0; x < NumFrag; x++)
//	{
//		Energy += FragEnergies[FragState[x]];
//		std::cout << "BE-DMET: -- Energy of Fragment " << x << " is " << FragEnergies[FragState[x]] << std::endl;
//		*Output << "BE-DMET: -- Energy of Fragment " << x << " is " << FragEnergies[FragState[x]] << std::endl;
//	}
//	Energy += Input.Integrals["0 0 0 0"];
//	return Energy;
//}

void Bootstrap::runDebug()
{
	std::cout << "BE-DMET: Beginning Bootstrap Embedding..." << std::endl;
	*Output << "BE-DMET: Beginning Bootstrap Embedding..." << std::endl;
	NewtonRaphson(); // Comment out to do one shot
	double Energy = CalcBEEnergy();
	// double Energy = CalcBEEnergyByFrag();
	std::cout << "BE-DMET: DMET Energy = " << Energy << std::endl;
	*Output << "BE-DMET: DMET Energy = " << Energy << std::endl;
}
