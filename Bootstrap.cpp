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

// This is a debug function for the time being
void Bootstrap::debugInit(InputObj Inp)
{
	NumAO = Inp.NumAO;
	NumOcc = Inp.NumOcc;
	NumFrag = Inp.FragmentOrbitals.size();
	FragmentOrbitals = Inp.FragmentOrbitals;
	EnvironmentOrbitals = Inp.EnvironmentOrbitals;
	Input = Inp;

	for (int x = 0; x < 10; x++)
	{
		std::vector< std::tuple <int , int, double > > FragmentBEPotential;
		FragmentBEPotential.push_back(std::tuple< int, int, double >(x + 1 % 10, FragmentOrbitals[x][2], 0));
		FragmentBEPotential.push_back(std::tuple< int, int, double >(x + 9 % 10, FragmentOrbitals[x][0], 0));
		BEPotential.push_back(FragmentBEPotential);
	}
}

std::vector< Eigen::MatrixXd > Bootstrap::CollectRDM(std::vector< std::vector< std::tuple< int, int, double> > > BEPot, double Mu, int ElecState)
{
	std::vector< Eigen::MatrixXd > AllBE1RDM;
	// std::vector< std::vector< Eigen::MatrixXd > > AllBE2RDM;
	for (int x = 0; x < NumFrag; x++)
	{
		Eigen::MatrixXd OneRDM;
		// std::vector< Eigen::MatrixXd > Fragment1RDM;
		// std::vector< Eigen::MatrixXd > Fragment2RDM;
		// for (int i = 0; i < BEPotential[x].size(); i++)
		// {
		// 	Eigen::MatrixXd OneRDM;
		// 	Eigen::MatrixXd TwoRDM;

		BEImpurityFCI(OneRDM, Input, x, RotationMatrices[x], Mu, ElecState, BEPot[x]);
			// Fragment1RDM.push_back(OneRDM);
			// Fragment2RDM.push_back(TwoRDM);
		// }
		// AllBE1RDM.push_back(Fragment1RDM);
		// AllBE2RDM.push_back(Fragment2RDM);
		AllBE1RDM.push_back(OneRDM);
	}
	return AllBE1RDM;
}

std::vector< double > Bootstrap::FragmentLoss(std::vector<Eigen::MatrixXd> DensityReference, Eigen::MatrixXd IterDensity, int FragmentIndex)
{
	std::vector< double > AllLosses;
	for (int MatchedOrbital = 0; MatchedOrbital < BEPotential[FragmentIndex].size(); MatchedOrbital++)
	{
		std::vector< int > FragPosImp, BathPosImp, FragPosBath, BathPosBath;
		GetCASPos(Input, FragmentIndex, FragPosImp, BathPosImp);
		GetCASPos(Input, std::get<0>(BEPotential[FragmentIndex][MatchedOrbital]), FragPosImp, BathPosImp);

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
			if (Input.FragmentOrbitals[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])][i] == std::get<0>(BEPotential[FragmentIndex][MatchedOrbital]))
			{
				break;
			}
			PElementBath++;
		}
		double Loss = DensityReference[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])].coeffRef(FragPosBath[PElementBath], FragPosBath[PElementBath]) - IterDensity.coeffRef(FragPosImp[PElementImp], FragPosImp[PElementImp]);
		Loss = Loss * Loss;
		AllLosses.push_back(Loss);
	}
}

Eigen::MatrixXd Bootstrap::CalcJacobian(Eigen::VectorXd &f)
{
	int NumConditions = 0;
	std::vector<int> NumFragCond(NumFrag);

	std::vector< Eigen::MatrixXd > OneRDMs;
	OneRDMs = CollectRDM(BEPotential, ChemicalPotential, State);

	f = Eigen::VectorXd::Zero(NumConditions);

	for (int x = 0; x < NumFrag; x++)
	{
		NumConditions += BEPotential[x].size();
		NumFragCond[x] = BEPotential[x].size();
	}

	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(NumConditions, NumConditions);

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
		auto BEMinusdLambda = BEPotential;

		// Collect all the density matrices for this iteration.
		Eigen::MatrixXd FragOneRDMMinusdLambda;
		BEImpurityFCI(FragOneRDMMinusdLambda, Input, x, RotationMatrices[x], ChemicalPotential, State, BEMinusdLambda[x]);
		std::vector<double> LossesMinus = FragmentLoss(OneRDMs, FragOneRDMMinusdLambda, x);

		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			// Make the + dLambda potential for the fragment.
			auto BEPlusdLambda = BEPotential;
			std::get<2>(BEPlusdLambda[x][i]) = std::get<2>(BEPotential[x][i]) + dLambda;

			// Collect all the density matrices for this iteration.
			Eigen::MatrixXd FragOneRDMPlusdLambda;
			BEImpurityFCI(FragOneRDMPlusdLambda, Input, x, RotationMatrices[x], ChemicalPotential, State, BEPlusdLambda[x]);
			std::vector<double> LossesPlus = FragmentLoss(OneRDMs, FragOneRDMPlusdLambda, x);

			// // Make the - dLambda potential for the fragment.
			// auto BEMinusdLambda = BEPotential;
			// std::get<2>(BEMinusdLambda[x][i]) = std::get<2>(BEPotential[x][i]) - dLambda;

			// // Collect all the density matrices for this iteration.
			// Eigen::MatrixXd FragOneRDMMinusdLambda;
			// BEImpurityFCI(FragOneRDMMinusdLambda, Input, x, RotationMatrices[x], ChemicalPotential, State, BEMinusdLambda[x]);
			// std::vector<double> LossesMinus = FragmentLoss(OneRDMs[x], FragOneRDMMinusdLambda, x);
			
			// Fill in J
			int JRow = 0;
			for (int j = 0; j < x; j++)
			{
				JRow += NumFragCond[x];
			}
			for (int j = 0; j < LossesPlus.size(); j++)
			{
				J(JRow + j, JCol) = (LossesPlus[j] - LossesMinus[j]) / (dLambda);
			}
			JCol++;
		}

		// Fill in f
		for (int j = 0; j < LossesMinus.size(); j++)
		{
			f[fCount] = LossesMinus[j];
			fCount++;
		}
	}
	return J;
}

void Bootstrap::CollectSchmidt(Eigen::MatrixXd MFDensity, std::ofstream &Output)
{
	for (int x = 0; x < NumFrag; x++)
	{
		Eigen::MatrixXd RotMat = Eigen::MatrixXd::Zero(NumAO, NumAO);
		int NumEnvVirt = NumAO - NumOcc - FragmentOrbitals.size();
		SchmidtDecomposition(MFDensity, RotMat, FragmentOrbitals[x], EnvironmentOrbitals[x], NumEnvVirt, Output);
		RotationMatrices.push_back(RotMat);
	}
}

void Bootstrap::VectorToBE(Eigen::VectorXd X)
{
	int xCount = 0;
	for (int x = 0; x < NumFrag; x++)
	{
		for (int i = 0; i < BEPotential.size(); i++)
		{
			std::get<2>(BEPotential[x][i]) = X[xCount];
			xCount++;
		}
	}
}

Eigen::VectorXd Bootstrap::BEToVector()
{
	Eigen::VectorXd X = Eigen::VectorXd::Zero(NumConditions);
	int xCount = 0;
	for (int x = 0; x < NumFrag; x++)
	{
		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			X[xCount] = std::get<2>(BEPotential[x][i]);
			xCount++;
		}
	}
}

void Bootstrap::OptMu()
{
	// We will do Newton Raphson to optimize the chemical potential.
	// First, intialize Mu to Chemical Potential and calculate intial loss.
	double dMu = 0.1;
	double Mu = ChemicalPotential;
	std::vector< Eigen::MatrixXd > Densities = CollectRDM(BEPotential, Mu, State);
	double Loss = CalcCostChemPot(Densities, Input);

	std::cout << "BE: Optimizing chemical potential." << std::endl;
	// Do Newton's method until convergence
	while (fabs(Loss) > 1e-6)
	{
		// Calculate Loss at Mu + dMu
		Densities = CollectRDM(BEPotential, Mu + dMu, State);
		double LossPlus = CalcCostChemPot(Densities, Input);
		double dLdMu = (LossPlus - Loss) / dMu;

		Mu = Mu - Loss / dLdMu;

		// Update Loss at new, updated Mu
		Densities = CollectRDM(BEPotential, Mu, State);
		Loss = CalcCostChemPot(Densities, Input);

		std::cout << "BE: Iteration yielded mu = " << Mu << " and Loss = " << Loss << std::endl;
	}
	std::cout << "BE: Chemical potential converged." << std::endl;
	// After everything is done, store Mu into ChemicalPotential for the rest of the object.
	ChemicalPotential = Mu;
	
}

void Bootstrap::NewtonRaphson()
{
	// First, vectorize Lambdas
	Eigen::VectorXd x = BEToVector();

	// Initialize J and f
	Eigen::VectorXd f;
	Eigen::MatrixXd J = CalcJacobian(f);

	std::cout << "BE: Optimizing site potential." << std::endl;

	while(f.squaredNorm() > 1e-6)
	{ 
		x = x - J.inverse() * f;
		VectorToBE(x); // Updates the BEPotential for the J and f update next.
		J = CalcJacobian(f); // Update here to check the loss.
		std::cout << "BE: Site potential obtained\n" << x << "BE: with loss \n" << f << std::endl;
	}
}

void Bootstrap::doBootstrap(InputObj &Input, Eigen::MatrixXd &MFDensity, std::vector< Eigen::MatrixXd > &TEST, std::ofstream &Output)
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

void Bootstrap::printDebug(std::ofstream &Output)
{
	for (int x = 0; x < RotationMatrices.size(); x++)
	{
		Output << RotationMatrices[x] << std::endl;
		Output << "\n";
	}
}

void Bootstrap::runDebug()
{
	OptMu();
	NewtonRaphson();
}
