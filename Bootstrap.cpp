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

	for (int x = 0; x < 8; x++)
	{
		std::vector< std::tuple <int , int, double > > FragmentBEPotential;
		FragmentBEPotential.push_back(std::tuple< int, int, double >((x + 1) % 8, FragmentOrbitals[x][2], 0));
		FragmentBEPotential.push_back(std::tuple< int, int, double >((x + 7) % 8, FragmentOrbitals[x][0], 0));
		BEPotential.push_back(FragmentBEPotential);

		std::vector< int >  CenterPos;
		CenterPos.push_back(1);
		BECenterPosition.push_back(CenterPos);
	}

	// Will be important to initialize this immediately when BE is implemented.
	for (int x = 0; x < NumFrag; x++)
	{
		NumConditions += BEPotential[x].size();
		NumFragCond.push_back(BEPotential[x].size());
	}
}

double Bootstrap::CalcCostChemPot(std::vector<Eigen::MatrixXd> Frag1RDMs, std::vector< std::vector< int > > BECenter, InputObj &Inp)
{
    double CF = 0;
    for(int x = 0; x < Frag1RDMs.size(); x++) // sum over fragments
    {
        std::vector< int > FragPos;
        std::vector< int > BathPos;
        GetCASPos(Inp, x , FragPos, BathPos);
        for(int i = 0; i < BECenter[x].size(); i++) // sum over diagonal matrix elements belonging to the fragment orbitals.
        {
            CF += Frag1RDMs[x](FragPos[BECenter[x][i]], FragPos[BECenter[x][i]]);
        }
    }
    CF -= Inp.NumElectrons;
    CF = CF * CF;
    return CF;
}

std::vector< Eigen::MatrixXd > Bootstrap::CollectRDM(std::vector< std::vector< std::tuple< int, int, double> > > BEPot, double Mu, int ElecState)
{
	std::vector< Eigen::MatrixXd > AllBE1RDM;
	// std::vector< std::vector< Eigen::MatrixXd > > AllBE2RDM;
	for (int x = 0; x < NumFrag; x++)
	{
		Eigen::MatrixXd OneRDM;
		if (x > 0 && isTS == true)
		{
			OneRDM = AllBE1RDM[0];
			AllBE1RDM.push_back(OneRDM);
			continue;
		}

		BEImpurityFCI(OneRDM, Input, x, RotationMatrices[x], Mu, ElecState, BEPot[x]);
		AllBE1RDM.push_back(OneRDM);
		std::cout << "Collect " << x << "\n" << OneRDM << std::endl;
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
		double Loss = DensityReference[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])].coeffRef(FragPosBath[PElementBath], FragPosBath[PElementBath]) - IterDensity.coeffRef(FragPosImp[PElementImp], FragPosImp[PElementImp]);
		Loss = Loss * Loss;
		AllLosses.push_back(Loss);
	}
	return AllLosses;
}

Eigen::MatrixXd Bootstrap::CalcJacobian(Eigen::VectorXd &f)
{
	std::vector< Eigen::MatrixXd > OneRDMs;
	OneRDMs = CollectRDM(BEPotential, ChemicalPotential, State);

	double LMu = CalcCostChemPot(OneRDMs, BECenterPosition, Input);
	double dMu = 1E-4;
	std::vector< Eigen::MatrixXd > DensitiesPlusdMu;
	DensitiesPlusdMu = CollectRDM(BEPotential, ChemicalPotential + dMu, State);
	double LMuPlus = CalcCostChemPot(DensitiesPlusdMu, BECenterPosition, Input);
	// We need every density matrix to calculation LMu, but since we only calculate one fragment each loop, we will use this vector
	// to store the different contributions to LMu per fragment, and then when we add the BE potential, we only affect one fragment
	// so we'll use the difference between LMu in that fragment to figure out the new LMu.
	std::vector<double> LMuByFrag;
	for (int x = 0; x < NumFrag; x++)
	{
		std::vector<Eigen::MatrixXd> SingleRDM;
		std::vector< std::vector< int > > SingleCenter;
		SingleRDM.push_back(OneRDMs[x]);
		SingleCenter.push_back(BECenterPosition[x]);
		double LMuX = CalcCostChemPot(SingleRDM, SingleCenter, Input);
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
				for (int iTS = 0; iTS < BEPotential[xTS].size(); iTS++)
				{
					int JRow = 0;
					for (int j = 0; j < xTS; j++)
					{
						JRow += NumFragCond[xTS];
					}
					for (int j = 0; j < BEPotential[xTS].size(); j++)
					{
						J(JRow + j, JCol) = J.coeffRef(j, JCol % BEPotential[xTS].size());
					}
					JCol++;
				}
				for (int j = 0; j < BEPotential[xTS].size(); j++)
				{
					f[fCount] = f[j];
					fCount++;
				}
			}
			break;
		}

		auto BEMinusdLambda = BEPotential;

		// Collect all the density matrices for this iteration.
		Eigen::MatrixXd FragOneRDMMinusdLambda;
		BEImpurityFCI(FragOneRDMMinusdLambda, Input, x, RotationMatrices[x], ChemicalPotential, State, BEMinusdLambda[x]);
		std::vector<double> LossesMinus = FragmentLoss(OneRDMs, FragOneRDMMinusdLambda, x);

		int JRow = 0;
		for (int j = 0; j < x; j++)
		{
			JRow += NumFragCond[j];
		}

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
			for (int j = 0; j < LossesPlus.size(); j++)
			{
				J(JRow + j, JCol) = (LossesPlus[j] - LossesMinus[j]) / (dLambda);
			}

			// Add in chemical potential portion.
			std::vector< Eigen::MatrixXd > SingleRDMPlus;
			std::vector< std::vector< int > > SingleBECenter;
			SingleRDMPlus.push_back(FragOneRDMPlusdLambda);
			SingleBECenter.push_back(BECenterPosition[x]);
			double LMuPlus = CalcCostChemPot(SingleRDMPlus, SingleBECenter, Input);
			J(J.rows() - 1, JCol) = (LMuPlus - LMuByFrag[x]) / dLambda;

			JCol++;
		}

		// Last column is derivative of each loss with respect to chemical potential.
		// The last element of this column is already handled.
		std::vector<double> LossesPlusMu = FragmentLoss(OneRDMs, DensitiesPlusdMu[x], x);
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
		for (int i = 0; i < BEPotential[x].size(); i++)
		{
			std::get<2>(BEPotential[x][i]) = X[xCount];
			xCount++;
		}
	}
	ChemicalPotential = X[X.size() - 1];
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
	X[X.size() - 1] = ChemicalPotential;
	return X;
}

void Bootstrap::OptMu()
{
	// We will do Newton Raphson to optimize the chemical potential.
	// First, intialize Mu to Chemical Potential and calculate intial loss.
	double dMu = 1E-4;
	double Mu = ChemicalPotential;
	std::vector< Eigen::MatrixXd > Densities = CollectRDM(BEPotential, Mu, State);
	double Loss = CalcCostChemPot(Densities, BECenterPosition, Input);

	std::cout << "BE: Optimizing chemical potential." << std::endl;
	// Do Newton's method until convergence
	while (fabs(Loss) > 1e-6)
	{
		// Calculate Loss at Mu + dMu
		Densities = CollectRDM(BEPotential, Mu + dMu, State);
		double LossPlus = CalcCostChemPot(Densities, BECenterPosition, Input);
		double dLdMu = (LossPlus - Loss) / dMu;

		Mu = Mu - Loss / dLdMu;

		// Update Loss at new, updated Mu
		Densities = CollectRDM(BEPotential, Mu, State);
		Loss = CalcCostChemPot(Densities, BECenterPosition, Input);

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

	std::cout << f << std::endl;
	std::cout << J << std::endl;
	return;

	std::cout << "BE: Optimizing site potential." << std::endl;

	while (f.squaredNorm() > 1e-6)
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
	// OptMu();
	NewtonRaphson();
}
