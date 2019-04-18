#pragma once
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
#include "Fragmenting.h"

class Bootstrap
{
public:
	std::string BEInputName = "matching.be";
	double BEEnergy;
	std::ofstream* Output;
	std::vector< std::vector< int > > FragmentOrbitals;
	std::vector< std::vector< int > > EnvironmentOrbitals;
	int NumAO;
	int NumOcc;
	int NumFrag;

	int NumConditions;
	std::vector<int> NumFragCond;
	bool isTS = false;
	bool MatchFullP = true;

	double ChemicalPotential = 0;
	InputObj Input;
	std::vector<InputObj> Inputs;
	std::vector<FCI> FCIs, FCIsBase;
	// std::vector<Eigen::MatrixXd> aOneRDMs, bOneRDMs;

	// Some definitions for excited state embedding
	int State = 0;
	std::vector<int> FragState;
	std::vector<int> BathState;
	int MaxState = 6;

	// Each element of this vector corresponds to a tuple for the BE FCI potential on each fragment.
	// Each vector is a different fragment.
	// Index 1 is overlapping fragment.
	// Index 2 - 5 is the orbital we want to match to that fragment -- We input this from the CURRENT fragment and we can search it in OTHER fragments.
	// Index 6 is the value of the potential on that orbital for this fragment.
	// Index 7 - 8 describe whether or not the potential is for alpha or beta spins.
	std::vector< std::vector< std::tuple< int, int, int, int, int, double, bool, bool > > > BEPotential;

	// Contains the INDEX of the center position orbital on each fragment.
	std::vector< std::vector< int > > aBECenterPosition, bBECenterPosition; // Need to implement a way to figure this out.

	std::vector< std::vector<int> > aFragPos, bFragPos, aBathPos, bBathPos;

	std::vector< Eigen::MatrixXd > RotationMatrices;
	std::vector< Eigen::MatrixXd > ImpurityDensities;
	std::vector< Eigen::MatrixXd > aRotationMatrices;
	std::vector< Eigen::MatrixXd > bRotationMatrices;

	void CollectSchmidt(std::vector<Eigen::MatrixXd>, std::ofstream&);
	void CollectSchmidt(std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::ofstream&);
	void ReadBEInput(); // Do this later.
	void debugInit(InputObj, std::ofstream&);
	void RunImpurityCalculations();
	void doBootstrap(InputObj&, std::vector<Eigen::MatrixXd>&, std::ofstream&);
	void doBootstrap(InputObj&, std::vector<Eigen::MatrixXd>&, std::vector< Eigen::MatrixXd >&, std::ofstream&);
	void printDebug(std::ofstream&);
	void runDebug();
	double CalcBEEnergyByFrag();
	int OrbitalToReducedIndex(int, int, bool);
	void InitFromFragmenting(Fragmenting);

private:
	double dLambda = 1E-6;
	double dMu = 1E-6;
	std::vector< double > FragmentLoss(std::vector< std::vector<Eigen::MatrixXd> >, std::vector<Eigen::MatrixXd>, int);
	void CollectRDM(std::vector< Eigen::MatrixXd > &, std::vector< Eigen::MatrixXd > &, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &,
                           std::vector< std::vector< std::tuple< int, int, int, int, int, double, bool, bool > > >, double);
	Eigen::MatrixXd CalcJacobian(Eigen::VectorXd&);
	void VectorToBE(Eigen::VectorXd);
	Eigen::VectorXd BEToVector();
	void NewtonRaphson();
	void OptMu();
	double CalcCostChemPot(std::vector< std::vector<Eigen::MatrixXd> >, std::vector< std::vector< int > >, std::vector<int>, InputObj&);
	double CalcCostChemPot(std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector< std::vector< int > >, std::vector< std::vector < int > >, std::vector<int>);
	std::vector<double> CalcCostLambda(std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, std::vector<std::vector< double > >, std::vector<std::vector< double > >, std::vector<std::vector< double > >, 
	Eigen::MatrixXd, Eigen::MatrixXd, std::vector<double>, std::vector<double>, std::vector<double>, int);
	double CalcBEEnergy();
	void CollectInputs();
	void UpdateFCIs();
	void PrintOneRDMs(std::vector< std::vector<Eigen::MatrixXd> >);
}; 
