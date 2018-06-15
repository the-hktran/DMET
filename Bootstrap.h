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

class Bootstrap
{
public:
	std::string BEInputName = "matching.be";
	std::vector< std::vector< int > > FragmentOrbitals;
	std::vector< std::vector< int > > EnvironmentOrbitals;
	int NumAO;
	int NumOcc;
	int NumFrag;

	int NumConditions;
	std::vector<int> NumFragCond;
	bool isTS = true;

	double ChemicalPotential = 0;
	InputObj Input;
	std::vector<InputObj> Inputs;

	// Some definitions for excited state embedding
	int State = 0;
	std::vector<int> FragState;
	int MaxStates = 2;

	// Each element of this vector corresponds to a tuple for the BE FCI potential on each fragment.
	// Each vector is a different fragment.
	// Index 1 is overlapping fragment.
	// Index 2 is the orbital we want to match to that fragment -- We input this from the CURRENT fragment and we can search it in OTHER fragments.
	// Index 3 is the value of the potential on that orbital for this fragment.
	std::vector< std::vector< std::tuple< int, int, double > > > BEPotential;

	// Contains the INDEX of the center position orbital on each fragment.
	std::vector< std::vector< int > > BECenterPosition; // Need to implement a way to figure this out.

	std::vector< Eigen::MatrixXd > RotationMatrices;
	std::vector< Eigen::MatrixXd > ImpurityDensities;

	void CollectSchmidt(Eigen::MatrixXd, std::ofstream&);
	void ReadBEInput(); // Do this later.
	void debugInit(InputObj);
	void RunImpurityCalculations();
	void doBootstrap(InputObj&, Eigen::MatrixXd&, std::vector< Eigen::MatrixXd >&, std::ofstream&);
	void printDebug(std::ofstream&);
	void runDebug();

private:
	double dLambda = 1E-4;
	double dMu = 1E-4;
	std::vector< double > FragmentLoss(std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, int);
	std::vector< Eigen::MatrixXd > CollectRDM(std::vector< std::vector< std::tuple< int, int, double> > >, double, int);
	Eigen::MatrixXd CalcJacobian(Eigen::VectorXd&);
	void VectorToBE(Eigen::VectorXd);
	Eigen::VectorXd BEToVector();
	void NewtonRaphson();
	void OptMu();
	double CalcCostChemPot(std::vector<Eigen::MatrixXd>, std::vector< std::vector< int > >, InputObj&);
	double CalcBEEnergy();
	void CollectInputs();
}; 
