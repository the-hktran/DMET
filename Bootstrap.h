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

class Bootstrap
{
public:
	std::string BEInputName = "matching.be";
	std::ofstream* Output;
	std::vector< std::vector< int > > FragmentOrbitals;
	std::vector< std::vector< int > > EnvironmentOrbitals;
	int NumAO;
	int NumOcc;
	int NumFrag;

	int NumConditions;
	std::vector<int> NumFragCond;
	bool isTS = false;

	double ChemicalPotential = 0;
	InputObj Input;
	std::vector<InputObj> Inputs;
	std::vector<FCI> FCIs;

	// Some definitions for excited state embedding
	int State = 0;
	std::vector<int> FragState;
	std::vector<int> BathState;
	int MaxState = 6;

	// Each element of this vector corresponds to a tuple for the BE FCI potential on each fragment.
	// Each vector is a different fragment.
	// Index 1 is overlapping fragment.
	// Index 2 is the orbital we want to match to that fragment -- We input this from the CURRENT fragment and we can search it in OTHER fragments.
	// Index 3 is the value of the potential on that orbital for this fragment.
	std::vector< std::vector< std::tuple< int, int, int, int, int, double > > > BEPotential;

	// Contains the INDEX of the center position orbital on each fragment.
	std::vector< std::vector< int > > BECenterPosition; // Need to implement a way to figure this out.

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

private:
	double dLambda = 1E-4;
	double dMu = 1E-1;
	std::vector< double > FragmentLoss(std::vector< std::vector<Eigen::MatrixXd> >, std::vector<Eigen::MatrixXd>, int);
	std::vector< std::vector< Eigen::MatrixXd > > CollectRDM(std::vector< std::vector< std::tuple< int, int, int, int, int, double> > >, double, int);
	Eigen::MatrixXd CalcJacobian(Eigen::VectorXd&);
	void VectorToBE(Eigen::VectorXd);
	Eigen::VectorXd BEToVector();
	void NewtonRaphson();
	void OptMu();
	double CalcCostChemPot(std::vector< std::vector<Eigen::MatrixXd> >, std::vector< std::vector< int > >, std::vector<int>, InputObj&);
	double CalcBEEnergy();
	void CollectInputs();
	void PrintOneRDMs(std::vector< std::vector<Eigen::MatrixXd> >);
}; 
