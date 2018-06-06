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

class Bootstrap
{
public:
	std::string BEInputName = "matching.be";
	std::vector< std::vector< int > > FragmentOrbitals;
	std::vector< std::vector< int > > EnvironmentOrbitals;
	int NumAO;
	int NumOcc;
	int NumFrag;

	double ChemicalPotential;

	// Each element of this vector corresponds to a tuple for the BE FCI potential on each fragment.
	// Each vector is a different fragment.
	// Index 1 is overlapping fragment.
	// Index 2 is the orbital we want to match to that fragment -- We input this from the CURRENT fragment and we can search it in OTHER fragments.
	// Index 3 is the value of the potential on that orbital for this fragment.
	std::vector< std::vector< std::tuple< int, int, double > > > BEPotential;

	std::vector< Eigen::MatrixXd > RotationMatrices;
	std::vector< Eigen::MatrixXd > ImpurityDensities;

	void CollectSchmidt(Eigen::MatrixXd, std::ofstream&);
	void ReadBEInput(); // Do this later.
	void debugInit(InputObj);
	void RunImpurityCalculations();
	void doBootstrap(InputObj&, Eigen::MatrixXd&, std::vector< Eigen::MatrixXd >&, std::ofstream&);
	void printDebug(std::ofstream&);

private:
	double dLambda = 0.1;
}; 
