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
#include <queue>
#include "DMET.cpp"

class Bootstrap
{
public:
	std::string BEInputName = "matching.be";
	std::vector< std::vector< int > > FragmentOrbitals;
	std::vector< std::vector< int > > EnvironmentOrbitals;
	int NumAO;
	int NumOcc;
	int NumFrag;

	// Each element of this vector corresponds to a tuple for the BE FCI potential on each fragment.
	// Index 1 is overlapping fragment.
	// Index 2 is the orbital we want to match to that fragment
	// Index 3 is the value of the potential on that orbital for this fragment.
	std::vector< std::vector< std::tuple< int, int, double > > > BEPotential;

	std::vector< Eigen::MatrixXd > RotationMatrices;
	void CollectSchmidt(Eigen::MatrixXd, std::ofstream&);
	void ReadBEInput(); // Do this later.
};

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

void doBootstrap(InputObj Input, Eigen::MatrixXd &MFDensity, std::ofstream &Output)
{
	Bootstrap BE;
	BE.FragmentOrbitals = Input.FragmentOrbitals;
	BE.EnvironmentOrbitals = Input.EnvironmentOrbitals;
	BE.NumAO = Input.NumAO;
	BE.NumOcc = Input.NumOcc;
	BE.NumFrag = Input.NumFragments;

	BE.CollectSchmidt(MFDensity, Output); // Runs a function to collect all rotational matrices in corresponding to each fragment.
}