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
//#include "NewtonRaphson.cpp"

// This is a debug function for the time being
void Bootstrap::debugInit(InputObj Input)
{
	NumAO = Input.NumAO;
	NumOcc = Input.NumOcc;
	NumFrag = Input.FragmentOrbitals.size();
	FragmentOrbitals = Input.FragmentOrbitals;
	EnvironmentOrbitals = Input.EnvironmentOrbitals;

	for (int x = 0; x < 10; x++)
	{
		std::vector< std::tuple <int , int, double > > FragmentBEPotential;
		FragmentBEPotential.push_back(std::tuple< int, int, double >(x + 1 % 10, FragmentOrbitals[x][2], 0));
		FragmentBEPotential.push_back(std::tuple< int, int, double >(x + 9 % 10, FragmentOrbitals[x][0], 0));
		BEPotential.push_back(FragmentBEPotential);
	}
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

void Bootstrap::doBootstrap(InputObj &Input, Eigen::MatrixXd &MFDensity, std::ofstream &Output)
{
	FragmentOrbitals = Input.FragmentOrbitals;
	EnvironmentOrbitals = Input.EnvironmentOrbitals;
	NumAO = Input.NumAO;
	NumOcc = Input.NumOcc;
	NumFrag = Input.NumFragments;

	CollectSchmidt(MFDensity, Output); // Runs a function to collect all rotational matrices in corresponding to each fragment.
}

void Bootstrap::printDebug(std::ofstream &Output)
{
	for (int x = 0; x < RotationMatrices.size(); x++)
	{
		Output << RotationMatrices[x] << std::endl;
		Output << "\n";
	}
}
