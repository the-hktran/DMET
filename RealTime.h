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
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>

class RealTime
{
public:
	InputObj Input;
	Eigen::MatrixXcd X;
	Eigen::MatrixXcd RotationMatrix;
	int FragmentIndex;
	int NumAOImp, NumVirt, NumCore, NumEnv;
	std::vector< Eigen::MatrixXcd > FragmentDensities;
	std::vector< Eigen::Tensor<std::complex<double>, 4> > Fragment2RDM;
	Eigen::MatrixXcd TDHam;
	Eigen::VectorXcd ImpurityEigenstate;
	
	// Things that should be in an FCI class.
	std::vector< std::vector< bool > > aStrings;
	std::vector< std::vector< bool > > bStrings;

	void Init(InputObj&, int, std::vector< Eigen::MatrixXd >&, std::vector< Eigen::Tensor<double, 4> >&, Eigen::MatrixXd, std::vector< Eigen::VectorXd >&);
	void FormX();
	Eigen::MatrixXcd UpdateR(double);
	Eigen::MatrixXcd UpdateH(double);
	Eigen::VectorXcd UpdateEigenstate(double);
	void UpdateRDM();
	
	void TimeUpdate(double, double);
};