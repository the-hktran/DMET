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

class RealTime
{
public:
	InputObj Input;
	Eigen::MatrixXd X;
	Eigen::MatrixXd RotationMatrix;
	int FragmentIndex;
	int NumAOImp, NumVirt, NumCore, NumEnv;
	std::vector< Eigen::MatrixXd > FragmentDensities;
	std::vector< Eigen::Tensor<double, 4> > Fragment2RDM;

	void Init(InputObj&, int, std::vector< Eigen::MatrixXd >&, std::vector< Eigen::Tensor<double, 4> >&, Eigen::MatrixXd);
	void FormX();
};