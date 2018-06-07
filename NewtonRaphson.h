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

#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/next.hpp> // For float_distance.
#include <tuple> // for std::tuple and std::make_tuple.
#include <boost/math/special_functions/cbrt.hpp> // For boost::math::cbrt.

#include "Bootstrap.h"
#include "Functions.h"

class BENewton
{
public:
	int FragmentIndex;
	Eigen::MatrixXd IterDensity;
	InputObj &Input;
	Eigen::MatrixXd RotationMatrix;
	double ChemicalPotential; 
	int State;
	int MatchedOrbital; // Which orbital we are matching.
	std::vector< std::vector< std::tuple< int, int, double> > > BEPotential;
    int NumConditions = BEPotential.size();

	std::vector< Eigen::MatrixXd > DensityReference;

	double dLoss(double);
	std::pair<double, double> BENewtonIteration(double const);
	double FCIwrtLambda(double);
	void FormDensityReference();
	double GetLambda();
};

class NewtonRaphson
{
public:
    Eigen::VectorXd x; // Input
    Eigen::MatrixXd J; // Jacobian at x
    Eigen::MatrixXd f; // Function at x

    void doNewton();
};