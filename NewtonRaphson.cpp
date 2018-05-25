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

	std::vector< Eigen::MatrixXd > DensityReference;

	void InitFromBE(Bootstrap);

	double dLoss(double);
	std::pair<double, double> BENewtonIteration(double const);
	double FCIwrtLambda(double);
	void FormDensityReference();
	double GetLambda();
};

void BENewton::FormDensityReference()
{
	std::vector< Eigen::MatrixXd > CollectFCIDensity;
	for (int x = 0; x < Input.NumFragments; x++)
	{
		Eigen::MatrixXd FragDensity;
		BEImpurityFCI(FragDensity, Input, FragmentIndex, RotationMatrix, ChemicalPotential, State, BEPotential[x]);
		CollectFCIDensity.push_back(FragDensity);
	}
	DensityReference = CollectFCIDensity;
}

void BENewton::InitFromBE(Bootstrap BE)
{
	//FragmentIndex
}

double BENewton::FCIwrtLambda(double Lambda)
{
	std::get<2>(BEPotential[FragmentIndex][MatchedOrbital]) = Lambda;
	BEImpurityFCI(IterDensity, Input, FragmentIndex, RotationMatrix, ChemicalPotential, State, BEPotential[FragmentIndex]);

	std::vector< int > FragPosImp, BathPosImp, FragPosBath, BathPosBath;
	GetCASPos(Input, FragmentIndex, FragPosImp, BathPosImp);
	GetCASPos(Input, std::get<0>(BEPotential[FragmentIndex][MatchedOrbital]), FragPosImp, BathPosImp);

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
		if (Input.FragmentOrbitals[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])][i] == std::get<0>(BEPotential[FragmentIndex][MatchedOrbital]))
		{
			break;
		}
		PElementBath++;
	}
	double Loss = DensityReference[std::get<0>(BEPotential[FragmentIndex][MatchedOrbital])].coeffRef(FragPosBath[PElementBath], FragPosBath[PElementBath]) - IterDensity.coeffRef(FragPosImp[PElementImp], FragPosImp[PElementImp]);

	return Loss;
}

double BENewton::dLoss(double Lambda)
{
	double StepSize = 0.1;
	double dL2 = FCIwrtLambda(Lambda + StepSize);
	double dL1 = FCIwrtLambda(Lambda - StepSize);
	return (dL2 - dL1) / (2 * StepSize);
}

std::pair<double, double> BENewton::BENewtonIteration(double const Lambda)
{
	double f = FCIwrtLambda(Lambda);
	double df = dLoss(Lambda);

	return std::make_pair(f, df);
}

// More or less copied this code from the boost documentation.
template <class T>
struct FunctorLoss
{ // Functor also returning 1st derivative.
  FunctorLoss()
  { 
  }
  std::pair<T, T> operator()(T const& x)
  { 
    // Return both f(x) and f'(x).
    return BENewton::BENewtonIteration(x); // std::make_pair(fx, dx);  // 'return' both fx and dx.
  }
};

template <class T>
T SolveForLambda()
{ 
  using namespace boost::math::tools;
  T guess = 0.0;
  T min = -10.0;
  T max = 10.0;
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  int get_digits = static_cast<int>(digits * 0.6);    // Accuracy doubles with each step, so stop when we have
                                                      // just over half the digits correct.
  const boost::uintmax_t maxit = 20;
  boost::uintmax_t it = maxit;
  T result = newton_raphson_iterate(FunctorLoss<T>(), guess, min, max, get_digits, it);
  return result;
}
// end boost code

double BENewton::GetLambda()
{
	double Lambda = SolveForLambda<double>();
	return Lambda;
}

double BENewtonSolver(Eigen::MatrixXd &DensityMatrix, InputObj &Input, int FragmentIndex, Eigen::MatrixXd &RotationMatrix, double ChemicalPotential, int State, std::tuple< int, int, double> BEPotential)
{
	return 0;
}