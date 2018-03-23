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
#include "FCIForBE.cpp"

#include <boost/math/tools/roots.hpp>

#include <boost/math/special_functions/next.hpp> // For float_distance.
#include <boost/math/special_functions/cbrt.hpp> // For boost::math::cbrt.

class BENewton
{
public:
	int FragmentIndex;
	Eigen::MatrixXd IterDensity;
	InputObj &Input;
	int FragmentIndex;
	Eigen::MatrixXd RotationMatrix;
	double ChemicalPotential; 
	int State;
	int MatchedOrbital; // Which orbital we are matching.
	std::vector< std::vector< std::tuple< int, int, double> > > BEPotential;

	std::vector< Eigen::MatrixXd > DensityReference;

	std::tuple<double, double> BENewtonMethod(double);
	double FCIwrtLambda(double);
};

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

template <class T>
struct cbrt_functor_noderiv
{
	//  cube root of x using only function - no derivatives.
	cbrt_functor_noderiv(T const& to_find_root_of) : a(to_find_root_of)
	{ /* Constructor just stores value a to find root of. */
	}
	T operator()(T const& x)
	{
		T fx = x*x*x - a; // Difference (estimate x^3 - a).
		return fx;
	}
private:
	T a; // to be 'cube_rooted'.
};

template <class T>
T cbrt_1(T x)
{ // return cube root of x using bracket_and_solve (no derivatives).
	using namespace std;  // Help ADL of std functions.
	using namespace boost::math;
	int exponent;
	frexp(x, &exponent); // Get exponent of z (ignore mantissa).
	T guess = ldexp(1., exponent / 3); // Rough guess is to divide the exponent by three.
	T factor = 2; // To multiply 
	int digits = std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy for type T.
												 // digits used to control how accurate to try to make the result.
	int get_digits = (digits * 3) / 4; // Near maximum (3/4) possible accuracy.
									   //cout  << ", std::numeric_limits<" << typeid(T).name()  << ">::digits = " << digits 
									   //   << ", accuracy " << get_digits << " bits."<< endl;

									   //boost::uintmax_t maxit = (std::numeric_limits<boost::uintmax_t>::max)();
									   // (std::numeric_limits<boost::uintmax_t>::max)() = 18446744073709551615 
									   // which is more than we might wish to wait for!!!  
									   // so we can choose some reasonable estimate of how many iterations may be needed.
	const boost::uintmax_t maxit = 10;
	boost::uintmax_t it = maxit; // Initally our chosen max iterations, but updated with actual.
								 // We could also have used a maximum iterations provided by any policy:
								 // boost::uintmax_t max_it = policies::get_max_root_iterations<Policy>();
	bool is_rising = true; // So if result if guess^3 is too low, try increasing guess.
	eps_tolerance<double> tol(get_digits);
	std::pair<T, T> r =
		bracket_and_solve_root(cbrt_functor_1<T>(x), guess, factor, is_rising, tol, it);

	// Can show how many iterations (this information is lost outside cbrt_1).
	cout << "Iterations " << maxit << endl;
	if (it >= maxit)
	{ // 
		cout << "Unable to locate solution in chosen iterations:"
			" Current best guess is between " << r.first << " and " << r.second << endl;
	}
	return r.first + (r.second - r.first) / 2;  // Midway between brackets.
} // T cbrt_1(T x)

double BENewtonSolver(Eigen::MatrixXd &DensityMatrix, InputObj &Input, int FragmentIndex, Eigen::MatrixXd &RotationMatrix, double ChemicalPotential, int State, std::tuple< int, int, double> BEPotential)
{

}