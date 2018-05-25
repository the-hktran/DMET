#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/CXX11/Tensor>
// Header files containg a list of all the functions used in the DMET code.

// DMET.cpp
void SchmidtDecomposition(Eigen::MatrixXd&, Eigen::MatrixXd&, std::vector< int >, std::vector< int >, int, std::ofstream&);
void GetCASPos(InputObj, int, std::vector< int >&, std::vector< int >&);
double OneElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int c, int d);
double TwoElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXd &RotationMatrix, int c, int d, int e, int f);
int ReducedIndexToOrbital(int c, InputObj Input, int FragmentIndex);
double OneElectronPlusCore(InputObj &Input, Eigen::MatrixXd &RotationMatrix, int FragmentIndex, int c, int d);

// FCIForBE.cpp
std::vector< double > BEImpurityFCI(Eigen::MatrixXd&, InputObj&, int, Eigen::MatrixXd&, double, int, std::vector< std::tuple< int, int, double> >);

// FCI.cpp
Eigen::Tensor<double, 4> Form2RDM(InputObj&, int, Eigen::VectorXf, std::vector< std::vector< bool > >, std::vector< std::vector< bool > >, Eigen::MatrixXd);
Eigen::MatrixXd Form1RDM(InputObj&, int, Eigen::VectorXd, std::vector< std::vector< bool > >, std::vector< std::vector< bool > >);
int BinomialCoeff(int, int); // n choose k
int Z_ForIndex(int, int, int, int);
int StringIndex(std::vector<int>, int);
void GetOrbitalString(int, int, int, std::vector<bool>&);
short int CountDifferences(std::vector<bool>, std::vector<bool>);
short int FindSign(std::vector<bool>, std::vector<bool>);
short int AnnihilationParity(std::vector< bool >, int);
std::vector<unsigned short int> ListOrbitals(std::vector<bool>);
std::vector<unsigned short int> ListDifference(std::vector<bool>, std::vector<bool>);
float TwoElectronIntegral(unsigned short int, unsigned short int, unsigned short int, unsigned short int, bool, bool, bool, bool, std::map<std::string, double>&, Eigen::MatrixXd&);
short int CountOrbitalPosition(unsigned short int, bool, std::vector<unsigned short int>, int);
short int CountSameImpurity(std::vector<bool>, std::vector<bool>, std::vector<int>);
std::vector< double > ImpurityFCI(Eigen::MatrixXd&, InputObj&, int, Eigen::MatrixXd&, double, int,  Eigen::Tensor<double, 4>&, Eigen::VectorXd&);

// ComplexDMET.cpp
std::complex<double> OneElectronPlusCore(InputObj&, Eigen::MatrixXcd&, int, int, int);
std::complex<double> TwoElectronEmbedding(std::map<std::string, double>&, Eigen::MatrixXcd&, int, int, int, int);
std::complex<double> OneElectronEmbedding(std::map<std::string, double>&, Eigen::MatrixXcd&, int, int);
Eigen::MatrixXcd TDHamiltonian(Eigen::MatrixXcd &DensityMatrix, InputObj&, int, Eigen::MatrixXcd&, double, int, Eigen::Tensor<std::complex<double>, 4>&, std::vector< std::vector< bool > >&, std::vector< std::vector< bool > >&);
Eigen::MatrixXcd Form1RDMComplex(InputObj&, int, Eigen::VectorXcd, std::vector< std::vector< bool > >, std::vector< std::vector< bool > >);
Eigen::Tensor<std::complex<double>, 4> Form2RDM(InputObj&, int, Eigen::VectorXcd, std::vector< std::vector< bool > >, std::vector< std::vector< bool > >, Eigen::MatrixXcd);