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
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

typedef vector<double> dv1;
typedef vector<int> iv1;
typedef vector<iv1> iv2;
typedef vector<iv2> iv3;
typedef vector<long unsigned int> luiv1;
typedef vector<MatrixXd> vMatrixXd;

class FCI
{
    public:
        InputObj Inp;

        int aElectrons, bElectrons, aElectronsActive, bElectronsActive, aActive, bActive, aOrbitals, bOrbitals, aCore, bCore, aVirtual, bVirtual,
            NumberOfEV, L, aDim, bDim, Dim;

        std::string Method; // "FCI", "CIS"

        std::vector<int> aCoreList, bCoreList, aActiveList, bActiveList, aVirtualList, bVirtualList;

        std::vector< std::vector<bool> > aStrings;
        std::vector< std::vector<bool> > bStrings;

        double *aOEI;
        double *bOEI;
        double *aaTEI;
        double *bbTEI;
        double *abTEI;
        double *aOEIPlusCore;
        double *bOEIPlusCore;
        double ENuc;
        bool doUnrestricted = false;

        int Conditioner = 10;
        int MaxIteration = 10000;

        double ChemicalPotential = 0.0;

        std::vector<double> Energies;
        std::vector<Eigen::MatrixXd> Eigenvectors;
        std::vector<double> Symmetries;
        std::vector<double> FCIErrors;

        // Some things I put in for sigmaFCI
        Eigen::MatrixXd Hamiltonian;
        Eigen::VectorXd SigmaFCIVector;
        Eigen::MatrixXd Henry1RDM;
        Eigen::Tensor<double, 4> Henry2RDM;

        std::vector< Eigen::MatrixXd > OneRDMs;
        std::vector< Eigen::MatrixXd > aOneRDMs;
        std::vector< Eigen::MatrixXd > bOneRDMs;
        std::vector< std::vector<double> > TwoRDMs;
        std::vector< std::vector<double> > aaTwoRDMs;
        std::vector< std::vector<double> > abTwoRDMs;
        std::vector< std::vector<double> > bbTwoRDMs;
        // std::vector< Eigen::Tensor<double, 4> > TwoRDMs;

        FCI();
        FCI(const FCI&);
        FCI(InputObj&);
        FCI(InputObj&, int, int, std::vector<int>, std::vector<int>, std::vector<int>);
        FCI(InputObj&, int, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>);

        void Copy(const FCI&);

        void ERIMapToArray(std::map<std::string, double>&);
        void ERIMapToArray(std::map<std::string, double>&, Eigen::MatrixXd RotationMatrix, std::vector<int> ActiveOrbitals);
        void ERIMapToArray(std::map<std::string, double>&, Eigen::MatrixXd aRotationMatrix, Eigen::MatrixXd bRotationMatrix, std::vector<int> aActiveList, std::vector<int> bActiveList);
        void ERIArrayToMap(std::map<std::string, double>&, std::map<std::string, double>&, std::map<std::string, double>&, std::map<std::string, double>&);
        void RotateERI(double*, double*, Eigen::MatrixXd, Eigen::MatrixXd, double*, double*, double*, double*, double*);
        void AddChemicalPotentialGKLC(std::vector<int>, std::vector<int>, double, double);
        void AddPotential(int, int, double, bool);
        void AddPotential(int, int, int, int, double, bool, bool);
        void runFCI();
        void getSpecificRDM(int, bool);
        void getRDM(bool);
        double calcImpurityEnergy(int, std::vector<int>, std::vector<int>);
        double calcImpurityEnergy(int, std::vector<int>);
        void CalcE1E2(int);
        Eigen::MatrixXd GenerateHamiltonian();
        void doSigmaFCI(double);
        double RDMFromHenryFCI(Eigen::VectorXd, int, Eigen::MatrixXd, Eigen::MatrixXd&);
        void DirectFCI();

        double ExpVal(Eigen::MatrixXd);

        void PrintERI(bool);
        Eigen::MatrixXd ProjectMatrix(std::vector<Eigen::MatrixXd>);
        std::vector<Eigen::MatrixXd> HalfFilledSchmidtBasis(int);
        std::vector<Eigen::MatrixXd> HalfFilledSchmidtBasis(Eigen::MatrixXd);
        Eigen::MatrixXd HFInFCISpace(Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>);

        std::vector<Eigen::MatrixXd> DirectProjection(Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, int);

        void dbgMyShitUp(std::map<std::string, double> &ERIMap, Eigen::MatrixXd Ra, Eigen::MatrixXd Rb);

    private:
        void InitFromInput(InputObj&);
        void GetOrbitalString(int, int, int, std::vector<bool>&);
        Eigen::Tensor<double, 4> Make2RDMTensor(std::vector<double>, int);
        std::vector<Eigen::MatrixXd> EigVecToMatrix(std::vector<Eigen::VectorXd>);

        // From troyfci.cpp
        bool FCIman(const int N, const int No, const int Nstr,
                    const int N0, const int NS,
                    const double *h, const double *V, 
                    vMatrixXd& Xi, dv1& Ei, dv1& Sym, dv1& Uncertainty, int& iter,
                    const int MAXITER, const double THRESH, const bool iprint);
        bool FCIman(const int N, const int No, const int Nstr,
                    const int N0, const int NS,
                    const double *ha, const double *hb, const double *Vbb, const double *Vab, const double *Vaa, // I think HongZhou swapped Vaa and Vbb somewhere in the code. I'll just reverse their order in the function.
                    vMatrixXd& Xi, dv1& Ei, dv1& Sym, dv1& Uncertainty, int& iter,
                    const int MAXITER, const double THRESH, const bool iprint); // This is the unrestricted version of Troy's FCI code.
        void RDM12(const int N, const int No, const int Nstr,
                   const MatrixXd& X, MatrixXd& D, dv1& G, const bool compute_rdm2);
        void URDM12(const int N, const int No, const int Nstr,
	                MatrixXd& X, MatrixXd& aD, MatrixXd& bD, dv1& aaG, dv1& abG, dv1& bbG, const bool compute_rdm2);
        void HX_(const int N, const int No, const int Nstr,
                 const MatrixXd& X, const double *h, const double *V, MatrixXd& XH);
        void UHX_(const int N, const int No, const int Nstr,
                  MatrixXd& X, const double *ha, const double *hb,
                  const double *Vaa, const double *Vab, const double *Vbb,
                  MatrixXd& XH);

        bool sigmaFCI(const int N, const int No, const int Nstr,
                    const int N0, const int NS,
                    const double *h, const double *V, const double w, vMatrixXd& Xi,
                    dv1& Ei, dv1& Sym, dv1& Uncertainty, int& iter,
                    const int MAXITER, const double THRESH, const bool iprint);
};