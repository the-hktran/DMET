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

        // std::vector< std::vector<bool> > aStrings;
        // std::vector< std::vector<bool> > bStrings;

        double *aOEI;
        double *bOEI;
        double *aaTEI;
        double *bbTEI;
        double *abTEI;
        double *aOEIPlusCore;
        double *bOEIPlusCore;
        double ENuc;

        int Conditioner = 10;
        int MaxIteration = 10000;

        double ChemicalPotential = 0.0;

        std::vector<double> Energies;
        std::vector<Eigen::MatrixXd> Eigenvectors;
        std::vector<double> Symmetries;
        std::vector<double> FCIErrors;

        std::vector< Eigen::MatrixXd > OneRDMs;
        std::vector< std::vector<double> > TwoRDMs;
        // std::vector< Eigen::Tensor<double, 4> > TwoRDMs;

        FCI(InputObj&);
        FCI(InputObj&, int, int, std::vector<int>, std::vector<int>, std::vector<int>);
        FCI(InputObj&, int, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>);

        void ERIMapToArray(std::map<std::string, double>&);
        void ERIMapToArray(std::map<std::string, double>&, Eigen::MatrixXd RotationMatrix, std::vector<int> ActiveOrbitals);
        void ERIMapToArray(std::map<std::string, double>&, std::map<std::string, double>&, std::map<std::string, double>&, Eigen::MatrixXd aRotationMatrix, Eigen::MatrixXd bRotationMatrix, std::vector<int> aActiveList, std::vector<int> bActiveList);
        void AddChemicalPotentialGKLC(std::vector<int>, double);
        void runFCI();
        void getSpecificRDM(int, bool);
        void getRDM(bool);
        double calcImpurityEnergy(int, std::vector<int>);

    private:
        void InitFromInput(InputObj&);
        void GetOrbitalString(int, int, int, std::vector<bool>&);
        Eigen::Tensor<double, 4> Make2RDMTensor(std::vector<double>, int);

        // From troyfci.cpp
        bool FCIman(const int N, const int No, const int Nstr,
                    const int N0, const int NS,
                    const double *h, const double *V, vMatrixXd& Xi,
                    dv1& Ei, dv1& Sym, dv1& Uncertainty, int& iter,
                    const int MAXITER, const double THRESH, const bool iprint);
        void RDM12(const int N, const int No, const int Nstr,
                   const MatrixXd& X, MatrixXd& D, dv1& G, const bool compute_rdm2);
        void HX_(const int N, const int No, const int Nstr,
                 const MatrixXd& X, const double *h, const double *V, MatrixXd& XH);
        void UHX_(const int N, const int No, const int Nstr,
                  MatrixXd& X, const double *ha, const double *hb,
                  const double *Vaa, const double *Vab, const double *Vbb,
                  MatrixXd& XH);
};