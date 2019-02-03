#pragma once
#include <Eigen/Dense>
#include <vector>

class InputObj
{
    public:
        void GetInputName();
        void SetNames(char*, char*, char*);
        void SetNames(std::string, std::string, std::string);
        void Set();
        std::map< std::string, double > Integrals;
        std::map< std::string, double > aaIntegrals;
        std::map< std::string, double > abIntegrals;
        std::map< std::string, double > baIntegrals;
        std::map< std::string, double > bbIntegrals;
        Eigen::MatrixXd OverlapMatrix;
        std::string IntegralsInput;
        std::string OverlapInput;
        std::string OutputName;
		unsigned short int NumOcc;
		unsigned short int NumSoln;
        unsigned short int NumElectrons;
        unsigned short int NumAO;
        std::vector< bool > Options;
        bool doScan = false;
        int ScanIntStart;
        int ScanIntEnd;
        double ScanValStart;
        double ScanValStep;
        int DensityOption = -1; // Will return an error if not set.
        Eigen::MatrixXd InitialCoeff;
        int MaxSCF = 5000;
        double StartNorm = 0.1;
        double StartLambda = 1;

        /* FCI definitions */
        int NumberOfEV = 1;

        /* Definitions for DMET */
        int NumFragments;
        std::vector< std::vector< int > > FragmentOrbitals;
        std::vector< std::vector< int > > EnvironmentOrbitals;
        std::vector<int> ImpurityStates;
        std::vector<int> BathStates;

		/* Some things that are nice to carry around */
		Eigen::MatrixXd SOrtho;
		Eigen::MatrixXd HCore;
		std::vector< int > OccupiedOrbitals;
		std::vector< int > VirtualOrbitals;
};