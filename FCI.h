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

class FCI
{
    public:
        int NumActive, NumVirt, NumCore, NumOcc, NumOrbitals,
            aElectrons, bElectrons, aElectronsActive, bElectronsActive, aActive, bActive, aOrbitals, bOrbitals, aCore, bCore, aVirtual, bVirtual,
            NumberOfEV, L, aDim, bDim, Dim;

        std::string Method; // "FCI", "CIS"

        // std::vector<int> aCoreList, bCoreList, aActiveList, bActiveList, aVirtualList, bVirtualList;

        std::vector< std::vector<bool> > aStrings;
        std::vector< std::vector<bool> > bStrings;

        FCI(InputObj&);
        FCI(InputObj&, int, int, std::vector<int>, std::vector<int>, std::vector<int>);
        FCI(InputObj&, int, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>);

        void Init();

    private:
        void InitFromInput(InputObj&);
        void GetOrbitalString(int, int, int, std::vector<bool>&);
};