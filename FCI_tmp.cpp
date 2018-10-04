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
#include "FCI.h"

/* 
   This is the initalization function when only the input object is given. In this case, we assume that the active space
   is the complete space 
*/
FCI::FCI(InputObj &Input)
{
    aElectrons = Input.NumOcc;
    bElectrons = Input.NumOcc;
    aActive = Input.NumAO;
    bActive = Input.NumAO;
    aCore = 0;
    bCore = 0;
    aVirtual = 0;
    bVirtual = 0;

    aDim = BinomialCoeff(aElectrons, aActive);
    bDim = BinomialCoeff(bElectrons, bActive);
    Dim = aDim * bDim;

    NumberOfEV = Input.NumberOfEV;
    L = NumberOfEV + 50;

    for (int i = 0; i < aDim; i++)
    {
        std::vector<bool> DetString;
        GetOrbitalString(i, aElectrons, aActive, DetString);
        for (int j = 0; j < aVirtual; j++)
        {
            DetString.push_back(false);
        }
        for (int j = 0; j < aCore; j++)
        {
            DetString.insert(DetString.begin(), true);
        }
        aStrings.push_back(DetString);
    }

    for (int i = 0; i < bDim; i++)
    {
        std::vector<bool> DetString;
        GetOrbitalString(i, bElectrons, bActive, DetString);
        for (int j = 0; j < bVirtual; j++)
        {
            DetString.push_back(false);
        }
        for (int j = 0; j < bCore; j++)
        {
            DetString.insert(DetString.begin(), true);
        }
        bStrings.push_back(DetString);
    }
}

FCI::FCI(InputObj &Input, int aElectronsActive, int bElectronsActive, std::vector<int> CoreList, std::vector<int> ActiveList, std::vector<int> VirtualList)
{
    aElectrons = Input.NumOcc;
    bElectrons = Input.NumOcc;
    aActive = ActiveList.size();
    bActive = ActiveList.size();
    aCore = CoreList.size();
    bCore = CoreList.size();
    aVirtual = VirtualList.size();
    bVirtual = VirtualList.size();
    aOrbitals = Input.NumAO;
    bOrbitals = Input.NumAO;

    aDim = BinomialCoeff(aElectronsActive, aActive);
    bDim = BinomialCoeff(bElectronsActive, bActive);
    Dim = aDim * bDim;

    NumberOfEV = Input.NumberOfEV;
    L = NumberOfEV + 50;

    /* Generate the strings. We have to insert the virtual and core orbitals according to their order in the list. */
    std::vector<int> CoreAndVirtualList = CoreList;
    CoreAndVirtualList.insert(CoreAndVirtualList.end(), VirtualList.begin(), VirtualList.end());
    std::sort(CoreAndVirtualList.begin(), CoreAndVirtualList.end());

    for (int i = 0; i < aDim; i++)
    {
        std::vector<bool> DetString;
        GetOrbitalString(i, aElectronsActive, aActive, DetString);
        for (int j = 0; j < CoreAndVirtualList.size(); j++)
        {
            DetString.insert(DetString.begin() + CoreAndVirtualList[j], std::find(CoreList.begin(), CoreList.end(), CoreAndVirtualList[j]) != CoreList.end());
        }
        aStrings.push_back(DetString);
    }

    for (int i = 0; i < bDim; i++)
    {
        std::vector<bool> DetString;
        GetOrbitalString(i, bElectronsActive, bActive, DetString);
        for (int j = 0; j < CoreAndVirtualList.size(); j++)
        {
            DetString.insert(DetString.begin() + CoreAndVirtualList[j], std::find(CoreList.begin(), CoreList.end(), CoreAndVirtualList[j]) != CoreList.end());
        }
        bStrings.push_back(DetString);
    }
}

FCI::FCI(InputObj &Input, int aElectronsActive, int bElectronsActive, std::vector<int> aCoreList, std::vector<int> aActiveList, std::vector<int> aVirtualList, std::vector<int> bCoreList, std::vector<int> bActiveList, std::vector<int> bVirtualList)
{
    aElectrons = Input.NumOcc;
    bElectrons = Input.NumOcc;
    aActive = aActiveList.size();
    bActive = bActiveList.size();
    aCore = aCoreList.size();
    bCore = bCoreList.size();
    aVirtual = aVirtualList.size();
    bVirtual = bVirtualList.size();
    aOrbitals = Input.NumAO;
    bOrbitals = Input.NumAO;

    aDim = BinomialCoeff(aElectronsActive, aActive);
    bDim = BinomialCoeff(bElectronsActive, bActive);
    Dim = aDim * bDim;

    NumberOfEV = Input.NumberOfEV;
    L = NumberOfEV + 50;

    /* Generate the strings. We have to insert the virtual and core orbitals according to their order in the list. */
    std::vector<int> aCoreAndVirtualList = aCoreList;
    aCoreAndVirtualList.insert(aCoreAndVirtualList.end(), aVirtualList.begin(), aVirtualList.end());
    std::sort(aCoreAndVirtualList.begin(), aCoreAndVirtualList.end());

    std::vector<int> bCoreAndVirtualList = bCoreList;
    aCoreAndVirtualList.insert(bCoreAndVirtualList.end(), bVirtualList.begin(), bVirtualList.end());
    std::sort(bCoreAndVirtualList.begin(), bCoreAndVirtualList.end());

    for (int i = 0; i < aDim; i++)
    {
        std::vector<bool> DetString;
        GetOrbitalString(i, aElectronsActive, aActive, DetString);
        for (int j = 0; j < aCoreAndVirtualList.size(); j++)
        {
            DetString.insert(DetString.begin() + aCoreAndVirtualList[j], std::find(aCoreList.begin(), aCoreList.end(), aCoreAndVirtualList[j]) != aCoreList.end());
        }
        aStrings.push_back(DetString);
    }

    for (int i = 0; i < bDim; i++)
    {
        std::vector<bool> DetString;
        GetOrbitalString(i, bElectronsActive, bActive, DetString);
        for (int j = 0; j < bCoreAndVirtualList.size(); j++)
        {
            DetString.insert(DetString.begin() + bCoreAndVirtualList[j], std::find(bCoreList.begin(), bCoreList.end(), bCoreAndVirtualList[j]) != bCoreList.end());
        }
        bStrings.push_back(DetString);
    }
}

/* This imposes an order onto the binary strings. The function takes an index and returns the corresponding binary string.
   We order the strings as such:
   0: 11000
   1: 10100
   2: 10010
   3: 10001
   4: 01100
   5: 01010
   6: 01001
   7: 00110
   8: 00101
   9: 00011
   We find the binary string recursively. We first ask, is the first digit 0 or 1. The first digit turns into a zero when
   all of the remaining electrons (n_electrons - 1) have permuted through all of the orbitals excluding the lowest one 
   (n_orbitals - 1). So if the index is above (n_orb - 1) choose (n_e - 1), then the first digit is zero. In this case, we
   ask ourselves the same problem for the remaining digits, but now we have one less orbital and we should subtract the number
   of permutations from the index. If the index is below, then the first digit is one. We fill that in and ask ourselves the 
   same question with the remaining digits. It is the same problem but with one less orbital AND one less electron, but the 
   index should not be changed. */
void FCI::GetOrbitalString(int Index, int NumElectrons, int NumOrbitals, std::vector<bool> &OrbitalString)
{
    if(NumOrbitals > 0) // Stop when we have chosen a digit for all orbitals. We take off an orbital each time we fill it.
    {
        int PreviousComb = BinomialCoeff(NumOrbitals - 1, NumElectrons - 1); // Number of ways for the higher electrons to permute in the higher orbitals.
        if (NumElectrons < 1) // If we don't have any electrons left, then all the remaining orbitals have to be empty.
        {
            OrbitalString.push_back(false);
            GetOrbitalString(Index, NumElectrons, NumOrbitals - 1, OrbitalString); // Move onto next orbital, remove one orbital from the list.
        }
        else if(Index < PreviousComb) // Means we have not finished all permutations and there is still an electron in the lowest orbital.
        {
            OrbitalString.push_back(true); // Put this electron there.
            GetOrbitalString(Index, NumElectrons - 1, NumOrbitals - 1, OrbitalString); // Consider the same problem, but with one less orbital and one less electron.
        }
        else // Means we have finished all those permuations and the electron in the first orbital should have moved.
        {
            Index -= PreviousComb; // Truncate the index, since we are considering a reduced problem.
            OrbitalString.push_back(false); // Empty orbital.
            GetOrbitalString(Index, NumElectrons, NumOrbitals - 1, OrbitalString); // Consider the same problem, but with one less orbital and a truncated index.
        }
    }
}