#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <utility>

class Fragmenting
{
    public:
        Eigen::MatrixXi AdjacencyMatrix;
        // First index is the fragment
        // Second index is the node in the fragment.
        std::vector< std::vector<int> > Fragments;
        std::vector< std::vector<int> > CenterPosition;

        bool isTS = false; // Is the system translationally symmetric?
        // This is a translator which translates orbitals in other fragments to identical orbitals in the first fragment
        // For the key: First int is the fragment number and second int is the orbital we wish to convert
        // For the value: The value is the corresponding orbital on the first fragment
        std::map<std::pair<int, int>, int> OrbitalAnalog;

        // First vector is the fragment
        // Second vector are the matching conditions for that one fragment.
        // First Tuple Element is the fragment which has the center position to match
        // Element 2-5 are the index of the RDM to match on the current fragment.
        // Element 6-7 are the spins
        std::vector< std::vector< std::tuple<int, int, int, int, int, bool, bool> > > MatchingConditions;

        void InitRing(int, int);
        void InitGrid(int, int);

        void PrintFrag();
        
    private:
        void OneCenterIteration();
        void OneCenterMatching(bool);
        void RingAnalogue();
};