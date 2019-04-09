#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tuple>

class Fragmenting
{
    public:
        Eigen::MatrixXi AdjacencyMatrix;
        // First index is the fragment
        // Second index is the node in the fragment. The first node is the center.
        std::vector< std::vector<int> > Fragments;
        std::vector< std::vector<int> > CenterPosition;

        // First vector is the fragment
        // Second vector are the matching conditions for that one fragment.
        // First Tuple Element is the fragment which has the center position to match
        // Element 2-5 are the index of the RDM to match on the current fragment.
        // Element 6-7 are the spins
        std::vector< std::vector< std::tuple<int, int, int, int, int, bool, bool> > > MatchingConditions;

        Fragmenting(int);
        
    private:
        void OneCenterIteration();
        void OneCenterMatching(bool);
};