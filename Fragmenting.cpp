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
#include "Bootstrap.h"
#include "Functions.h"
#include "NewtonRaphson.h"
#include "FCI.h"
#include "Fragmenting.h"

void Fragmenting::OneCenterIteration()
{
    Fragments.clear();
    CenterPosition.clear();
    int NumFrag = AdjacencyMatrix.cols();
    for (int x = 0; x < NumFrag; x++)
    {
        std::vector<int> tmpVec;
        tmpVec.push_back(x);
        CenterPosition.push_back(tmpVec);
        std::vector<int> xFrag;
        xFrag.push_back(x);
        for (int i = 0; i < AdjacencyMatrix.rows(); i++)
        {
            if (x == i) continue;
            if (AdjacencyMatrix.coeffRef(i, x) == 1) xFrag.push_back(i);
        }
        Fragments.push_back(xFrag);
    }
}

void Fragmenting::OneCenterMatching(bool Match2RDM)
{
    MatchingConditions.clear();
    for (int x = 0; x < AdjacencyMatrix.cols(); x++)
    {
        std::vector< std::tuple<int, int, int, int, int, bool, bool> > tmpVec;
        for (int i = 1; i < Fragments[x].size(); i++)
        {
            int FragMatch;
            // Find which other fragment has this element as the center.
            for (int y = 0; y < CenterPosition.size(); y++)
            {
                if (CenterPosition[y] == Fragments[x][i])
                {
                    FragMatch = y;
                    break;
                }
            }
            tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], -1, -1, true, true));
            tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], -1, -1, false, false));
            if (Match2RDM)
            {
                tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], Fragments[x][i], Fragments[x][i], true, true));
                tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], Fragments[x][i], Fragments[x][i], true, false));
                tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], Fragments[x][i], Fragments[x][i], false, false));
            }
        }
        MatchingConditions.push_back(tmpVec);
    }
}