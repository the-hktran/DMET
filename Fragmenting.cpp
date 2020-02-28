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
#include <utility>
#include "Bootstrap.h"
#include "Functions.h"
#include "NewtonRaphson.h"
#include "FCI.h"
#include "Fragmenting.h"

void Fragmenting::InitRing(int N, int BEDegree)
{
    int BEDeg = BEDegree;
    if (BEDeg % 2 != 1)
    {
        std::cout << "BE-DMET: Improper bootstrap degree! Setting degree to " << BEDegree + 1 << std::endl;
        BEDeg++;
    }
    int Neighbors = (BEDeg - 1) / 2;

    AdjacencyMatrix = Eigen::MatrixXi::Zero(N, N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < Neighbors; j++)
        {
            AdjacencyMatrix(i, (i + (j + 1)) % N) = 1;
            AdjacencyMatrix(i, (i + N - (j + 1)) % N) = 1;
        }
    }

    OneCenterIteration();
    OneCenterMatching(false);
    isTS = true;
    RingAnalogue();
}

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
        sort(xFrag.begin(), xFrag.end());
        Fragments.push_back(xFrag);
    }
}

void Fragmenting::OneCenterMatching(bool Match2RDM)
{
    MatchingConditions.clear();
    for (int x = 0; x < AdjacencyMatrix.cols(); x++)
    {
        std::vector< std::tuple<int, int, int, int, int, bool, bool> > tmpVec;
        for (int i = 0; i < Fragments[x].size(); i++)
        {
            // Center pieces should not be matched here.
            if (Fragments[x][i] == CenterPosition[x][0]) continue;
            int FragMatch;
            // Find which other fragment has this element as the center.
            for (int y = 0; y < CenterPosition.size(); y++)
            {
                if (CenterPosition[y][0] == Fragments[x][i])
                {
                    FragMatch = y;
                    break;
                }
            }
            tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], -1, -1, true, true));
            // tmpVec.push_back(std::make_tuple(FragMatch, Fragments[x][i], Fragments[x][i], -1, -1, false, false)); // Comment this out when matching full density.
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

void Fragmenting::RingAnalogue()
{
    int NumFrag = AdjacencyMatrix.cols();
    for (int x = 0; x < NumFrag; x++)
    {
        for (int i = 0; i < Fragments[x].size(); i++)
        {
            OrbitalAnalog[std::make_pair(x, Fragments[x][i])] = (NumFrag + (Fragments[x][i] - x)) % NumFrag;
        }
    }
}

void Fragmenting::InitGrid(int Nx, int Ny)
{
    int N = Nx * Ny;
    AdjacencyMatrix = Eigen::MatrixXi::Zero(N, N);
    for (int i = 0; i < N; i++)
    {
        // Coordinate of i'th atom on the grid.
        int xPos = i % Nx;
        int yPos = floor(i / Nx);

        // Check to see if there is something to the left and right, check to make sure the i'th atom is not on the left or right edge.
        if (xPos != 0)
        {
            AdjacencyMatrix(i, i - 1) = 1;
        }
        if (xPos != Nx - 1)
        {
            AdjacencyMatrix(i, i + 1) = 1;
        }

        // Same thing with the top and bottom.
        if (yPos != 0)
        {
            AdjacencyMatrix(i, i - Nx) = 1;
        }
        if (yPos != Ny - 1)
        {
            AdjacencyMatrix(i, i + Nx) = 1;
        }
    }

    OneCenterIteration();
    OneCenterMatching(false);
}

void Fragmenting::PrintFrag()
{
    for (int x = 0; x < Fragments.size(); x++)
    {
        std::cout << "Frag " << x << std::endl;
        for (int i = 0; i < Fragments[x].size(); i++)
        {
            std::cout << Fragments[x][i] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "Matching Conditions" << std::endl;
    for (int x = 0; x < MatchingConditions.size(); x++)
    {
        for (int i = 0; i < MatchingConditions[x].size(); i++)
        {
            std::cout << x << "\t" << i << "\t" << std::get<0>(MatchingConditions[x][i]) << "\t" << std::get<1>(MatchingConditions[x][i]) << "\t" << std::get<2>(MatchingConditions[x][i]) << std::endl;
        }
    }
}