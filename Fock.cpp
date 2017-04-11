#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>

double Metric(int NumElectrons, Eigen::MatrixXd &FirstDensityMatrix, Eigen::MatrixXd &SecondDensityMatrix);
double BiasMatrixElement(int Row, int Col, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, Eigen::MatrixXd &CurrentDensity, int NumElectrons);

/// <summary>
/// Calculates the electron-electron repulsion term for each matrix element of the Fock matrix: 
/// F_mn = sum_ij D_ij [2 (mn|ij) - (mi|jn)]
/// </summary>
/// <param name="m">
/// Row of the Fock matrix.
/// </param>
/// <param name="n">
/// Column of the Fock matrix.
/// </param>
/// <param name="DensityMatrix">
/// Density matrix of the current iteration.
/// </param>
/// <param name="Integrals">
/// Map to value of two electron integrals.
/// </param>
double ExchangeTerm(int m, int n, Eigen::MatrixXd &DensityMatrix, std::map<std::string, double> &Integrals)
{
    double XTerm = 0; // 
    for(int i = 0; i < DensityMatrix.rows(); i++)
    {
        for(int j = 0; j < DensityMatrix.cols(); j++)
        {
            XTerm += DensityMatrix(i, j) * (2 * Integrals[std::to_string(m + 1) + " " + std::to_string(n + 1) + " " + std::to_string(i + 1) + " " + std::to_string(j + 1)]
                                              - Integrals[std::to_string(m + 1) + " " + std::to_string(i + 1) + " " + std::to_string(j + 1) + " " + std::to_string(n + 1)]);
        }
    }
    return XTerm;
}

/// <summary>
/// Takes a density matrix and calculates the corresponding Fock matrix.
/// </summary>
/// <param name="FockMatrix">
/// Container for Fock matrix.
/// </param>
/// <param name="DensityMatrix">
/// Density matrix of the current iteration.
/// </param>
/// <param name="Integrals">
/// Map to value of two electron integrals.
/// </param>
/// <param name="Bias">
/// Vector of triples containing the biasing potential parameters (density matrix, N, lambda) for the metadynamics bias.
/// </param>
/// <param name="NumElectrons">
/// Number of electrons in the system.
/// </param>
void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &DensityMatrix, std::map<std::string, double> &Integrals, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int NumElectrons)
{
    for(int m = 0; m < FockMatrix.rows(); m++)
    {
        for(int n = m; n < FockMatrix.cols(); n++)
        {
            FockMatrix(m, n) = Integrals[std::to_string(m + 1) + " " + std::to_string(n + 1) + " 0 0"] // This is HCore
                             + ExchangeTerm(m, n, DensityMatrix, Integrals)
                             + BiasMatrixElement(m, n, Bias, DensityMatrix, NumElectrons); // Metadynamics bias.
            FockMatrix(n, m) = FockMatrix(m, n);
        }
    }
}
