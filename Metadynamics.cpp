#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <tuple>
#include "ReadInput.h"
#include <fstream>
#include <map>
#include <stdlib.h> 
#include <algorithm> // std::sort
#include <iomanip>

double SCF(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int SolnNum, Eigen::MatrixXd &DensityMatrix, InputObj &Input, std::ofstream &Output, Eigen::MatrixXd &SOrtho, Eigen::MatrixXd &HCore, std::vector< double > &AllEnergies, Eigen::MatrixXd &CoeffMatrix, std::vector<int> &OccupiedOrbitals, std::vector<int> &VirtualOrbitals, int &SCFCount, int MaxSCF);
void BuildFockMatrix(Eigen::MatrixXd &FockMatrix, Eigen::MatrixXd &DensityMatrix, std::map<std::string, double> &Integrals, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, int NumElectrons);
void GenerateRandomDensity(Eigen::MatrixXd &DensityMatrix);

/// <summary>
/// This function makes a new density matrix to be used for the next metadynamics iteration. 
/// It switches an occupied orbitals with a random virtual orbital
/// </summary>
/// <param name="DensityMatrix">
/// Current density matrix. The new density matrix is stored here.
/// </param>
/// <param name="CoeffMatrix">
/// Current coefficient matrix. This is what gets rotated and the new density matrix is calculated from this.
/// </param>
/// <param name="OccupiedOrbitals">
/// Vector of occupied orbitals.
/// </param>
/// <param name="VirtualOrbitals">
/// Vector of virtual orbitals.
/// </param>
void NewDensityMatrix(Eigen::MatrixXd &DensityMatrix, Eigen::MatrixXd &CoeffMatrix, std::vector<int> OccupiedOrbitals, std::vector<int> VirtualOrbitals)
{
    int ExcludedOcc = rand() % OccupiedOrbitals.size();
    int IncludedVirt = rand() % VirtualOrbitals.size();

    double Cos = 1 - 2 * (rand() / RAND_MAX); // A random value between -1 and 1;
    double Sin = sqrt(1 - Cos * Cos); // Corresponding sin value.
	rand() % 2 == 0 ? Sin = Sin : Sin = -1 * Sin; // Random parity.

    Eigen::MatrixXd RotatedCoeff = CoeffMatrix;
    for (int i = 0; i < RotatedCoeff.rows(); i++)
    {
        RotatedCoeff(i, OccupiedOrbitals[ExcludedOcc]) = Cos * CoeffMatrix(i, OccupiedOrbitals[ExcludedOcc]) - Sin * CoeffMatrix(i, VirtualOrbitals[IncludedVirt]);
    }
    for (int i = 0; i < DensityMatrix.rows(); i++)
	{
		for (int j = 0; j < DensityMatrix.cols(); j++)
		{
			double DensityElement = 0;
			for (int k = 0; k < OccupiedOrbitals.size(); k++)
			{
				DensityElement += RotatedCoeff(i, OccupiedOrbitals[k]) * RotatedCoeff(j, OccupiedOrbitals[k]);
			}
			DensityMatrix(i, j) = DensityElement;
		}
	}
}

/// <summary>
/// Changes the biasing parameters to increase the width of the biasing potential when
/// the SCF method converges to the same solution. N_x is increased (increase height)
/// while lambda_x is decreased (increase width).
/// </summary>
/// <param name="Bias">
/// List of biases containing the parameters for each bias. The new parameters are stored here.
/// </param>
/// <param name="WhichSoln">
/// Indicates which solution (order agrees with order in Bias) SCF has converged to. We only
/// increment the bias associated with this solution.
/// </param>
void ModifyBias(std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, short int WhichSoln)
{
    if(WhichSoln == -1) // Means the solution was positive, not reconverged. Don't do anything.
    {
        return;
    }
    double BiasScale = 1.1; // Scale to increase and decrease parameters. Hard coded for now.
    double NewNorm = std::get<1>(Bias[WhichSoln]) * BiasScale; // Increase height of Gaussian.
    double NewLambda = std::get<2>(Bias[WhichSoln]) / BiasScale; // Increase width of Gaussian (lambda is the inverse variance).
    if(NewNorm > 100)
    {
        NewNorm = 100 * (rand() / RAND_MAX);
    }
    if(NewLambda < 1E-10)
    {
        NewLambda = 20 * (rand() / RAND_MAX) + 1E-10;
    }
    std::tuple< Eigen::MatrixXd, double, double > NewTuple = std::make_tuple(std::get<0>(Bias[WhichSoln]), NewNorm, NewLambda);
    Bias[WhichSoln] = NewTuple;
}

/// <summary>
/// Calculates the distance metric between two density matrices. Equation (7) 
/// in AJW Thom and M Head-Gordon, Phys. Rev. Lett., 101, 193001 (2008)
/// </summary>
/// <param name="NumElectrons">
/// Number of electrons.
/// </param>
/// <param name="FirstDensityMatrix">
/// Density matrix.
/// </param>
/// <param name="SecondDensityMatrix">
/// Other density matrix. We calculate the distance between these matrices.
/// </param>
double Metric(int NumElectrons, Eigen::MatrixXd &FirstDensityMatrix, Eigen::MatrixXd &SecondDensityMatrix)
{
    double d = 0;
    for(int i = 0; i < FirstDensityMatrix.rows(); i++)
    {
        for(int j = 0; j < FirstDensityMatrix.cols(); j++)
        {
            d += FirstDensityMatrix(i, j) * SecondDensityMatrix(j, i);
        }
    }
    d = (double)NumElectrons - d;
    return d;
}

/// <summary>
/// This calculates the form of the biasing potential in the modified Fock matrix. 
/// This is the second term in Equation (9) of AJW Thom and M Head-Gordon, Phys. Rev. Lett., 101, 193001 (2008)
/// </summary>
/// <param name="Bias">
/// List of biases. There is a term for each bias.
/// </param>
/// <param name="CurrentDensity">
/// The current density matrix of the iteration. We want the distance of previous solutions to this density.
/// </param>
/// <param name="NumElectrons">
/// Number of electrons. Needed for metric.
/// </param>
double BiasMatrixElement(int Row, int Col, std::vector< std::tuple< Eigen::MatrixXd, double, double > > &Bias, Eigen::MatrixXd &CurrentDensity, int NumElectrons)
{
    double BiasElement = 0;
    for(int i = 0; i < Bias.size(); i++)
    {
        BiasElement += std::get<0>(Bias[i])(Row, Col) * std::get<1>(Bias[i]) * std::get<2>(Bias[i]) * exp(-1 * std::get<2>(Bias[i]) * Metric(NumElectrons, CurrentDensity, std::get<0>(Bias[i])));
    }
    return BiasElement;
}

int InitMetadynamics(int argc, char* argv[])
{
    InputObj Input;
    if(argc == 4)
    {
        Input.SetNames(argv[1], argv[2], argv[3]);
    }
    else
    {
        Input.GetInputName();
    }

    /* This part will do a scan. It repeats much of the part below. */
    if(Input.doScan)
    {
        std::ofstream TotalOutput(Input.OutputName);
        TotalOutput << "Self-Consistent Field Metadynamics Scan\n" << std::endl;
        TotalOutput << "Steps from " << Input.ScanIntStart << " to " << Input.ScanIntEnd << std::endl;
        for(int IT = Input.ScanIntStart; IT <= Input.ScanIntEnd; IT++) // Loop over all scan iterations. Basically just pasted the rest of main into here.
        {
            /* Initialize a new object for each iteration */
            InputObj IterationInput;
            std::string IterationIntegralsName = Input.IntegralsInput + "_" + std::to_string(IT); // (integralsname)_1, ...
            std::string IterationOverlapName = Input.OverlapInput + "_" + std::to_string(IT);
            std::string IterationOutputName = Input.OutputName + "_" + std::to_string(IT);
            IterationInput.SetNames(IterationIntegralsName, IterationOverlapName, IterationOutputName);
            IterationInput.Set();

            /* Any mention of "Input" after this point is a bug. */
            std::ofstream IterationOutput(IterationInput.OutputName);

            IterationOutput << "Self-Consistent Field Metadynamics Calculation" << std::endl;
            IterationOutput << "\n" << IterationInput.NumSoln << " solutions desired." << std::endl;

            IterationOutput << "\n" << "Settings:" << std::endl;
            IterationOutput << "Number of Orbitals = " << IterationInput.NumAO << std::endl;
            IterationOutput << "Number of Electrons = " << IterationInput.NumElectrons << std::endl;
            IterationOutput << "Use DIIS?: " << IterationInput.Options[0] << std::endl;
            IterationOutput << "Use MOM?: " << IterationInput.Options[1] << std::endl;
            IterationOutput << "Density choice: " << IterationInput.DensityOption << "\n" << std::endl;

            Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemS(IterationInput.OverlapMatrix);
            Eigen::SparseMatrix< double > LambdaSOrtho(IterationInput.NumAO, IterationInput.NumAO); // Holds the inverse sqrt matrix of eigenvalues of S ( Lambda^-1/2 )
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            for(int i = 0; i < IterationInput.NumAO; i++)
            {
                tripletList.push_back(T(i, i, 1 / sqrt(EigensystemS.eigenvalues()[i])));
            }
            LambdaSOrtho.setFromTriplets(tripletList.begin(), tripletList.end());
            
            Eigen::MatrixXd SOrtho = EigensystemS.eigenvectors() * LambdaSOrtho * EigensystemS.eigenvectors().transpose();

            /* Initialize the density matrix. We're going to be smart about it and use the correct ground state density
            corresponding to Q-Chem outputs. Q-Chem uses an MO basis for its output, so the density matrix has ones
            along the diagonal for occupied orbitals. */
            Eigen::MatrixXd DensityMatrix = Eigen::MatrixXd::Zero(IterationInput.NumAO, IterationInput.NumAO); //
            for(int i = 0; i < IterationInput.NumOcc; i++)
            {
                DensityMatrix(i, i) = 1;
            }

            std::vector< std::tuple< Eigen::MatrixXd, double, double > > Bias; // Tuple containing DensityMatrix, N_x, lambda_x
            Eigen::MatrixXd HCore(IterationInput.NumAO, IterationInput.NumAO);
            Eigen::MatrixXd ZeroMatrix = Eigen::MatrixXd::Zero(IterationInput.NumAO, IterationInput.NumAO);
            BuildFockMatrix(HCore, ZeroMatrix, IterationInput.Integrals, Bias, IterationInput.NumElectrons); // Form HCore (D is zero)

            double Energy;

            std::vector< double > AllEnergies;
            Eigen::MatrixXd CoeffMatrix = Eigen::MatrixXd::Zero(IterationInput.NumAO, IterationInput.NumAO);
            std::vector<int> OccupiedOrbitals(IterationInput.NumOcc);
            std::vector<int> VirtualOrbitals(IterationInput.NumAO - IterationInput.NumOcc);
            for(int i = 0; i < IterationInput.NumAO; i++)
            {
                if(i < IterationInput.NumOcc)
                {
                    OccupiedOrbitals[i] = i;
                }
                else
                {
                    VirtualOrbitals[i - IterationInput.NumOcc] = i;
                }
            }

            int SCFCount = 0;
            for(int i = 0; i < IterationInput.NumSoln; i++)
            {
                std::tuple< Eigen::MatrixXd, double, double > tmpTuple;
                NewDensityMatrix(DensityMatrix, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals); // CoeffMatrix is zero so this doesn't do anything the  first time.
                Energy = SCF(Bias, i + 1, DensityMatrix, IterationInput, IterationOutput, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, IterationInput.MaxSCF);
                if(SCFCount >= IterationInput.MaxSCF && IterationInput.MaxSCF != -1) 
                {
                    std::cout << "SCF MetaD: Maximum number of SCF iterations reached." << std::endl;
                    break;
                }
                tmpTuple = std::make_tuple(DensityMatrix, 0.1, 1);
                Bias.push_back(tmpTuple);
            }

            /* This ends the single point calculation. Now the output file records the information. */
            std::sort(AllEnergies.begin(), AllEnergies.end()); // Adiabatic representation!
            TotalOutput << Input.ScanValStart + IT * Input.ScanValStep;
            for(int i = 0; i < AllEnergies.size(); i++)
            {
                TotalOutput << "\t" << AllEnergies[i];
            }
            TotalOutput << std::endl;
        } // end iterations loop
        return 0;
    } // end scan loop

    /* Everything below here is for a single point calculation. We handle scans before this and terminate the program
       before this point. */
    Input.Set();

	std::ofstream Output(Input.OutputName);
    Output << std::fixed << std::setprecision(10);

    /* Set up the output file. */

	Output << "Self-Consistent Field Metadynamics Calculation" << std::endl;
	Output << "\n" << Input.NumSoln << " solutions desired." << std::endl;

    Output << "\n" << "Settings:" << std::endl;
    Output << "Number of Orbitals = " << Input.NumAO << std::endl;
    Output << "Number of Electrons = " << Input.NumElectrons << std::endl;
    Output << "Use DIIS?: " << Input.Options[0] << std::endl;
    Output << "Use MOM?: " << Input.Options[1] << std::endl;
    Output << "Density choice: " << Input.DensityOption << "\n" << std::endl;

    /* We calculate S^-1/2, which is used to put everything in an orthogonal basis */
    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > EigensystemS(Input.OverlapMatrix);
    Eigen::SparseMatrix< double > LambdaSOrtho(Input.NumAO, Input.NumAO); // Holds the inverse sqrt matrix of eigenvalues of S ( Lambda^-1/2 )
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    for(int i = 0; i < Input.NumAO; i++)
    {
        tripletList.push_back(T(i, i, 1 / sqrt(EigensystemS.eigenvalues()[i])));
    }
    LambdaSOrtho.setFromTriplets(tripletList.begin(), tripletList.end());
    
    Eigen::MatrixXd SOrtho = EigensystemS.eigenvectors() * LambdaSOrtho * EigensystemS.eigenvectors().transpose(); // S^-1/2

    /* Initialize the density matrix. We're going to be smart about it and use the correct ground state density
       corresponding to Q-Chem outputs. Q-Chem uses an MO basis for its output, so the density matrix has ones
       along the diagonal for occupied orbitals. */
    Eigen::MatrixXd DensityMatrix = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO); //
    for(int i = 0; i < Input.NumOcc; i++)
    {
        DensityMatrix(i, i) = 1;
    }
    if(Input.OverlapInput == "c")
    {
        DensityMatrix = Input.InitialCoeff * Input.InitialCoeff.transpose();
        std::cout << Input.InitialCoeff << std::endl;
    }
    
    std::vector< std::tuple< Eigen::MatrixXd, double, double > > Bias; // Tuple containing DensityMatrix, N_x, lambda_x
    Eigen::MatrixXd HCore(Input.NumAO, Input.NumAO); // T_e + V_eN
    Eigen::MatrixXd ZeroMatrix = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
    BuildFockMatrix(HCore, ZeroMatrix, Input.Integrals, Bias, Input.NumElectrons); // Form HCore (D is zero)

    double Energy;

    std::vector< double > AllEnergies; // Stores all SCF energies. Used to check if solution is unique.
    Eigen::MatrixXd CoeffMatrix = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
    std::vector<int> OccupiedOrbitals(Input.NumOcc);
    std::vector<int> VirtualOrbitals(Input.NumAO - Input.NumOcc);
    /* Initialize the virtual and occupied orbitals. We choose the occupied orbitals to be the
       lowest n / 2 orbitals first */
    for(int i = 0; i < Input.NumAO; i++)
    {
        if(i < Input.NumOcc)
        {
            OccupiedOrbitals[i] = i;
        }
        else
        {
            VirtualOrbitals[i - Input.NumOcc] = i;
        }
    }

    int SCFCount = 0; // Number of SCF iterations. Allows us to terminate after a maximum number of iterations.
    for(int i = 0; i < Input.NumSoln; i++)
    {
        std::tuple< Eigen::MatrixXd, double, double > tmpTuple;
        if(Input.StartLambda == 10) // This gets solution 9. No good reason, just random.
        {
            NewDensityMatrix(DensityMatrix, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals); // CoeffMatrix is zero so this doesn't do anything the  first time.
        }
        Energy = SCF(Bias, i + 1, DensityMatrix, Input, Output, SOrtho, HCore, AllEnergies, CoeffMatrix, OccupiedOrbitals, VirtualOrbitals, SCFCount, Input.MaxSCF);
        if(SCFCount >= Input.MaxSCF && Input.MaxSCF != -1) // Means we have exceeded maximum SCF iterations, and we didn't ask to do it indefinately.
        {
            std::cout << "SCF MetaD: Maximum number of SCF iterations reached." << std::endl;
            break;
        }
        tmpTuple = std::make_tuple(DensityMatrix, Input.StartNorm, Input.StartLambda); // Add a new bias for the new solution. Starting N_x and lambda_x are here.
        Bias.push_back(tmpTuple);
    }

    return 0;
}
