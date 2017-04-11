#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include "ReadInput.h"
#include <Eigen/Dense>
#include <string>

void InputObj::GetInputName()
{
    std::cout << "SCF MetaD: Enter integral filename:\nSCF MetaD: ";
    std::cin >> IntegralsInput;
    std::cout << "SCF MetaD: Enter overlap filename:\nSCF MetaD: ";
    std::cin >> OverlapInput;
    std::cout << "SCF MetaD: Enter output filename:\nSCF MetaD: ";
    std::cin >> OutputName;
    std::cout << "SCF MetaD: Do a scan? (1 / 0):\nSCF MetaD: ";
    std::cin >> doScan;
    if(doScan)
    {
        std::cout << "SCF MetaD: Enter starting integer:\nSCF MetaD: ";
        std::cin >> ScanIntStart;
        std::cout << "SCF MetaD: Enter ending integer:\nSCF MetaD: ";
        std::cin >> ScanIntEnd;
        std::cout << "SCF MetaD: Enter starting value:\nSCF MetaD: ";
        std::cin >> ScanValStart;
        std::cout << "SCF MetaD: Enter step size:\nSCF MetaD: ";
        std::cin >> ScanValStep;
    }
}

void InputObj::SetNames(char* Int, char* Overlap, char* Out)
{
    IntegralsInput = Int;
    OverlapInput = Overlap;
    OutputName = Out;
}

void InputObj::SetNames(std::string Int, std::string Overlap, std::string Out)
{
    IntegralsInput = Int;
    OverlapInput = Overlap;
    OutputName = Out;
}

/*****                                    FORMAT OF THE INPUT FILE                                 *****/
/* The input file is taken from the integrals computed in QChem (version 4.4). To obtain these integrals,
   the line "CORRELATION idump" must be added to the %rem section of the QChem input. This outputs a list
   of integrals in the format 
                (ij|kl)     i   j   k   l
   in the file titled "INTDUMP". Further modification must be made to this file in order to use it in my
   program. The first two lines must be deleted and the following entered (separated by spaces)
            Number of orbitals
			Number of electrons
            Number of solutions desired
            Use DIIS? (1 / 0)
            Use MOM? (1 / 0)
            How to step through density matrix space (2 / 1 / 0) = (Complete Random / Random Normalized by Trace / Rotation)
            Maximum number of SCF iterations. Use -1 for infinite.
            Starting N_x
            Starting lambda_x
   As an example, here is the first few lines of an input file for H2, in a space of four orbitals with 
   two electrons and we looking for 10 solutions. We want to use DIIS and MOM and we will choose a new density
   by rotating an occupied and unoccupied orbital. We proceed for at most 10000 SCF iterations. The starting norm is 0.1
   and the starting lambda is 1.
            4 2 10
            1 1 0
            10000
            0.1 1
                0.64985185942031   1   1   1   1
                0.16712550470738   1   3   1   1
                0.080102886434995  1   2   1   2
                0.07936780580498   1   4   1   2
                (And the rest of the integrals)                                                        */
void InputObj::Set()
{
    std::ifstream IntegralsFile(IntegralsInput.c_str());
    IntegralsFile >> NumAO >> NumElectrons >> NumSoln;

    /* DIIS and MOM options */
    bool tmpBool1;
    bool tmpBool2; 
    IntegralsFile >> tmpBool1 >> tmpBool2 >> DensityOption;
    Options.push_back(tmpBool1); // Use DIIS
    Options.push_back(tmpBool2); // Use MOM
    IntegralsFile >> MaxSCF >> StartNorm >> StartLambda;

    double tmpDouble;
    int tmpInt1, tmpInt2, tmpInt3, tmpInt4;
    while(!IntegralsFile.eof())
    {
        /* We have to include all 8-fold permuation symmetries. This holds each integral in chemistry notation. We represent
        (ij|kl) as "i j k l". h_ij is "i j 0 0", as given in QChem. */
        IntegralsFile >> tmpDouble >> tmpInt1 >> tmpInt2 >> tmpInt3 >> tmpInt4;
        // IntegralsFile >> tmpInt1 >> tmpInt2 >> tmpInt3 >> tmpInt4 >> tmpDouble;
        Integrals[std::to_string(tmpInt1) + " " + std::to_string(tmpInt2) + " " + std::to_string(tmpInt3) + " " + std::to_string(tmpInt4)] = tmpDouble;
        Integrals[std::to_string(tmpInt3) + " " + std::to_string(tmpInt4) + " " + std::to_string(tmpInt1) + " " + std::to_string(tmpInt2)] = tmpDouble;
        Integrals[std::to_string(tmpInt2) + " " + std::to_string(tmpInt1) + " " + std::to_string(tmpInt4) + " " + std::to_string(tmpInt3)] = tmpDouble;
        Integrals[std::to_string(tmpInt4) + " " + std::to_string(tmpInt3) + " " + std::to_string(tmpInt2) + " " + std::to_string(tmpInt1)] = tmpDouble;
        Integrals[std::to_string(tmpInt2) + " " + std::to_string(tmpInt1) + " " + std::to_string(tmpInt3) + " " + std::to_string(tmpInt4)] = tmpDouble;
        Integrals[std::to_string(tmpInt4) + " " + std::to_string(tmpInt3) + " " + std::to_string(tmpInt1) + " " + std::to_string(tmpInt2)] = tmpDouble;
        Integrals[std::to_string(tmpInt1) + " " + std::to_string(tmpInt2) + " " + std::to_string(tmpInt4) + " " + std::to_string(tmpInt3)] = tmpDouble;
        Integrals[std::to_string(tmpInt3) + " " + std::to_string(tmpInt4) + " " + std::to_string(tmpInt2) + " " + std::to_string(tmpInt1)] = tmpDouble;
    }
    NumOcc = (NumElectrons + 1) / 2; // Floor division. This is RHF so just divide electrons by two.

    OverlapMatrix = Eigen::MatrixXd::Identity(NumAO, NumAO);
	/*std::ifstream OverlapFile(OverlapInput.c_str());
    OverlapMatrix = Eigen::MatrixXd::Zero(NumAO, NumAO);
    while(!OverlapFile.eof())
    {
        OverlapFile >> tmpInt1 >> tmpInt2 >> tmpDouble;
        OverlapMatrix(tmpInt1 - 1 ,tmpInt2 - 1) = tmpDouble;
        OverlapMatrix(tmpInt2 - 1, tmpInt1 - 1) = tmpDouble;
    }*/
    // Read the overlap input here. Since Q-Chem uses a MO basis, we just set the overlap to be the identity. I should fix this when
    // I use something other than Q-Chem, but then I would have to change the integrals too, so...

    if(OverlapInput == "c") 
    {
        std::ifstream C("c");
        InitialCoeff = Eigen::MatrixXd::Zero(NumAO, NumOcc);
        for(int j = 0; j < NumOcc; j++)
        {
            for(int i = 0; i < NumAO; i++)
            {
                double tmpDouble;
                C >> tmpDouble;
                InitialCoeff(i, j) = tmpDouble;
            }
        }
    }

    if(DensityOption != 1 && DensityOption != 0 && DensityOption != 2)
    {
        std::cerr << "Something is wrong in the input file." << std::endl;
        std::string tmpString;
        std::getline(std::cin, tmpString);
    }
}