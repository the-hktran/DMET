#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>

#include "RealTime.h"
#include "Functions.h"

void RealTime::Init(InputObj &Inp, int x, std::vector< Eigen::MatrixXd > &FragDensities, std::vector< Eigen::Tensor<double, 4> > &Frag2RDM, Eigen::MatrixXd RMat, std::vector< Eigen::VectorXd > &FragEigenstate)
{
	Input = Inp;
	FragmentIndex = x;
	for (int i = 0; i < FragDensities.size(); i++)
	{
		FragmentDensities.push_back(FragDensities[i].cast< std::complex<double> >());
		Fragment2RDM.push_back(Frag2RDM[i].cast< std::complex<double> >());
	}
	RotationMatrix = RMat;
	ImpurityEigenstate = FragEigenstate[x].cast< std::complex<double> >();

	NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();
	NumVirt = Input.NumAO - NumAOImp - Input.NumOcc;
	NumCore = Input.NumOcc - NumAOImp;
	NumEnv = Input.EnvironmentOrbitals[FragmentIndex].size();

	X = Eigen::MatrixXcd::Zero(Input.NumAO, Input.NumAO);
}

void RealTime::FormX()
{
	std::vector< int > FragPos;
	std::vector< int > BathPos;
	GetCASPos(Input, FragmentIndex, FragPos, BathPos);

	// Between core and virtual:
	for (int a = 0; a < NumVirt; a++)
	{
		for (int c = NumVirt + NumAOImp; c < NumEnv; c++)
		{
			int VirtOrbital = Input.EnvironmentOrbitals[FragmentIndex][a];
			int CoreOrbital = Input.EnvironmentOrbitals[FragmentIndex][c];
			std::complex<double> Xac = OneElectronPlusCore(Input, RotationMatrix, FragmentIndex, VirtOrbital, CoreOrbital);
			for (int i = 0; i < 2 * NumAOImp; i++)
			{
				for (int j = 0; j < 2 * NumAOImp; j++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					int jOrb = ReducedIndexToOrbital(j, Input, FragmentIndex);
					Xac += (TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, VirtOrbital, jOrb, CoreOrbital)
						- 0.5 * TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, VirtOrbital, CoreOrbital, jOrb)) * FragmentDensities[FragmentIndex].coeffRef(i, j);
				}
			}
			X(VirtOrbital, CoreOrbital) = Xac;
			X(CoreOrbital, VirtOrbital) = std::conj(Xac);
		}
	}

	// Betweeen bath and virtual
	// First, get pure bath density.
	Eigen::MatrixXcd PBath = Eigen::MatrixXcd::Zero(NumAOImp, NumAOImp);
	for (int ib = 0; ib < NumAOImp; ib++)
	{
		for (int jb = 0; jb < NumAOImp; jb++)
		{
			std::cout << ib << jb << std::endl;
			int iBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + ib];
			int jBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + jb];
			PBath(ib, jb) = FragmentDensities[FragmentIndex].coeffRef(BathPos[ib], BathPos[jb]);
		}
	}
	Eigen::MatrixXcd  InversePBath = PBath.inverse();

	for (int a = 0; a < NumVirt; a++)
	{
		for (int ib = 0; ib < NumAOImp; ib++)
		{
			int VirtOrb = Input.EnvironmentOrbitals[FragmentIndex][a];
			int BathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + ib];
			std::complex<double> Xai = 0;
			for (int jb = 0; jb < NumAOImp; jb++)
			{
				std::complex<double> Xai_jb = 0;
				for (int i = 0; i < 2 * NumAOImp; i++) // First sum
				{
					int jBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + jb];
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					Xai += (OneElectronEmbedding(Input.Integrals, RotationMatrix, VirtOrb, iOrb) * FragmentDensities[FragmentIndex].coeffRef(i, BathPos[jb]));
				}
				for (int j = 0; j < 2 * NumAOImp; j++) // Second sum
				{
					int jOrb = ReducedIndexToOrbital(j, Input, FragmentIndex);
					for (int k = 0; k < 2 * NumAOImp; k++)
					{
						int kOrb = ReducedIndexToOrbital(k, Input, FragmentIndex);
						for (int l = 0; l < 2 * NumAOImp; l++)
						{
							int lOrb = ReducedIndexToOrbital(l, Input, FragmentIndex);
							Xai += TwoElectronEmbedding(Input.Integrals, RotationMatrix, jOrb, VirtOrb, kOrb, lOrb) * Fragment2RDM[FragmentIndex](k, l, j, BathPos[ib]);
						}
					}
				} // End second sum
				Xai += Xai_jb * InversePBath.coeffRef(jb, ib);
			} // End Sum over bath orbitals
			X(VirtOrb, BathOrb) = Xai;
			X(BathOrb, VirtOrb) = std::conj(Xai);
		}
	} // End all virt-bath interactions.

	// Between bath and core
	// First, generate Pbath bar.
	Eigen::MatrixXcd PBathBar = 2 * Eigen::MatrixXcd::Identity(PBath.rows(), PBath.cols()) - PBath;
	Eigen::MatrixXcd InversePBathBar = PBathBar.inverse();
	for (int ib = 0; ib < NumAOImp; ib++)
	{
		int iBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + ib];
		for (int c = NumVirt + NumAOImp; c < NumEnv; c++)
		{
			int cOrb = Input.EnvironmentOrbitals[FragmentIndex][c];
			std::complex<double> Xic = 0;
			for (int jb = 0; jb < NumAOImp; jb++)
			{
				int jBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + jb];
				std::complex<double> Xic_jb = 0;
				
				Xic_jb = 2.0 * OneElectronPlusCore(Input, RotationMatrix, FragmentIndex, jBathOrb, cOrb);

				for (int i = 0; i < 2 * NumAOImp; i++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					Xic_jb -= FragmentDensities[FragmentIndex].coeffRef(BathPos[jb], i) * OneElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, cOrb);
				}

				for (int i = 0; i < 2 * NumAOImp; i++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					for (int j = 0; j < 2 * NumAOImp; j++)
					{
						int jOrb = ReducedIndexToOrbital(j, Input, FragmentIndex);
						Xic_jb += (TwoElectronEmbedding(Input.Integrals, RotationMatrix, jOrb, jBathOrb, iOrb, cOrb)
							- TwoElectronEmbedding(Input.Integrals, RotationMatrix, jOrb, jBathOrb, cOrb, iOrb)) * FragmentDensities[FragmentIndex].coeffRef(i, j);
					}
				}

				for (int i = 0; i < 2 * NumAOImp; i++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					for (int j = 0; j < 2 * NumAOImp; j++)
					{
						int jOrb = ReducedIndexToOrbital(j, Input, FragmentIndex);
						for (int k = 0; k < 2 * NumAOImp; k++)
						{
							int kOrb = ReducedIndexToOrbital(k, Input, FragmentIndex);
							Xic_jb -= TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, jOrb, kOrb, cOrb) * Fragment2RDM[FragmentIndex](k, BathPos[jb], j, i);
						}
					}
				} // End last inner sum.
				Xic += Xic_jb * InversePBathBar.coeffRef(iBathOrb, jBathOrb);
			} // End outer sum over bath states.
			X(iBathOrb, cOrb) = Xic;
			X(cOrb, iBathOrb) = std::conj(Xic);
		}
	} // End loop over X matrix elements.

	// Print outs to debug
	// std::cout << "PBath\n" << PBath << std::endl;
	// std::cout << "InversePBath\n" << InversePBath << std::endl;
	// std::cout << "PBathBar\n" << PBathBar << std::endl;
	// std::cout << "InversePBathBar\n" << InversePBathBar << std::endl;
	// std::cout << "P\n" << FragmentDensities[FragmentIndex] << std::endl;
	// std::cout << "h^tilde\n";
	// for (int i = 0; i < 8; i++)
	// {
	// 	for (int j = 0; j < 8; j++)
	// 	{
	// 		std::cout << OneElectronPlusCore(Input, RotationMatrix, FragmentIndex, i, j) << "\t";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << "V\n" << std::endl;
	// for (int i = 0; i < 8; i++)
	// {
	// 	for (int j = 0; j < 8; j++)
	// 	{
	// 		std::cout << i << "\t" << j << std::endl;
	// 		for (int k = 0; k < 8; k++)
	// 		{
	// 			for (int l = 0; l < 8; l++)
	// 			{
	// 				std::cout << TwoElectronEmbedding(Input.Integrals, RotationMatrix, i, j, k, l) << "\t";
	// 			}
	// 			std::cout << std::endl;
	// 		}
	// 		std::cout << "\n\n";
	// 	}
	// }
}

// The flow of updates is: Update R -> Update H -> Update Cs -> Update RDM

/* Updates rotation matrix with X matrix. The new rotation matrix is normalized. */
Eigen::MatrixXcd RealTime::UpdateR(double TimeStep)
{
	std::complex<double> ImUnit(0.0, 1.0);
	Eigen::MatrixXcd dR = RotationMatrix * X / (ImUnit) * TimeStep;
	RotationMatrix = RotationMatrix + dR;
	for (int i = 0; i < RotationMatrix.cols(); i++)
	{
		RotationMatrix.col(i).normalize();
	}
	return RotationMatrix;
}

/* Updates the Hamiltonian using the new rotation matrix */
Eigen::MatrixXcd RealTime::UpdateH(double ChemPotential)
{
	Eigen::MatrixXcd tmpMat1;
	Eigen::Tensor<std::complex<double>, 4> tmpMat2;
	// This also generates aStrings and bStrings
	TDHam = TDHamiltonian(tmpMat1, Input, FragmentIndex, RotationMatrix, ChemPotential, 0, tmpMat2, aStrings, bStrings);
	return TDHam;
}

/* Updates the FCI coefficients using the new H */
Eigen::VectorXcd RealTime::UpdateEigenstate(double TimeStep)
{
	std::complex<double> ImUnit(0.0, 1.0);
	ImpurityEigenstate = ImpurityEigenstate + TimeStep * TDHam * ImpurityEigenstate / ImUnit;
	ImpurityEigenstate.normalize();
	return ImpurityEigenstate;
}

/* Updates the new RDMs using the new coefficients */
void RealTime::UpdateRDM()
{
	FragmentDensities[FragmentIndex] = Form1RDM(Input, FragmentIndex, ImpurityEigenstate, aStrings, bStrings);
	Fragment2RDM[FragmentIndex] = Form2RDM(Input, FragmentIndex, ImpurityEigenstate, aStrings, bStrings, FragmentDensities[FragmentIndex]);
}

void RealTime::TimeUpdate(double TimeStep, double ChemPot)
{
	FormX();
	UpdateR(TimeStep);
	UpdateH(ChemPot);
	UpdateEigenstate(TimeStep);
	UpdateRDM();
}

void RealTime::RunTimeEvolution(double EndTime, double TimeStep, double ChemPot, int QDRedPos)
{
	double t = 0;
	Times.push_back(t);
	Properties.push_back(std::real(FragmentDensities[FragmentIndex].coeffRef(QDRedPos, QDRedPos)));

	while (t < EndTime)
	{
		t += TimeStep;
		TimeUpdate(TimeStep, ChemPot);
		Times.push_back(t);
		Properties.push_back(std::real(FragmentDensities[FragmentIndex].coeffRef(QDRedPos, QDRedPos)));
		std::cout << t << "\t" << std::real(FragmentDensities[FragmentIndex].coeffRef(QDRedPos, QDRedPos)) << std::endl;
	}
}

void RealTime::PrintToOutput(std::ofstream &Output)
{
	Output << "Results of time evolution:" << std::endl;
	for (int i = 0; i < Times.size(); i++)
	{
		Output << Times[i] << "\t" << Properties[i] << std::endl;
	}
}