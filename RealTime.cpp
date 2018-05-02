#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include <RealTime.h>
#include <Functions.h>

void RealTime::Init(InputObj &Inp, int x, std::vector< Eigen::MatrixXd > &FragDensities)
{
	Input = Inp;
	FragmentIndex = x;
	FragmentDensities = FragDensities;

	int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();
	int NumVirt = Input.NumAO - NumAOImp - Input.NumOcc;
	int NumCore = Input.NumOcc - NumAOImp;
	int NumEnv = Input.EnvironmentOrbitals[FragmentIndex].size();

	X = Eigen::MatrixXd::Zero(Input.NumAO, Input.NumAO);
}

Eigen::MatrixXd RealTime::FormX()
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
			double Xac = OneElectronPlusCore(Input, RotationMatrix, FragmentIndex, VirtOrbital, CoreOrbital);
			for (int i = 0; i < 2 * NumAOImp; i++)
			{
				for (int j = 0; j < 2 * NumAOImp; j++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					int jOrb = ReducedIndexToOrbital(j, Input, FragmentIndex);
					Xac += (TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, VirtOrbital, jOrb, CoreOrbital)
						- 0.5 * TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, VirtOrbital, CoreOrbital, jOrb)) * FragmentDensities[FragmentIndex].coeffRef(iOrb, jOrb);
				}
			}
			X(VirtOrbital, CoreOrbital) = Xac;
			X(CoreOrbital, VirtOrbital) = Xac;
		}
	}

	// Betweeen bath and virtual
	// First, get pure bath density.
	Eigen::MatrixXd PBath = Eigen::MatrixXd::Zero(NumAOImp, NumAOImp);
	for (int ib = 0; ib < NumAOImp; ib++)
	{
		for (int jb = 0; jb < NumAOImp; jb++)
		{
			int iBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + ib];
			int jBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + jb];
			PBath(ib, jb) = FragmentDensities[FragmentIndex].coeffRef(iBathOrb, jBathOrb);
		}
	}
	Eigen::MatrixXd  InversePBath = PBath.inverse();

	for (int a = 0; a < NumVirt; a++)
	{
		for (int ib = 0; ib < NumAOImp; ib++)
		{
			int VirtOrb = Input.EnvironmentOrbitals[FragmentIndex][a];
			int BathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + ib];
			double Xai = 0;
			for (int jb = 0; jb < NumAOImp; jb++)
			{
				double Xai_jb = 0;
				for (int i = 0; i < 2 * NumAOImp; i++) // First sum
				{
					int jBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + jb];
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					Xai += (OneElectronEmbedding(Input.Integrals, RotationMatrix, VirtOrb, iOrb) * FragmentDensities[FragmentIndex].coeffRef(iOrb, jBathOrb));
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
							Xai += TwoElectronEmbedding(Input.Integrals, RotationMatrix, jOrb, VirtOrb, kOrb, lOrb) * Fragment2RDM[FragmentIndex](kOrb, lOrb, jOrb, BathOrb);
						}
					}
				} // End second sum
				Xai += Xai_jb * InversePBath.coeffRef(jb, ib);
			} // End Sum over bath orbitals
			X(VirtOrb, BathOrb) = Xai;
			X(BathOrb, VirtOrb) = Xai;
		}
	} // End all virt-bath interactions.

	// Between bath and core
	// First, generate Pbath bar.
	Eigen::MatrixXd PBathBar = 2 * Eigen::MatrixXd::Identity(PBath.rows(), PBath.cols()) - PBath;
	Eigen::MatrixXd InversePBathBar = PBathBar.inverse();
	for (int ib = 0; ib < NumAOImp; ib++)
	{
		int iBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + ib];
		for (int c = NumVirt + NumAOImp; c < NumEnv; c++)
		{
			int cOrb = Input.EnvironmentOrbitals[FragmentIndex][c];
			double Xic = 0;
			for (int jb = 0; jb < NumAOImp; jb++)
			{
				int jBathOrb = Input.EnvironmentOrbitals[FragmentIndex][NumVirt + jb];
				double Xic_jb = 0;
				
				Xic_jb = 2 * OneElectronPlusCore(Input, RotationMatrix, FragmentIndex, jBathOrb, cOrb);

				for (int i = 0; i < 2 * NumAOImp; i++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					Xic_jb -= FragmentDensities[FragmentIndex].coeffRef(jBathOrb, iOrb) * OneElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, cOrb);
				}

				for (int i = 0; i < 2 * NumAOImp; i++)
				{
					int iOrb = ReducedIndexToOrbital(i, Input, FragmentIndex);
					for (int j = 0; j < 2 * NumAOImp; j++)
					{
						int jOrb = ReducedIndexToOrbital(j, Input, FragmentIndex);
						Xic_jb += (TwoElectronEmbedding(Input.Integrals, RotationMatrix, jOrb, jBathOrb, iOrb, cOrb)
							- TwoElectronEmbedding(Input.Integrals, RotationMatrix, jOrb, jBathOrb, cOrb, iOrb)) * FragmentDensities[FragmentIndex].coeffRef(iOrb, jOrb);
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
							Xic_jb -= TwoElectronEmbedding(Input.Integrals, RotationMatrix, iOrb, jOrb, kOrb, cOrb) * Fragment2RDM[FragmentIndex](kOrb, jBathOrb, jOrb, iOrb);
						}
					}
				} // End last inner sum.
				Xic += Xic_jb * InversePBathBar.coeffRef(iBathOrb, jBathOrb);
			} // End outer sum over bath states.
			X(iBathOrb, cOrb) = Xic;
			X(cOrb, iBathOrb) = Xic;
		}
	} // End loop over X matrix elements.

}