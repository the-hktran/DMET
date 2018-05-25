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
#include <queue>
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Bootstrap.h"
#include "Functions.h"
#include "RealTime.h"

std::complex<double> OneElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXcd &RotationMatrix, int c, int d)
{
	std::complex<double> hcd = 0;
	for (int p = 0; p < RotationMatrix.rows(); p++)
	{
		for (int q = 0; q < RotationMatrix.rows(); q++)
		{
			hcd += std::conj(RotationMatrix(p, c)) * Integrals[std::to_string(p + 1) + " " + std::to_string(q + 1) + " 0 0"] * RotationMatrix(q, d);
		}
	}
	return hcd;
}

std::complex<double> TwoElectronIntegral(unsigned short int m, unsigned short int n, unsigned short int k, unsigned short int l, bool m_isAlpha, bool n_isAlpha, bool k_isAlpha, bool l_isAlpha, std::map<std::string, double> &Integrals, Eigen::MatrixXcd &RotationMatrix)
{
	std::complex<double> mknl = 0; // First term. (mk|nl)
	std::complex<double> mlnk = 0; // Second term. (ml|nk)

					 /* Deal with first term first */
	if ((m_isAlpha != k_isAlpha) || (n_isAlpha != l_isAlpha)) // Means spin component is different.
	{
		mknl = 0;
	}
	else
	{
		mknl = TwoElectronEmbedding(Integrals, RotationMatrix, m - 1, n - 1, k - 1, l - 1);
		// Integrals[std::to_string(m) + " " + std::to_string(k) + " " + std::to_string(n) + " " + std::to_string(l)];
	}
	/* Now, the second term */
	if ((m_isAlpha != l_isAlpha) || (n_isAlpha != k_isAlpha))
	{
		mlnk = 0;
	}
	else
	{
		mlnk = TwoElectronEmbedding(Integrals, RotationMatrix, m - 1, n - 1, l - 1, k - 1);
		// Integrals[std::to_string(m) + " " + std::to_string(l) + " " + std::to_string(n) + " " + std::to_string(k)];
	}
	return mknl - mlnk;
}

/* Equation 11 in J.S. Kretchmer and G.K-L. Chan, Preprint: https://arxiv.org/abs/1609.07678, (2016). */
std::complex<double> TwoElectronEmbedding(std::map<std::string, double> &Integrals, Eigen::MatrixXcd &RotationMatrix, int c, int d, int e, int f)
{
	std::complex<double> Vcdef = 0;
	for (int p = 0; p < RotationMatrix.rows(); p++)
	{
		for (int q = 0; q < RotationMatrix.rows(); q++)
		{
			for (int r = 0; r < RotationMatrix.rows(); r++)
			{
				for (int s = 0; s < RotationMatrix.rows(); s++)
				{
					Vcdef += std::conj(RotationMatrix(p, c)) * std::conj(RotationMatrix(q, d)) * Integrals[std::to_string(p + 1) + " " + std::to_string(r + 1) + " " + std::to_string(q + 1) + " " + std::to_string(s + 1)] * RotationMatrix(r, e) * RotationMatrix(s, f);
				}
			}
		}
	}
	return Vcdef;
}

/* This is tilde h_cd, which is the one electron component of the Hamiltonian in the embedding basis, plus XC with the core
electrons. */
std::complex<double> OneElectronPlusCore(InputObj &Input, Eigen::MatrixXcd &RotationMatrix, int FragmentIndex, int c, int d)
{
	std::complex<double> tildehcd = 0;
	tildehcd = OneElectronEmbedding(Input.Integrals, RotationMatrix, c, d);
	for (int u = 0; u < Input.NumOcc - Input.FragmentOrbitals[FragmentIndex].size(); u++) // XC with core
	{
		int uu = Input.EnvironmentOrbitals[FragmentIndex][Input.EnvironmentOrbitals[FragmentIndex].size() - 1 - u];
		std::complex<double> Vcudu = TwoElectronEmbedding(Input.Integrals, RotationMatrix, c, uu, d, uu);
		std::complex<double> Vcuud = TwoElectronEmbedding(Input.Integrals, RotationMatrix, c, uu, uu, d);
		tildehcd += (2.0 * Vcudu - Vcuud);
	}
	return tildehcd;
}

/* This calculates the elements < a_i,alpha^\dagger a_j,alpha + a_i,beta^\dagger a_j,beta > */
Eigen::MatrixXcd Form1RDMComplex(InputObj &Input, int FragmentIndex, Eigen::VectorXcd Eigenvector, std::vector< std::vector< bool > > aStrings, std::vector< std::vector< bool > > bStrings)
{
	std::vector<int> FragPos;
	std::vector<int> BathPos;
	GetCASPos(Input, FragmentIndex, FragPos, BathPos);

	int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();

	Eigen::MatrixXcd DensityMatrix(2 * NumAOImp, 2 * NumAOImp);
	for (int i = 0; i < 2 * NumAOImp; i++)
	{
		for (int j = 0; j < 2 * NumAOImp; j++)
		{
			std::complex<double> DijA = 0;
			/* We formulate the alpha and beta density matrices separately, and add them together by element. */
			// Determine which orbital we are looking at, since i and j loop over the CAS index and not the real orbitals.
			int iOrbital = ReducedIndexToOrbital(i, Input, FragmentIndex);
			int jOrbital = ReducedIndexToOrbital(j, Input, FragmentIndex);

			// First, make the alpha density matrix.
			for (int ai = 0; ai < aStrings.size(); ai++)
			{
				for (int bi = 0; bi < bStrings.size(); bi++)
				{
					for (int aj = 0; aj < aStrings.size(); aj++)
					{
						for (int bj = 0; bj < bStrings.size(); bj++)
						{
							std::vector< bool > BraAnnil = aStrings[ai];
							std::vector< bool > KetAnnil = aStrings[aj];
							short int BraSign = 1;
							short int KetSign = 1;
							if (aStrings[ai][iOrbital] && aStrings[aj][jOrbital]) // If bra contains i and ket contains j, then we annihilate the orbitals.
							{
								BraSign = AnnihilationParity(BraAnnil, iOrbital);
								KetSign = AnnihilationParity(KetAnnil, jOrbital);
								BraAnnil[iOrbital] = false; // bra with i annihilated
								KetAnnil[jOrbital] = false; // ket with j annhiliated
							}
							else // Annihilated the bra or ket to zero
							{
								continue; // zero contribution
							}

							// Count the differences in the bra and ket, which is the sum of differences in the alpha and beta
							// components : <alpha|alpha><beta|beta>
							int Diff = CountDifferences(BraAnnil, KetAnnil) + CountDifferences(bStrings[bi], bStrings[bj]);
							if (Diff > 0)
							{
								continue;
							}
							DijA += (double)BraSign * (double)KetSign * std::conj(Eigenvector[ai + bi * aStrings.size()]) * Eigenvector[aj + bj * aStrings.size()];
						}
					}
				}
			}
			DensityMatrix(i, j) = DijA;

			// Now, do the same to make the beta density matrix. See above comments for explanation.
			std::complex<double> DijB = 0;
			for (int ai = 0; ai < aStrings.size(); ai++)
			{
				for (int bi = 0; bi < bStrings.size(); bi++)
				{
					for (int aj = 0; aj < aStrings.size(); aj++)
					{
						for (int bj = 0; bj < bStrings.size(); bj++)
						{
							std::vector< bool > BraAnnil = bStrings[bi];
							std::vector< bool > KetAnnil = bStrings[bj];
							short int BraSign = 1;
							short int KetSign = 1;
							if (bStrings[bi][iOrbital] && bStrings[bj][jOrbital])
							{
								BraSign = AnnihilationParity(BraAnnil, iOrbital);
								KetSign = AnnihilationParity(KetAnnil, jOrbital);
								BraAnnil[iOrbital] = false;
								KetAnnil[jOrbital] = false;
							}
							else // Annihilated the bra or ket to zero
							{
								continue;
							}

							int Diff = CountDifferences(BraAnnil, KetAnnil) + CountDifferences(aStrings[ai], aStrings[aj]);
							if (Diff > 0)
							{
								continue;
							}
							DijB += (double)BraSign * (double)KetSign * std::conj(Eigenvector[ai + bi * aStrings.size()]) * Eigenvector[aj + bj * aStrings.size()];
						}
					}
				}
			}
			DensityMatrix(i, j) += DijB;
		} // end loop over j
	} // end loop over i
	return DensityMatrix;
}

/* Calculates the 2RDM with elements P_ijkl = <a^dagger_i a^dagger_j a_k a_l>
However, note that given the definition of P_{ij|kl} = < a_j^dagger a_l^dagger a_i a_k > this means that the element TwoRDM(i,j,k,l) actually corresponds to P_{ki|lj} */
Eigen::Tensor<std::complex<double>, 4> Form2RDM(InputObj &Input, int FragmentIndex, Eigen::VectorXcd Eigenvector, std::vector< std::vector< bool > > aStrings, std::vector< std::vector< bool > > bStrings, Eigen::MatrixXcd OneRDM)
{
	std::vector<int> FragPos;
	std::vector<int> BathPos;
	GetCASPos(Input, FragmentIndex, FragPos, BathPos);

	int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();

	/* Note that in order to convert to spacial orbitals, the actual element we are calculating is
	< a^t_i,alpha a^t_j,alpha a_k,alpha a_l,alpha + a^t_i,alpha a^t_j,beta a_k,alpha a_l,beta
	a^t_i,beta a^t_j,alpha a_k,beta a_l,alpha + a^t_i,beta a^t_j,beta a_k,beta a_l,beta >
	We will refer to these as elements 1, 2, 3, and 4 respectively. */
	Eigen::Tensor< std::complex<double>, 4 > TwoRDM(2 * NumAOImp, 2 * NumAOImp, 2 * NumAOImp, 2 * NumAOImp);
	#pragma omp parallel for
	for (int i = 0; i < 2 * NumAOImp; i++)
	{
		// This is the orbital associated with the index, in the CAS index.
		int iOrbital = ReducedIndexToOrbital(i, Input, FragmentIndex);
		for (int j = 0; j < 2 * NumAOImp; j++)
		{
			int jOrbital = ReducedIndexToOrbital(j, Input, FragmentIndex);
			for (int k = 0; k < 2 * NumAOImp; k++)
			{
				int kOrbital = ReducedIndexToOrbital(k, Input, FragmentIndex);
				for (int l = 0; l < 2 * NumAOImp; l++)
				{
					int lOrbital = ReducedIndexToOrbital(l, Input, FragmentIndex);
					std::complex<double> Pijkl = 0;
					// Element 1 (alpha, alpha)
					for (int aBra = 0; aBra < aStrings.size(); aBra++)
					{
						for (int bBra = 0; bBra < bStrings.size(); bBra++)
						{
							for (int aKet = 0; aKet < aStrings.size(); aKet++)
							{
								for (int bKet = 0; bKet < bStrings.size(); bKet++)
								{
									// First annihilate one orbital in the bra and ket.
									std::vector< bool > aBraAnnihilate = aStrings[aBra];
									std::vector< bool > bBraAnnihilate = bStrings[bBra];
									std::vector< bool > aKetAnnihilate = aStrings[aKet];
									std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

									if (aBraAnnihilate[iOrbital] && aKetAnnihilate[lOrbital])
									{
										tmpInt1 = AnnihilationParity(aBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										aBraAnnihilate[iOrbital] = false;
										aKetAnnihilate[lOrbital] = false;
									}
									else
									{
										continue;
									}
									// Now annihilate another orbital, if it was already annihilated, the term is annihilated to zero and we move on.
									if (aBraAnnihilate[jOrbital] && aKetAnnihilate[kOrbital])
									{
										tmpInt1 = AnnihilationParity(aBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										aBraAnnihilate[jOrbital] = false;
										aKetAnnihilate[kOrbital] = false;
									}
									else
									{
										continue;
									}

									int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
									if (Diff > 0)
									{
										continue;
									}
									Pijkl += (double)BraSign * (double)KetSign * std::conj(Eigenvector[aBra + bBra * aStrings.size()]) * Eigenvector[aKet + bKet * aStrings.size()];
								}
							}
						}
					} // end aBra loop.

					  // Element 2 (alpha, beta)
					for (int aBra = 0; aBra < aStrings.size(); aBra++)
					{
						for (int bBra = 0; bBra < bStrings.size(); bBra++)
						{
							for (int aKet = 0; aKet < aStrings.size(); aKet++)
							{
								for (int bKet = 0; bKet < bStrings.size(); bKet++)
								{
									// Annihilate orbital i, j, k, and l in respective strings. There's no possibility for overlap so we can just do it all at once.
									std::vector< bool > aBraAnnihilate = aStrings[aBra];
									std::vector< bool > bBraAnnihilate = bStrings[bBra];
									std::vector< bool > aKetAnnihilate = aStrings[aKet];
									std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

									if (aBraAnnihilate[iOrbital] && bBraAnnihilate[jOrbital] && aKetAnnihilate[lOrbital] && bKetAnnihilate[kOrbital])
									{
										tmpInt1 = AnnihilationParity(aBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										aBraAnnihilate[iOrbital] = false;
										aKetAnnihilate[lOrbital] = false;

										tmpInt1 = AnnihilationParity(bBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										bBraAnnihilate[jOrbital] = false;
										bKetAnnihilate[kOrbital] = false;
									}
									else
									{
										continue;
									}

									int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
									if (Diff > 0)
									{
										continue;
									}
									Pijkl += (double)BraSign * (double)KetSign * std::conj(Eigenvector[aBra + bBra * aStrings.size()]) * Eigenvector[aKet + bKet * aStrings.size()];
								}
							}
						}
					} // end aBra loop.

					  // Element 3 (beta, alpha)
					for (int aBra = 0; aBra < aStrings.size(); aBra++)
					{
						for (int bBra = 0; bBra < bStrings.size(); bBra++)
						{
							for (int aKet = 0; aKet < aStrings.size(); aKet++)
							{
								for (int bKet = 0; bKet < bStrings.size(); bKet++)
								{
									// Annihilate orbital i, j, k, and l in respective strings. There's no possibility for overlap so we can just do it all at once.
									std::vector< bool > aBraAnnihilate = aStrings[aBra];
									std::vector< bool > bBraAnnihilate = bStrings[bBra];
									std::vector< bool > aKetAnnihilate = aStrings[aKet];
									std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

									if (bBraAnnihilate[iOrbital] && aBraAnnihilate[jOrbital] && bKetAnnihilate[lOrbital] && aKetAnnihilate[kOrbital])
									{
										tmpInt1 = AnnihilationParity(bBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										bBraAnnihilate[iOrbital] = false;
										bKetAnnihilate[lOrbital] = false;

										tmpInt1 = AnnihilationParity(aBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(aKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										aBraAnnihilate[jOrbital] = false;
										aKetAnnihilate[kOrbital] = false;
									}
									else
									{
										continue;
									}

									int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
									if (Diff > 0)
									{
										continue;
									}
									Pijkl += (double)BraSign * (double)KetSign * std::conj(Eigenvector[aBra + bBra * aStrings.size()]) * Eigenvector[aKet + bKet * aStrings.size()];
								}
							}
						}
					} // end aBra loop.

					  // Element 4 (beta, beta)
					for (int aBra = 0; aBra < aStrings.size(); aBra++)
					{
						for (int bBra = 0; bBra < bStrings.size(); bBra++)
						{
							for (int aKet = 0; aKet < aStrings.size(); aKet++)
							{
								for (int bKet = 0; bKet < bStrings.size(); bKet++)
								{
									std::vector< bool > aBraAnnihilate = aStrings[aBra];
									std::vector< bool > bBraAnnihilate = bStrings[bBra];
									std::vector< bool > aKetAnnihilate = aStrings[aKet];
									std::vector< bool > bKetAnnihilate = bStrings[bKet];
									short int BraSign = 1;
									short int KetSign = 1;
									short int tmpInt1 = 1;
									short int tmpInt2 = 1;

									if (bBraAnnihilate[iOrbital] && bKetAnnihilate[lOrbital])
									{
										tmpInt1 = AnnihilationParity(bBraAnnihilate, iOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, lOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										bBraAnnihilate[iOrbital] = false;
										bKetAnnihilate[lOrbital] = false;
									}
									else
									{
										continue;
									}

									if (bBraAnnihilate[jOrbital] && bKetAnnihilate[kOrbital])
									{
										tmpInt1 = AnnihilationParity(bBraAnnihilate, jOrbital);
										tmpInt2 = AnnihilationParity(bKetAnnihilate, kOrbital);
										BraSign *= tmpInt1;
										KetSign *= tmpInt2;
										bBraAnnihilate[jOrbital] = false;
										bKetAnnihilate[kOrbital] = false;
									}
									else
									{
										continue;
									}

									int Diff = CountDifferences(aBraAnnihilate, aKetAnnihilate) + CountDifferences(bBraAnnihilate, bKetAnnihilate);
									if (Diff > 0)
									{
										continue;
									}
									Pijkl += (double)BraSign * (double)KetSign * std::conj(Eigenvector[aBra + bBra * aStrings.size()]) * Eigenvector[aKet + bKet * aStrings.size()];
								}
							}
						}
					} // end aBra loop.

					  // - delta_jk D_il
					  //if (j == k)
					  //{
					  //    Pijkl -= OneRDM(i, l);
					  //}

					TwoRDM(i, j, k, l) = Pijkl;
				} // l
			} // k
		} // j
	} // i
	return TwoRDM;
}

Eigen::MatrixXcd TDHamiltonian(Eigen::MatrixXcd &DensityMatrix, InputObj &Input, int FragmentIndex, Eigen::MatrixXcd &RotationMatrix, double ChemicalPotential, int State, Eigen::Tensor<std::complex<double>, 4> &TwoRDM, std::vector< std::vector< bool > > &aStringsOut, std::vector< std::vector< bool > > &bStringsOut)
{
	int NumAOImp = Input.FragmentOrbitals[FragmentIndex].size();
	int NumVirt = Input.NumAO - NumAOImp - Input.NumOcc;
	int NumCore = Input.NumOcc - NumAOImp;
	int NumEnv = Input.EnvironmentOrbitals[FragmentIndex].size();

	/* There doesn't seem to be any method that chooses number of electrons */
	int aElectrons = Input.NumOcc; // 2 * NumAOImp + NumCore;
	int bElectrons = Input.NumOcc; // 2 * NumAOImp + NumCore;
	int aElectronsCAS = NumAOImp; // N_occ - N_core?????????????
	int bElectronsCAS = NumAOImp;
	int aOrbitals = Input.NumAO;
	int bOrbitals = Input.NumAO;
	int aCAS = 2 * NumAOImp;// + NumVirt;
	int bCAS = 2 * NumAOImp;// + NumVirt;
	int NumberOfEV = 1; // Input.NumberOfEV; // Number of eigenvalues desired from Davidson Diagonalization
						// int aDim = BinomialCoeff(aOrbitals, aElectrons);
						// int bDim = BinomialCoeff(bOrbitals, bElectrons);
	int aDim = BinomialCoeff(aCAS, aElectronsCAS);
	int bDim = BinomialCoeff(bCAS, aElectronsCAS);
	int Dim = aDim * bDim;
	int L = NumberOfEV + 50; // Dimension of starting subspace in Davidson Diagonalization
	while (L > Dim)
	{
		L /= 10;
	}

	double MatTol = 1E-12; // Zeros elements below this threshold, significiantly reduces storage requirements.

	std::vector< std::vector<bool> > aStrings;
	std::vector< std::vector<bool> > bStrings;

	// std::ofstream Output(Input.OutputName);
	// Output << "FCI Calculation\n\nInput File: " << Input.InputName << "\n\nNumber of Alpha Electrons: " << aElectrons << 
	// "\nNumber of Alpha Orbitals: " << aOrbitals << "\nNumber of Beta Electrons: " << bElectrons << "\nNumber of Beta Orbitals: "
	// << bOrbitals << "\nDimension of Space: " << aDim << " x " << bDim << " = " << Dim << "\n\nLooking for " << NumberOfEV << 
	// " solutions.\n" << std::endl;

	// clock_t Start = clock();
	double Start = omp_get_wtime();

	std::cout << "FCI: Generating all determinant binary representations and enumerating determinants with differences... ";
	for (int i = 0; i < aDim; i++)
	{
		std::vector<bool> tmpVec;
		GetOrbitalString(i, aElectronsCAS, aCAS, tmpVec); // Get binary string for active space orbitals.
		for (int a = 0; a < NumVirt; a++) // Insert frozen virtual orbitals into string
		{
			tmpVec.insert(tmpVec.begin() + Input.EnvironmentOrbitals[FragmentIndex][a], false);
		}
		for (int c = 0; c < NumCore; c++) // Insert the core orbitals into the string.
		{
			tmpVec.insert(tmpVec.begin() + Input.EnvironmentOrbitals[FragmentIndex][NumVirt + NumAOImp + c], true);
		}
		aStrings.push_back(tmpVec);
	}
	for (int i = 0; i < bDim; i++)
	{
		std::vector<bool> tmpVec;
		GetOrbitalString(i, bElectronsCAS, bCAS, tmpVec); // Get binary string for active space orbitals.
		for (int a = 0; a < NumVirt; a++) // Insert frozen virtual orbitals into string
		{
			tmpVec.insert(tmpVec.begin() + Input.EnvironmentOrbitals[FragmentIndex][a], false);
		}
		for (int c = 0; c < NumCore; c++) // Insert the core orbitals into the string.
		{
			tmpVec.insert(tmpVec.begin() + Input.EnvironmentOrbitals[FragmentIndex][NumVirt + NumAOImp + c], true);
		}
		bStrings.push_back(tmpVec);
	}

	std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > aSingleDifference; // i index, j index, sign, list of different orbitals.
	std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > aDoubleDifference;
	unsigned short int tmpInt;
	std::tuple<unsigned short int, unsigned short int, short int, std::vector<unsigned short int>> tmpTuple;

	for (unsigned int i = 0; i < aDim; i++)
	{
		for (unsigned int j = i + 1; j < aDim; j++)
		{
			tmpInt = CountDifferences(aStrings[i], aStrings[j]);
			if (tmpInt == 1)
			{
				short int tmpInt2 = FindSign(aStrings[i], aStrings[j]);
				std::vector<unsigned short int> tmpVec = ListDifference(aStrings[i], aStrings[j]);
				tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
				aSingleDifference.push_back(tmpTuple);
			}
			if (tmpInt == 2)
			{
				short int tmpInt2 = FindSign(aStrings[i], aStrings[j]);
				std::vector<unsigned short int> tmpVec = ListDifference(aStrings[i], aStrings[j]);
				tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
				aDoubleDifference.push_back(tmpTuple);
			}
		}
	}

	std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > bSingleDifference;
	std::vector< std::tuple<unsigned int, unsigned int, short int, std::vector<unsigned short int>> > bDoubleDifference;
	for (unsigned int i = 0; i < bDim; i++)
	{
		for (unsigned int j = i + 1; j < bDim; j++)
		{
			tmpInt = CountDifferences(bStrings[i], bStrings[j]);
			if (tmpInt == 1)
			{
				short int tmpInt2 = FindSign(bStrings[i], bStrings[j]);
				std::vector<unsigned short int> tmpVec = ListDifference(bStrings[i], bStrings[j]);
				tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
				bSingleDifference.push_back(tmpTuple);
			}
			if (tmpInt == 2)
			{
				short int tmpInt2 = FindSign(bStrings[i], bStrings[j]);
				std::vector<unsigned short int> tmpVec = ListDifference(bStrings[i], bStrings[j]);
				tmpTuple = std::make_tuple(i, j, tmpInt2, tmpVec);
				bDoubleDifference.push_back(tmpTuple);
			}
		}
	}
	aStringsOut = aStrings;
	bStringsOut = bStrings;

	unsigned int NonzeroElements = Dim + aSingleDifference.size() * bDim * 2 + bSingleDifference.size() * aDim * 2 + aDoubleDifference.size() * bDim * 2
		+ bDoubleDifference.size() * aDim * 2 + aSingleDifference.size() * bSingleDifference.size() * 4;
	//Output << "Number of alpha singles: " << aSingleDifference.size() << "\nNumber of beta singles: " << bSingleDifference.size() 
	//<< "\nNumber of alpha doubles: " << aDoubleDifference.size() << "\nNumber of beta doubles: " << bDoubleDifference.size() 
	//<< "\nChecking " << NonzeroElements << " elements.\n" << std::endl;

	std::cout << "done.\nFCI: Commencing with matrix initialization... " << std::endl;
	Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> Ham(Dim, Dim);
	// Ham.reserve(Eigen::VectorXi::Constant(Dim,NonzeroElements));
	// clock_t Timer = clock();
	double Timer = omp_get_wtime();

	typedef Eigen::Triplet< std::complex<double> > T;
	std::vector<T> tripletList;
	// std::vector< std::vector<T> > tripletList_Private(NumThreads);

	tripletList.reserve(NonzeroElements);

	/* The basis of the matrix is ordered by reverse lexicographic ordering (A,B) where A is the A'th  alpha orbital
	and B is the B'th beta orbital. Essentially, this means we have beta blocks and inside each block is a matrix for
	the alpha elements. */

	/* Diagonal Elements */
	/* Since I order the orbitals the same way in the bra and ket, there should be no sign change. There is a one electron
	component (single particle Hamiltonian), two electron component (coulombic repulsion), and zero electron component
	(nuclear repulsion). The nuclear repulsion term only appears in the diagonal. */
	double NuclearEnergy = Input.Integrals["0 0 0 0"]; // Nuclear repulsion, will shift total energy and needs to be added to diagonal.

													   // Isolated Core-Core interaction. This is excluded in the DMET impurity solver.
	std::complex<double> CoreInteraction = 0;
	for (int c = 0; c < 2 * NumCore; c++)
	{
		int cc = c % NumCore; // Loops back to handle both spins.
		int CoreOrbital1 = Input.EnvironmentOrbitals[FragmentIndex][NumEnv - 1 - cc] + 1; // Count from 1
		bool c_isAlpha = true;
		if (c > NumCore - 1)
		{
			c_isAlpha = false;
		}
		for (int d = c + 1; d < 2 * NumCore; d++)
		{
			int dd = d % NumCore;
			int CoreOrbital2 = Input.EnvironmentOrbitals[FragmentIndex][NumEnv - 1 - dd] + 1;
			bool d_isAlpha = true;
			if (d > NumCore - 1)
			{
				d_isAlpha = false;
			}
			CoreInteraction += TwoElectronIntegral(CoreOrbital1, CoreOrbital2, CoreOrbital1, CoreOrbital2, c_isAlpha, d_isAlpha, c_isAlpha, d_isAlpha, Input.Integrals, RotationMatrix);
		}
		CoreInteraction += OneElectronEmbedding(Input.Integrals, RotationMatrix, CoreOrbital1 - 1, CoreOrbital1 - 1);
	}

	std::vector< std::vector<unsigned short int> > aOrbitalList; // [Determinant Number][Occupied Orbital]
	std::vector< std::vector<unsigned short int> > bOrbitalList;
	for (unsigned short int i = 0; i < aDim; i++)
	{
		aOrbitalList.push_back(ListOrbitals(aStrings[i]));
	}
	for (unsigned short int j = 0; j < bDim; j++)
	{
		bOrbitalList.push_back(ListOrbitals(bStrings[j]));
	}
#pragma omp parallel for
	for (int i = 0; i < aDim; i++) // Loop through every matrix element
	{
		// int Thread = omp_get_thread_num();
		std::vector<T> tripletList_Private;
		for (unsigned int j = 0; j < bDim; j++) // See above comment.
		{
			std::complex<double> tmpDoubleD = 0;
			/* Zero electron operator */
			// tmpDoubleD += NuclearEnergy; // Nuclear potential.
			/* One electron operator */
			for (int ii = 0; ii < aOrbitalList[i].size(); ii++)
			{
				tmpDoubleD += OneElectronEmbedding(Input.Integrals, RotationMatrix, aOrbitalList[i][ii] - 1, aOrbitalList[i][ii] - 1);
			}
			for (int jj = 0; jj < bOrbitalList[j].size(); jj++)
			{
				tmpDoubleD += OneElectronEmbedding(Input.Integrals, RotationMatrix, bOrbitalList[j][jj] - 1, bOrbitalList[j][jj] - 1);
			}

			/* Two electron operator in the notation <mn||mn> */
			std::vector<unsigned short int> abOrbitalList = aOrbitalList[i]; // List of all orbitals, starting with alpha.
			abOrbitalList.insert(abOrbitalList.end(), bOrbitalList[j].begin(), bOrbitalList[j].end());
			for (unsigned short int n = 0; n < aElectrons + bElectrons; n++) // Sum over occupied orbitals n
			{
				for (unsigned short int m = n + 1; m < aElectrons + bElectrons; m++) // Sum over m > n
				{
					bool n_isAlpha = true;
					bool m_isAlpha = true;
					if (n > aElectrons - 1) n_isAlpha = false; // Means we have looped through the alpha orbitals and are now looking at a beta orbital
					if (m > aElectrons - 1) m_isAlpha = false;
					tmpDoubleD += TwoElectronIntegral(abOrbitalList[m], abOrbitalList[n], abOrbitalList[m], abOrbitalList[n], m_isAlpha, n_isAlpha, m_isAlpha, n_isAlpha, Input.Integrals, RotationMatrix);
				}
			}

			tmpDoubleD -= CoreInteraction; // Removes the core-core interaction.

			int NumSameImp = CountSameImpurity(aStrings[i], aStrings[i], Input.FragmentOrbitals[FragmentIndex]) + CountSameImpurity(bStrings[j], bStrings[j], Input.FragmentOrbitals[FragmentIndex]); // This totals the number of impurity orbitals in the alpha and beta lists.
			tmpDoubleD -= ChemicalPotential * (double)NumSameImp; // Form of chemical potential matrix element.

																  // tripletList_Private[Thread].push_back(T(i + j * aDim, i + j * aDim, tmpDoubleD));
			tripletList_Private.push_back(T(i + j * aDim, i + j * aDim, tmpDoubleD));
		}
#pragma omp critical
		tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
	}
	std::cout << "FCI: ...diagonal elements completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
	// Output << "Diagonal elements generated in " << (omp_get_wtime() - Timer)  << " seconds." << std::endl;
	// Timer = clock();
	Timer = omp_get_wtime();

	/*
	Now we begin setting the nonzero off-diagonal elements. We separate this into three groups.
	1) Elements differing by one spin-orbital (one orbital in alpha and beta electrons)
	This is achieved by looping through single differences of alpha (beta) and choosing
	to keep beta (alpha) diagonal so that there is only one difference.
	2) Elements differing by two spin-orbitals of the same spin.
	This is achieved by looping through double differences of alpha (beta) and choosing
	to keep beta (alpha) diagonal so that there is only two differences.
	3) Elements differing by two spin-orbitals of different spin.
	This is achieved by looping through single differences of both alpha and beta spaces
	and gives us two differences composed of a one different alpha and one different beta
	orbital.
	*/

	/*
	We start with Group 1. The matrix of single differences in alpha is block diagonal in beta states,
	but off diagonal in alpha states within that block.
	|      *   * |            |            |
	|        *   |            |            |
	|          * |            |            |
	|            |            |            |
	|____________|____________|____________|
	|            |      *   * |            |
	|            |        *   |            |
	|            |          * |            |
	|            |            |            |
	|____________|____________|____________|
	|            |            |     *   *  |
	|            |            |       *    |
	|            |            |         *  |
	|            |            |            |
	|____________|____________|____________|
	We denote the bra <...mn...|
	and the ket |...pn...>

	To find these elements, we are going to calculation <m|h|p> using the list of differences. Then we construct
	an orbital list for this basis function and loop through all shared orbitals, meaning without m and p, and
	calculate the two electron operator contribution.

	Matrix Elements: <m|h|p> + sum_n <mn||pn>
	*/
#pragma omp parallel for
	for (int i = 0; i < aSingleDifference.size(); i++)
	{
		// int Thread = omp_get_thread_num();
		std::vector<T> tripletList_Private;
		unsigned int Index1, Index2;
		std::complex<double> tmpDouble1 = 0;
		// First, add the one electron contribution.
		tmpDouble1 += OneElectronEmbedding(Input.Integrals, RotationMatrix, std::get<3>(aSingleDifference[i])[0] - 1, std::get<3>(aSingleDifference[i])[1] - 1); // Input.Integrals[std::to_string(std::get<3>(aSingleDifference[i])[0]) + " " + std::to_string(std::get<3>(aSingleDifference[i])[1]) + " 0 0"];

																																								 // Now, two electron contribution
		for (unsigned int j = 0; j < bDim; j++)
		{
			std::complex<double> tmpDouble2 = 0;
			std::vector<unsigned short int> KetOrbitalList = aOrbitalList[std::get<1>(aSingleDifference[i])]; // To make the orbital list of the bra, we take the list of the current alpha determinant...
			KetOrbitalList.insert(KetOrbitalList.end(), bOrbitalList[j].begin(), bOrbitalList[j].end()); // ...and append the beta determinant, which is the same for bra and ket.
			int pos_m = CountOrbitalPosition(std::get<3>(aSingleDifference[i])[1], true, KetOrbitalList, aElectrons); // Position in ket of orbital missing in bra.
			for (unsigned short int n = 0; n < KetOrbitalList.size(); n++) // Sum over electrons in the Ket.
			{
				bool n_isAlpha = true; // Checks if we are looking at an alpha electron.
				if (n > aElectrons - 1) n_isAlpha = false; // There are aElectrons alpha orbitals at the front of the list. After this, we are done looping over alpha orbitals.
				if (n + 1 == pos_m) continue; // n shouldn't loop over different orbitals. We're looping over ket orbitals, so ignore the m.
				tmpDouble2 += TwoElectronIntegral(std::get<3>(aSingleDifference[i])[0], KetOrbitalList[n], std::get<3>(aSingleDifference[i])[1], KetOrbitalList[n], true, n_isAlpha, true, n_isAlpha, Input.Integrals, RotationMatrix);
				// For this case, we know that m and p orbitals are alpha. n may or may not be alpha depending on the index of the sum.
			}

			if (std::norm(tmpDouble1 + tmpDouble2) < MatTol) continue;

			Index1 = std::get<0>(aSingleDifference[i]) + j * aDim; // Diagonal in beta states. Hop to other beta blocks.
			Index2 = std::get<1>(aSingleDifference[i]) + j * aDim;

			// tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
			// tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
			tripletList_Private.push_back(T(Index1, Index2, (double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2)));
			tripletList_Private.push_back(T(Index2, Index1, std::conj((double)std::get<2>(aSingleDifference[i])*(tmpDouble1 + tmpDouble2))));
		}
#pragma omp critical
		tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
	}

	/*
	The matrix of single differences in beta is not block diagonal, but is diagonal within the
	beta blocks that are off diagonal.
	|            |            |            |
	|            |  *         |            |
	|            |     *      |            |
	|            |        *   |            |
	|____________|____________|____________|
	|            |            |            |
	|            |            |  *         |
	|            |            |     *      |
	|            |            |        *   |
	|____________|____________|____________|
	|            |            |            |
	|            |            |            |
	|            |            |            |
	|            |            |            |
	|____________|____________|____________|

	The matrix elements for bra and ket
	<...mn...|
	|...pn...>
	are: <m|h|p> + sum_n <mn||pn>
	*/
#pragma omp parallel for
	for (int i = 0; i < bSingleDifference.size(); i++)
	{
		// int Thread = omp_get_thread_num();
		std::vector<T> tripletList_Private;
		unsigned int Index1, Index2;
		std::complex<double> tmpDouble1 = 0;
		// First, add the one electron contribution.
		tmpDouble1 += OneElectronEmbedding(Input.Integrals, RotationMatrix, std::get<3>(bSingleDifference[i])[0] - 1, std::get<3>(bSingleDifference[i])[1] - 1); // Input.Integrals[std::to_string(std::get<3>(bSingleDifference[i])[0]) + " " + std::to_string(std::get<3>(bSingleDifference[i])[1]) + " 0 0"];

																																								 // Now, two electron contribution
		for (unsigned int j = 0; j < aDim; j++)
		{
			std::complex<double> tmpDouble2 = 0;
			std::vector<unsigned short int> KetOrbitalList = aOrbitalList[j]; // Same as before, but now we set the alpha orbital in front and then append the beta orbitals.
			KetOrbitalList.insert(KetOrbitalList.end(), bOrbitalList[std::get<1>(bSingleDifference[i])].begin(), bOrbitalList[std::get<1>(bSingleDifference[i])].end());
			int pos_m = CountOrbitalPosition(std::get<3>(bSingleDifference[i])[1], false, KetOrbitalList, aElectrons); // Position in ket of orbital missing in bra.
			for (unsigned short int n = 0; n < KetOrbitalList.size(); n++) // Sum over orbitals in Ket.
			{
				bool n_isAlpha = true; // Checks if we are looking at an alpha electron.
				if (n > aElectrons - 1) n_isAlpha = false; // Finished looping over all alpha electrons.
				if (n + 1 == pos_m) continue; // n shouldn't loop over different orbitals. We're looping over ket orbitals, so ignore the m.
				tmpDouble2 += TwoElectronIntegral(std::get<3>(bSingleDifference[i])[0], KetOrbitalList[n], std::get<3>(bSingleDifference[i])[1], KetOrbitalList[n], false, n_isAlpha, false, n_isAlpha, Input.Integrals, RotationMatrix);
				// In this case, both the unique orbitals are beta orbitals.
			}

			if (std::norm(tmpDouble1 + tmpDouble2) < MatTol) continue;

			Index1 = std::get<0>(bSingleDifference[i]) * aDim + j; // Loop through each same alpha state in each beta block.
			Index2 = std::get<1>(bSingleDifference[i]) * aDim + j;

			// tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
			// tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
			tripletList_Private.push_back(T(Index1, Index2, (double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2)));
			tripletList_Private.push_back(T(Index2, Index1, std::conj((double)std::get<2>(bSingleDifference[i]) * (tmpDouble1 + tmpDouble2))));
		}
#pragma omp critical
		tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
	}

	std::cout << "FCI: ...elements differing by one spin-orbital completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
	// Output << "Elements differing by one spin-orbital generated in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;

	// Timer = clock();
	Timer = omp_get_wtime();

	/* Now Group 2. The elements of the matrix for two differences, exclusively alpha or beta spin-orbitals, has the same
	matrix form as before. We have to loop through the other spins having no differences.
	The notation used to denote the bra and ket is
	<...mn...|
	|...pq...>
	and the matrix element is <mn||pq> */
#pragma omp parallel for
	for (int i = 0; i < aDoubleDifference.size(); i++)
	{
		// int Thread = omp_get_thread_num();
		std::vector<T> tripletList_Private;
		unsigned int Index1, Index2;
		for (unsigned int j = 0; j < bDim; j++)
		{
			/* This case is easier than the previous cases in that we do not need to obtain the list of similar orbitals,
			we only need to calculate the two electron integral involving the two differing orbitals and we know that
			both of these orbitals hold alpha electrons. */
			std::complex<double> tmpDouble;
			tmpDouble = TwoElectronIntegral(std::get<3>(aDoubleDifference[i])[0], std::get<3>(aDoubleDifference[i])[1], std::get<3>(aDoubleDifference[i])[2], std::get<3>(aDoubleDifference[i])[3], true, true, true, true, Input.Integrals, RotationMatrix);
			// The four electron differences, all of them alpha electrons.

			if (std::norm(tmpDouble) < MatTol) continue;

			Index1 = std::get<0>(aDoubleDifference[i]) + j * aDim;
			Index2 = std::get<1>(aDoubleDifference[i]) + j * aDim;

			// tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
			// tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
			tripletList_Private.push_back(T(Index1, Index2, (double)std::get<2>(aDoubleDifference[i]) * tmpDouble));
			tripletList_Private.push_back(T(Index2, Index1, std::conj((double)std::get<2>(aDoubleDifference[i]) * tmpDouble)));
		}
#pragma omp critical
		tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
	}

#pragma omp parallel for
	for (int i = 0; i < bDoubleDifference.size(); i++)
	{
		// int Thread = omp_get_thread_num();
		std::vector<T> tripletList_Private;
		unsigned int Index1, Index2;

		for (unsigned int j = 0; j < aDim; j++)
		{
			std::complex<double> tmpDouble;
			tmpDouble = TwoElectronIntegral(std::get<3>(bDoubleDifference[i])[0], std::get<3>(bDoubleDifference[i])[1], std::get<3>(bDoubleDifference[i])[2], std::get<3>(bDoubleDifference[i])[3], false, false, false, false, Input.Integrals, RotationMatrix);
			// The four electron differences, all of them beta electrons.

			if (std::norm(tmpDouble) < MatTol) continue;

			Index1 = std::get<0>(bDoubleDifference[i]) * aDim + j; // Loop through each same alpha state in each beta block.
			Index2 = std::get<1>(bDoubleDifference[i]) * aDim + j;

			// tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
			// tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
			tripletList_Private.push_back(T(Index1, Index2, (double)std::get<2>(bDoubleDifference[i]) * tmpDouble));
			tripletList_Private.push_back(T(Index2, Index1, std::conj((double)std::get<2>(bDoubleDifference[i]) * tmpDouble)));
		}
#pragma omp critical
		tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
	}

	/* Now Group 3. Unlike before, we don't have to loop over alpha or beta having no differences. We simply loop
	over both alpha and beta having one difference. */
	int H2OMemoryWorkAround = 0;
	if (aSingleDifference.size() > 45000) // This is a workaround for the case of H2O. Cut down memory costs
	{
		// MatTol = 1E-12;
		H2OMemoryWorkAround = 17000; // This is how many differences I exclude from each alpha and beta string.
									 // For H2O, the memory requirements are just too large. My work around is to increase the tolerance,
									 // and remove the highest excitations, which shouldn't contribute a great deal to the ground state.
									 // This is not systematic and I am not considering specific excitaitons.
	}
#pragma omp parallel for
	for (int i = 0; i < aSingleDifference.size() - H2OMemoryWorkAround; i++)
	{
		// int Thread = omp_get_thread_num();
		std::vector<T> tripletList_Private;
		unsigned int Index1, Index2;
		for (unsigned int j = 0; j < bSingleDifference.size() - H2OMemoryWorkAround; j++)
		{
			std::complex<double> tmpDouble;
			tmpDouble = TwoElectronIntegral(std::get<3>(aSingleDifference[i])[0], std::get<3>(bSingleDifference[j])[0], std::get<3>(aSingleDifference[i])[1], std::get<3>(bSingleDifference[j])[1], true, false, true, false, Input.Integrals, RotationMatrix);
			/* There is one alpha and one beta orbital mismatched between the bra and ket. According to the formula, we put the bra unique orbitals in first,
			which are the first two arguments, the first one alpha and the second one beta. Then the next two arguments are the unique orbitals
			of the ket. We know whether these electrons are alpha or beta. */

			if (std::norm(tmpDouble) < MatTol) continue;
			Index1 = std::get<0>(aSingleDifference[i]) + aDim * std::get<0>(bSingleDifference[j]);
			Index2 = std::get<1>(aSingleDifference[i]) + aDim * std::get<1>(bSingleDifference[j]);
			// Note that the sign is the product of the signs of the alpha and beta strings. This is because we can permute them independently.
			// tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
			// tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
			tripletList_Private.push_back(T(Index1, Index2, (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
			tripletList_Private.push_back(T(Index2, Index1, std::conj((double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble)));

			/* We have to be a little more careful in this case. We want the upper triangle, but this only gives us half
			of the upper triangle. In particular, the upper half of each beta block in upper triangle of the full matrix
			are the only nonzero elements. We want the whole beta block in the upper triangle of the full matrix to be
			nonzero where needed.
			|            |      *   * |      *   * |
			|            |        *   |        *   |
			|            |          * |          * |
			|            |            |            |
			|____________|____________|____________|
			|            |            |      *   * |
			|            |            |        *   |
			|            |            |          * |
			|            |            |            |
			|____________|____________|____________|
			|            |            |            |
			|            |            |            |
			|            |            |            |
			|            |            |            |
			|____________|____________|____________|
			So we have to include some transposed elements too. It is enough to transpose the alpha indices. this
			transposes each block above, and we end up with a fully upper triangular matrix. */
			Index1 = std::get<1>(aSingleDifference[i]) + aDim * std::get<0>(bSingleDifference[j]);
			Index2 = std::get<0>(aSingleDifference[i]) + aDim * std::get<1>(bSingleDifference[j]); // Note that first and second are switched for alpha here.

																								   // tripletList_Private[Thread].push_back(T(Index1, Index2 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
																								   // tripletList_Private[Thread].push_back(T(Index2, Index1 , (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
																								   /* IDK why but this is the culprit to the memory issue */
			tripletList_Private.push_back(T(Index1, Index2, (double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble));
			tripletList_Private.push_back(T(Index2, Index1, std::conj((double)std::get<2>(aSingleDifference[i]) * (double)std::get<2>(bSingleDifference[j]) * tmpDouble)));
		}
#pragma omp critical
		tripletList.insert(tripletList.end(), tripletList_Private.begin(), tripletList_Private.end());
	}


	std::cout << "FCI: ...elements differing by two spin-orbitals completed in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
	// Output << "Elements differing by two spin-orbitals generated in " << (omp_get_wtime() - Timer) << " seconds." << std::endl;

	// for(int Thread = 0; Thread < NumThreads; Thread++)
	// {
	//     tripletList.insert(tripletList.end(), tripletList_Private[Thread].begin(), tripletList_Private[Thread].end());
	//     std::vector<T>().swap(tripletList_Private[Thread]); // Free up memory.
	// }

	Ham.setFromTriplets(tripletList.begin(), tripletList.end());
	std::vector<T>().swap(tripletList); // This clears the memory, but I'm unsure if it does or doesn't increase the memory usage to do so

	std::cout << "FCI: Hamiltonian initialization took " << (omp_get_wtime() - Start) << " seconds." << std::endl;
	// Output << "\nHamiltonian initialization took " << (omp_get_wtime() - Start) << " seconds." << std::endl;

	/* Prints matrix, used for error checking on my part */
	// Eigen::MatrixXf HD = Ham;
	// std::ofstream PrintHam("printham.out");
	// PrintHam << HD << std::endl;
	// return 0;

	/* These section is for Davidson diagonalization */
	// Timer = omp_get_wtime();
	// std::cout << "FCI: Beginning Davidson Diagonalization... " << std::endl;
	// std::vector< double > DavidsonEV;
	// Davidson(Ham, Dim, NumberOfEV, L, DavidsonEV);
	// std::cout << "FCI: ...done" << std::endl;
	// std::cout << "FCI: Davidson Diagonalization took " << (omp_get_wtime() - Timer) << " seconds." << std::endl;
	// // Output << "\nDavidson Diagonalization took " << (omp_get_wtime() - Timer) << " seconds.\nThe eigenvalues are" << std::endl;
	// for(int k = 0; k < NumberOfEV; k++)
	// {
	//     // Output << "\n" << DavidsonEV[k];
	// }

	/* This section is for direct diagonalization. Uncomment if desired. */
	return Ham;
}