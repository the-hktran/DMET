/* This file includes algorithms to solve for the eigenvalues and eigenvectors of
   a matrix of interest. The Eigen library will be used for the matrix classes and
   matrix / vector operators. The second method is the Davidson Diagonalization 
   algorithm that is an iterative algorithm approximating the lowest eigenvalues and
   eigenvectors of a sparse, real-symmetric matrix.
*/

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>

/* 
   This is the modified Gram Schmidt procedure. (Means that it orthonalizes against itself iteratively instead of one large
   calculataion. This helps avoid numerical instability. Though, a Givens rotation may be a better way to orthogonalize
   the basis). Since I need some vectors to be unnormalized, this function does not automatically orthogonalize the new vector
*/
void GramSchmidt(Eigen::VectorXf &NextVector, const std::vector< Eigen::VectorXf > &PreviousVectors)
{
    for(int i = 0; i < PreviousVectors.size(); i++)
    {
        NextVector = NextVector - ((PreviousVectors[i].dot(NextVector))/PreviousVectors[i].dot(PreviousVectors[i])) * PreviousVectors[i];
    }
    // OrthogonalVector = OrthogonalVector / sqrt(OrthogonalVector.dot(OrthogonalVector)); // Need to check tolerance before normalization.
}

/* 
   The following function takes a set of vectors and reorthogonalizes them using GramSchmidt, which
   allows us to avoid the accumulation of numerical error
*/
void Reorthogonalize(std::vector< Eigen::VectorXf > &BVectors)
{
    std::vector< Eigen::VectorXf > NewBVectors;
    NewBVectors.push_back(BVectors[0]);
    // Eigen::VectorXf b;
    for(int k = 1; k < BVectors.size(); k++)
    {
        GramSchmidt(BVectors[k], NewBVectors);
        BVectors[k] /= BVectors[k].norm();
        NewBVectors.push_back(BVectors[k]);
    }
}
/* 
   The following is the Davidson algorithm to diagonalize a matrix.
   The formulation is taken directly from Lui's paper and his steps are marked in the program. 
*/
void Davidson(Eigen::SparseMatrix<float, Eigen::RowMajor> &Ham, int Dim, int NumberOfEV, int L, std::vector<double> &DavidsonEV) // The Hamiltonian, the dimension of the Hamiltonian, the number of eigenvalues we want to find (Lui calls this M), and the starting size of the subspace.
{
    std::vector< Eigen::VectorXf > BVectors; // This holds the basis of our subspace. We add to this list each iteration.

    double Tolerance = 1E-5;

    for(int i = 0; i < L; i++)
    {
        /***** Step 1 *****/
        Eigen::VectorXf b(Dim); // Starting set of orthornmal vectors. There are L of them and they have Dim components.
        /* We initialize b to be random, orthonormal vectors, for now. */
        for(int j = 0; j < Dim; j++)
        {
            // b[j] = rand();
            if(i == j)
            {
                b[j] = 1;
            }
            else
            {
                b[j] = 0;
            }
        }
        b = b.normalized();
        GramSchmidt(b, BVectors);
        b = b.normalized();
        BVectors.push_back(b); // Add to list of b vectors.
    }
    /* 
       BVectors now contains a vector of Eigen::VectorXd which are orthonormal. We will consider the Hamiltonian matrix restricted
       to this subspace, call it G.
    */

    int MAX_STEPS = 100000; // Not important, should converge before reaching this. Can be hard coded.
    for(int Step = 0; Step < MAX_STEPS; Step++)
    {
        std::cout << "FCI: Davidson iteration " << Step + 1 << std::endl; // To show us that something is happening.

        std::vector< Eigen::VectorXf > HbVectors; // Hamiltonian applied to the b vectors. Better to store these since we need them later.
        for(int i = 0; i < L; i++)
        {
            Eigen::VectorXf Hb = Ham * BVectors[i];
            HbVectors.push_back(Hb);
        }

        /***** Step 2 *****/
        /* 
           Here, we construct the L x L matrix G, which has elements <bi| H | bj>. This is the Hamiltonian
           in the subspace spanned by the current set of b vectors.
        */
        Eigen::MatrixXf G =  Eigen::MatrixXf::Zero(L, L); // Hamiltonian in smaller subspace. G in Lui's paper.
        for(int i = 0; i < L; i++) // Place diagonal elements first.
        {
            G(i, i) = BVectors[i].dot(HbVectors[i]); // <bi | H | bi>
        }
        for(int i = 0; i < L; i++) // Now offdiagonal elements.
        {
            for(int j = i + 1; j < L; j++)
            {
                G(i, j) = BVectors[i].dot(HbVectors[j]); // <bi | H | bj>
                G(j, i) = G(i, j); // Symmetric matrix.
            }
        }

        Eigen::SelfAdjointEigenSolver< Eigen::MatrixXf > EigensystemG(G); // I think how Eigen orders eigenvectors here determines which ones we converge towards.
        /***** Step 3 *****/
        std::vector< Eigen::VectorXf > FVectors; // Correction vectors fk in Lui's paper.
        std::vector< Eigen::VectorXf > DVectors; // These are the residual vectors, called dk in Lui's paper.

        int NumberFound = 0; // Counter for how many eigenvalues are found. Loop will terminate if this hits the desired number.

        /* Calculate the list of residual vectors, which we further use to test convergence. */
        for(int k = 0; k < NumberOfEV; k++)
        {
            Eigen::VectorXf ResidualK = Eigen::VectorXf::Zero(Dim); // Holds the k'th residual vector.
            for(int i = 0; i < L; i++) // Now we sum over terms that make the residual vector.
            {
                ResidualK += EigensystemG.eigenvectors().col(k)[i] * (HbVectors[i] - (EigensystemG.eigenvalues()[k] * BVectors[i]));
            }
            DVectors.push_back(ResidualK); // Add this to the list of residual vectors.

            if(fabs(ResidualK.dot(ResidualK)) < Tolerance) // If the eigenvector of G is the eigenvector of H, then we should have zero norm.
            {
                std::cout << "FCI: Eigenvalue " << k << " converged with value " << EigensystemG.eigenvalues()[k] << std::endl;
                // << " and eigenvector\n" << EigensystemG.eigenvectors().col(k) << std::endl;
                NumberFound++; // Count that we've converged.
            }
        }

        if(NumberFound == NumberOfEV) // This means we have found all the eigenvalues we wanted to find, and we can exit the loop.
        {
            std::cout << "FCI: All eigenvalues found. Terminating loop." << std::endl;
            for(int k = 0; k < NumberOfEV; k++)
            {
                DavidsonEV.push_back(EigensystemG.eigenvalues()[k]);
            }
            break;
        }

        /* Calculate the list of correction vectors, now that we have the residual vectors. */
        for(int k = 0; k < NumberOfEV; k++)
        {   
            Eigen::VectorXf fk(Dim); // Holds the k'th correction vector.
            for(int i = 0; i < Dim; i++)
            {
                fk[i] = DVectors[k][i] / (EigensystemG.eigenvalues()[k] - Ham.coeff(i,i));
            }
            /***** Step 4 *****/
            fk /= fk.norm(); // Normalize.
            FVectors.push_back(fk);
        }

        /***** Step 5 ******/
        /* Now we begin adding correction vectors to the list of b vectors, depending on their norm. */
        int m = 0; // Counts how many vectors we add. Can either be 1 up to M.
        for(int k = 0; k < NumberOfEV; k++)
        {
            GramSchmidt(FVectors[k], BVectors); // Orthogonalize them, but don't normalize.
            if(fabs(FVectors[k].dot(FVectors[k])) > Tolerance || k == 0) // If their norm is large enough, add them to the list. Also add the first one always.
            {
                FVectors[k] /= FVectors[k].norm();
                BVectors.push_back(FVectors[k]); // Normalized and then added to list.
                m++;
            }
        }

        /***** Step 6 *****/
        Reorthogonalize(BVectors); // To avoid numerical error, we reorthogonalize this set.

        /***** Step 7 *****/
        L += m;

        if(L > Dim) // This means we have a complete basis. The solution should be exact.
        {
            std::cout << "FCI: The subspace dimension has surpassed dimension of whole space." << std::endl;
            for(int k = 0; k < NumberOfEV; k++)
            {
                DavidsonEV.push_back(EigensystemG.eigenvalues()[k]);
            }
            break;
        }

        if(Step + 1 == MAX_STEPS) // We made it to the end without exhausting the basis or finding all eigenvectors. Something is wrong.
        {
            std::cout << "Davidson iterations terminated before any conditions were fulfilled. The method did not converge." << std::endl;
        }
    } // End Davidson Iteration for loop
}

// int main()
// {
//     std::srand(0);

//     Eigen::SparseMatrix<double> M(50,50);
//     for(int i = 0; i < 50; i++)
//     {
//         for(int j = 0; j < 50; j++)
//     for(int i = 0; i < 50; i++)
//     {
//         for(int j = 0; j < 50; j++)
//         {
//             if(i != j) 
//             else
//             {
//                 if(i < 5)
//                 {
//                     M.insert(i, i) = 1 + 0.1 * ((double)i - 1);
//                 }
//                 else
//                 {
//                     M.insert(i, i) = 2 * (double)i - 1;
//                 }
//             }
//         }
//     }

//     Davidson(M, 50, 4, 4);

//     Eigen::MatrixXd MDense = M;
//     Eigen::EigenSolver< Eigen::MatrixXd > ES(MDense);
//     std::cout << ES.eigenvalues().real() << std::endl;
//     return 0;
// }
