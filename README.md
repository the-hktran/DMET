Excited State Density Matrix Embedding Theory
==============

Thesis Project<br />
MPhil in Scientific Computing<br />
Churchill College<br />
University of Cambridge

*This work was produced as part of the thesis requirements of the MPhil in Scientific Computing course I undertook as a student of Churchill College, University of Cambridge. This work was done under the supervision of Dr. Alex Thom and with funding from the Sir Winston Churchill Foundation of the USA.*

## Introduction

This program is an extension to Density Matrix Embedding Theory (Phys. Rev. Lett. 109, 186404 (2013)) which allows for direct embedding of excited states. 

## Compiling

The ESDMET program uses the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library for linear algebra. The program also utilizes the C++11 standard and has parallelization implemented using [OpenMP](http://www.openmp.org/). Note that OpenMP cannot be disabled as timing is done through OpenMP functions. The following should allow for compilation of the program.

```
% g++ -std=c++11 -I /path/to/eigen DMET.cpp BFGS.cpp FCI.cpp Diagonalization.cpp Metadynamics.cpp SCF.cpp Fock.cpp ReadInput.cpp -O3 -o ESDMET
```

## Running the Program

The program takes three command line inputs. These are, in order, the input filename, the overlap matrix filename, the output filename. The first input is the file that contains all the settings and values of the integrals. The second input contains the values of each overlap matrix element. If the overlap matrix is the identity, simply enter "e" for this option. The third input is the filename of the output. Alternatively the program can be simply ran without any command line inputs. A prompt will ask for these to be input individually.

## Format of the Input File

The input file is formatted as such. First, a string of input parameters should be listed. In order, they are:
- (Integer, Positive) The number of spacial orbitals.
- (Integer, Positive) The number of electrons.
- (Integer, Positive) The number of SCF solutions desired.
- (Integer, 1 / 0) Option to use / not use DIIS error.
- (Integer, 1 / 0) Option to use / not use MOM.
- (Integer, 2 / 1 / 0) Option to change converged density matrix by Randomization / Randomization with unity trace / Rotation of orbitals.
- (Integer, Positive) Maximum number of SCF cycles.
- (Double, Positive) Starting Norm of the biasing potential.
- (Double, Positive) Starting Lambda of the biasing potential.
- (Integer, Positive) Number of impurities. The next following lines define the impurities. There should be as many lines after this line as the number of impurities.
  - (Integer, Positive) Number of orbitals in the current impurity. The following lines define the number of the orbitals. There should be as many lines as number of orbitals in the current impurity plus two.
    - (Integer, Non-Negative) The label of the orbitals to be included in this impurity.
    - (Integer, Non-Negative) The impurity state to be chosen for this fragment.
    - (Integer, Non-Negative) The bath state to be chosen for this fragment.
  - [Same as above]
	
Next, the values for the two electron integrals (nm|kl) are listed in the following format
```
(nm|kl)     n     m     k     l
```
It should be noted that the nuclear repulsion has n, m, k, and l set to zero and the one electron integrals are labelled by n and m while k and l are set to zero. This is the format of Q-Chem.

Below is an example input for H<sub>2</sub>-H<sub>2</sub>-H<sub>2</sub>. There are 6 electrons in 6 orbitals. 10 SCF solutions are desired, which will be used as bath states for the embedding. DIIS is not used, MOM is used, and the density matrix is randomized in the SCFMD method. 1000 SCF iterations is the maximum. The starting norm is 0.1 and the starting lambda is 1. There are 3 impurities in total. The first has 2 orbitals, orbital 0 and 1. We will embed the ground impurity solution in the ground bath solution. The second has 2 orbitals, orbitals 2 and 3. We will embed the first excited impurity solution in the ground bath solution. The third has 2 orbitals, orbitals 4 and 5. We will embed the ground impurity solution in the ground bath solution.

```
6 6 10
0 1 2
1000
0.1 1
3
2 0 1 0 0
2 2 3 1 0
2 4 5 0 0
 0.86029666242200	1	1	1	1
-0.00569069357932	1	1	1	2
  (And the rest of the integrals)
```

## Format of the Overlap Matrix File

The values for the overlap matrix are listed in the following format. The overlap matrix is symmetrized, so only the upper triangle needs to be specified.
```
i     j     <i|j>
```
Below is an example for H<sub>2</sub>O in a STO-3G basis.   
```
    1     1    1.000000000000000
    2     1    0.236703936510848
    2     2    1.000000000000000
    3     1   -0.000000000000000
    3     2    0.000000000000000
    3     3    1.000000000000000
	(And the rest of the overlap integrals)
```
