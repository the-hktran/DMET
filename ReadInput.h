#include <Eigen/Dense>
#include <vector>

class InputObj
{
    public:
        void GetInputName();
        void SetNames(char*, char*, char*);
        void SetNames(std::string, std::string, std::string);
        void Set();
        std::map< std::string, double > Integrals;
        Eigen::MatrixXd OverlapMatrix;
        std::string IntegralsInput;
        std::string OverlapInput;
        std::string OutputName;
		unsigned short int NumOcc;
		unsigned short int NumSoln;
        unsigned short int NumElectrons;
        unsigned short int NumAO;
        std::vector< bool > Options;
        bool doScan = false;
        int ScanIntStart;
        int ScanIntEnd;
        double ScanValStart;
        double ScanValStep;
        int DensityOption = -1; // Will return an error if not set.
        Eigen::MatrixXd InitialCoeff;
        int MaxSCF = 5000;
        double StartNorm = 0.1;
        double StartLambda = 1;
};