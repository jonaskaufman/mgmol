// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory.
// Written by J.-L. Fattebert, D. Osei-Kuffuor and I.S. Dunn.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "DistMatrix.h"
#include "SquareLocalMatrices.h"
#include "global.h"
#include "random.h"

class GramMatrix;

class Power
{
private:
    static Timer compute_tm_;
    static Timer compute_gen_tm_;

    static std::ostream* os_;

    MATDTYPE diff2(
        std::vector<MATDTYPE>& y, std::vector<MATDTYPE>& v, const MATDTYPE theta,
        const bool verbose);

    MATDTYPE power(LocalMatrices<MATDTYPE>& A, std::vector<MATDTYPE>& y,
        const int maxits, const double epsilon, const bool verbose);

    // use shift to target highest or lowest eigenvalue
    double shift_;

    std::vector<DISTMATDTYPE> vec1_;
    std::vector<DISTMATDTYPE> vec2_;

public:
    Power(const int n)
        : shift_(0.), vec1_(generate_rand(n)), vec2_(generate_rand(n))
    {
        
    }

    void computeEigenInterval(
        SquareLocalMatrices<MATDTYPE>& A, double& emin, double& emax,
        const double epsilon = 1.e-2, const bool verbose = false);
    void computeGenEigenInterval(
        dist_matrix::DistMatrix<DISTMATDTYPE>& mat, GramMatrix& gm,
        std::vector<double>& interval, const int maxits, const double pad);

    static void printTimers(std::ostream& os)
    {
        compute_tm_.print(os);
        compute_gen_tm_.print(os);
    }
};
