// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "LocGridOrbitals.h"
#include "MGmol.h"

#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

// Save the wavefunction snapshots
template <class OrbitalsType>
int MGmol<OrbitalsType>::save_orbital_snapshot(std::string file_path, OrbitalsType& orbitals)
{
    std::cout << "rom.cc: save_orbital_snapshot: " << file_path << std::endl;
    std::string snapshot_filename = file_path;
    struct stat s; 
    if (stat(file_path.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR)
            snapshot_filename = file_path + "/orbital";
        else if (s.st_mode & S_IFREG)
        {
            snapshot_filename = file_path + "_orbital";
        }
        else
        {
            std::cout << file_path << " exists but is not a directory or a file." << std::endl;
            return 1;
        }
    }

    const int dim = orbitals.getLocNumpt();
    const int totalSamples = orbitals.chromatic_number();

    CAROM::Options svd_options(dim, totalSamples, 1);
    CAROM::BasisGenerator basis_generator(svd_options, false, snapshot_filename);

    for (int i = 0; i < totalSamples; ++i)
        basis_generator.takeSample(orbitals.getPsi(i), 0.0, 0.01);

    basis_generator.writeSnapshot();

    return 0;
}

template class MGmol<LocGridOrbitals>;
template class MGmol<ExtendedGridOrbitals>;
