// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Control.h"
#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "mgmol_run.h"

#include <cassert>
#include <iostream>
#include <time.h>
#include <vector>
#include <cmath>
#include <torch/script.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    int mpirc = MPI_Init(&argc, &argv);
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Initialization failed!!!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    /*
     * Initialize general things, like magma, openmp, IO, ...
     */
    mgmol_init(comm);

    /*
     * read runtime parameters
     */
    std::string input_filename("");
    std::string lrs_filename;
    std::string constraints_filename("");

    float total_spin = 0.;
    bool with_spin   = false;

    po::variables_map vm;

    // read from PE0 only
    if (MPIdata::onpe0)
    {
        read_config(argc, argv, vm, input_filename, lrs_filename,
            constraints_filename, total_spin, with_spin);
    }

    MGmol_MPI::setup(comm, std::cout, with_spin);
    MGmol_MPI& mmpi      = *(MGmol_MPI::instance());
    MPI_Comm global_comm = mmpi.commGlobal();

    /*
     * Setup control struct with run time parameters
     */
    Control::setup(global_comm, with_spin, total_spin);
    Control& ct = *(Control::instance());

    ct.setOptions(vm);

    int ret = ct.checkOptions();
    if (ret < 0) return ret;

    mmpi.bcastGlobal(input_filename);
    mmpi.bcastGlobal(lrs_filename);

    // Enter main scope
    {
        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "Construct MGmol object..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }

        MGmolInterface* mgmol;
        if (ct.isLocMode())
            mgmol = new MGmol<LocGridOrbitals>(global_comm, *MPIdata::sout,
                input_filename, lrs_filename, constraints_filename);
        else
            mgmol = new MGmol<ExtendedGridOrbitals>(global_comm, *MPIdata::sout,
                input_filename, lrs_filename, constraints_filename);

        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "MGmol setup..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }
        mgmol->setup();

        if (MPIdata::onpe0)
        {
            std::cout << "-------------------------" << std::endl;
            std::cout << "Setup done..." << std::endl;
            std::cout << "-------------------------" << std::endl;
        }

        // here we just use the atomic positions read in and used
        // to initialize MGmol
        std::vector<double> positions;
        mgmol->getAtomicPositions(positions);
        std::vector<short> anumbers;
        mgmol->getAtomicNumbers(anumbers);
        if (MPIdata::onpe0)
        {
            std::cout << "Positions:" << std::endl;
            std::vector<short>::iterator ita = anumbers.begin();
            for (std::vector<double>::iterator it = positions.begin();
                 it != positions.end(); it += 3)
            {
                std::cout << *ita;
                for (int i = 0; i < 3; i++)
                    std::cout << "    " << *(it + i);
                std::cout << std::endl;
                ita++;
            }
        }

        // compute energy and forces using all MPI tasks
        // expect positions to be replicated on all MPI tasks
        std::vector<double> forces;
	double eks;
	bool run_dft = true;
	if (run_dft) {
        eks
            = mgmol->evaluateEnergyAndForces(positions, anumbers, forces);
        mgmol->dumpRestart();

        // print out results
        if (MPIdata::onpe0)
        {
            std::cout << "Eks1 : " << eks << std::endl;
            std::cout << "Forces2 :" << std::endl;
            for (std::vector<double>::iterator it = forces.begin();
                 it != forces.end(); it += 3)
            {
                for (int i = 0; i < 3; i++)
                    std::cout << "    " << *(it + i);
                std::cout << std::endl;
            }
        }
	}
	

        // compute energy and forces again using wavefunctions
        // from previous call
        Mesh* mymesh           = Mesh::instance();
        const pb::Grid& mygrid = mymesh->grid();

        std::shared_ptr<ProjectedMatricesInterface> projmatrices
            = mgmol->getProjectedMatrices();

        ExtendedGridOrbitals orbitals("new_orbitals", mygrid, mymesh->subdivx(),
            ct.numst, ct.bcWF, projmatrices.get(), nullptr, nullptr, nullptr,
            nullptr);

        const pb::PEenv& myPEenv = mymesh->peenv();
        HDFrestart h5file("WF", myPEenv, ct.out_restart_file_type);
        orbitals.read_hdf5(h5file);
	
	// load torch model
	std::cout << "Loading torch model" << std::endl;
	torch::jit::script::Module encoder;
	torch::jit::script::Module decoder;
	try {
            encoder = torch::jit::load("traced_pod_enc.pt");
            decoder = torch::jit::load("traced_pod_dec.pt");
  	}
  	catch (const c10::Error& e) {
    	    std::cerr << "Error loading the model\n";
            return -1;
   	}
  	std::cout << "ok" << std::endl;

	// load orbitals and convert data format
	std::cout << "Converting orbital data" << std::endl;
	int n_psi = orbitals.chromatic_number();
	int d_psi = orbitals.getLocNumpt();
	std::vector<torch::jit::IValue> encoder_input;
	std::vector<at::Tensor> orbital_tensors;
	for (int i = 0; i < n_psi; ++i) {
	    orbital_tensors.push_back(torch::from_blob(orbitals.getPsi(i), {262144}, torch::TensorOptions().dtype(torch::CppTypeToScalarType<ORBDTYPE>())));
	}
	at::Tensor original_orbitals = at::stack(orbital_tensors);
	//std::cout << original_orbitals.type() << std::endl;
	original_orbitals = original_orbitals.toType(torch::CppTypeToScalarType<float>());
	//std::cout << original_orbitals.type() << std::endl;

	//std::cout << original_orbitals.numel() << std::endl;
	auto original_norm = original_orbitals.norm();
	std::cout << "Norm: " << original_norm << std::endl;
	std::vector<float> v(original_orbitals.data_ptr<float>(), original_orbitals.data_ptr<float>() + original_orbitals.numel());
	float v_norm = 0;
	for (auto i : v)
	{
	    v_norm += i*i;
	}
	v_norm = sqrt(v_norm);
	std::cout << "Norm (explicit): " << v_norm << std::endl;
	std::cout << "Shape: " << at::_shape_as_tensor(original_orbitals) << std::endl;
	encoder_input.push_back(original_orbitals);

	// encode orbitals
	std::cout << "Encoding" << std::endl;
	auto codes = encoder.forward(encoder_input).toTensor();
	std::cout << "Encoded shape: " << at::_shape_as_tensor(codes) << std::endl;
	//std::cout << codes.type() << std::endl;

	// decode	
	std::cout << "Decoding" << std::endl;	
	std::vector<torch::jit::IValue> decoder_input;
	decoder_input.push_back(codes);
	auto approx_orbitals = decoder.forward(decoder_input).toTensor();
	std::cout << "Decoded shape: " << at::_shape_as_tensor(approx_orbitals) << std::endl;
	//std::cout << approx_orbitals.type() << std::endl;

	// calculate error
	at::Tensor diff = original_orbitals - approx_orbitals;
	std::cout << "Relative error: " << diff.norm() / original_norm << std::endl;

	// set orbitals to reconstructed orbitals
	approx_orbitals = approx_orbitals.toType(torch::CppTypeToScalarType<ORBDTYPE>()).contiguous();
	//std::cout << approx_orbitals.type() << std::endl;
	for (int i = 0; i < n_psi; ++i) {
	    pb::GridFunc<ORBDTYPE> gf_psi(mymesh->grid(), ct.bcWF[0], ct.bcWF[1], ct.bcWF[2]);	    
	    gf_psi.assign(approx_orbitals[i].data_ptr<ORBDTYPE>());
            orbitals.setPsi(gf_psi, i);
	}
	
        //
        // evaluate energy and forces again, with wavefunctions
        // frozen to solution of previous problem
        //

        // reset initial DM to test iterative solve for it
        projmatrices->setDMuniform(ct.getNelSpin(), 0);
        ct.dm_inner_steps = 50;
        eks               = mgmol->evaluateDMandEnergyAndForces(
            &orbitals, positions, anumbers, forces);

        // print out results
        if (MPIdata::onpe0)
        {
            std::cout << "Eks2 : " << eks << std::endl;
            std::cout << "Forces2 :" << std::endl;
            for (std::vector<double>::iterator it = forces.begin();
                 it != forces.end(); it += 3)
            {
                for (int i = 0; i < 3; i++)
                    std::cout << "    " << *(it + i);
                std::cout << std::endl;
            }
        }

        delete mgmol;

    } // close main scope

    mgmol_finalize();

    mpirc = MPI_Finalize();
    if (mpirc != MPI_SUCCESS)
    {
        std::cerr << "MPI Finalize failed!!!" << std::endl;
    }

    time_t tt;
    time(&tt);
    if (onpe0) std::cout << " Run ended at " << ctime(&tt) << std::endl;

    return 0;
}
