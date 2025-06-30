#!/bin/tcsh

setenv OMP_NUM_THREADS 1
set ncpus = 1
set maindir = /usr/workspace/kaufman/ROMFPMD/mgmol_libtorch_test_clean/mgmol

setenv LD_LIBRARY_PATH ${maindir}/build_quartz/libROM/build/lib:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH ${maindir}/install_quartz/lib:$LD_LIBRARY_PATH

set exe = testDMandEnergyAndForces_libtorch
cp $maindir/build_quartz/tests/$exe .

set datadir = $maindir/examples/PinnedH2O
cp $datadir/coords_test1.in .
ln -s -f $maindir/potentials/pseudo.O_ONCV_PBE_SG15 .
ln -s -f $maindir/potentials/pseudo.H_ONCV_PBE_SG15 .

set cfg_offline = mgmol.cfg

source $maindir/scripts/modules.quartz

srun -p pdebug -n $ncpus $exe -c $cfg_offline -i coords_test1.in >test.out &

date
