export OMP_NUM_THREADS=2
ROOT=../../
mpirun -np 1 $ROOT/genic/MP-GenIC paramfile.genic || exit 1
mpirun -np 1 $ROOT/gadget/MP-Gadget paramfile.gadget || exit 1
