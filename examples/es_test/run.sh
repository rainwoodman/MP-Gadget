export OMP_NUM_THREADS=2

ROOT=../../

mpirun -np 2 $ROOT/genic/MP-GenIC paramfile.genic && \
mpirun -np 2 $ROOT/gadget/MP-Gadget paramfile.gadget 
