OPTIMIZE = -O3 -g -Wall -ffast-math -march=native
#-fnocommon fixes linker errors on Mac.
#Without it uninitialised variables are not included in static libraries.
# For some reason apple clang doesn't build openmp into their binary.
# You must install openmpi from homebrew using gcc first with, eg, "brew install open-mpi --cc=gcc-9 --build-from-source"
OPTIMIZE += -fno-common -fopenmp
# Find the sdk path on Mac
OPT += -isysroot $(shell xcrun -sdk macosx --show-sdk-path)

GSL_INCL = $(shell pkg-config --cflags gsl)
GSL_LIBS = $(shell pkg-config --libs gsl)

OPT += -DVALGRIND     # allow debugging with valgrind, disable the GADGET memory allocator.
#OPT += -DDEBUG      # print a lot of debugging messages
#Use alternative OpenMP locks, instead of the pthread spinlocks. Required on mac.
OPT += -DNO_OPENMP_SPINLOCK
#OPT	+=  -DNO_ISEND_IRECV_IN_DOMAIN     #sparse MPI_Alltoallv do not use ISEND IRECV
