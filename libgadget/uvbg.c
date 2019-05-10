/*=============================================================================
 * A first pass, inefficient implementation of a patchy UV ionising background
 * calculation.  This is deliberately dumb and mega inefficient, but is as
 * close to the implementation in Meraxes as I can reasonably make it.  This
 * will act as a baseline for further iterations that can take advantage of
 * calculations already happening with the PM forces etc. and taking advantage
 * of the careful domain decompostion already in place.
============================================================================*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <bigfile-mpi.h>

#include "uvbg.h"
#include "utils.h"
#include "allvars.h"
#include "partmanager.h"
#include "petapm.h"


// TODO(smutch): This should be a parameter.
static const int uvbg_dim = 64;

static void assign_slabs(UVBGgrids *grids)
{
    message(0, "Assigning slabs to MPI cores...\n");

    // Allocations made in this function are free'd in `free_reionization_grids`.
    fftwf_mpi_init();

    // Assign the slab size
    int n_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Use fftw to find out what slab each rank should get
    ptrdiff_t local_nix, local_ix_start;
    ptrdiff_t local_n_complex = fftwf_mpi_local_size_3d(uvbg_dim, uvbg_dim, uvbg_dim / 2 + 1, MPI_COMM_WORLD, &local_nix, &local_ix_start);

    // let every rank know...
    ptrdiff_t* slab_nix = malloc(sizeof(ptrdiff_t) * n_ranks); ///< array of number of x cells of every rank
    grids->slab_nix = slab_nix;
    MPI_Allgather(&local_nix, sizeof(ptrdiff_t), MPI_BYTE, slab_nix, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);

    ptrdiff_t *slab_ix_start = malloc(sizeof(ptrdiff_t) * n_ranks); ///< array first x cell of every rank
    grids->slab_ix_start = slab_ix_start;
    slab_ix_start[0] = 0;
    for (int ii = 1; ii < n_ranks; ii++)
        slab_ix_start[ii] = slab_ix_start[ii - 1] + slab_nix[ii - 1];

    ptrdiff_t *slab_n_complex = malloc(sizeof(ptrdiff_t) * n_ranks); ///< array of allocation counts for every rank
    grids->slab_n_complex = slab_n_complex;
    MPI_Allgather(&local_n_complex, sizeof(ptrdiff_t), MPI_BYTE, slab_n_complex, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);
}

static void malloc_grids(UVBGgrids *grids)
{
    int this_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    ptrdiff_t slab_n_complex = grids->slab_n_complex[this_rank];

    grids->deltax = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    grids->deltax_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    // grids->uvphot = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    // grids->uvphot_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    // grids->xHI = fftwf_alloc_real((size_t)slab_n_real);
    // grids->J21 = fftwf_alloc_real((size_t)slab_n_real);
}

static void free_grids(UVBGgrids *grids)
{
    free(grids->slab_n_complex);
    free(grids->slab_ix_start);
    free(grids->slab_nix);

    // fftwf_free(grids->J_21);
    // fftwf_free(grids->xHI);
    // fftwf_free(grids->uvphot_filtered);
    // fftwf_free(grids->uvphot);
    fftwf_free(grids->deltax_filtered);
    fftwf_free(grids->deltax);
    // fftwf_free(grids->z_at_ionization);
}


static inline int pos_to_ngp(double x, double side, int nx)
{
    int ind = (int)nearbyint(x / side * (double)nx);

    if (ind > nx - 1)
        ind = 0;

    assert(ind > -1);

    return ind;
}


static inline int compare_ptrdiff(const void* a, const void* b)
{
    ptrdiff_t result = *(ptrdiff_t*)a - *(ptrdiff_t*)b;

    return (int)result;
}


static int searchsorted(void* val,
    void* arr,
    int count,
    size_t size,
    int (*compare)(const void*, const void*),
    int imin,
    int imax)
{
    // check if we need to init imin and imax
    if ((imax < 0) && (imin < 0)) {
        imin = 0;
        imax = count - 1;
    }

    // test if we have found the result
    if ((imax - imin) < 0)
        return imax;
    else {
        // calculate midpoint to cut set in half
        int imid = imin + ((imax - imin) / 2);
        void* arr_val = (void*)(((char*)arr + imid * size));

        // three-way comparison
        if (compare(arr_val, val) > 0)
            // key is in lower subset
            return searchsorted(val, arr, count, size, compare, imin, imid - 1);
        else if (compare(arr_val, val) < 0)
            // key is in upper subset
            return searchsorted(val, arr, count, size, compare, imid + 1, imax);
        else
            // key has been found
            return imid;
    }
}


typedef enum index_type {
    INDEX_PADDED = 5674,
    INDEX_REAL,
    INDEX_COMPLEX_HERM,
} index_type;

static inline int grid_index(int i, int j, int k, int dim, index_type type)
{
    int ind = -1;

    switch (type) {
    case INDEX_PADDED:
        ind = k + (2 * (dim / 2 + 1)) * (j + dim * i);
        break;
    case INDEX_REAL:
        ind = k + dim * (j + dim * i);
        break;
    case INDEX_COMPLEX_HERM:
        ind = k + (dim / 2 + 1) * (j + dim * i);
        break;
    default:
        endrun(1, "Unknown indexing type in `grid_index`.");
        break;
    }

    return ind;
}


static void populate_grids(UVBGgrids *grids)
{
    // TODO(smutch): Is this stored somewhere globally?
    int nranks = -1, this_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    ptrdiff_t *slab_nix = grids->slab_nix;
    ptrdiff_t *slab_ix_start = grids->slab_ix_start;

    // create a buffer on each rank which is as large as the largest LOGICAL allocation on any single rank
    int buffer_size = 0;
    for (int ii = 0; ii < nranks; ii++)
        if (slab_nix[ii] > buffer_size)
            buffer_size = (int)slab_nix[ii];

    buffer_size *= uvbg_dim * uvbg_dim;
    float *buffer = fftwf_alloc_real((size_t)buffer_size);

    // I am going to reuse the RegionInd member which is from petapm.  I
    // *think* this is ok as we are doing this after the grav calculations.
    // This is a potentially stupid way to do things anyway and will most
    // definitely need to be changed! There is no way we should have to search
    // all of the particles to find out what slab they sit on, and then loop
    // through all particles again n_slab times!
    double box_size = All.BoxSize;
    for(int ii = 0; ii < PartManager->NumPart; ii++) {
        if((!P[ii].IsGarbage) && (!P[ii].Swallowed) && (P[ii].Type < 5)) {
            ptrdiff_t ix = pos_to_ngp(P[ii].Pos[0], box_size, uvbg_dim);
            P[ii].RegionInd = searchsorted(&ix, slab_ix_start, nranks, sizeof(ptrdiff_t), compare_ptrdiff, -1, -1);
        } else {
            P[ii].RegionInd = -1;
        }
    }


    // enum property {
        // prop_gas,
        // prop_stars
    // };
    // for (int prop = prop_gas; prop <= prop_stars; prop++) {

    for (int i_r = 0; i_r < nranks; i_r++) {

        // init the buffer
        for (int ii = 0; ii < buffer_size; ii++)
            buffer[ii] = (float)0.;

        // fill the local buffer for this slab
        unsigned int count = 0;
        for(int ii = 0; ii < PartManager->NumPart; ii++) {
            if(P[ii].RegionInd == i_r) {
                int ix = (int)(pos_to_ngp(P[ii].Pos[0], box_size, uvbg_dim) - slab_ix_start[i_r]);
                int iy = pos_to_ngp(P[ii].Pos[1], box_size, uvbg_dim);
                int iz = pos_to_ngp(P[ii].Pos[2], box_size, uvbg_dim);

                int ind = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);

                buffer[ind] += P[ii].Mass;
                count++;
            }
        }

        message(0, "Added %d particles to grid.\n", count);

        // reduce on to the correct rank
        if (this_rank == i_r)
            MPI_Reduce(MPI_IN_PLACE, buffer, (int)buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        else
            MPI_Reduce(buffer, buffer, (int)buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);

        if (this_rank == i_r)
            for (int ix = 0; ix < slab_nix[i_r]; ix++)
                for (int iy = 0; iy < uvbg_dim; iy++)
                    for (int iz = 0; iz < uvbg_dim; iz++) {
                        float val = buffer[grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL)];
                        if (val < 0)
                            val = 0;
                        grids->deltax[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)] = val;
                    }
    }


    fftwf_free(buffer);
}


void calculate_uvbg()
{
    message(0, "Creating UVBG grids.\n");

    UVBGgrids grids;
    assign_slabs(&grids);
    malloc_grids(&grids);

    populate_grids(&grids);

    // DEBUG =========================================================================================
    int this_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    int local_nix = grids.slab_nix[this_rank];
    int grid_size = (size_t)(local_nix * uvbg_dim * uvbg_dim);
    float* grid = (float*)calloc(grid_size, sizeof(float));
    for (int ii = 0; ii < local_nix; ii++)
        for (int jj = 0; jj < uvbg_dim; jj++)
            for (int kk = 0; kk < uvbg_dim; kk++)
                grid[grid_index(ii, jj, kk, uvbg_dim, INDEX_REAL)] = (grids.deltax)[grid_index(ii, jj, kk, uvbg_dim, INDEX_PADDED)];

    FILE *fout;
    char fname[128];
    sprintf(fname, "output/dump_r%03d.dat", this_rank);
    if((fout = fopen(fname, "wb")) == NULL) {
      endrun(1, "poop...");
    }
    fwrite(grid, sizeof(float), grid_size, fout);
    fclose(fout);
    free(grid);
    // ===============================================================================================

    free_grids(&grids);

    walltime_measure("/UVBG/CreateGrids");
}
