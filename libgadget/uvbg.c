/*=============================================================================
 * A first pass, inefficient implementation of a patchy UV ionising background
 * calculation.  This is deliberately dumb and mega inefficient, but is as
 * close to the implementation in Meraxes as I can reasonably make it.  This
 * will act as a baseline for further iterations that can take advantage of
 * calculations already happening with the PM forces etc. and taking advantage
 * of the careful domain decompostion already in place.
============================================================================*/

// TODO(smutch): Mallocs and frees should follow rest of code

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <ctype.h>
#include <bigfile-mpi.h>
#include <complex.h>
#include <fftw3.h>

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
    ptrdiff_t slab_n_real = grids->slab_nix[this_rank] * uvbg_dim * uvbg_dim;

    grids->deltax = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    grids->deltax_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    grids->uvphot = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    grids->uvphot_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    grids->xHI = fftwf_alloc_real((size_t)slab_n_real);
    grids->J21 = fftwf_alloc_real((size_t)slab_n_real);
}

static void free_grids(UVBGgrids *grids)
{
    free(grids->slab_n_complex);
    free(grids->slab_ix_start);
    free(grids->slab_nix);

    fftwf_free(grids->J21);
    fftwf_free(grids->xHI);
    fftwf_free(grids->uvphot_filtered);
    fftwf_free(grids->uvphot);
    fftwf_free(grids->deltax_filtered);
    fftwf_free(grids->deltax);
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

    // create buffers on each rank which is as large as the largest LOGICAL allocation on any single rank
    int buffer_size = 0;
    for (int ii = 0; ii < nranks; ii++)
        if (slab_nix[ii] > buffer_size)
            buffer_size = (int)slab_nix[ii];

    buffer_size *= uvbg_dim * uvbg_dim;
    float *buffer_deltax = fftwf_alloc_real((size_t)buffer_size);
    float *buffer_uvphot = fftwf_alloc_real((size_t)buffer_size);

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


    for (int i_r = 0; i_r < nranks; i_r++) {

        // init the buffers
        for (int ii = 0; ii < buffer_size; ii++) {
            buffer_deltax[ii] = (float)0.;
            buffer_uvphot[ii] = (float)0.;
        }

        // fill the local buffer for this slab
        unsigned int count_deltax = 0, count_uvphot = 0;
        for(int ii = 0; ii < PartManager->NumPart; ii++) {
            if(P[ii].RegionInd == i_r) {
                int ix = (int)(pos_to_ngp(P[ii].Pos[0], box_size, uvbg_dim) - slab_ix_start[i_r]);
                int iy = pos_to_ngp(P[ii].Pos[1], box_size, uvbg_dim);
                int iz = pos_to_ngp(P[ii].Pos[2], box_size, uvbg_dim);

                int ind = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);

                buffer_deltax[ind] += P[ii].Mass;
                count_deltax++;

                if(P[ii].Type == 4) {
                    buffer_uvphot[ind] += P[ii].Mass;
                    count_uvphot++;
                }

            }
        }

        message(0, "Added %d particles to deltax grid.\n", count_deltax);
        message(0, "Added %d particles to uvphot grid.\n", count_uvphot);

        // reduce on to the correct rank
        if (this_rank == i_r) {
            MPI_Reduce(MPI_IN_PLACE, buffer_deltax, (int)buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        }
        else
            MPI_Reduce(buffer_deltax, buffer_deltax, (int)buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);

        if (this_rank == i_r) {
            MPI_Reduce(MPI_IN_PLACE, buffer_uvphot, (int)buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        }
        else
            MPI_Reduce(buffer_uvphot, buffer_uvphot, (int)buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);


        if (this_rank == i_r)
            for (int ix = 0; ix < slab_nix[i_r]; ix++)
                for (int iy = 0; iy < uvbg_dim; iy++)
                    for (int iz = 0; iz < uvbg_dim; iz++) {
                        grids->deltax[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)] = buffer_deltax[grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL)];
                        grids->uvphot[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)] = buffer_uvphot[grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL)];
                    }
    }


    fftwf_free(buffer_uvphot);
    fftwf_free(buffer_deltax);
}


static void filter(fftwf_complex* box, int local_ix_start, int slab_nx, int grid_dim, float R)
{
    const int filter_type = 0;  // TODO(smutch): Make this an option
    int middle = grid_dim / 2;
    float box_size = (float)All.BoxSize;
    float delta_k = (float)(2.0 * M_PI / box_size);

    // Loop through k-box
    for (int n_x = 0; n_x < slab_nx; n_x++) {
        float k_x;
        int n_x_global = n_x + local_ix_start;

        if (n_x_global > middle)
            k_x = (n_x_global - grid_dim) * delta_k;
        else
            k_x = n_x_global * delta_k;

        for (int n_y = 0; n_y < grid_dim; n_y++) {
            float k_y;

            if (n_y > middle)
                k_y = (n_y - grid_dim) * delta_k;
            else
                k_y = n_y * delta_k;

            for (int n_z = 0; n_z <= middle; n_z++) {
                float k_z = n_z * delta_k;

                float k_mag = sqrtf(k_x * k_x + k_y * k_y + k_z * k_z);

                float kR = k_mag * R; // Real space top-hat

                switch (filter_type) {
                case 0: // Real space top-hat
                    if (kR > 1e-4)
                        box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] *= (fftwf_complex)(3.0 * (sinf(kR) / powf(kR, 3) - cosf(kR) / powf(kR, 2)));
                    break;

                case 1: // k-space top hat
                    kR *= 0.413566994; // Equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
                    if (kR > 1)
                        box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] = (fftwf_complex)0.0;
                    break;

                case 2: // Gaussian
                    kR *= 0.643; // Equates integrated volume to the real space top-hat
                    box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] *= (fftwf_complex)(powf((float)M_E,
                        (float)(-kR * kR / 2.0)));
                    break;

                default:
                    if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {
                        endrun(1, "ReionFilterType type %d is undefined!", filter_type);
                    }
                    break;
                }
            }
        }
    } // End looping through k box
}


static double RtoM(double R)
{
    // All in internal units
    const int filter = 0;  // TODO(smutch): Make this an option
    double OmegaM = All.CP.Omega0;
    double RhoCrit = 200.;  // TODO(smutch): This should probably be Bryan & Norman value (z dep)

    switch (filter) {
    case 0: //top hat M = (4/3) PI <rho> R^3
        return (4.0 / 3.0) * M_PI * pow(R, 3) * (OmegaM * RhoCrit);
    case 1: //gaussian: M = (2PI)^1.5 <rho> R^3
        return pow(2 * M_PI, 1.5) * OmegaM * RhoCrit * pow(R, 3);
    default: // filter not defined
        endrun(1, "Unrecognised RtoM filter (%d).", filter);
        break;
    }

    return -1;
}

static void create_plans(UVBGgrids *grids)
{
    grids->plan_dft_r2c = fftwf_mpi_plan_dft_r2c_3d(uvbg_dim, uvbg_dim, uvbg_dim,
            grids->deltax, (fftwf_complex*)grids->deltax,
            MPI_COMM_WORLD, FFTW_PATIENT);
}

static void destroy_plans(UVBGgrids *grids)
{
    fftwf_destroy_plan(grids->plan_dft_r2c);
}

static void find_HII_bubbles(UVBGgrids *grids)
{
    /* This function is based on find_hII_bubbles from 21cmFAST, but has been
     * largely rewritten. */

    // TODO(smutch): TAKE A VERY VERY CLOSE LOOK AT UNITS!!!!

    double box_size = All.BoxSize; // Mpc/h
    double pixel_volume = pow(box_size / (double)uvbg_dim, 3); // (Mpc/h)^3
    double cell_length_factor = 0.620350491;
    double total_n_cells = pow((double)uvbg_dim, 3);
    int local_nix = (int)(grids->slab_nix[run_globals.mpi_rank]);
    int slab_n_real = local_nix * uvbg_dim * uvbg_dim;
    double ReionNionPhotPerBary = 4000.;  // TODO(smutch): replace this!
    float J21_aux = 0;
    double J21_aux_constant = 0;
    double density_over_mean = 0;
    double sfr_density = 0;
    double f_coll_stars = 0;
    int i_real = 0;
    int i_padded = 0;

    // This parameter choice is sensitive to noise on the cell size, at least for the typical
    // cell sizes in RT simulations. It probably doesn't matter for larger cell sizes.
    if ((box_size / (double)uvbg_dim) < 1.0) // Fairly arbitrary length based on 2 runs Sobacchi did
        cell_length_factor = 1.0;

    // Init J21 and xHI
    float* J21 = grids->J21;
    float* xHI = grids->xHI;
    for (int ii = 0; ii < slab_n_real; ii++) {
        J21[ii] = 0.0;
        xHI[ii] = (float)1.0;
    }

    // Forward fourier transform to obtain k-space fields
    float* deltax = grids->deltax;
    fftwf_complex* deltax_unfiltered = (fftwf_complex*)deltax; // WATCH OUT!
    fftwf_complex* deltax_filtered = grids->deltax_filtered;
    fftwf_execute_dft_r2c(grids->plan_dft_r2c, deltax, deltax_unfiltered);

    float* uvphot = grids.uvphot;
    fftwf_complex* uvphot_unfiltered = (fftwf_complex*)uvphot; // WATCH OUT!
    fftwf_complex* uvphot_filtered = grids->uvphot_filtered;
    fftwf_execute_dft_r2c(grids->plan_dft_r2c, uvphot, uvphot_unfiltered);

    // Remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
    // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
    int this_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    int slab_n_complex = (int)(grids->slab_n_complex[run_globals.mpi_rank]);
    for (int ii = 0; ii < slab_n_complex; ii++) {
        deltax_unfiltered[ii] /= total_n_cells;
        stars_unfiltered[ii] /= total_n_cells;
    }

    // Loop through filter radii
    // TODO(smutch): These should be parameters
    double ReionRBubbleMax = 20.34; // Mpc/h
    double ReionRBubbleMin = 0.4068; // Mpc/h
    double R = fmin(ReionRBubbleMax, cell_length_factor * box_size); // Mpc/h
    double ReionDeltaRFactor = 1.1;
    double ReionGammaHaloBias = 2.0;

    bool flag_last_filter_step = false;

    // TODO(smutch): Cont here!
    while (!flag_last_filter_step) {
        // check to see if this is our last filtering step
        if (((R / ReionDeltaRFactor) <= (cell_length_factor * box_size / (double)uvbg_dim))
            || ((R / ReionDeltaRFactor) <= ReionRBubbleMin)) {
            flag_last_filter_step = true;
            R = cell_length_factor * box_size / (double)uvbg_dim;
        }

        // DEBUG
        // mlog("R = %.2e (h=0.678 -> %.2e)", MLOG_MESG, R, R/0.678);
        mlog(".", MLOG_CONT);

        // copy the k-space grids
        memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
        memcpy(stars_filtered, stars_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
        memcpy(sfr_filtered, sfr_unfiltered, sizeof(fftwf_complex) * slab_n_complex);

        // do the filtering unless this is the last filter step
        int local_ix_start = (int)(run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank]);
        if (!flag_last_filter_step) {
            filter(deltax_filtered, local_ix_start, local_nix, uvbg_dim, (float)R);
            filter(stars_filtered, local_ix_start, local_nix, uvbg_dim, (float)R);
            filter(sfr_filtered, local_ix_start, local_nix, uvbg_dim, (float)R);
        }

        // inverse fourier transform back to real space
        plan = fftwf_mpi_plan_dft_c2r_3d(uvbg_dim, uvbg_dim, uvbg_dim, deltax_filtered, (float*)deltax_filtered, run_globals.mpi_comm, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        plan = fftwf_mpi_plan_dft_c2r_3d(uvbg_dim, uvbg_dim, uvbg_dim, stars_filtered, (float*)stars_filtered, run_globals.mpi_comm, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        plan = fftwf_mpi_plan_dft_c2r_3d(uvbg_dim, uvbg_dim, uvbg_dim, sfr_filtered, (float*)sfr_filtered, run_globals.mpi_comm, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        // Perform sanity checks to account for aliasing effects
        for (int ix = 0; ix < local_nix; ix++)
            for (int iy = 0; iy < uvbg_dim; iy++)
                for (int iz = 0; iz < uvbg_dim; iz++) {
                    i_padded = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);
                    ((float*)deltax_filtered)[i_padded] = fmaxf(((float*)deltax_filtered)[i_padded], -1 + REL_TOL);
                    ((float*)stars_filtered)[i_padded] = fmaxf(((float*)stars_filtered)[i_padded], 0.0);
                    ((float*)sfr_filtered)[i_padded] = fmaxf(((float*)sfr_filtered)[i_padded], 0.0);
                }

        // #ifdef DEBUG
        //   {
        //     char fname_debug[STRLEN];
        //     sprintf(fname_debug, "post_stars-%d_R%.3f.dat", run_globals.mpi_rank, R);
        //     FILE *fd_debug = fopen(fname_debug, "wb");
        //     for(int ix = 0; ix < run_globals.reion_grids.slab_nix[run_globals.mpi_rank]; ix++)
        //       for(int iy = 0; iy < uvbg_dim; iy++)
        //         for(int iz = 0; iz < uvbg_dim; iz++)
        //           fwrite(&(run_globals.reion_grids.stars_filtered[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)]), sizeof(float), 1, fd_debug);
        //     fclose(fd_debug);
        //   }
        //   {
        //     char fname_debug[STRLEN];
        //     sprintf(fname_debug, "post_sfr-%d_R%.3f.dat", run_globals.mpi_rank, R);
        //     FILE *fd_debug = fopen(fname_debug, "wb");
        //     for(int ix = 0; ix < run_globals.reion_grids.slab_nix[run_globals.mpi_rank]; ix++)
        //       for(int iy = 0; iy < uvbg_dim; iy++)
        //         for(int iz = 0; iz < uvbg_dim; iz++)
        //           fwrite(&(run_globals.reion_grids.sfr_filtered[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)]), sizeof(float), 1, fd_debug);
        //     fclose(fd_debug);
        //   }
        // #endif

        /*
     * Main loop through the box...
     */

        J21_aux_constant = (1.0 + redshift) * (1.0 + redshift) / (4.0 * M_PI)
            * run_globals.params.physics.ReionAlphaUV * PLANCK
            * 1e21 // * run_globals.params.physics.ReionEscapeFrac
            * R * units->UnitLength_in_cm * ReionNionPhotPerBary / PROTONMASS
            * units->UnitMass_in_g / pow(units->UnitLength_in_cm, 3) / units->UnitTime_in_s;

        // DEBUG
        // for (int ix=0; ix<local_nix; ix++)
        //   for (int iy=0; iy<uvbg_dim; iy++)
        //     for (int iz=0; iz<uvbg_dim; iz++)
        //       if (((float *)sfr_filtered)[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)] > 0)
        //       {
        //         mlog("J21_aux ==========", MLOG_OPEN);
        //         mlog("redshift = %.2f", MLOG_MESG, redshift);
        //         mlog("alpha = %.2e", MLOG_MESG, run_globals.params.physics.ReionAlphaUV);
        //         mlog("PLANCK = %.2e", MLOG_MESG, PLANCK);
        //         mlog("ReionEscapeFrac = %.2f", MLOG_MESG, ReionEscapeFrac);
        //         mlog("ReionNionPhotPerBary = %.1f", MLOG_MESG, ReionNionPhotPerBary);
        //         mlog("UnitMass_in_g = %.2e", MLOG_MESG, units->UnitMass_in_g);
        //         mlog("UnitLength_in_cm = %.2e", MLOG_MESG, units->UnitLength_in_cm);
        //         mlog("PROTONMASS = %.2e", MLOG_MESG, PROTONMASS);
        //         mlog("-> J21_aux_constant = %.2e", MLOG_MESG, J21_aux_constant);
        //         mlog("pixel_volume = %.2e", MLOG_MESG, pixel_volume);
        //         mlog("sfr_density[%d, %d, %d] = %.2e", MLOG_MESG, ((float *)sfr_filtered)[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)] / pixel_volume);
        //         mlog("-> J21_aux = %.2e", MLOG_MESG, ((float *)sfr_filtered)[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)] / pixel_volume * J21_aux_constant);
        //         mlog("==========", MLOG_CLOSE);
        //         if (run_globals.mpi_rank == 0)
        //           ABORT(EXIT_SUCCESS);
        //       }

        for (int ix = 0; ix < local_nix; ix++)
            for (int iy = 0; iy < uvbg_dim; iy++)
                for (int iz = 0; iz < uvbg_dim; iz++) {
                    i_real = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);
                    i_padded = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);

                    density_over_mean = 1.0 + (double)((float*)deltax_filtered)[i_padded];

                    f_coll_stars = (double)((float*)stars_filtered)[i_padded] / (RtoM(R) * density_over_mean)
                        * (4.0 / 3.0) * M_PI * pow(R, 3.0) / pixel_volume;

                    sfr_density = (double)((float*)sfr_filtered)[i_padded] / pixel_volume; // In internal units

                    // #ifdef DEBUG
                    //           if(abs(redshift - 23.074) < 0.01)
                    //             debug("%d, %d, %d, %d, %g, %g, %g, %g, %g, %g, %g, %g, %g, %g, %g\n",
                    //                 run_globals.mpi_rank, ix, iy, iz,
                    //                 R, density_over_mean, f_coll_stars,
                    //                 ((float *)stars_filtered)[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)],
                    //                 RtoM(R), (4.0/3.0)*M_PI*pow(R,3.0), pixel_volume, 1.0/ReionEfficiency, sfr_density, ((float*)sfr_filtered)[grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED)], (float)(sfr_density * J21_aux_constant));
                    // #endif

                    if (flag_ReionUVBFlag)
                        J21_aux = (float)(sfr_density * J21_aux_constant);

                    // Check if ionised!
                    if (f_coll_stars > 1.0 / ReionEfficiency) // IONISED!!!!
                    {
                        // If it is the first crossing of the ionisation barrier for this cell (largest R), let's record J21
                        if (xH[i_real] > REL_TOL)
                            if (flag_ReionUVBFlag)
                                J21[i_real] = J21_aux;

                        // Mark as ionised
                        xH[i_real] = 0;

                        // Record radius
                        r_bubble[i_real] = (float)R;
                    }
                    // Check if this is the last filtering step.
                    // If so, assign partial ionisations to those cells which aren't fully ionised
                    else if (flag_last_filter_step && (xH[i_real] > REL_TOL))
                        xH[i_real] = (float)(1.0 - f_coll_stars * ReionEfficiency);

                    // Check if new ionisation
                    float* z_in = run_globals.reion_grids.z_at_ionization;
                    if ((xH[i_real] < REL_TOL) && (z_in[i_real] < 0)) // New ionisation!
                    {
                        z_in[i_real] = (float)redshift;
                        if (flag_ReionUVBFlag)
                            run_globals.reion_grids.J21_at_ionization[i_real] = J21_aux * (float)ReionGammaHaloBias;
                    }
                }
        // iz

        R /= ReionDeltaRFactor;
    }

    // Find the volume and mass weighted neutral fractions
    // TODO: The deltax grid will have rounding errors from forward and reverse
    //       FFT. Should cache deltax slabs prior to ffts and reuse here.
    double volume_weighted_global_xH = 0.0;
    double mass_weighted_global_xH = 0.0;
    double mass_weight = 0.0;

    for (int ix = 0; ix < local_nix; ix++)
        for (int iy = 0; iy < uvbg_dim; iy++)
            for (int iz = 0; iz < uvbg_dim; iz++) {
                i_real = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);
                i_padded = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);
                volume_weighted_global_xH += (double)xH[i_real];
                density_over_mean = 1.0 + (double)((float*)deltax_filtered)[i_padded];
                mass_weighted_global_xH += (double)(xH[i_real]) * density_over_mean;
                mass_weight += density_over_mean;
            }

    MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_xH, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_xH, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &mass_weight, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);

    volume_weighted_global_xH /= total_n_cells;
    mass_weighted_global_xH /= mass_weight;
    run_globals.reion_grids.volume_weighted_global_xH = volume_weighted_global_xH;
    run_globals.reion_grids.mass_weighted_global_xH = mass_weighted_global_xH;
    
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
