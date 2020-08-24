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
#include <bigfile.h>
#include <bigfile-mpi.h>
#include <complex.h>
#include <fftw3.h>
#include <stdbool.h>
#include <gsl/gsl_integration.h>
#include <assert.h>

#include "uvbg.h"
#include "utils.h"
#include "allvars.h"
#include "partmanager.h"
#include "petapm.h"
#include "physconst.h"
#include "walltime.h"

// TODO(smutch): See if something equivalent is defined anywhere else
#define FLOAT_REL_TOL (float)1e-5

struct UVBGgrids_type UVBGgrids;

double integrand_time_to_present(double a, void *dummy)
{
    double omega_m = All.CP.Omega0;
    double omega_k = All.CP.OmegaK;
    double omega_lambda = All.CP.OmegaLambda;

    return 1 / sqrt(omega_m / a + omega_k + omega_lambda * a * a);
}

double time_to_present(double a)
{
#define WORKSIZE 1000
    gsl_function F;
    gsl_integration_workspace* workspace;
    double time;
    double result;
    double abserr;

    double hubble;
    hubble = All.CP.Hubble / All.UnitTime_in_s * All.CP.HubbleParam;
    //jdavies(second to Myr conversion)
    double s_to_Myr = 3.17098e-14;

    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = &integrand_time_to_present;

    gsl_integration_qag(&F, a, 1.0, 1.0 / hubble,
        1.0e-8, WORKSIZE, GSL_INTEG_GAUSS21, workspace, &result, &abserr);

    time = 1 / hubble * result * s_to_Myr;

    gsl_integration_workspace_free(workspace);

    // return time to present as a function of redshift
    return time;
}

static void assign_slabs()
{
    message(0, "Assigning slabs to MPI cores...\n");

    int uvbg_dim = All.UVBGdim;
    // Allocations made in this function are free'd in `free_reionization_grids`.
    fftwf_mpi_init();

    // Assign the slab size
    int n_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Use fftw to find out what slab each rank should get
    ptrdiff_t local_nix, local_ix_start;
    ptrdiff_t local_n_complex = fftwf_mpi_local_size_3d(uvbg_dim, uvbg_dim, uvbg_dim / 2 + 1, MPI_COMM_WORLD, &local_nix, &local_ix_start);

    // let every rank know...
    ptrdiff_t* slab_nix = mymalloc("slab_nix",sizeof(ptrdiff_t) * n_ranks); ///< array of number of x cells of every rank
    UVBGgrids.slab_nix = slab_nix;
    MPI_Allgather(&local_nix, sizeof(ptrdiff_t), MPI_BYTE, slab_nix, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);

    ptrdiff_t *slab_ix_start = mymalloc("slab_ix_start",sizeof(ptrdiff_t) * n_ranks); ///< array first x cell of every rank
    UVBGgrids.slab_ix_start = slab_ix_start;
    slab_ix_start[0] = 0;
    for (int ii = 1; ii < n_ranks; ii++)
    {
        slab_ix_start[ii] = slab_ix_start[ii - 1] + slab_nix[ii - 1];
        message(0,"rank %d got slab size %d starting at %d\n",ii,slab_nix[ii],slab_ix_start[ii]);
    }
    ptrdiff_t *slab_n_complex = mymalloc("slab_n_complex",sizeof(ptrdiff_t) * n_ranks); ///< array of allocation counts for every rank
    UVBGgrids.slab_n_complex = slab_n_complex;
    MPI_Allgather(&local_n_complex, sizeof(ptrdiff_t), MPI_BYTE, slab_n_complex, sizeof(ptrdiff_t), MPI_BYTE, MPI_COMM_WORLD);
}

void malloc_permanent_uvbg_grids()
{
    UVBGgrids.last_a = All.Time;
    int uvbg_dim = All.UVBGdim;
    size_t grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;

    // Note that these are full grids stored on every rank!
    UVBGgrids.J21 = mymalloc("J21", sizeof(float) * grid_n_real);
    UVBGgrids.stars = mymalloc("stars", sizeof(float) * grid_n_real);
    UVBGgrids.prev_stars = mymalloc("stars", sizeof(float) * grid_n_real);

    for(size_t ii=0; ii < grid_n_real; ii++) {
        UVBGgrids.J21[ii] = 0.0f;
    }
    for(size_t ii=0; ii < grid_n_real; ii++) {
        UVBGgrids.stars[ii] = 0.0f;
    }
    for(size_t ii=0; ii < grid_n_real; ii++) {
        UVBGgrids.prev_stars[ii] = 0.0f;
    }
}

void free_permanent_uvbg_grids()
{
    myfree(UVBGgrids.prev_stars);
    myfree(UVBGgrids.stars);
    myfree(UVBGgrids.J21);
}

static void malloc_grids()
{
    int this_rank;
    int uvbg_dim = All.UVBGdim;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    ptrdiff_t slab_n_complex = UVBGgrids.slab_n_complex[this_rank];
    ptrdiff_t slab_n_real = UVBGgrids.slab_nix[this_rank] * uvbg_dim * uvbg_dim;
    
    UVBGgrids.deltax = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    UVBGgrids.deltax_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    UVBGgrids.stars_slab = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    UVBGgrids.stars_slab_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    UVBGgrids.sfr = fftwf_alloc_real((size_t)(slab_n_complex * 2));  // padded for in-place FFT
    UVBGgrids.sfr_filtered = fftwf_alloc_complex((size_t)(slab_n_complex));
    UVBGgrids.xHI = fftwf_alloc_real((size_t)slab_n_real);
    UVBGgrids.z_at_ionization = fftwf_alloc_real((size_t)(slab_n_real));
    UVBGgrids.J21_at_ionization = fftwf_alloc_real((size_t)(slab_n_real));

    // Init grids for which values persist for the entire simulation
    for(ptrdiff_t ii=0; ii < slab_n_real; ++ii) {
        UVBGgrids.z_at_ionization[ii] = -999.0f;
        UVBGgrids.J21_at_ionization[ii] = -999.0f;
    }

    UVBGgrids.volume_weighted_global_xHI = 1.0f;
    UVBGgrids.mass_weighted_global_xHI = 1.0f;
}

static void free_grids()
{
    myfree(UVBGgrids.slab_n_complex);
    myfree(UVBGgrids.slab_ix_start);
    myfree(UVBGgrids.slab_nix);

    fftwf_free(UVBGgrids.J21_at_ionization);
    fftwf_free(UVBGgrids.z_at_ionization);
    fftwf_free(UVBGgrids.xHI);
    fftwf_free(UVBGgrids.stars_slab);
    fftwf_free(UVBGgrids.stars_slab_filtered);
    fftwf_free(UVBGgrids.deltax_filtered);
    fftwf_free(UVBGgrids.deltax);
    fftwf_free(UVBGgrids.sfr);
    fftwf_free(UVBGgrids.sfr_filtered);
}


int pos_to_ngp(double x, double side, int nx)
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


int grid_index(int i, int j, int k, int dim, index_type type)
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


static void populate_grids()
{
    // TODO(smutch): Is this stored somewhere globally?
    int uvbg_dim = All.UVBGdim;
    int nranks = -1, this_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    ptrdiff_t *slab_nix = UVBGgrids.slab_nix;
    ptrdiff_t *slab_ix_start = UVBGgrids.slab_ix_start;

    // create buffers on each rank which is as large as the largest LOGICAL allocation on any single rank
    int buffer_size = 0;
    for (int ii = 0; ii < nranks; ii++)
        if (slab_nix[ii] > buffer_size)
            buffer_size = (int)slab_nix[ii];

    buffer_size *= uvbg_dim * uvbg_dim;
    float *buffer_mass = fftwf_alloc_real((size_t)buffer_size);
    float *buffer_stars_slab = fftwf_alloc_real((size_t)buffer_size);
    float *buffer_sfr = fftwf_alloc_real((size_t)buffer_size);

    //RegionInd no longer global, allocate array for slab decomposition (Jdavies)
    //TODO (jdavies): still find a better way to do this (with mymalloc?)
    int *UVRegionInd = mymalloc("UVRegionInd",sizeof(int) * PartManager->NumPart);

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
            UVRegionInd[ii] = searchsorted(&ix, slab_ix_start, nranks, sizeof(ptrdiff_t), compare_ptrdiff, -1, -1);
        } else {
            UVRegionInd[ii] = -1;
        }
    }


    for (int i_r = 0; i_r < nranks; i_r++) {

        const int ix_start = slab_ix_start[i_r];
        const int nix = slab_nix[i_r];

        // init the buffers
        for (int ii = 0; ii < buffer_size; ii++) {
            buffer_mass[ii] = (float)0.;
            buffer_stars_slab[ii] = (float)0.;
            buffer_sfr[ii] = (float)0.;
        }


        // fill the local buffer for this slab
        // TODO(smutch): This should become CIC
        unsigned int count_mass = 0;
        for(int ii = 0; ii < PartManager->NumPart; ii++) {
            if(UVRegionInd[ii] == i_r) {
                int ix = pos_to_ngp(P[ii].Pos[0], box_size, uvbg_dim) - ix_start;
                int iy = pos_to_ngp(P[ii].Pos[1], box_size, uvbg_dim);
                int iz = pos_to_ngp(P[ii].Pos[2], box_size, uvbg_dim);

                int ind = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);

                buffer_mass[ind] += P[ii].Mass;
                count_mass++;
            }
        }

        message(0, "Added %d particles to mass grid.\n", count_mass);

        // reduce on to the correct rank
        if (this_rank == i_r) {
            MPI_Reduce(MPI_IN_PLACE, buffer_mass, buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        }
        else
            MPI_Reduce(buffer_mass, buffer_mass, buffer_size, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);

        MPI_Reduce(UVBGgrids.stars + grid_index(ix_start, 0, 0, uvbg_dim, INDEX_REAL), buffer_stars_slab, nix*uvbg_dim*uvbg_dim, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        MPI_Reduce(UVBGgrids.prev_stars + grid_index(ix_start, 0, 0, uvbg_dim, INDEX_REAL), buffer_sfr, nix*uvbg_dim*uvbg_dim, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        
        // TODO(smutch): These could perhaps be precalculated?
        const float inv_dt = (float)(1.0 / (time_to_present(UVBGgrids.last_a) - time_to_present(All.Time)));
        message(0, "UVBG calculation dt = %.2e Myr\n", (1.0 / inv_dt) * All.UnitTime_in_s / SEC_PER_MEGAYEAR);

        for(int ii=0; ii < buffer_size; ii++) {
            buffer_sfr[ii] = (buffer_stars_slab[ii] - buffer_sfr[ii]) * inv_dt;
        }

        if (this_rank == i_r) {
            const double tot_n_cells = uvbg_dim * uvbg_dim * uvbg_dim;
            const double deltax_conv_factor = tot_n_cells / (All.CP.RhoCrit * All.CP.Omega0 * All.BoxSize * All.BoxSize * All.BoxSize);
            for (int ix = 0; ix < slab_nix[i_r]; ix++)
                for (int iy = 0; iy < uvbg_dim; iy++)
                    for (int iz = 0; iz < uvbg_dim; iz++) {
                        // TODO(smutch): The buffer will need to be a double for precision...
                        const int ind_real = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);
                        const int ind_pad = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);
                        const float mass = buffer_mass[ind_real];
                        UVBGgrids.deltax[ind_pad] = mass * (float)deltax_conv_factor - 1.0f;
                        UVBGgrids.sfr[ind_pad] = buffer_sfr[ind_real];
                        UVBGgrids.stars_slab[ind_pad] = buffer_stars_slab[ind_real];
                    }
        }
    }

    fftwf_free(buffer_stars_slab);
    fftwf_free(buffer_mass);
    myfree(UVRegionInd);

    // set the last_a value so we can calulate dt for the SFR at the next call
    UVBGgrids.last_a = All.Time;
    // copy the current stellar mass grid so we can work out the SFRs
    // WATCH OUT: prev_stars is a union with J21 to save memory.  Until
    // find_HII_bubbles is called again, J21 will be invalid!
    const size_t grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    memcpy(UVBGgrids.prev_stars, UVBGgrids.stars, sizeof(float) * grid_n_real);
}


static void filter(fftwf_complex* box, const int local_ix_start, const int slab_nx, const int grid_dim, const float R)
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
    double RhoCrit = All.CP.RhoCrit;

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

static void create_plans()
{
    int uvbg_dim = All.UVBGdim;
    UVBGgrids.plan_dft_r2c = fftwf_mpi_plan_dft_r2c_3d(uvbg_dim, uvbg_dim, uvbg_dim,
            UVBGgrids.deltax, (fftwf_complex*)UVBGgrids.deltax,
            MPI_COMM_WORLD, FFTW_PATIENT);
    UVBGgrids.plan_dft_c2r = fftwf_mpi_plan_dft_c2r_3d(uvbg_dim, uvbg_dim, uvbg_dim,
            (fftwf_complex*)UVBGgrids.deltax, UVBGgrids.deltax,
            MPI_COMM_WORLD, FFTW_PATIENT);
}

static void destroy_plans()
{
    fftwf_destroy_plan(UVBGgrids.plan_dft_c2r);
    fftwf_destroy_plan(UVBGgrids.plan_dft_r2c);
}

static void find_HII_bubbles()
{
    /* This function is based on find_hII_bubbles from 21cmFAST, but has been
     * largely rewritten. */

    // TODO(smutch): TAKE A VERY VERY CLOSE LOOK AT UNITS!!!!

    message(0, "Calling find_HII_bubbles.\n");

    int this_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    int uvbg_dim = All.UVBGdim;
    double box_size = All.BoxSize; // Mpc/h comoving
    double pixel_volume = pow(box_size / (double)uvbg_dim, 3); // (Mpc/h)^3 comoving
    double cell_length_factor = 0.620350491;
    double total_n_cells = pow((double)uvbg_dim, 3);
    int local_nix = (int)(UVBGgrids.slab_nix[this_rank]);
    int slab_n_real = local_nix * uvbg_dim * uvbg_dim;
    int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    double density_over_mean = 0;
    double sfr_density = 0;
    double f_coll_stars = 0;
    int i_real = 0;
    int i_padded = 0;
    const double redshift = 1.0 / (All.Time) - 1.;

    // This parameter choice is sensitive to noise on the cell size, at least for the typical
    // cell sizes in RT simulations. It probably doesn't matter for larger cell sizes.
    if ((box_size / (double)uvbg_dim) < 1.0) // Fairly arbitrary length based on 2 runs Sobacchi did
        cell_length_factor = 1.0;

    // Init J21 and xHI
    float* xHI = UVBGgrids.xHI;
    for (int ii = 0; ii < slab_n_real; ii++) {
        xHI[ii] = 1.0f;
    }
    float* J21 = UVBGgrids.J21;
    for (int ii = 0; ii < grid_n_real; ii++) {
        J21[ii] = 0.0f;
    }

    // Forward fourier transform to obtain k-space fields
    float* deltax = UVBGgrids.deltax;
    fftwf_complex* deltax_unfiltered = (fftwf_complex*)deltax; // WATCH OUT!
    fftwf_complex* deltax_filtered = UVBGgrids.deltax_filtered;
    fftwf_execute_dft_r2c(UVBGgrids.plan_dft_r2c, deltax, deltax_unfiltered);

    float* stars_slab = UVBGgrids.stars_slab;
    fftwf_complex* stars_slab_unfiltered = (fftwf_complex*)stars_slab; // WATCH OUT!
    fftwf_complex* stars_slab_filtered = UVBGgrids.stars_slab_filtered;
    fftwf_execute_dft_r2c(UVBGgrids.plan_dft_r2c, stars_slab, stars_slab_unfiltered);

    float* sfr = UVBGgrids.sfr;
    fftwf_complex* sfr_unfiltered = (fftwf_complex*)sfr; // WATCH OUT!
    fftwf_complex* sfr_filtered = UVBGgrids.sfr_filtered;
    fftwf_execute_dft_r2c(UVBGgrids.plan_dft_r2c, sfr, sfr_unfiltered);

    // Remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
    // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
    int slab_n_complex = (int)(UVBGgrids.slab_n_complex[this_rank]);
    for (int ii = 0; ii < slab_n_complex; ii++) {
        deltax_unfiltered[ii] /= total_n_cells;
        stars_slab_unfiltered[ii] /= total_n_cells;
        sfr_unfiltered[ii] /= total_n_cells;
    }

    // Loop through filter radii
    //(jdavies): get the parameters
    double ReionRBubbleMax = All.ReionRBubbleMax;
    double ReionRBubbleMin = All.ReionRBubbleMin;
    double ReionDeltaRFactor = All.ReionDeltaRFactor;
    double ReionGammaHaloBias = All.ReionGammaHaloBias;
    const double ReionNionPhotPerBary = All.ReionNionPhotPerBary;
    double alpha_uv = All.AlphaUV;
    double EscapeFraction = All.EscapeFraction;

    double R = fmin(ReionRBubbleMax, cell_length_factor * box_size); // Mpc/h

    // TODO(smutch): tidy this up!
    // The following is based on Sobacchi & Messinger (2013) eqn 7
    // with f_* removed and f_b added since we define f_coll as M_*/M_tot rather than M_vir/M_tot,
    // and also with the inclusion of the effects of the Helium fraction.
    const double Y_He = 1.0 - HYDROGEN_MASSFRAC;
    const double BaryonFrac = All.CP.OmegaBaryon / All.CP.Omega0;
    double ReionEfficiency = 1.0 / BaryonFrac * ReionNionPhotPerBary * EscapeFraction / (1.0 - 0.75 * Y_He);

    bool flag_last_filter_step = false;
    
    //(jdavies) J21 total debug variables
    double total_J21 = 0.;
    double max_coll = 0.;

    while (!flag_last_filter_step) {
        // check to see if this is our last filtering step
        if (((R / ReionDeltaRFactor) <= (cell_length_factor * box_size / (double)uvbg_dim))
            || ((R / ReionDeltaRFactor) <= ReionRBubbleMin)) {
            flag_last_filter_step = true;
            R = cell_length_factor * box_size / (double)uvbg_dim;
        }

        // copy the k-space grids
        memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
        memcpy(stars_slab_filtered, stars_slab_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
        memcpy(sfr_filtered, sfr_unfiltered, sizeof(fftwf_complex) * slab_n_complex);

        // do the filtering unless this is the last filter step
        int local_ix_start = (int)(UVBGgrids.slab_ix_start[this_rank]);
        if (!flag_last_filter_step) {
            filter(deltax_filtered, local_ix_start, local_nix, uvbg_dim, (float)R);
            filter(stars_slab_filtered, local_ix_start, local_nix, uvbg_dim, (float)R);
            filter(sfr_filtered, local_ix_start, local_nix, uvbg_dim, (float)R);
        }

        // inverse fourier transform back to real space
        fftwf_execute_dft_c2r(UVBGgrids.plan_dft_c2r, deltax_filtered, (float*)deltax_filtered);
        fftwf_execute_dft_c2r(UVBGgrids.plan_dft_c2r, stars_slab_filtered, (float*)stars_slab_filtered);
        fftwf_execute_dft_c2r(UVBGgrids.plan_dft_c2r, sfr_filtered, (float*)sfr_filtered);

        // Perform sanity checks to account for aliasing effects
        for (int ix = 0; ix < local_nix; ix++)
            for (int iy = 0; iy < uvbg_dim; iy++)
                for (int iz = 0; iz < uvbg_dim; iz++) {
                    i_padded = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);
                    ((float*)deltax_filtered)[i_padded] = fmaxf(((float*)deltax_filtered)[i_padded], -1 + FLOAT_REL_TOL);
                    ((float*)stars_slab_filtered)[i_padded] = fmaxf(((float*)stars_slab_filtered)[i_padded], 0.0f);
                    ((float*)sfr_filtered)[i_padded] = fmaxf(((float*)sfr_filtered)[i_padded], 0.0f);
                }


        // ============================================================================================================
        // {
        //     // DEBUG HERE
        //     const int grid_size = (int)(local_nix * uvbg_dim * uvbg_dim);
        //     float* grid = (float*)calloc(grid_size, sizeof(float));
        //     int count_gtz = 0;
        //     for (int ii = 0; ii < local_nix; ii++)
        //         for (int jj = 0; jj < uvbg_dim; jj++)
        //             for (int kk = 0; kk < uvbg_dim; kk++) {
        //                 grid[grid_index(ii, jj, kk, uvbg_dim, INDEX_REAL)] = ((float*)deltax_filtered)[grid_index(ii, jj, kk, uvbg_dim, INDEX_PADDED)];
        //                 if (grid[grid_index(ii, jj, kk, uvbg_dim, INDEX_REAL)] > 0)
        //                     count_gtz++;
        //             }

        //     message(0, "count_gtz for filter R=%.2f = %d\n", R, count_gtz);

        //     BigFile fout;
        //     char fname[256];
        //     sprintf(fname, "output/filterstep-%.2f.bf", R);
        //     big_file_mpi_create(&fout, fname, MPI_COMM_WORLD);
        //     BigBlock block;
        //     int n_ranks;
        //     MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
        //     big_file_mpi_create_block(&fout, &block, "deltax", "=f4", 1, n_ranks, uvbg_dim*uvbg_dim*uvbg_dim, MPI_COMM_WORLD);
        //     BigBlockPtr ptr = {0};
        //     int start_elem = this_rank > 1 ? UVBGgrids.slab_nix[this_rank - 1]*uvbg_dim*uvbg_dim : 0;
        //     big_block_seek(&block, &ptr, start_elem);
        //     BigArray arr = {0};
        //     big_array_init(&arr, grid, "=f4", 1, (size_t[]){grid_size}, NULL);
        //     big_block_mpi_write(&block, &ptr, &arr, 1, MPI_COMM_WORLD);
        //     big_block_mpi_close(&block, MPI_COMM_WORLD);
        //     big_file_mpi_close(&fout, MPI_COMM_WORLD);

        //     free(grid);
        // }
        // ============================================================================================================

        const double J21_aux_constant = (1.0 + redshift) * (1.0 + redshift) / (4.0 * M_PI)
            * alpha_uv * PLANCK * 1e21
            * R * All.UnitLength_in_cm * ReionNionPhotPerBary / PROTONMASS
            * All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3) / All.UnitTime_in_s;

        // Main loop through the box...
        for (int ix = 0; ix < local_nix; ix++)
            for (int iy = 0; iy < uvbg_dim; iy++)
                for (int iz = 0; iz < uvbg_dim; iz++) {
                    i_real = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);
                    i_padded = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);

                    density_over_mean = 1.0 + (double)((float*)deltax_filtered)[i_padded];

                    f_coll_stars = (double)((float*)stars_slab_filtered)[i_padded] / (RtoM(R) * density_over_mean)
                        * (4.0 / 3.0) * M_PI * pow(R, 3.0) / pixel_volume;

                    sfr_density = (double)((float*)sfr_filtered)[i_padded] / pixel_volume; // In internal units

                    const float J21_aux = (float)(sfr_density * J21_aux_constant);

                    max_coll = fmax(f_coll_stars,max_coll);

                    // Check if ionised!
                    if (f_coll_stars > (1.0 / ReionEfficiency)) // IONISED!!!!
                    {
                        // If it is the first crossing of the ionisation barrier for this cell (largest R), let's record J21
                        if (xHI[i_real] > FLOAT_REL_TOL) {
                            const int i_grid_real = grid_index(ix + local_ix_start, iy, iz, uvbg_dim, INDEX_REAL);
                            J21[i_grid_real] = J21_aux;
                            total_J21 += J21_aux;
                        }

                        // Mark as ionised
                        xHI[i_real] = 0;

                        // TODO(smutch): Do we want to implement this?
                        // r_bubble[i_real] = (float)R;
                    }
                    else if (flag_last_filter_step && (xHI[i_real] > FLOAT_REL_TOL)) {
                        // Check if this is the last filtering step.
                        // If so, assign partial ionisations to those cells which aren't fully ionised
                        xHI[i_real] = (float)(1.0 - f_coll_stars * ReionEfficiency);
                    }

                    // Check if new ionisation
                    float* z_in = UVBGgrids.z_at_ionization;
                    if ((xHI[i_real] < FLOAT_REL_TOL) && (z_in[i_real] < 0)) // New ionisation!
                    {
                        z_in[i_real] = (float)redshift;
                        UVBGgrids.J21_at_ionization[i_real] = J21_aux * ReionGammaHaloBias;
                    }
                } // iz

        R /= ReionDeltaRFactor;
    }

    // Reduce the J21 grid onto all ranks
    MPI_Allreduce(MPI_IN_PLACE, UVBGgrids.J21, grid_n_real, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // DEBUG ==========================================================================================================
    // {
    //     int n_ranks;
    //     MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    //     for(int i_rank = 0; i_rank < n_ranks; i_rank++)
    //     {
    //         if(this_rank == i_rank)
    //         {
    //             BigFile fout;
    //             char fname[256];
    //             sprintf(fname, "output/J21_check-rank%03d.bf", i_rank);
    //             big_file_create(&fout, fname);
    //             BigBlock block;
    //             big_file_create_block(&fout, &block, "J21", "=f4", 1, 1, (size_t[]){grid_n_real});
    //             BigArray arr = {0};
    //             big_array_init(&arr, UVBGgrids.J21, "=f4", 1, (size_t[]){grid_n_real}, NULL);
    //             BigBlockPtr ptr = {0};
    //             big_block_write(&block, &ptr, &arr);
    //             big_block_close(&block);
    //             big_file_close(&fout);
    //         }
    //     }
    // }
    // ================================================================================================================

    // Find the volume and mass weighted neutral fractions
    // TODO: The deltax grid will have rounding errors from forward and reverse
    //       FFT. Should cache deltax slabs prior to ffts and reuse here.
    double volume_weighted_global_xHI = 0.0;
    double mass_weighted_global_xHI = 0.0;
    double mass_weight = 0.0;
    double total_stars = 0.0;
    int ionised_cells = 0;

    for (int ix = 0; ix < local_nix; ix++)
        for (int iy = 0; iy < uvbg_dim; iy++)
            for (int iz = 0; iz < uvbg_dim; iz++) {
                i_real = grid_index(ix, iy, iz, uvbg_dim, INDEX_REAL);
                i_padded = grid_index(ix, iy, iz, uvbg_dim, INDEX_PADDED);
                volume_weighted_global_xHI += (double)xHI[i_real];
                density_over_mean = 1.0 + (double)((float*)deltax_filtered)[i_padded];
                mass_weighted_global_xHI += (double)(xHI[i_real]) * density_over_mean;
                mass_weight += density_over_mean;
                total_stars += (double)((float*)stars_slab_filtered)[i_padded];
                ionised_cells += xHI[i_real] < 0.1;
            }

    MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_xHI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_xHI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mass_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_stars, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_J21, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_coll, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ionised_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    volume_weighted_global_xHI /= total_n_cells;
    mass_weighted_global_xHI /= mass_weight;
    UVBGgrids.volume_weighted_global_xHI = (float)volume_weighted_global_xHI;
    UVBGgrids.mass_weighted_global_xHI = (float)mass_weighted_global_xHI;

    message(0,"vol weighted xhi : %f\n",volume_weighted_global_xHI);
    message(0,"mass weighted xhi : %f\n",mass_weighted_global_xHI);
    message(0,"sum of stars : %f\n",total_stars);
    message(0,"sum of J21 : %f\n",total_J21);
    message(0,"Reionefficiency : %f\n",ReionEfficiency);
    message(0,"max collapsed frac : %f\n",max_coll);
    message(0,"ionised cells : %d\n",ionised_cells);
}

void save_uvbg_grids(int SnapshotFileCount)
{
    int n_ranks;
    int this_rank=-1;
    int uvbg_dim = All.UVBGdim;
    int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    //malloc new grid for star grid reduction on one rank
    //TODO:use bigfile_mpi to write star/XHI grids and/or slabs
    float* star_buffer;
    if(this_rank == 0)
    {
        star_buffer = mymalloc("star_buffer", sizeof(float) * grid_n_real);
        for(int ii=0;ii<grid_n_real;ii++)
        {
            star_buffer[ii] = 0.0;
            if(SnapshotFileCount==6 && UVBGgrids.stars[ii] > 1e-5)
            {
                message(0,"star grid = %f at %d\n",UVBGgrids.stars[ii],ii);
            }
            if(SnapshotFileCount==6 && UVBGgrids.J21[ii] > 1e-5)
            {
                message(0,"J21 grid = %f at %d\n",UVBGgrids.J21[ii],ii);
            }
        }
    }
    MPI_Reduce(UVBGgrids.stars, star_buffer, grid_n_real, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    //TODO(jdavies): a better write function, probably using petaio stuff
    //These grids should have been reduced onto all ranks
    if(this_rank == 0)
    {

        BigFile fout;
        char fname[256];
        sprintf(fname, "%s/UVgrids_%03d", All.OutputDir,SnapshotFileCount);
        message(0, "saving uv grids to %s \n", fname);
        big_file_create(&fout, fname);

        //J21 block
        BigBlock block;
        big_file_create_block(&fout, &block, "J21", "=f4", 1, 1, (size_t[]){grid_n_real});
        BigArray arr = {0};
        big_array_init(&arr, UVBGgrids.J21, "=f4", 1, (size_t[]){grid_n_real}, NULL);
        BigBlockPtr ptr = {0};
        big_block_write(&block, &ptr, &arr);
        big_block_close(&block);

        message(0,"saved J21\n");

        //xHI grid is in slabs
        /*//xHI block
        BigBlock block2;
        big_file_create_block(&fout, &block2, "xHI", "=f4", 1, 1, (size_t[]){grid_n_real});
        BigArray arr2 = {0};
        big_array_init(&arr2, UVBGgrids.xHI, "=f4", 1, (size_t[]){grid_n_real}, NULL);
        BigBlockPtr ptr2 = {0};
        big_block_write(&block2, &ptr2, &arr2);
        big_block_close(&block2);

        message(0,"saved XHI\n");*/

        //stars block
        BigBlock block3;
        big_file_create_block(&fout, &block3, "stars", "=f4", 1, 1, (size_t[]){grid_n_real});
        BigArray arr3 = {0};
        big_array_init(&arr3, star_buffer, "=f4", 1, (size_t[]){grid_n_real}, NULL);
        BigBlockPtr ptr3 = {0};
        big_block_write(&block3, &ptr3, &arr3);
        big_block_close(&block3);

        big_file_close(&fout);

        myfree(star_buffer);
        message(0,"saved stars\n");
   }
}

void calculate_uvbg()
{
    walltime_measure("/Misc");
    message(0, "Calculating UVBG grids.\n");

    assign_slabs();
    malloc_grids();
    
    create_plans();
    walltime_measure("/UVBG/create_plans");

    populate_grids();
    walltime_measure("/UVBG/populate_grids");

    // DEBUG =========================================================================================
    // int this_rank;
    // int uvbg_dim = All.UVBGdim;
    // MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    // int local_nix = UVBGgrids.slab_nix[this_rank];
    // int grid_size = (size_t)(local_nix * uvbg_dim * uvbg_dim);
    // float* grid = (float*)calloc(grid_size, sizeof(float));
    // for (int ii = 0; ii < local_nix; ii++)
    //     for (int jj = 0; jj < uvbg_dim; jj++)
    //         for (int kk = 0; kk < uvbg_dim; kk++)
    //             grid[grid_index(ii, jj, kk, uvbg_dim, INDEX_REAL)] = (UVBGgrids.deltax)[grid_index(ii, jj, kk, uvbg_dim, INDEX_PADDED)];

    // FILE *fout;
    // char fname[128];
    // sprintf(fname, "output/dump_r%03d.dat", this_rank);
    // if((fout = fopen(fname, "wb")) == NULL) {
    //   endrun(1, "poop...");
    // }
    // fwrite(grid, sizeof(float), grid_size, fout);
    // fclose(fout);
    // free(grid);
    // walltime_measure("/Misc");
    // ===============================================================================================


    message(0, "Away to call find_HII_bubbles...\n");
    find_HII_bubbles();

    walltime_measure("/UVBG/find_HII_bubbles");

    destroy_plans();
    free_grids();

    walltime_measure("/UVBG");
}
