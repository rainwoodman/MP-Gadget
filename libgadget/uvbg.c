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
#include <stdbool.h>
#include <gsl/gsl_integration.h>
#include <assert.h>
//#include <fftw3.h>

#include "uvbg.h"
#include "cosmology.h"
#include "utils.h"
#include "allvars.h"
#include "partmanager.h"
#include "petapm.h"
#include "physconst.h"
#include "walltime.h"

// TODO(smutch): See if something equivalent is defined anywhere else
#define FLOAT_REL_TOL (float)1e-5

static struct UVBGParams {
    /*filter scale parameters*/
    double ReionRBubbleMax;
    double ReionRBubbleMin;
    double ReionDeltaRFactor;
    int ReionFilterType;

    /*J21 calculation parameters*/
    double ReionGammaHaloBias;
    double ReionNionPhotPerBary;
    //double AlphaUV;
    double EscapeFraction;

} uvbg_params;

struct UVBGgrids_type UVBGgrids;

/*set uvbg parameters*/
void set_uvbg_params(ParameterSet * ps) {

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask==0)
    {
        uvbg_params.ReionFilterType = param_get_int(ps, "ReionFilterType");
        uvbg_params.ReionRBubbleMax = param_get_double(ps, "ReionRBubbleMax");
        uvbg_params.ReionRBubbleMin = param_get_double(ps, "ReionRBubbleMin");
        uvbg_params.ReionDeltaRFactor = param_get_double(ps, "ReionDeltaRFactor");
        uvbg_params.ReionGammaHaloBias = param_get_double(ps, "ReionGammaHaloBias");
        uvbg_params.ReionNionPhotPerBary = param_get_double(ps, "ReionNionPhotPerBary");
        //uvbg_params.AlphaUV = param_get_double(ps, "AlphaUV");
        uvbg_params.EscapeFraction = param_get_double(ps, "EscapeFraction");
    }
    MPI_Bcast(&uvbg_params, sizeof(struct UVBGParams), MPI_BYTE, 0, MPI_COMM_WORLD);

}

double integrand_time_to_present(double a, void *dummy)
{
    return 1 / a / hubble_function(&All.CP,a);
}

//time_to_present in Myr
double time_to_present(double a)
{
#define WORKSIZE 1000
    gsl_function F;
    gsl_integration_workspace* workspace;
    double time;
    double result;
    double abserr;

    double hubble;
    hubble = All.CP.Hubble / All.UnitTime_in_Megayears * All.CP.HubbleParam;
    //jdavies(second to Myr conversion)

    workspace = gsl_integration_workspace_alloc(WORKSIZE);
    F.function = &integrand_time_to_present;

    gsl_integration_qag(&F, a, 1.0, 1.0 / hubble,
        1.0e-8, WORKSIZE, GSL_INTEG_GAUSS21, workspace, &result, &abserr);

    //convert to Myr and multiply by h
    time = result / (hubble/All.CP.Hubble);

    gsl_integration_workspace_free(workspace);

    // return time to present as a function of redshift
    return time;
}

static void assign_slabs()
{
    message(0, "Assigning slabs to MPI cores...\n");

    int uvbg_dim = All.UVBGdim;
    // Allocations made in this function are free'd in `free_reionization_grids`.

    //TODO: have the flags stored somewhere so it's not both here and in create_plans
    unsigned pfft_flags = PFFT_PADDED_R2C|PFFT_TRANSPOSED_NONE;

    // Assign the slab size
    int n_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    ptrdiff_t n[3] = {uvbg_dim,uvbg_dim,uvbg_dim};
    int np[2];

    /* try to find a square 2d decomposition */
    int i;
    for(i = sqrt(n_ranks) + 1; i >= 0; i --) {
        if(n_ranks % i == 0) break;
    }
    np[0] = i;
    np[1] = n_ranks / i;

    message(0, "Using 2D Task mesh for UVBG %td x %td \n", np[0], np[1]);
    if( pfft_create_procmesh_2d(MPI_COMM_WORLD, np[0], np[1], &UVBGgrids.comm_cart_2d) ){
        endrun(0, "Error: This test file only works with %td processes.\n", np[0]*np[1]);
    }

    MPI_Comm comm_cart_2d = UVBGgrids.comm_cart_2d;

    PetaPMRegion* local_r_region = &(UVBGgrids.local_r_region);
    PetaPMRegion* local_c_region = &(UVBGgrids.local_c_region);

    // find out what slab each rank should get
    ptrdiff_t local_n_complex = pfft_local_size_dft_r2c_3d(n, comm_cart_2d, pfft_flags, local_r_region->size, local_r_region->offset, local_c_region->size, local_c_region->offset);

    petapm_region_init_strides(local_r_region);
    petapm_region_init_strides(local_c_region);
    
    message(1,"local_r_region strides are (%d,%d,%d)\n",local_r_region->strides[0],local_r_region->strides[1],local_r_region->strides[2]);
    message(1,"local_c_region strides are (%d,%d,%d)\n",local_c_region->strides[0],local_c_region->strides[1],local_c_region->strides[2]);
    message(1,"slab size (%d) (%d,%d,%d) starting at (%d,%d,%d)\n Outputs size (%d,%d,%d) starting at (%d,%d,%d)\n"
           ,local_n_complex,local_r_region->size[0],local_r_region->size[1],local_r_region->size[2]
           ,local_r_region->offset[0],local_r_region->offset[1],local_r_region->offset[2]
           ,local_c_region->size[0],local_c_region->size[1],local_c_region->size[2]
           ,local_c_region->offset[0],local_c_region->offset[1],local_c_region->offset[2]);


    // let every rank know...
    ptrdiff_t* slab_ni = mymalloc("slab_ni",sizeof(ptrdiff_t) * n_ranks * 3); ///< array of number of x cells of every rank
    UVBGgrids.slab_ni = slab_ni;
    MPI_Allgather(local_r_region->size, sizeof(ptrdiff_t)*3, MPI_BYTE, slab_ni, sizeof(ptrdiff_t)*3, MPI_BYTE, MPI_COMM_WORLD);

    ptrdiff_t *slab_i_start = mymalloc("slab_i_start",sizeof(ptrdiff_t) * n_ranks * 3); ///< array first x cell of every rank
    UVBGgrids.slab_i_start = slab_i_start;
    MPI_Allgather(local_r_region->offset, sizeof(ptrdiff_t)*3, MPI_BYTE, slab_i_start, sizeof(ptrdiff_t)*3, MPI_BYTE, MPI_COMM_WORLD);

    ptrdiff_t* slab_no = mymalloc("slab_no",sizeof(ptrdiff_t) * n_ranks * 3); ///< array of number of x cells of every rank
    UVBGgrids.slab_no = slab_no;
    MPI_Allgather(local_c_region->size, sizeof(ptrdiff_t)*3, MPI_BYTE, slab_no, sizeof(ptrdiff_t)*3, MPI_BYTE, MPI_COMM_WORLD);

    ptrdiff_t *slab_o_start = mymalloc("slab_o_start",sizeof(ptrdiff_t) * n_ranks * 3); ///< array first x cell of every rank
    UVBGgrids.slab_o_start = slab_o_start;
    MPI_Allgather(local_c_region->offset, sizeof(ptrdiff_t)*3, MPI_BYTE, slab_o_start, sizeof(ptrdiff_t)*3, MPI_BYTE, MPI_COMM_WORLD);

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
    UVBGgrids.prev_stars = mymalloc("prev_stars", sizeof(float) * grid_n_real);

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
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    ptrdiff_t slab_n_complex = UVBGgrids.slab_n_complex[this_rank];
    ptrdiff_t slab_n_real = UVBGgrids.slab_ni[3*this_rank] * UVBGgrids.slab_ni[3*this_rank + 1] * All.UVBGdim;
  
    //NOTES: slab_n_real for real grids calculated after inverse
    //slab_n_complex for grids calculated in k-space that are inverse transformed (a bit bigger than slab_n_real/2)
    //2 * slab_n_complex for real grids that are padded for in-place fft (a bit bigger than slab_n_real)
    
    UVBGgrids.deltax = mymalloc("deltax", 2*slab_n_complex * sizeof(double));
    UVBGgrids.deltax_filtered = (pfft_complex *) mymalloc("deltax_filtered", slab_n_complex * sizeof(pfft_complex));
    UVBGgrids.stars_slab = mymalloc("stars_slab", 2*slab_n_complex * sizeof(double));
    UVBGgrids.stars_slab_filtered = (pfft_complex *) mymalloc("stars_slab_filtered", slab_n_complex * sizeof(pfft_complex));
    UVBGgrids.sfr = mymalloc("sfr", 2*slab_n_complex * sizeof(double));
    UVBGgrids.sfr_filtered = (pfft_complex *) mymalloc("sfr_filtered", slab_n_complex * sizeof(pfft_complex));

    //TODO: these grids were useful outputs in meraxes but are not used here yet
    UVBGgrids.xHI = mymalloc("xHI", slab_n_real * sizeof(float));
    UVBGgrids.z_at_ionization = mymalloc("z_at_ion", slab_n_real * sizeof(float));
    UVBGgrids.J21_at_ionization = mymalloc("J21_at_ion", slab_n_real * sizeof(float));

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
    //grids were alloc'd after slab decomp, so free these first
    myfree(UVBGgrids.J21_at_ionization);
    myfree(UVBGgrids.z_at_ionization);
    myfree(UVBGgrids.xHI);
    myfree(UVBGgrids.sfr_filtered);
    myfree(UVBGgrids.sfr);
    myfree(UVBGgrids.stars_slab_filtered);
    myfree(UVBGgrids.stars_slab);
    myfree(UVBGgrids.deltax_filtered);
    myfree(UVBGgrids.deltax);
    
    myfree(UVBGgrids.slab_n_complex);
    myfree(UVBGgrids.slab_o_start);
    myfree(UVBGgrids.slab_no);
    myfree(UVBGgrids.slab_i_start);
    myfree(UVBGgrids.slab_ni);
}


int pos_to_ngp(double x, double Offset, double side, int nx)
{
    //subtract offset
    double corrpos = x - Offset;

    //periodic box, corrpos now in [0,side] 
    while (corrpos < 0)
        corrpos += side;
    while (corrpos > side)
        corrpos -= side;
    
    //find nearest gridpoint, NOTE: can round to [0,nx]
    int ind = (int)nearbyint((corrpos) / side * (double)nx);

    //deal with ind == nx case by wrapping
    if (ind > nx - 1)
        ind = 0;

    //make sure we are in bounds
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

//NOTE: this is less efficient than searchsorted, but works with the 3d decomp
//TODO: optimize
//ix_start is no longer guaranteed to be sorted, but there should only be one correct slab per coordinate
int find_UV_region(int* coords, ptrdiff_t* slab_i_start, ptrdiff_t* slab_ni, int count)
{
    int x_ll,y_ll,x_ul,y_ul;
    //int z_ll, z_ul;
    //count is number of slabs, not size of array
    for(int i=0;i<count;i++)
    {
        x_ll = slab_i_start[3*i];
        y_ll = slab_i_start[3*i + 1];
        //z_ll = slab_i_start[3*i + 2];
        x_ul = x_ll + slab_ni[3*i];
        y_ul = y_ll + slab_ni[3*i + 1];
        //z_ul = z_ll + slab_ni[3*i + 2];
        
        if(coords[0] >= x_ll && coords[0] < x_ul
                && coords[1] >= y_ll && coords[1] < y_ul)
                //&& coords[2] >= z_ll && coords[2] < z_ul)
        {
            return i;
        }
    }
    //this shouldn't happen
    endrun(0,"particle (%d,%d,%d) outside all UV regions???\n",coords[0],coords[1],coords[2]);
}

int grid_index(int i, int j, int k, ptrdiff_t strides[3])
{
    return k*strides[2] + j*strides[1] + i*strides[0];
}

static void populate_grids()
{
    int uvbg_dim = All.UVBGdim;
    int nranks = -1, this_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    //TODO:replace these with region structs gathered here
    ptrdiff_t *slab_ni = UVBGgrids.slab_ni;
    ptrdiff_t *slab_i_start = UVBGgrids.slab_i_start;
    //full grid strides
    ptrdiff_t grid_strides[3] = {uvbg_dim*uvbg_dim,uvbg_dim,1};

    // create buffers on each rank which is as large as the largest LOGICAL allocation on any single rank
    // TODO: I need the unpadded dimensions here for the reduce, so I assume 2D decomp with z kept intact
    // if I want to generalise I'll need to store unpadded dimensions somewhere
    int buffer_size = 0;
    for (int ii = 0; ii < nranks; ii++)
    {
        int temp = (int)(slab_ni[3*ii]*slab_ni[3*ii+1]*uvbg_dim);
        if (temp > buffer_size)
            buffer_size = temp;
    }

    float *buffer_mass = mymalloc("buffer_mass",(size_t)buffer_size * sizeof(float));
    float *buffer_stars_slab = mymalloc("buffer_stars_slab",(size_t)buffer_size * sizeof(float));
    float *buffer_sfr = mymalloc("buffer_sfr",(size_t)buffer_size * sizeof(float));

    //RegionInd no longer global, allocate array for slab decomposition (Jdavies)
    int *UVRegionInd = mymalloc("UVRegionInd",sizeof(int) * PartManager->NumPart);

    // This is a potentially stupid way to do things anyway and will most
    // definitely need to be changed! There is no way we should have to search
    // all of the particles to find out what slab they sit on, and then loop
    // through all particles again n_slab times!
    double box_size = All.BoxSize;

    // fill UVRegionInd with the index of the slab each particle is on, before filling the mass slabs
    #pragma omp parallel for
    for(int ii = 0; ii < PartManager->NumPart; ii++) {
        if((!P[ii].IsGarbage) && (!P[ii].Swallowed) && (P[ii].Type < 5)) {
            int ix[3];
            ix[0] = pos_to_ngp(P[ii].Pos[0],PartManager->CurrentParticleOffset[0], box_size, uvbg_dim);
            ix[1] = pos_to_ngp(P[ii].Pos[1],PartManager->CurrentParticleOffset[1], box_size, uvbg_dim);
            ix[2] = pos_to_ngp(P[ii].Pos[2],PartManager->CurrentParticleOffset[2], box_size, uvbg_dim);
            //UVRegionInd[ii] = searchsorted(&ix, slab_i_start, nranks, sizeof(ptrdiff_t), compare_ptrdiff, -1, -1);
            UVRegionInd[ii] = find_UV_region(ix, slab_i_start, slab_ni, nranks);
        } else {
            UVRegionInd[ii] = -1;
        }
    }


    for (int i_r = 0; i_r < nranks; i_r++) {
        const int ix_start[3] = {slab_i_start[3*i_r],slab_i_start[3*i_r + 1],slab_i_start[3*i_r + 2]};
        const int nix[3] = {slab_ni[3*i_r],slab_ni[3*i_r + 1],slab_ni[3*i_r + 2]};
        //unpadded strides
        ptrdiff_t slab_strides[3] = {uvbg_dim*nix[1],uvbg_dim,1};

        //init the buffers
        for (int ii = 0; ii < buffer_size; ii++) {
            buffer_mass[ii] = 0.f;
            buffer_stars_slab[ii] = 0.f;
            buffer_sfr[ii] = 0.f;
        }


        // fill the local buffer for this slab
        // TODO(smutch): This should become CIC
        unsigned int count_mass = 0;
        #pragma omp parallel for reduction(+:count_mass)
        for(int ii = 0; ii < PartManager->NumPart; ii++) {
            if(UVRegionInd[ii] == i_r) {
                int ix = pos_to_ngp(P[ii].Pos[0],PartManager->CurrentParticleOffset[0], box_size, uvbg_dim) - ix_start[0];
                int iy = pos_to_ngp(P[ii].Pos[1],PartManager->CurrentParticleOffset[1], box_size, uvbg_dim) - ix_start[1];
                int iz = pos_to_ngp(P[ii].Pos[2],PartManager->CurrentParticleOffset[2], box_size, uvbg_dim) - ix_start[2];
 
                if(ix<0 || ix>nix[0] || iy<0 || iy>nix[1] || iz<0 || iz>uvbg_dim)
                    endrun(0,"particle (%d,%d,%d) outside its UV region %d\n",ix,iy,iz,i_r);

                int ind = grid_index(ix, iy, iz, slab_strides);

                #pragma omp atomic update
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

        //TODO(jdavies): this is going to be a bad way to communicate data, find a better way to do strided reductions
        //TODO(jdavies): build a buffer with dimensions of the padded grids, then reduce once straight onto the UVBGgrids struct
        //TODO(jdavies): this essentially means reverse the order of this reduction and the assignment loop below
        //TODO(jdavies): replaced nix[1]*nix[2] with nix[1]*uvbg_dim because i need the unbuffered dimension, this will break in any z decomposition
        for(int ix=0;ix<nix[0];ix++){
            MPI_Reduce(UVBGgrids.stars + grid_index(ix+ix_start[0], ix_start[1], 0, grid_strides), buffer_stars_slab + grid_index(ix, 0, 0, slab_strides)
                    , nix[1]*uvbg_dim, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
            MPI_Reduce(UVBGgrids.prev_stars + grid_index(ix+ix_start[0], ix_start[1], 0, grid_strides), buffer_sfr + grid_index(ix, 0, 0, slab_strides)
                    , nix[1]*uvbg_dim, MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
            }

        //MPI_Reduce(UVBGgrids.stars + grid_index(ix_start, 0, 0, uvbg_dim, INDEX_REAL), buffer_stars_slab, nix[0]*nix[1]*nix[2], MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        //MPI_Reduce(UVBGgrids.prev_stars + grid_index(ix_start, 0, 0, uvbg_dim, INDEX_REAL), buffer_sfr, nix[0]*nix[1]*nix[2], MPI_FLOAT, MPI_SUM, i_r, MPI_COMM_WORLD);
        
        // TODO(smutch): These could perhaps be precalculated?
        const double inv_dt = (1.0 / (time_to_present(UVBGgrids.last_a) - time_to_present(All.Time)));
        message(0, "UVBG calculation dt = %.2e Myr\n", (1.0 / inv_dt));

        // currently buffer_sfr is equal to prev_stars (see above MPI_Reduce), so we subtract the star buffer
        // and divide by the time between now the last calculation to get the sfr
        #pragma omp parallel for
        for(int ii=0; ii < buffer_size; ii++) {
            buffer_sfr[ii] = (buffer_stars_slab[ii] - buffer_sfr[ii]) * (float)inv_dt / All.UnitTime_in_Megayears;
        }

        if (this_rank == i_r) {
            const double tot_n_cells = uvbg_dim * uvbg_dim * uvbg_dim;
            const double deltax_conv_factor = tot_n_cells / (All.CP.RhoCrit * All.CP.Omega0 * All.BoxSize * All.BoxSize * All.BoxSize);
            #pragma omp parallel for collapse(3)
            for (int ix = 0; ix < nix[0]; ix++)
                for (int iy = 0; iy < nix[1]; iy++)
                    for (int iz = 0; iz < uvbg_dim; iz++) {
                        // TODO(smutch): The buffer will need to be a double for precision...
                        const int ind_real = grid_index(ix, iy, iz, slab_strides);
                        const int ind_pad = grid_index(ix, iy, iz, UVBGgrids.local_r_region.strides);
                        const double mass = (double)buffer_mass[ind_real];
                        UVBGgrids.deltax[ind_pad] = (mass * deltax_conv_factor - 1.0);
                        UVBGgrids.sfr[ind_pad] = (double)buffer_sfr[ind_real];
                        UVBGgrids.stars_slab[ind_pad] = (double)buffer_stars_slab[ind_real];
                    }
        }
    }

    myfree(UVRegionInd);
    myfree(buffer_sfr);
    myfree(buffer_stars_slab);
    myfree(buffer_mass);

    // set the last_a value so we can calulate dt for the SFR at the next call
    UVBGgrids.last_a = All.Time;
    // copy the current stellar mass grid so we can work out the SFRs
    const size_t grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    memcpy(UVBGgrids.prev_stars, UVBGgrids.stars, sizeof(float) * grid_n_real);
}


static void filter(pfft_complex* box, ptrdiff_t* local_o_start, ptrdiff_t* slab_no, ptrdiff_t* strides, const int grid_dim, const double R)
{
    const int filter_type = uvbg_params.ReionFilterType;
    int middle = grid_dim / 2;
    double box_size = All.BoxSize;
    double delta_k = (2.0 * M_PI / box_size);

    // Loop through k-box
    // (jdavies): outer loop ONLY threaded here, not perfectly nested
    #pragma omp parallel for
    for (int n_x = 0; n_x < slab_no[0]; n_x++) {
        double k_x;
        int n_x_global = n_x + local_o_start[0];

        if (n_x_global > middle)
            k_x = (n_x_global - grid_dim) * delta_k;
        else
            k_x = n_x_global * delta_k;

        for (int n_y = 0; n_y < slab_no[1]; n_y++) {
            double k_y;
            int n_y_global = n_y + local_o_start[1];

            if (n_y_global > middle)
                k_y = (n_y_global - grid_dim) * delta_k;
            else
                k_y = n_y_global * delta_k;

            //TODO: make sure this is correct with padding etc
            for (int n_z = 0; n_z < slab_no[2]; n_z++) {
                double k_z = n_z * delta_k;

                double k_mag = sqrtf(k_x * k_x + k_y * k_y + k_z * k_z);

                double kR = k_mag * R; // Real space top-hat

                switch (filter_type) {
                case 0: // Real space top-hat
                    if (kR > 1e-4)
                        box[grid_index(n_x, n_y, n_z, strides)] *= (pfft_complex)(3.0 * (sinf(kR) / powf(kR, 3) - cosf(kR) / powf(kR, 2)));
                    break;

                case 1: // k-space top hat
                    kR *= 0.413566994; // Equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
                    if (kR > 1)
                        box[grid_index(n_x, n_y, n_z, strides)] = (pfft_complex)0.0;
                    break;

                case 2: // Gaussian
                    kR *= 0.643; // Equates integrated volume to the real space top-hat
                    box[grid_index(n_x, n_y, n_z, strides)] *= (pfft_complex)(pow(M_E,
                        (-kR * kR / 2.0)));
                    break;

                default:
                    if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {
                        endrun(1, "ReionFilterType type %d is undefined!\n", filter_type);
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
        endrun(1, "Unrecognised RtoM filter (%d).\n", filter);
        break;
    }

    return -1;
}

static void create_plans()
{
    int uvbg_dim = All.UVBGdim;
    ptrdiff_t n[3] = {uvbg_dim,uvbg_dim,uvbg_dim};
    //TODO: have the flags stored somewhere so it's not both here and in assign_slabs

    UVBGgrids.plan_dft_r2c = pfft_plan_dft_r2c_3d(n, UVBGgrids.deltax,
            (pfft_complex*)UVBGgrids.deltax, UVBGgrids.comm_cart_2d,
            PFFT_FORWARD, PFFT_PADDED_R2C|PFFT_PATIENT|PFFT_TRANSPOSED_NONE);
    UVBGgrids.plan_dft_c2r = pfft_plan_dft_c2r_3d(n, (pfft_complex*)UVBGgrids.deltax,
            UVBGgrids.deltax, UVBGgrids.comm_cart_2d,
            PFFT_BACKWARD, PFFT_PADDED_C2R|PFFT_PATIENT|PFFT_TRANSPOSED_NONE);
}

static void destroy_plans()
{
    pfft_destroy_plan(UVBGgrids.plan_dft_c2r);
    pfft_destroy_plan(UVBGgrids.plan_dft_r2c);
    MPI_Comm_free(&UVBGgrids.comm_cart_2d);
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
    
    //get the shapes of the local blocks
    PetaPMRegion r_region = UVBGgrids.local_r_region;
    PetaPMRegion c_region = UVBGgrids.local_c_region;
    
    int slab_n_real = r_region.size[0] * r_region.size[1] * uvbg_dim;
    int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;

    //full grid strides
    ptrdiff_t grid_strides[3] = {uvbg_dim*uvbg_dim,uvbg_dim,1};
    //unpadded strides
    ptrdiff_t slab_strides[3] = {uvbg_dim*r_region.size[1],uvbg_dim,1};

    /*for(int k=0;k<3;k++){
        message(1,"dim %d, complex local size = %d, offset = %d, region size = %d, offset = %d, strides = %d\n"
                ,k,local_no[k],local_o_start[k],c_region.size[k],c_region.offset[k],c_region.strides[k]);

    }*/
    
    //MAKE SURE THESE ARE PRIVATE IN THREADED LOOPS
    double density_over_mean = 0;
    double sfr_density = 0;
    double f_coll_stars = 0;
    int i_real = 0;
    int i_padded = 0;
    
    const double redshift = 1.0 / (All.Time) - 1.;

    double hubble_time;
    //hubble time in internal units
    hubble_time = 1 / (hubble_function(&All.CP,All.Time) * All.CP.HubbleParam);
    message(0,"hubble time is %.3e internal, %.3e s, %.3e Myr\n",hubble_time,hubble_time * All.UnitTime_in_s, hubble_time * All.UnitTime_in_Megayears);

    // This parameter choice is sensitive to noise on the cell size, at least for the typical
    // cell sizes in RT simulations. It probably doesn't matter for larger cell sizes.
    // TODO(jdavies): look into the units here and apply a UnitLength_in_blahblah factor, I think this is meant to switch on 1Mpc resolution
    if ((box_size / (double)uvbg_dim) < 1.0) // Fairly arbitrary length based on 2 runs Sobacchi did
        cell_length_factor = 1.0;

    // Init J21 and xHI
    // TODO: i could add threading here but the overhead might be more than array initialization
    float* xHI = UVBGgrids.xHI;
    for (int ii = 0; ii < slab_n_real; ii++) {
        xHI[ii] = 1.0f;
    }
    float* J21 = UVBGgrids.J21;
    for (int ii = 0; ii < grid_n_real; ii++) {
        J21[ii] = 0.0f;
    }

    // Forward fourier transform to obtain k-space fields
    double* deltax = UVBGgrids.deltax;
    pfft_complex* deltax_unfiltered = (pfft_complex*)deltax; // WATCH OUT!
    pfft_complex* deltax_filtered = UVBGgrids.deltax_filtered;
    pfft_execute_dft_r2c(UVBGgrids.plan_dft_r2c, deltax, deltax_unfiltered);

    double* stars_slab = UVBGgrids.stars_slab;
    pfft_complex* stars_slab_unfiltered = (pfft_complex*)stars_slab; // WATCH OUT!
    pfft_complex* stars_slab_filtered = UVBGgrids.stars_slab_filtered;
    pfft_execute_dft_r2c(UVBGgrids.plan_dft_r2c, stars_slab, stars_slab_unfiltered);

    double* sfr = UVBGgrids.sfr;
    pfft_complex* sfr_unfiltered = (pfft_complex*)sfr; // WATCH OUT!
    pfft_complex* sfr_filtered = UVBGgrids.sfr_filtered;
    pfft_execute_dft_r2c(UVBGgrids.plan_dft_r2c, sfr, sfr_unfiltered);

    // Remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
    // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
    int slab_n_complex = (int)(UVBGgrids.slab_n_complex[this_rank]);
    //TODO(jdavies): use simd here?
    #pragma omp parallel for
    for (int ii = 0; ii < slab_n_complex; ii++) {
        deltax_unfiltered[ii] /= total_n_cells;
        stars_slab_unfiltered[ii] /= total_n_cells;
        sfr_unfiltered[ii] /= total_n_cells;
    }

    // Loop through filter radii
    //(jdavies): get the parameters
    double ReionRBubbleMax = uvbg_params.ReionRBubbleMax;
    double ReionRBubbleMin = uvbg_params.ReionRBubbleMin;
    double ReionDeltaRFactor = uvbg_params.ReionDeltaRFactor;
    double ReionGammaHaloBias = uvbg_params.ReionGammaHaloBias;
    const double ReionNionPhotPerBary = uvbg_params.ReionNionPhotPerBary;
    double alpha_uv = All.AlphaUV;
    double EscapeFraction = uvbg_params.EscapeFraction;

    double R = fmin(ReionRBubbleMax, cell_length_factor * box_size); // Mpc/h

    // TODO(smutch): tidy this up!
    // The following is based on Sobacchi & Messinger (2013) eqn 7
    // with f_* removed and f_b added since we define f_coll as M_*/M_tot rather than M_vir/M_tot,
    // and also with the inclusion of the effects of the Helium fraction.
    const double Y_He = 1.0 - HYDROGEN_MASSFRAC;
    const double BaryonFrac = All.CP.OmegaBaryon / All.CP.Omega0;
    double ReionEfficiency = 1.0 / BaryonFrac * ReionNionPhotPerBary * EscapeFraction / (1.0 - 0.75 * Y_He);

    bool flag_last_filter_step = false;
    
    while (!flag_last_filter_step) {
        // check to see if this is our last filtering step
        if (((R / ReionDeltaRFactor) <= (cell_length_factor * box_size / (double)uvbg_dim))
            || ((R / ReionDeltaRFactor) <= ReionRBubbleMin)) {
            flag_last_filter_step = true;
            R = cell_length_factor * box_size / (double)uvbg_dim;
        }
        //message(0, "filter step R = %.3e, Rmax = %.3e, Rmin = %.3e, delta = %.3e\n",R,ReionRBubbleMax,ReionRBubbleMin,ReionDeltaRFactor);

        // copy the k-space grids
        memcpy(deltax_filtered, deltax_unfiltered, sizeof(pfft_complex) * slab_n_complex);
        memcpy(stars_slab_filtered, stars_slab_unfiltered, sizeof(pfft_complex) * slab_n_complex);
        memcpy(sfr_filtered, sfr_unfiltered, sizeof(pfft_complex) * slab_n_complex);

        
        // do the filtering unless this is the last filter step
        if (!flag_last_filter_step) {
            filter(deltax_filtered, c_region.offset, c_region.size, c_region.strides, uvbg_dim, R);
            filter(stars_slab_filtered, c_region.offset, c_region.size, c_region.strides, uvbg_dim, R);
            filter(sfr_filtered, c_region.offset, c_region.size, c_region.strides, uvbg_dim, R);
        }

        // inverse fourier transform back to real space
        pfft_execute_dft_c2r(UVBGgrids.plan_dft_c2r, deltax_filtered, (double*)deltax_filtered);
        pfft_execute_dft_c2r(UVBGgrids.plan_dft_c2r, stars_slab_filtered, (double*)stars_slab_filtered);
        pfft_execute_dft_c2r(UVBGgrids.plan_dft_c2r, sfr_filtered, (double*)sfr_filtered);

        // Perform sanity checks to account for aliasing effects
        // NOTE: these went from COMPLEX_HERM to PADDED dimensions (same size) after c2r
        // z-loop only goes to uvbg_dim to not loop over the padding
        #pragma omp parallel for private(i_padded)
        for (int ix = 0; ix < r_region.size[0]; ix++)
            for (int iy = 0; iy < r_region.size[1]; iy++)
                for (int iz = 0; iz < uvbg_dim; iz++) {
                    i_padded = grid_index(ix, iy, iz, r_region.strides);
                    ((double*)deltax_filtered)[i_padded] = fmax(((double*)deltax_filtered)[i_padded], -1 + FLOAT_REL_TOL);
                    ((double*)stars_slab_filtered)[i_padded] = fmax(((double*)stars_slab_filtered)[i_padded], 0.0f);
                    ((double*)sfr_filtered)[i_padded] = fmax(((double*)sfr_filtered)[i_padded], 0.0f);
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
        //     int start_elem = this_rank > 1 ? UVBGgrids.slab_ni[this_rank - 1]*uvbg_dim*uvbg_dim : 0;
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

        // Main loop through the box... again not over the padding
        #pragma omp parallel for collapse(3) private(i_real,i_padded,density_over_mean,f_coll_stars,sfr_density)
        for (int ix = 0; ix < r_region.size[0]; ix++)
            for (int iy = 0; iy < r_region.size[1]; iy++)
                for (int iz = 0; iz < uvbg_dim; iz++) {
                    i_real = grid_index(ix, iy, iz, slab_strides);
                    i_padded = grid_index(ix, iy, iz, r_region.strides);

                    density_over_mean = 1.0 + ((double*)deltax_filtered)[i_padded];

                    f_coll_stars = ((double*)stars_slab_filtered)[i_padded] / (RtoM(R) * density_over_mean)
                        * (4.0 / 3.0) * M_PI * pow(R, 3.0) / pixel_volume;

                    //TODO(jdavies): NOT THE ACTUAL SFR DENSITY, the rates functions don't work well with the bursty sfr
                    //this is total cumulative sfr smoothed over hubble time
                    sfr_density = ((double*)stars_slab_filtered)[i_padded] / hubble_time / pixel_volume; // In internal units
                    //sfr_density = (double)((float*)sfr_filtered)[i_padded] / pixel_volume; // In internal units

                    const float J21_aux = (float)(sfr_density * J21_aux_constant);

                    // Check if ionised!
                    if (f_coll_stars > (1.0 / ReionEfficiency)) // IONISED!!!!
                    {
                        // If it is the first crossing of the ionisation barrier for this cell (largest R), let's record J21
                        if (xHI[i_real] > FLOAT_REL_TOL) {
                            const int i_grid_real = grid_index(ix + r_region.offset[0], iy + r_region.offset[1], iz + r_region.offset[2], grid_strides);
                            J21[i_grid_real] = J21_aux;
                        }

                        // Mark as ionised
                        xHI[i_real] = 0.;

                        // TODO(smutch): Do we want to implement this?
                        // r_bubble[i_real] = (float)R;
                    }
                    else if (flag_last_filter_step && (xHI[i_real] > FLOAT_REL_TOL)) {
                        // Check if this is the last filtering step.
                        // If so, assign partial ionisations to those cells which aren't fully ionised
                        xHI[i_real] = (float)(1.0 - f_coll_stars * ReionEfficiency);
                    }

                    // Check if new ionisation
                    // TODO: if this is needed we should make these grids permanent
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

    //TODO: this directive is ridiculous and I doubt the parallelisation does much here
    #pragma omp parallel for collapse(3) reduction(+:volume_weighted_global_xHI) reduction(+:density_over_mean) reduction(+:mass_weighted_global_xHI) reduction(+:mass_weight) private(i_real,i_padded)
    for (int ix = 0; ix < r_region.size[0]; ix++)
        for (int iy = 0; iy < r_region.size[1]; iy++)
            for (int iz = 0; iz < uvbg_dim; iz++) {
                i_real = grid_index(ix, iy, iz, slab_strides);
                i_padded = grid_index(ix, iy, iz, r_region.strides);
                volume_weighted_global_xHI += (double)xHI[i_real];
                density_over_mean = 1.0 + ((double*)deltax_filtered)[i_padded];
                mass_weighted_global_xHI += (double)(xHI[i_real]) * density_over_mean;
                mass_weight += density_over_mean;
            }

    MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_xHI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_xHI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mass_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /*MPI_Allreduce(MPI_IN_PLACE, &min_J21, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_J21, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_coll, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);*/

    volume_weighted_global_xHI /= total_n_cells;
    mass_weighted_global_xHI /= mass_weight;
    UVBGgrids.volume_weighted_global_xHI = volume_weighted_global_xHI;
    UVBGgrids.mass_weighted_global_xHI = mass_weighted_global_xHI;

    message(0,"vol weighted xhi : %f\n",volume_weighted_global_xHI);
    message(0,"mass weighted xhi : %f\n",mass_weighted_global_xHI);
}

void save_uvbg_grids(int SnapshotFileCount)
{
    int n_ranks;
    int this_rank=-1;
    int uvbg_dim = All.UVBGdim;
    int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    int starcount=0;
    double startotal=0.;
    int jcount=0;
    double jtotal=0.;
    //malloc new grid for star grid reduction on one rank
    //TODO:use bigfile_mpi to write star/XHI grids and/or slabs
    float* star_buffer;
    if(this_rank == 0)
    {
        star_buffer = mymalloc("star_buffer", sizeof(float) * grid_n_real);
        for(int ii=0;ii<grid_n_real;ii++)
        {
            star_buffer[ii] = 0.0;
        }
    }
    MPI_Reduce(UVBGgrids.stars, star_buffer, grid_n_real, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    //print some debug stats
    if(this_rank == 0)
    {
        for(int ii=0;ii<grid_n_real;ii++)
        {
            startotal += star_buffer[ii];
            jtotal += UVBGgrids.J21[ii];
            if(star_buffer[ii] > 0)
                starcount++;
            if(UVBGgrids.J21[ii] > 0)
                jcount++;
        }
    }
    message(0,"star grid has %d nonzero cells, %e total\n",starcount,startotal);
    message(0,"J21 grid has %d nonzero cells, %e total\n",jcount,jtotal);

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


//read stellar grid for checkpoint start
//at the moment I just divide by number of tasks since only the sum is used.
//TODO: a better decomposition
void read_star_grids(int snapnum)
{
    int n_ranks;
    int this_rank=-1;
    int uvbg_dim = All.UVBGdim;
    int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    //TODO(jdavies): a better write function, probably using petaio stuff
    if(this_rank == 0)
    {

        BigFile fin;
        char fname[256];
        sprintf(fname, "%s/UVgrids_%03d", All.OutputDir,snapnum);
        message(0, "reading star grid from %s \n", fname);
        big_file_open(&fin, fname);

        //stars block
        BigBlock block;
        big_file_open_block(&fin, &block, "stars");
        BigArray arr = {0};
        big_array_init(&arr, UVBGgrids.stars, "=f4", 1, (size_t[]){grid_n_real}, NULL);
        BigBlockPtr ptr = {0};
        big_block_read(&block, &ptr, &arr);
        big_block_close(&block);

        big_file_close(&fin);

        //TODO:(jdavies) now here things get strange, since the grids on all ranks are
        //summed together before the excursion set, I need a way to distribute them
        //from a checkpointed grid. Since particle info from previous snapshots is not
        //read in, I simply divide it evenly here. This has no effect currently but might
        //be confusing later on if we want to do something with local star grids
        for(int ii=0; ii<grid_n_real;ii++)
        {
            UVBGgrids.stars[ii] = UVBGgrids.stars[ii] / n_ranks;
        }

   }
   //send a copy of the divided grid to each rank
   MPI_Bcast(UVBGgrids.stars,grid_n_real,MPI_FLOAT,0,MPI_COMM_WORLD);
}

void calculate_uvbg()
{
    walltime_measure("/Misc");
    message(0, "Calculating UVBG grids.\n");

    assign_slabs();
    message(0, "Slabs Assigned...\n");
    malloc_grids();
    message(0, "Grids Allocated...\n");
    
    create_plans();
    message(0, "Plans Created...\n");
    walltime_measure("/UVBG/create_plans");

    populate_grids();
    message(0, "Grids Populated...\n");
    walltime_measure("/UVBG/populate_grids");

    // DEBUG =========================================================================================
    // int this_rank;
    // int uvbg_dim = All.UVBGdim;
    // MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);
    // int local_nix = UVBGgrids.slab_ni[this_rank];
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
    
    //TODO(jdavies):remove this
    UVBGgrids.debug_printed = 0;

    walltime_measure("/UVBG/find_HII_bubbles");
    
    //debug :save checkpoint to snap 999
    //save_uvbg_grids(999);

    destroy_plans();
    free_grids();
    //endrun(0,"first UVBG complete\n");

    walltime_measure("/UVBG");
}
