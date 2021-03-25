/*=============================================================================
 * An implementation of a patchy UV ionising background
 * calculation. This code utilises the decomposition and communication
 * in the long-range force code in petapm.c, some new functions have been
 * written in petapm.c to accomodate the order of operations and multiple grids
 * present in the reionisation model
============================================================================*/

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
#include <assert.h>

#include "uvbg.h"
#include "cosmology.h"
#include "utils.h"
#include "allvars.h"
#include "partmanager.h"
#include "slotsmanager.h"
#include "petapm.h"
#include "physconst.h"
#include "walltime.h"
#include "petaio.h"
#include "fof.h"

// TODO(smutch): See if something equivalent is defined anywhere else
#define FLOAT_REL_TOL (float)1e-5

static struct UVBGParams {
    /*filter scale parameters*/
    double ReionRBubbleMax;
    double ReionRBubbleMin;
    double ReionDeltaRFactor;
    int ReionFilterType;
    int RtoMFilterType;

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
        uvbg_params.RtoMFilterType = param_get_int(ps, "RtoMFilterType");
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

int grid_index(int i, int j, int k, ptrdiff_t strides[3])
{
    return k*strides[2] + j*strides[1] + i*strides[0];
}

static double RtoM(double R)
{
    // All in internal units
    const int filter = uvbg_params.RtoMFilterType;
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

void save_uvbg_grids(int SnapshotFileCount, PetaPM * pm)
{
    int n_ranks;
    int this_rank=-1;
    int grid_n = pm->real_space_region.size[0] * pm->real_space_region.size[1] * pm->real_space_region.size[2];
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    int xhi_neutral=0;
    int xhi_ionised=0;
    int j_gtz=0;

    //print some debug stats
    for(int ii=0;ii<grid_n;ii++)
    {
        if(UVBGgrids.J21[ii] > FLOAT_REL_TOL)
            j_gtz++;
        if(UVBGgrids.xHI[ii] < FLOAT_REL_TOL)
            xhi_neutral++;
        if(UVBGgrids.xHI[ii] > (1 - FLOAT_REL_TOL))
            xhi_ionised++;
    }
    MPI_Allreduce(MPI_IN_PLACE, &xhi_neutral, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &xhi_ionised, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &j_gtz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    message(0,"J21 grid has %d nonzero cells out of %d total cells \n",j_gtz,grid_n);
    message(0,"XHI grid has %d fully neutral cells and %d fully ionised cells \n",xhi_neutral,xhi_ionised);

    //TODO(jdavies): finish this grid writing function
    BigFile fout;
    char fname[256];
    sprintf(fname, "%s/UVgrids_%03d", All.OutputDir,SnapshotFileCount);
    message(0, "saving uv grids to %s \n", fname);

    if(0 != big_file_mpi_create(&fout, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    big_file_mpi_create(&fout, fname, MPI_COMM_WORLD);

    BigBlock bh;
    if(0 != big_file_mpi_create_block(&fout, &bh, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                big_file_get_error_message());
    }

    if(
    (0 != big_block_set_attr(&bh, "volume_weighted_global_xHI", &(UVBGgrids.volume_weighted_global_xHI), "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "mass_weighted_global_xHI", &(UVBGgrids.mass_weighted_global_xHI), "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "scale_factor", &All.Time, "f8", 1)) ) {
        endrun(0, "Failed to write attributes %s\n",
                    big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }

    //TODO: think about the cartesian communicator in the PetaPM struct
    //and the mapping between ranks, indices and positions

    //J21 block
    BigArray arr = {0};
    big_array_init(&arr, UVBGgrids.J21, "=f4", 1, (size_t[]){grid_n}, NULL);
    petaio_save_block(&fout,"J21",&arr,0);

    message(0,"saved J21\n");

    //xHI block
    BigArray arr2 = {0};
    big_array_init(&arr2, UVBGgrids.xHI, "=f4", 1, (size_t[]){grid_n}, NULL);
    petaio_save_block(&fout,"XHI",&arr2,0);

    message(0,"saved XHI\n");

    if(0 != big_file_mpi_close(&fout, MPI_COMM_WORLD)){
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
}

//Simple region initialization (taken from zeldovich.c)
//TODO: look into _prepare (gravpm.c) and see if its worth implementing anything there
static PetaPMRegion * makeregion(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions) {
    PetaPMRegion * regions = mymalloc2("Regions", sizeof(PetaPMRegion));
    int NumPart = PartManager->NumPart;
    int k;
    int r = 0;
    int i;
    double min[3] = {pm->BoxSize, pm->BoxSize, pm->BoxSize};
    double max[3] = {0, 0, 0.};

    for(i = 0; i < NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            if(min[k] > P[i].Pos[k])
            min[k] = P[i].Pos[k];
            if(max[k] < P[i].Pos[k])
            max[k] = P[i].Pos[k];
        }
    }

    for(k = 0; k < 3; k ++) {
        regions[r].offset[k] = floor(min[k] / pm->BoxSize * pm->Nmesh - 1);
        regions[r].size[k] = ceil(max[k] / pm->BoxSize * pm->Nmesh + 2);
        regions[r].size[k] -= regions[r].offset[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(&regions[r]);
    *Nregions = 1;
    return regions;
}

//this is applied as global_transfer, dividing by n_cells due to the forward-reverse FFT
static void divide_by_ncell(PetaPM * pm, int64_t k2, int k[3], pfft_complex * value){
        int total_n_cells = (double)(All.UVBGdim * All.UVBGdim * All.UVBGdim);
        *value /= total_n_cells;
}

//transfer functions that applies a certain filter (top-hat or gaussian)
static void filter_pm(PetaPM * pm, int64_t k2, int k[3], pfft_complex * value)
{
    const int filter_type = uvbg_params.ReionFilterType;
    double k_mag = sqrt(k2) * (2 * M_PI / pm->Nmesh) * (pm->Nmesh / pm->BoxSize);

    double kR = k_mag * pm->G; // Radius is stored in the G variable

    switch (filter_type) {
    case 0: // Real space top-hat
        if (kR > 1e-4)
            *value *= (pfft_complex)(3.0 * (sinf(kR) / powf(kR, 3) - cosf(kR) / powf(kR, 2)));
        break;

    case 1: // k-space top hat
        kR *= 0.413566994; // Equates integrated volume to the real space top-hat (9pi/2)^(-1/3)
        if (kR > 1)
            *value = (pfft_complex)0.0;
        break;

    case 2: // Gaussian
        kR *= 0.643; // Equates integrated volume to the real space top-hat
        *value *= (pfft_complex)(pow(M_E,(-kR * kR / 2.0)));
        break;

    default:
        endrun(1, "ReionFilterType type %d is undefined!\n", filter_type);
        break;
    }
}

//print some statistics of the reion grids for debugging
static void print_reion_debug_info(PetaPM * pm_mass, float * J21, float * xHI, double * mass_real, double * star_real, double * sfr_real)
{
    double min_J21 = 1e30;
    double max_J21 = 0;
    double min_mass = 1e30;
    double max_mass = 0;
    double min_star = 1e30;
    double max_star = 0;
    double min_sfr = 1e30;
    double max_sfr = 0;
    int neutral_count = 0;
    int ion_count = 0;
    int pm_idx = 0;
    int uvbg_dim = All.UVBGdim;
    int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
    #pragma omp parallel for collapse(3) reduction(+:neutral_count,ion_count) reduction(min:min_J21,min_mass,min_star,min_sfr) reduction(max:max_J21,max_mass,max_star,max_sfr) private(pm_idx)
    for (int ix = 0; ix < pm_mass->real_space_region.size[0]; ix++)
        for (int iy = 0; iy < pm_mass->real_space_region.size[1]; iy++)
            for (int iz = 0; iz < pm_mass->real_space_region.size[2]; iz++) {
                pm_idx = grid_index(ix, iy, iz, pm_mass->real_space_region.strides);

                if(xHI[pm_idx] > 1 - FLOAT_REL_TOL)
                    neutral_count += 1;
                if(xHI[pm_idx] < FLOAT_REL_TOL)
                    ion_count += 1;
                min_J21 = min_J21 < J21[pm_idx] ? min_J21 : J21[pm_idx];
                max_J21 = max_J21 > J21[pm_idx] ? max_J21 : J21[pm_idx];
                min_mass = min_mass < mass_real[pm_idx] ? min_mass : mass_real[pm_idx];
                max_mass = max_mass > mass_real[pm_idx] ? max_mass : mass_real[pm_idx];
                min_star = min_star < star_real[pm_idx] ? min_star : star_real[pm_idx];
                max_star = max_star > star_real[pm_idx] ? max_star : star_real[pm_idx];
                min_sfr = min_sfr < sfr_real[pm_idx] ? min_sfr : sfr_real[pm_idx];
                max_sfr = max_sfr > sfr_real[pm_idx] ? max_sfr : sfr_real[pm_idx];

            }
    MPI_Allreduce(MPI_IN_PLACE, &neutral_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ion_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_J21, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_J21, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_mass, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_mass, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_star, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_star, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_sfr, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_sfr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double n_ratio = (double)neutral_count / (double)grid_n_real;
    double i_ratio = (double)ion_count / (double)grid_n_real;
    
    message(0,"neutral cells : %d, ion cells %d, ratio(%d) N %f ion %f\n",neutral_count, ion_count, grid_n_real, n_ratio, i_ratio);
    message(0,"min J21 : %e | max J21 %e\n",min_J21,max_J21);
    message(0,"min mass : %e | max mass %e\n",min_mass,max_mass);
    message(0,"min star : %e | max star %e\n",min_star,max_star);
    message(0,"min sfr : %e | max sfr %e\n",min_sfr,max_sfr);
}


//takes filtered mass, star, sfr grids and calculates J21 and neutral fractions onto a grid
//which is placed in the mass grid out on the last call of this function.
static void reion_loop_pm(PetaPM * pm_mass, PetaPM * pm_star, PetaPM * pm_sfr,
        double * mass_real, double * star_real, double * sfr_real, int last_step)
{
    //MAKE SURE THESE ARE PRIVATE IN THREADED LOOPS
    double density_over_mean = 0;
    double sfr_density = 0;
    double f_coll_stars = 0;
    int pm_idx = 0;

    double R = pm_mass->G;
    
    const double redshift = 1.0 / (All.Time) - 1.;
    
    // Loop through filter radii
    //(jdavies): get the parameters
    //double ReionGammaHaloBias = uvbg_params.ReionGammaHaloBias;
    const double ReionNionPhotPerBary = uvbg_params.ReionNionPhotPerBary;
    double alpha_uv = All.AlphaUV;
    double EscapeFraction = uvbg_params.EscapeFraction;

    // TODO(smutch): tidy this up!
    // The following is based on Sobacchi & Messinger (2013) eqn 7
    // with f_* removed and f_b added since we define f_coll as M_*/M_tot rather than M_vir/M_tot,
    // and also with the inclusion of the effects of the Helium fraction.
    const double Y_He = 1.0 - HYDROGEN_MASSFRAC;
    const double BaryonFrac = All.CP.OmegaBaryon / All.CP.Omega0;
    double ReionEfficiency = 1.0 / BaryonFrac * ReionNionPhotPerBary * EscapeFraction / (1.0 - 0.75 * Y_He);
    
    const double tot_n_cells = pm_mass->Nmesh * pm_mass->Nmesh * pm_mass->Nmesh; 
    const double pixel_volume = pm_mass->CellSize * pm_mass->CellSize * pm_mass->CellSize;
    const double deltax_conv_factor = tot_n_cells / (All.CP.RhoCrit * All.CP.Omega0 * All.BoxSize * All.BoxSize * All.BoxSize);

    float* J21 = UVBGgrids.J21;
    float* xHI = UVBGgrids.xHI;
    
    // Perform sanity checks to account for aliasing effects
    #pragma omp parallel for collapse(3) private(pm_idx)
    for (int ix = 0; ix < pm_mass->real_space_region.size[0]; ix++)
        for (int iy = 0; iy < pm_mass->real_space_region.size[1]; iy++)
            for (int iz = 0; iz < pm_mass->real_space_region.size[2]; iz++) {
                pm_idx = grid_index(ix, iy, iz, pm_mass->real_space_region.strides);
                mass_real[pm_idx] = fmax(mass_real[pm_idx], 0.0);
                star_real[pm_idx] = fmax(star_real[pm_idx], 0.0);
                sfr_real[pm_idx] = fmax(sfr_real[pm_idx], 0.0);
            }


    const double J21_aux_constant = (1.0 + redshift) * (1.0 + redshift) / (4.0 * M_PI)
        * alpha_uv * PLANCK * 1e21
        * R * All.UnitLength_in_cm * ReionNionPhotPerBary / PROTONMASS
        * All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3) / All.UnitTime_in_s;

    const double hubble_time = 1 / (hubble_function(&All.CP,All.Time) * All.CP.HubbleParam);

    // Main loop through the box
    #pragma omp parallel for collapse(3) private(pm_idx,density_over_mean,f_coll_stars,sfr_density)
    for (int ix = 0; ix < pm_mass->real_space_region.size[0]; ix++)
        for (int iy = 0; iy < pm_mass->real_space_region.size[1]; iy++)
            for (int iz = 0; iz < pm_mass->real_space_region.size[2]; iz++) {
                pm_idx = grid_index(ix, iy, iz, pm_mass->real_space_region.strides);

                //convert mass to delta
                density_over_mean = mass_real[pm_idx] * deltax_conv_factor;

                /*TODO: ask Simon about this part of the model where we use mass in a sphere of radius R
                 * at the density of the central cell */
                f_coll_stars = star_real[pm_idx] / (RtoM(R) * density_over_mean)
                    * (4.0 / 3.0) * M_PI * R * R * R / pixel_volume;

                //TODO(jdavies): NOT THE ACTUAL SFR DENSITY, the rates functions don't work well with the bursty sfr
                //this is total cumulative sfr smoothed over hubble time
                sfr_density = star_real[pm_idx] / hubble_time / pixel_volume; // In internal units
                //sfr_density = sfr_real[pm_idx] / pixel_volume / (All.UnitMass_in_g / SOLAR_MASS) * (All.UnitTime_in_s / SEC_PER_YEAR); // In internal units

                const float J21_aux = (float)(sfr_density * J21_aux_constant);

                // Check if ionised!
                if (f_coll_stars > (1.0 / ReionEfficiency)) // IONISED!!!!
                {
                    // If it is the first crossing of the ionisation barrier for this cell (largest R), let's record J21
                    if (xHI[pm_idx] > FLOAT_REL_TOL) {
                        J21[pm_idx] = J21_aux;
                    }

                    // Mark as ionised
                    xHI[pm_idx] = 0.0f;

                    // TODO(smutch): Do we want to implement this?
                    // r_bubble[i_real] = (float)R;
                }
                //TODO: implement CellSizeFactor
                else if (last_step && (xHI[pm_idx] > FLOAT_REL_TOL)) {
                    // Check if this is the last filtering step.
                    // If so, assign partial ionisations to those cells which aren't fully ionised
                     xHI[pm_idx] = (float)(1.0 - f_coll_stars * ReionEfficiency);
                }
                
            } // iz
    // Find the volume and mass weighted neutral fractions
    // TODO: The deltax grid will have rounding errors from forward and reverse
    //       FFT. Should cache deltax slabs prior to ffts and reuse here.
    // TODO: this could be put in the above loop
    if(last_step){

#ifdef DEBUG
        print_reion_debug_info(pm_mass,J21,xHI,mass_real,star_real,sfr_real);
#endif

        double volume_weighted_global_xHI = 0.0;
        double mass_weighted_global_xHI = 0.0;
        double mass_weight = 0.0;
        int uvbg_dim = All.UVBGdim;
        int grid_n_real = uvbg_dim * uvbg_dim * uvbg_dim;
        //TODO: this directive is ridiculous and I doubt the parallelisation does much here
        #pragma omp parallel for collapse(3) reduction(+:volume_weighted_global_xHI,mass_weighted_global_xHI,mass_weight) private(pm_idx,density_over_mean)
        for (int ix = 0; ix < pm_mass->real_space_region.size[0]; ix++)
            for (int iy = 0; iy < pm_mass->real_space_region.size[1]; iy++)
                for (int iz = 0; iz < pm_mass->real_space_region.size[2]; iz++) {
                    pm_idx = grid_index(ix, iy, iz, pm_mass->real_space_region.strides);
                    volume_weighted_global_xHI += (double)(xHI[pm_idx]);
                    
                    density_over_mean = deltax_conv_factor * mass_real[pm_idx];
                    mass_weighted_global_xHI += (double)(xHI[pm_idx]) * density_over_mean;
                    mass_weight += density_over_mean;

                    //if we are on the last step, we re_use the mass grid to store J21 so it can be read out
                    mass_real[pm_idx] = (double)(J21[pm_idx]);
                }
    
        MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_xHI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_xHI, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &mass_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        volume_weighted_global_xHI /= grid_n_real;
        mass_weighted_global_xHI /= mass_weight;
        UVBGgrids.volume_weighted_global_xHI = volume_weighted_global_xHI;
        UVBGgrids.mass_weighted_global_xHI = mass_weighted_global_xHI;
        message(0,"vol weighted xhi : %f\n",volume_weighted_global_xHI);
        message(0,"mass weighted xhi : %f\n",mass_weighted_global_xHI);
    }

}

//readout J21 from grid to particle
static void readout_J21(PetaPM * pm, int i, double * mesh, double weight) {
    if (P[i].Type == 0)
        SPHP(i).local_J21 += weight * mesh[0];
}

//TODO:split up into more functions
void calculate_uvbg(PetaPM * pm_mass, PetaPM * pm_star, PetaPM * pm_sfr, FOFGroups * fof, int WriteSnapshot, int SnapshotFileCount){
    //setup filter radius range
    double Rmax = uvbg_params.ReionRBubbleMax;
    double Rmin = uvbg_params.ReionRBubbleMin;
    double Rdelta = uvbg_params.ReionDeltaRFactor;
    
    //define particle structure with the info petapm needs
    PetaPMParticleStruct pstruct = {
        P,
        sizeof(P[0]),
        (char*) &P[0].Pos[0]  - (char*) P,
        (char*) &P[0].Mass  - (char*) P,
        /* Regions allocated inside _prepare*/
        NULL,
        /* By default all particles are active. For hybrid neutrinos set below.*/
        NULL,
        PartManager->NumPart,
    };
    PetaPMReionPartStruct = {
        (char*) &P[0].Type  - (char*) P,
        (char*) &P[0].PI  - (char*) P,
        (char*) &P[0].GrNr  - (char*) P,
        SphP,
        sizeof(SphP[0]),
        (char*) &SphP[0].Sfr  - (char*) SphP, //TODO: make sure you are using the right object here
        fof->Group,
        sizeof(fof->Group[0]),
        (char*) &fof->Group[0].Mass - (char*) &fof->Group,
    };
    PetaPMGlobalFunctions global_functions = {NULL, NULL, divide_by_ncell};
    
    //TODO: set this up with all the filtering/reion loops
    static PetaPMFunctions functions [] =
    {
        {"Reionisation", filter_pm, readout_J21},
        {NULL, NULL, NULL},
    };

    //Reset local J21
    for(int ii = 0; ii < PartManager->NumPart; ii++) {
        if(P[ii].Type == 0) {
            SPHP(ii).local_J21 = 0.;
        }
    }
    
    /* initialize J21 for grid and particles */
    int grid_n = pm_mass->real_space_region.size[0] 
        * pm_mass->real_space_region.size[1] 
        * * pm_mass->real_space_region.size[2];

    UVBGgrids.J21 = mymalloc("J21", sizeof(float) * grid_n);
    float * J21 = UVBGgrids.J21;
    UVBGgrids.xHI = mymalloc("xHI", sizeof(float) * grid_n);
    float * xHI = UVBGgrids.xHI;
    
    for (int ii = 0; ii < grid_n; ii++) {
        J21[ii] = 0.0f;
        xHI[ii] = 1.0f;
    }

    message(0, "Away to call find_HII_bubbles...\n");
    petapm_reion(pm_mass,pm_star,pm_sfr,makeregion,&global_functions
            ,functions,&pstruct,&rstruct,reion_loop_pm,Rmax,Rmin,Rdelta,NULL);

    //TODO: a particle loop that detects new ionisations, saves J21_at_ion and z_at_ion
    //TODO: multiply J21_at_ion with halo bias??

    walltime_measure("/UVBG/find_HII_bubbles");

    //since J21 is output to particles, we should only need to write these grids for debugging
    //This function is currently WIP
    //TODO: test the new grid-saving before including it in debug
#if 0
    if(WriteSnapshot) {
        save_uvbg_grids(SnapshotFileCount,&pm_mass);
        message(0,"uvbg saved\n");
    }
    walltime_measure("/UVBG/save");
#endif
    myfree(UVBGgrids.xHI);
    myfree(UVBGgrids.J21);
   
    walltime_measure("/UVBG");
}
