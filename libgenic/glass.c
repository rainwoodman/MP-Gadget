#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include <gsl/gsl_rng.h>

#include "allvars.h"
#include "proto.h"

#include <libgadget/petapm.h>
#include <libgadget/walltime.h>
#include <libgadget/utils.h>
#include <libgadget/powerspectrum.h>
#include <libgadget/gravity.h>

static void potential_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_x_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_y_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_z_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value);
static void readout_force_x(PetaPM *pm, int i, double * mesh, double weight);
static void readout_force_y(PetaPM *pm, int i, double * mesh, double weight);
static void readout_force_z(PetaPM *pm, int i, double * mesh, double weight);
static PetaPMFunctions functions [] =
{
    {"ForceX", force_x_transfer, readout_force_x},
    {"ForceY", force_y_transfer, readout_force_y},
    {"ForceZ", force_z_transfer, readout_force_z},
    {NULL, NULL, NULL},
};

static PetaPMGlobalFunctions global_functions = {measure_power_spectrum, NULL, potential_transfer};

static PetaPMRegion * _prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions);

static void glass_force(PetaPM * pm, double t_f, struct ic_part_data * ICP, const int NumPart);
static void glass_stats(struct ic_part_data * ICP, int NumPart);

int
setup_glass(IDGenerator * idgen, PetaPM * pm, double shift, int seed, double mass, struct ic_part_data * ICP, const double UnitLength_in_cm, const char * OutputDir)
{
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_ranlxd1);
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    gsl_rng_set(rng, seed + ThisTask);
    memset(ICP, 0, idgen->NumPart*sizeof(struct ic_part_data));

    int i;
    /* Note: this loop should nto be omp because
     * of the call to gsl_rng_uniform*/
    for(i = 0; i < idgen->NumPart; i ++) {
        int k;
        idgen_create_pos_from_index(idgen, i, &ICP[i].Pos[0]);
        /* a spread of 3 will kill most of the grid anisotropy structure;
         * and still being local */
        for(k = 0; k < 3; k++) {
            double rand = idgen->BoxSize / idgen->Ngrid * 3 * (gsl_rng_uniform(rng) - 0.5);
            ICP[i].Pos[k] += shift + rand;
        }
        ICP[i].Mass = mass;
    }

    gsl_rng_free(rng);

    char * fn = fastpm_strdup_printf("powerspectrum-glass-%08X", seed);
    glass_evolve(pm, 14, fn, ICP, idgen->NumPart, UnitLength_in_cm, OutputDir);
    myfree(fn);

    return idgen->NumPart;
}

void glass_evolve(PetaPM * pm, int nsteps, char * pkoutname, struct ic_part_data * ICP, const int NumPart, const double UnitLength_in_cm, const char * OutputDir)
{
    int i;
    int step = 0;
    double t_x = 0;
    double t_v = 0;
    double t_f = 0;

    /*Allocate memory for a power spectrum*/
    powerspectrum_alloc(pm->ps, pm->Nmesh, omp_get_max_threads(), 0, pm->BoxSize*UnitLength_in_cm);

    glass_force(pm, t_x, ICP, NumPart);

    /* Our pick of the units ensures there is an oscillation period of 2 * M_PI.
     *
     * (don't ask me how this worked -- I think I failed this problem in physics undergrad. )
     * We use 4 steps per oscillation, and end at
     *
     * 12 + 1 = 13, the first time phase is M_PI / 2, a close encounter to the minimum.
     *
     * */
    for(step = 0; step < nsteps; step++) {
        /* leap-frog, K D D F K */
        double dt = M_PI / 2; /* step size */
        double hdt = 0.5 * dt; /* half a step */
        int d;
        /*
         * Use inverted gravity with a damping term proportional to the velocity.
         *
         * The magic setup was studied in
         *      https://github.com/rainwoodman/fastpm-python/blob/1be020b/Example/Glass.ipynb
         * */

        /* Kick */
        for(i = 0; i < NumPart; i ++) {
            for(d = 0; d < 3; d ++) {
                /* mind the damping term */
                ICP[i].Vel[d] += (ICP[i].Disp[d] - ICP[i].Vel[d]) * hdt;
            }
        }
        t_x += hdt;

        /* Drift */
        for(i = 0; i < NumPart; i ++) {
            for(d = 0; d < 3; d ++) {
                ICP[i].Pos[d] += ICP[i].Vel[d] * dt;
           }
        }
        t_v += dt;

        glass_force(pm, t_x, ICP, NumPart);
        t_f = t_x;

        /* Kick */
        for(i = 0; i < NumPart; i ++) {
            for(d = 0; d < 3; d ++) {
                /* mind the damping term */
                ICP[i].Vel[d] += (ICP[i].Disp[d] - ICP[i].Vel[d]) * hdt;
            }
        }

        t_x += hdt;
        message(0, "Generating glass, step = %d, t_f= %g, t_v = %g, t_x = %g\n", step, t_f / (2 * M_PI), t_v / (2 *M_PI), t_x / (2 * M_PI));
        glass_stats(ICP, NumPart);

        /*Now save the power spectrum*/
        powerspectrum_save(pm->ps, OutputDir, pkoutname, t_f, 1.0);
    }

    /*We are done with the power spectrum, free it*/
    powerspectrum_free(pm->ps);
}


static void
glass_stats(struct ic_part_data * ICP, int NumPart) {
    int i;
    double disp2 = 0;
    double vel2 = 0;
    double n = NumPart;
    for(i = 0; i < NumPart; i++)
    {
        int k;
        for(k = 0; k < 3; k++)
        {
            double dis = ICP[i].Disp[k];
            double vel = ICP[i].Vel[k];
            disp2 += dis * dis;
            vel2 += vel * vel;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &disp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vel2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    message(0, "Force std = %g, vel std = %g\n", sqrt(disp2 / n), sqrt(vel2 / n));
}

struct ic_prep_data
{
    struct ic_part_data * curICP;
    int NumPart;
};
/*Global to pass the particle data to the readout functions*/
static struct ic_part_data * curICP;

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.*/
static void glass_force(PetaPM * pm, double t_f, struct ic_part_data * ICP, const int NumPart) {

    PetaPMParticleStruct pstruct = {
        ICP,
        sizeof(ICP[0]),
        (char*) &ICP[0].Pos[0]  - (char*) ICP,
        (char*) &ICP[0].Mass  - (char*) ICP,
        NULL,
        NULL,
        NumPart,
        0, //no star info needed here
        0, //no sfr info needed here
        0, //no pi info needed here
        NULL, //no SPHP needed here
    };
    curICP = ICP;

    int i;
    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        ICP[i].Disp[0] = ICP[i].Disp[1] = ICP[i].Disp[2] = 0;
    }

    powerspectrum_zero(pm->ps);

    struct ic_prep_data icprep = {ICP, NumPart};
    /*
     * we apply potential transfer immediately after the R2C transform,
     * Therefore the force transfer functions are based on the potential,
     * not the density.
     * */
    petapm_force(pm, _prepare, &global_functions, functions, &pstruct, &icprep);

    powerspectrum_sum(pm->ps);
    walltime_measure("/LongRange");
}

static double pot_factor;

static PetaPMRegion *
_prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions)
{
    struct ic_prep_data * icprep = (struct ic_prep_data *) userdata;
    int NumPart = icprep->NumPart;
    struct ic_part_data * ICP = icprep->curICP;

    /* dimensionless invert gravity;
     *
     * pot = nabla ^ -2 delta
     *
     * (2pi / L) ** -2 is for kint -> k.
     * Need to divide by mean mass per cell to get delta */
    pot_factor = -1 * (-1) * pow(2 * M_PI / pm->BoxSize, -2);

    PetaPMRegion * regions = mymalloc2("Regions", sizeof(PetaPMRegion));
    int k;
    int r = 0;
    int i;
    double min[3] = {pm->BoxSize, pm->BoxSize, pm->BoxSize};
    double max[3] = {0, 0, 0.};
    double totmass = 0;

    for(i = 0; i < NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            if(min[k] > ICP[i].Pos[k])
                min[k] = ICP[i].Pos[k];
            if(max[k] < ICP[i].Pos[k])
                max[k] = ICP[i].Pos[k];
        }

        totmass += ICP[i].Mass;
    }

    MPI_Allreduce(MPI_IN_PLACE, &totmass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* 1 / pow(pm->Nmesh, 3) is included by the FFT, so just use total mass. */
    pot_factor /= totmass;

    for(k = 0; k < 3; k ++) {
        regions[r].offset[k] = floor(min[k] / pm->BoxSize * pm->Nmesh - 1);
        regions[r].size[k] = ceil(max[k] / pm->BoxSize * pm->Nmesh + 2);
        regions[r].size[k] -= regions[r].offset[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(&regions[r]);
    *Nregions = 1;

    walltime_measure("/PMgrav/Regions");
    return regions;
}


/********************
 * transfer functions for
 *
 * potential from mass in cell
 *
 * and
 *
 * force from potential
 *
 *********************/

static void potential_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex *value) {

    double f = 1.0;
    const double smth = 1.0 / k2;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 pm->Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    /*
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout
     * I don't understand the second yet!
     * */
    const double fac = pot_factor * smth * f * f;

    /*Compute the power spectrum*/
    powerspectrum_add_mode(pm->ps, k2, kpos, value, f, pm->Nmesh);

    if(k2 == 0) {
        /* Remove zero mode corresponding to the mean.*/
        value[0][0] = 0.0;
        value[0][1] = 0.0;
        return;
    }

    value[0][0] *= fac;
    value[0][1] *= fac;
}

/* the transfer functions for force in fourier space applied to potential */
/* super lanzcos in CH6 P 122 Digital Filters by Richard W. Hamming */
static double diff_kernel(double w) {
/* order N = 1 */
/*
 * This is the same as GADGET-2 but in fourier space:
 * see gadget-2 paper and Hamming's book.
 * c1 = 2 / 3, c2 = 1 / 12
 * */
    return 1 / 6.0 * (8 * sin (w) - sin (2 * w));
}

static void force_transfer(PetaPM *pm, int k, pfft_complex * value) {
    double tmp0;
    double tmp1;
    /*
     * negative sign is from force_x = - Del_x pot
     *
     * filter is   i K(w)
     * */
    double fac = -1 * diff_kernel (k * (2 * M_PI / pm->Nmesh)) * (pm->Nmesh / pm->BoxSize);
    tmp0 = - value[0][1] * fac;
    tmp1 = value[0][0] * fac;
    value[0][0] = tmp0;
    value[0][1] = tmp1;
}
static void force_x_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[0], value);
}
static void force_y_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[1], value);
}
static void force_z_transfer(PetaPM *pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[2], value);
}
static void readout_force_x(PetaPM *pm, int i, double * mesh, double weight) {
    curICP[i].Disp[0] += weight * mesh[0];
}
static void readout_force_y(PetaPM *pm, int i, double * mesh, double weight) {
    curICP[i].Disp[1] += weight * mesh[0];
}
static void readout_force_z(PetaPM *pm, int i, double * mesh, double weight) {
    curICP[i].Disp[2] += weight * mesh[0];
}
