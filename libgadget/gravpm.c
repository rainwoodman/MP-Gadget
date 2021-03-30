#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "utils.h"

#include "allvars.h"
#include "partmanager.h"
#include "forcetree.h"
#include "petapm.h"
#include "domain.h"
#include "walltime.h"
#include "gravity.h"

#include "cosmology.h"
#include "neutrinos_lra.h"

static int pm_mark_region_for_node(int startno, int rid, int * RegionInd, const ForceTree * tt);
static void convert_node_to_region(PetaPM * pm, PetaPMRegion * r, struct NODE * Nodes);

static int hybrid_nu_gravpm_is_active(int i);
static void potential_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void compute_neutrino_power(PetaPM * pm);
static void force_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void force_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void readout_potential(PetaPM * pm, int i, double * mesh, double weight);
static void readout_force_x(PetaPM * pm, int i, double * mesh, double weight);
static void readout_force_y(PetaPM * pm, int i, double * mesh, double weight);
static void readout_force_z(PetaPM * pm, int i, double * mesh, double weight);
static PetaPMFunctions functions [] =
{
    {"Potential", NULL, readout_potential},
    {"ForceX", force_x_transfer, readout_force_x},
    {"ForceY", force_y_transfer, readout_force_y},
    {"ForceZ", force_z_transfer, readout_force_z},
    {NULL, NULL, NULL},
};

static PetaPMGlobalFunctions global_functions = {NULL, NULL, potential_transfer};

static PetaPMRegion * _prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions);

void
gravpm_init_periodic(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G) {
    petapm_init(pm, BoxSize, Asmth, Nmesh, G, MPI_COMM_WORLD);

    /*Initialise the kspace neutrino code if it is enabled.
     * Mpc units are used to match power spectrum code.*/
    if(All.MassiveNuLinRespOn) {
        global_functions.global_readout = measure_power_spectrum;
        global_functions.global_analysis = compute_neutrino_power;
    }
}

/* Computes the gravitational force on the PM grid
 * and saves the total matter power spectrum.*/
void
gravpm_force(PetaPM * pm, ForceTree * tree) {
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
        0, //no star info needed here
        0, //no sfr info needed here
        0, //no pi info needed here
        NULL, //no SPHP needed here
    };

    if(All.HybridNeutrinosOn && particle_nu_fraction(&All.CP.ONu.hybnu, All.Time, 0) == 0.)
        pstruct.active = &hybrid_nu_gravpm_is_active;

    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        P[i].GravPM[0] = P[i].GravPM[1] = P[i].GravPM[2] = 0;
    }

    /*
     * we apply potential transfer immediately after the R2C transform,
     * Therefore the force transfer functions are based on the potential,
     * not the density.
     * */
    petapm_force(pm, _prepare, &global_functions, functions, &pstruct, tree);
    powerspectrum_sum(pm->ps);
    /*Now save the power spectrum*/
    powerspectrum_save(pm->ps, All.OutputDir, "powerspectrum", All.Time, GrowthFactor(&All.CP, All.Time, 1.0));
    /* Save the neutrino power if it is allocated*/
    if(pm->ps->logknu)
        powerspectrum_nu_save(pm->ps, All.OutputDir, "powerspectrum-nu", All.Time);
    /*We are done with the power spectrum, free it*/
    powerspectrum_free(pm->ps);
    walltime_measure("/LongRange");
}

static PetaPMRegion * _prepare(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions) {
    /*
     *
     * walks down the tree, identify nodes that contains local mass and
     * are sufficiently large in volume.
     *
     * for each nodes, a mesh region is created.
     * the particles in a node are linked to their hosting region
     * (each particle belongs
     * to exactly one region even though it may be covered by two)
     *
     * */
    ForceTree * tree = (ForceTree *) userdata;
    /* In worst case, each topleave becomes a region: thus
     * NTopLeaves is sufficient */
    PetaPMRegion * regions = mymalloc2("Regions", sizeof(PetaPMRegion) * tree->NTopLeaves);
    pstruct->RegionInd = mymalloc2("RegionInd", PartManager->NumPart * sizeof(int));
    int r = 0;

    int no = tree->firstnode; /* start with the root */
    while(no >= 0) {

        if(!(tree->Nodes[no].f.DependsOnLocalMass)) {
            /* node doesn't contain particles on this process, do not open */
            no = tree->Nodes[no].sibling;
            continue;
        }
        if(
            /* node is large */
           (tree->Nodes[no].len <= pm->BoxSize / pm->Nmesh * 24)
           ||
            /* node is a top leaf */
            ( !tree->Nodes[no].f.InternalTopLevel && (tree->Nodes[no].f.TopLevel) )
                ) {
            regions[r].no = no;
            r ++;
            /* do not open */
            no = tree->Nodes[no].sibling;
            continue;
        }
        /* open */
        no = tree->Nodes[no].s.suns[0];
    }

    *Nregions = r;
    int maxNregions;
    MPI_Reduce(&r, &maxNregions, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    message(0, "max number of regions is %d\n", maxNregions);

    int64_t i;
    int numswallowed = 0;
    #pragma omp parallel for reduction(+: numswallowed)
    for(i =0; i < PartManager->NumPart; i ++) {
        /* Swallowed black hole particles stick around but should not gravitate.
         * Short-range is handled by not adding them to the tree. */
        if(All.BlackHoleOn && P[i].Swallowed && P[i].Type==5){
            pstruct->RegionInd[i] = -2;
            numswallowed++;
        }
        else
            pstruct->RegionInd[i] = -1;
    }

    /* now lets mark particles to their hosting region */
    int64_t numpart = 0;
#pragma omp parallel for reduction(+: numpart)
    for(r = 0; r < *Nregions; r++) {
        regions[r].numpart = pm_mark_region_for_node(regions[r].no, r, pstruct->RegionInd, tree);
        numpart += regions[r].numpart;
    }
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(pstruct->RegionInd[i] == -1) {
            message(1, "i = %d father %d not assigned to a region\n", i, force_get_father(i, tree));
        }
    }
    /* All particles shall have been processed just once. Otherwise we die */
    if((numpart+numswallowed) != PartManager->NumPart) {
        endrun(1, "Processed only %ld particles out of %ld\n", numpart, PartManager->NumPart);
    }
    for(r =0; r < *Nregions; r++) {
        convert_node_to_region(pm, &regions[r], tree->Nodes);
    }
    /*This is done to conserve memory during the PM step*/
    if(force_tree_allocated(tree)) force_tree_free(tree);

    /*Allocate memory for a power spectrum*/
    powerspectrum_alloc(pm->ps, pm->Nmesh, omp_get_max_threads(), All.MassiveNuLinRespOn, pm->BoxSize*All.UnitLength_in_cm);

    walltime_measure("/PMgrav/Regions");
    return regions;
}

static int pm_mark_region_for_node(int startno, int rid, int * RegionInd, const ForceTree * tree) {
    int numpart = 0;
    int no = startno;
    int endno = tree->Nodes[startno].sibling;
    while(no >= 0 && no != endno)
    {
        struct NODE * nop = &tree->Nodes[no];
        if(nop->f.ChildType == PARTICLE_NODE_TYPE) {
            int i;
            for(i = 0; i < nop->s.noccupied; i++) {
                int p = nop->s.suns[i];
                RegionInd[p] = rid;
#ifdef DEBUG
                /* Check for particles outside of the node. This should never happen,
                * unless there is a bug in tree build, or the particles are being moved.*/
                int k;
                for(k = 0; k < 3; k ++) {
                    double l = P[p].Pos[k] - tree->Nodes[startno].center[k];
                    l = fabs(l * 2);
                    if (l > tree->Nodes[startno].len) {
                        if(l > tree->Nodes[startno].len * (1+ 1e-7))
                        endrun(1, "enlarging node size from %g to %g, due to particle of type %d at %g %g %g id=%ld\n",
                            tree->Nodes[startno].len, l, P[p].Type, P[p].Pos[0], P[p].Pos[1], P[p].Pos[2], P[p].ID);
                        tree->Nodes[startno].len = l;
                    }
                }
#endif
            }
            numpart += nop->s.noccupied;
            /* Move to sibling*/
            no = nop->sibling;
        }
        /* Generally this function should be handed top leaves which do not contain pseudo particles.
         * However, sometimes it will receive an internal top node which can, if that top node is small enough.
         * In this case, just move to the sibling: no particles to mark here.*/
        else if(nop->f.ChildType == PSEUDO_NODE_TYPE)
            no = nop->sibling;
        else if(nop->f.ChildType == NODE_NODE_TYPE)
            no = nop->s.suns[0];
        else
            endrun(122, "Unrecognised Node type %d, memory corruption!\n", nop->f.ChildType);
    }
    return numpart;
}


static void convert_node_to_region(PetaPM * pm, PetaPMRegion * r, struct NODE * Nodes) {
    int k;
    double cellsize = pm->BoxSize / pm->Nmesh;
    int no = r->no;
#if 0
    printf("task = %d no = %d len = %g hmax = %g center = %g %g %g\n",
            ThisTask, no, Nodes[no].len, Nodes[no].hmax,
            Nodes[no].center[0],
            Nodes[no].center[1],
            Nodes[no].center[2]);
#endif
    for(k = 0; k < 3; k ++) {
        r->offset[k] = floor((Nodes[no].center[k] - Nodes[no].len * 0.5) / cellsize);
        int end = (int) ceil((Nodes[no].center[k] + Nodes[no].len * 0.5) / cellsize) + 1;
        r->size[k] = end - r->offset[k] + 1;
        r->center[k] = Nodes[no].center[k];
    }

    /* setup the internal data structure of the region */
    petapm_region_init_strides(r);

    r->len  = Nodes[no].len;
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

/* unnormalized sinc function sin(x) / x */
static double sinc_unnormed(double x) {
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}

/* Update the model prediction of LinResp neutrino power spectrum.
 * This should happen after the CFT is computed,
 * and after powerspectrum_add_mode() has been called,
 * but before potential_transfer is called.*/
static void compute_neutrino_power(PetaPM * pm) {
    if(!All.MassiveNuLinRespOn)
        return;
    Power * ps = pm->ps;
    /*Note the power spectrum is now in Mpc units*/
    powerspectrum_sum(ps);
    int i;
    /*Get delta_cdm_curr , which is P(k)^1/2.*/
    for(i=0; i<ps->nonzero; i++) {
        ps->Power[i] = sqrt(ps->Power[i]);
    }
    /*Get the neutrino power.*/
    delta_nu_from_power(ps, &All.CP, All.Time, All.TimeIC);

    /*Initialize the interpolation for the neutrinos*/
    ps->nu_spline = gsl_interp_alloc(gsl_interp_linear,ps->nonzero);
    ps->nu_acc = gsl_interp_accel_alloc();
    gsl_interp_init(ps->nu_spline,ps->logknu,ps->delta_nu_ratio,ps->nonzero);
    /*Zero power spectrum, which is stored with the neutrinos*/
    powerspectrum_zero(ps);
}

/* Compute the power spectrum of the fourier transformed grid in value.
 * Store it in the PowerSpectrum structure */
void
powerspectrum_add_mode(Power * PowerSpectrum, const int64_t k2, const int kpos[3], pfft_complex * const value, const double invwindow, double Nmesh)
{
    if(k2 == 0) {
        /* Save zero mode corresponding to the mean as the normalisation factor.*/
        PowerSpectrum->Norm = (value[0][0] * value[0][0] + value[0][1] * value[0][1]);
        return;
    }
    /* Measure power spectrum: we don't want the zero mode.
     * Some modes with k_z = 0 or N/2 have weight 1, the rest have weight 2.
     * This is because of the symmetry of the real fft. */
    if(k2 > 0) {
        /*How many bins per unit (log) interval in k?*/
        const double binsperunit=(PowerSpectrum->size-1)/log(sqrt(3) * Nmesh/2.0);
        int kint=floor(binsperunit*log(k2)/2.);
        int w;
        const double keff = sqrt(kpos[0]*kpos[0]+kpos[1]*kpos[1]+kpos[2]*kpos[2]);
        const double m = (value[0][0] * value[0][0] + value[0][1] * value[0][1]);
        /*Make sure we do not overflow (although this should never happen)*/
        if(kint >= PowerSpectrum->size)
            return;
        if(kpos[2] == 0 || kpos[2] == Nmesh/2) w = 1;
        else w = 2;
        /*Make sure we use thread-local memory to avoid racing.*/
        const int index = kint + omp_get_thread_num() * PowerSpectrum->size;
        /*Multiply P(k) by inverse window function*/
        PowerSpectrum->Power[index] += w * m * invwindow * invwindow;
        PowerSpectrum->Nmodes[index] += w;
        PowerSpectrum->kk[index] += w * keff;
    }

}

/*Just read the power spectrum, without changing the input value.*/
void
measure_power_spectrum(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value) {
    double f = 1.0;
    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / pm->Nmesh;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    powerspectrum_add_mode(pm->ps, k2, kpos, value, f, pm->Nmesh);
}

static void
potential_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex *value)
{
    const double asmth2 = pow((2 * M_PI) * pm->Asmth / pm->Nmesh,2);
    double f = 1.0;
    const double smth = exp(-k2 * asmth2) / k2;
        /* fac is - 4pi G     (L / 2pi) **2 / L ** 3
     *        Gravity       k2            DFT (dk **3, but )
     * */
    const double pot_factor = - pm->G / (M_PI * pm->BoxSize);	/* to get potential */


    /* the CIC deconvolution kernel is
     *
     * sinc_unnormed(k_x L / 2 Nmesh) ** 2
     *
     * k_x = kpos * 2pi / L
     *
     * */
    int k;
    for(k = 0; k < 3; k ++) {
        double tmp = (kpos[k] * M_PI) / pm->Nmesh;
        tmp = sinc_unnormed(tmp);
        f *= 1. / (tmp * tmp);
    }
    /*
     * first decovolution is CIC in par->mesh
     * second decovolution is correcting readout
     * I don't understand the second yet!
     * */
    const double fac = pot_factor * smth * f * f;
    Power * ps = pm->ps;

    /*Add neutrino power if desired*/
    if(All.MassiveNuLinRespOn && k2 > 0) {
        /* Change the units of k to match those of logkk*/
        double logk2 = log(sqrt(k2) * 2 * M_PI / ps->BoxSize_in_MPC);
        /* Floating point roundoff and the binning means there may be a mode just beyond the box size.*/
        if(logk2 < ps->logknu[0] && logk2 > ps->logknu[0]-log(2) )
            logk2 = ps->logknu[0];
        else if( logk2 > ps->logknu[ps->nonzero-1])
            logk2 = ps->logknu[ps->nonzero-1];
        /* Note get_neutrino_powerspec returns Omega_nu / (Omega0 -OmegaNu) * delta_nu / P_cdm^1/2, which is dimensionless.
         * So below is: M_cdm * delta_cdm (1 + Omega_nu/(Omega0-OmegaNu) (delta_nu / delta_cdm))
         *            = M_cdm * (delta_cdm (Omega0 - OmegaNu)/Omega0 + Omega_nu/Omega0 delta_nu) * Omega0 / (Omega0-OmegaNu)
         *            = M_cdm * Omega0 / (Omega0-OmegaNu) * (delta_cdm (1 - f_nu)  + f_nu delta_nu) )
         *            = M_cdm * Omega0 / (Omega0-OmegaNu) * delta_t
         *            = (M_cdm + M_nu) * delta_t
         * This is correct for the forces, and gives the right power spectrum,
         * once we multiply PowerSpectrum.Norm by (Omega0 / (Omega0 - OmegaNu))**2 */
        const double nufac = 1 + ps->nu_prefac * gsl_interp_eval(ps->nu_spline,ps->logknu,
                                                                       ps->delta_nu_ratio,logk2,ps->nu_acc);
        value[0][0] *= nufac;
        value[0][1] *= nufac;
    }

    /*Compute the power spectrum*/
    powerspectrum_add_mode(ps, k2, kpos, value, f, pm->Nmesh);
    if(k2 == 0) {
        if(All.MassiveNuLinRespOn) {
            const double MtotbyMcdm = All.CP.Omega0/(All.CP.Omega0 - pow(All.Time,3)*get_omega_nu_nopart(&All.CP.ONu, All.Time));
            ps->Norm *= MtotbyMcdm*MtotbyMcdm;
        }
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

/*This function decides if a particle is actively gravitating; tracers are not.*/
static int hybrid_nu_gravpm_is_active(int i) {
    if (P[i].Type == All.FastParticleType)
        return 0;
    else
        return 1;
}

static void force_transfer(PetaPM * pm, int k, pfft_complex * value) {
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
static void force_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[0], value);
}
static void force_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[1], value);
}
static void force_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    force_transfer(pm, kpos[2], value);
}
static void readout_potential(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].Potential += weight * mesh[0];
}
static void readout_force_x(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].GravPM[0] += weight * mesh[0];
}
static void readout_force_y(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].GravPM[1] += weight * mesh[0];
}
static void readout_force_z(PetaPM * pm, int i, double * mesh, double weight) {
    P[i].GravPM[2] += weight * mesh[0];
}
