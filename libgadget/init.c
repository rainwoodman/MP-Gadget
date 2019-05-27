#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_sf_gamma.h>

#include "utils.h"

#include "allvars.h"
#include "cooling.h"
#include "forcetree.h"
#include "density.h"

#include "timefac.h"
#include "petaio.h"
#include "domain.h"
#include "walltime.h"
#include "slotsmanager.h"
#include "hydra.h"
#include "sfr_eff.h"
#include "exchange.h"
#include "fof.h"
#include "timestep.h"
#include "timebinmgr.h"
#include "cosmology.h"
#include "uvbg.h"

/*! \file init.c
 *  \brief code for initialisation of a simulation from initial conditions
 */

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
struct global_data_all_processes All;

/*Set the parameters of the hydro module*/
void
set_init_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /* Start reading the values */
        param_get_string2(ps, "InitCondFile", All.InitCondFile, sizeof(All.InitCondFile));
        param_get_string2(ps, "OutputDir", All.OutputDir, sizeof(All.OutputDir));
        param_get_string2(ps, "SnapshotFileBase", All.SnapshotFileBase, sizeof(All.SnapshotFileBase));
        param_get_string2(ps, "FOFFileBase", All.FOFFileBase, sizeof(All.FOFFileBase));
        param_get_string2(ps, "EnergyFile", All.EnergyFile, sizeof(All.EnergyFile));
        All.OutputEnergyDebug = param_get_int(ps, "OutputEnergyDebug");
        param_get_string2(ps, "CpuFile", All.CpuFile, sizeof(All.CpuFile));

        All.CP.CMBTemperature = param_get_double(ps, "CMBTemperature");
        All.CP.RadiationOn = param_get_int(ps, "RadiationOn");
        All.CP.Omega0 = param_get_double(ps, "Omega0");
        All.CP.OmegaBaryon = param_get_double(ps, "OmegaBaryon");
        All.CP.OmegaLambda = param_get_double(ps, "OmegaLambda");
        All.CP.Omega_fld = param_get_double(ps, "Omega_fld");
        if(All.CP.OmegaLambda > 0 && All.CP.Omega_fld > 0)
            endrun(0, "Cannot have OmegaLambda and Omega_fld (evolving dark energy) at the same time!\n");
        All.CP.w0_fld = param_get_double(ps,"w0_fld");
        All.CP.wa_fld = param_get_double(ps,"wa_fld");
        All.CP.Omega_ur = param_get_double(ps, "Omega_ur");
        All.CP.HubbleParam = param_get_double(ps, "HubbleParam");

        All.OutputPotential = param_get_int(ps, "OutputPotential");
        All.OutputTimebins = param_get_int(ps, "OutputTimebins");
        All.OutputHeliumFractions = param_get_int(ps, "OutputHeliumFractions");
        All.OutputDebugFields = param_get_int(ps, "OutputDebugFields");

        All.TimeMax = param_get_double(ps, "TimeMax");
        All.Asmth = param_get_double(ps, "Asmth");
        All.ShortRangeForceWindowType = param_get_enum(ps, "ShortRangeForceWindowType");
        All.Nmesh = param_get_int(ps, "Nmesh");

        All.CoolingOn = param_get_int(ps, "CoolingOn");
        All.HydroOn = param_get_int(ps, "HydroOn");
        All.DensityOn = param_get_int(ps, "DensityOn");
        All.TreeGravOn = param_get_int(ps, "TreeGravOn");
        All.LightconeOn = param_get_int(ps, "LightconeOn");
        All.FastParticleType = param_get_int(ps, "FastParticleType");
        All.PairwiseActiveFraction = param_get_double(ps, "PairwiseActiveFraction");
        All.TimeLimitCPU = param_get_double(ps, "TimeLimitCPU");
        All.AutoSnapshotTime = param_get_double(ps, "AutoSnapshotTime");
        All.TimeBetweenSeedingSearch = param_get_double(ps, "TimeBetweenSeedingSearch");
        All.RandomParticleOffset = param_get_double(ps, "RandomParticleOffset");

        All.PartAllocFactor = param_get_double(ps, "PartAllocFactor");
        All.SlotsIncreaseFactor = param_get_double(ps, "SlotsIncreaseFactor");

        All.SnapshotWithFOF = param_get_int(ps, "SnapshotWithFOF");

        All.RandomSeed = param_get_int(ps, "RandomSeed");

        All.BlackHoleOn = param_get_int(ps, "BlackHoleOn");
        All.WriteBlackHoleDetails = param_get_int(ps,"WriteBlackHoleDetails");

        All.StarformationOn = param_get_int(ps, "StarformationOn");
        All.WindOn = param_get_int(ps, "WindOn");
        All.MetalReturnOn = param_get_int(ps, "MetalReturnOn");
        All.MaxDomainTimeBinDepth = param_get_int(ps, "MaxDomainTimeBinDepth");
        All.InitGasTemp = param_get_double(ps, "InitGasTemp");

        /*Massive neutrino parameters*/
        All.MassiveNuLinRespOn = param_get_int(ps, "MassiveNuLinRespOn");
        All.HybridNeutrinosOn = param_get_int(ps, "HybridNeutrinosOn");
        All.CP.MNu[0] = param_get_double(ps, "MNue");
        All.CP.MNu[1] = param_get_double(ps, "MNum");
        All.CP.MNu[2] = param_get_double(ps, "MNut");
        All.HybridVcrit = param_get_double(ps, "Vcrit");
        All.HybridNuPartTime = param_get_double(ps, "NuPartTime");
        if(All.MassiveNuLinRespOn && !All.CP.RadiationOn)
            endrun(2, "You have enabled (kspace) massive neutrinos without radiation, but this will give an inconsistent cosmology!\n");
        /*End massive neutrino parameters*/

        if(All.StarformationOn == 0)
        {
            if(All.WindOn == 1) {
                endrun(1, "You try to use the code with wind enabled,\n"
                          "but you did not switch on starformation.\nThis mode is not supported.\n");
            }
        } else {
            if(All.CoolingOn == 0)
            {
                endrun(1, "You try to use the code with star formation enabled,\n"
                          "but you did not switch on cooling.\nThis mode is not supported.\n");
            }
        }
    }
    MPI_Bcast(&All, sizeof(All), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void check_omega(int generations);
static void check_positions(void);
void check_smoothing_length(double * MeanSpacing, const double BoxSize);

static void
setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp, const inttime_t Ti_Current);

/*! This function reads the initial conditions, and allocates storage for the
 *  tree(s). Various variables of the particle data are initialised and An
 *  intial domain decomposition is performed. If SPH particles are present,
 *  the initial SPH smoothing lengths are determined.
 */
inttime_t init(int RestartSnapNum, DomainDecomp * ddecomp)
{
    int i;

    /*Add TimeInit and TimeMax to the output list*/
    if (RestartSnapNum < 0) {
        /* allow a first snapshot at IC time; */
        setup_sync_points(All.TimeIC, All.TimeMax, 0.0, All.SnapshotWithFOF);
    } else {
        /* skip dumping the exactly same snapshot */
        setup_sync_points(All.TimeIC, All.TimeMax, All.TimeInit, All.SnapshotWithFOF);
        /* If TimeInit is not in a sensible place on the integer timeline
         * (can happen if the outputs changed since it was written)
         * start the integer timeline anew from TimeInit */
        inttime_t ti_init = ti_from_loga(log(All.TimeInit)) % TIMEBASE;
        if(round_down_power_of_two(ti_init) != ti_init) {
            message(0,"Resetting integer timeline (as %x != %x) to current snapshot\n",ti_init, round_down_power_of_two(ti_init));
            setup_sync_points(All.TimeInit, All.TimeMax, All.TimeInit, All.SnapshotWithFOF);
        }
    }

    inttime_t Ti_Current = init_timebins(All.TimeInit);

    /* Important to set the global time before reading in the snapshot time as it affects the GT funcs for IO. */
    set_global_time(Ti_Current);

    /*Read the snapshot*/
    petaio_read_snapshot(RestartSnapNum, MPI_COMM_WORLD);

    domain_test_id_uniqueness(PartManager);

    check_omega(get_generations());

    check_positions();

    if(RestartSnapNum == -1)
        check_smoothing_length(All.MeanSeparation, All.BoxSize);

    /* As the above will mostly take place
     * on Task 0, there will be a lot of imbalance*/
    MPIU_Barrier(MPI_COMM_WORLD);

    fof_init(All.MeanSeparation[1]);

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)	/* initialize sph_properties */
    {
        int j;
        P[i].Ti_drift = Ti_Current;

        if(All.BlackHoleOn && RestartSnapNum == -1 && P[i].Type == 5 )
        {
            /* Note: Gadget-3 sets this to the seed black hole mass.*/
            BHP(i).Mass = P[i].Mass;

            /* Touch up potentially zero BH smoothing lengths, since they have historically not been saved in the snapshots.
             * Anything non-zero would work, but since BH tends to be in high density region,
             *  use a small number */
            if(P[i].Hsml == 0)
                P[i].Hsml = 0.01 * All.MeanSeparation[0];
        }

        if(All.MetalReturnOn && P[i].Type == 4 )
        {
            /* Touch up zero star smoothing lengths, not saved in the snapshots.*/
            if(P[i].Hsml == 0)
                P[i].Hsml = 0.1 * All.MeanSeparation[0];
        }

        if(All.BlackHoleOn && P[i].Type == 5)
        {
            for(j = 0; j < 3; j++) {
                BHP(i).DFAccel[j] = 0;
                BHP(i).DragAccel[j] = 0;
            }
        }

        P[i].Key = PEANO(P[i].Pos, All.BoxSize);

        if(P[i].Type != 0) continue;

        for(j = 0; j < 3; j++)
        {
            SPHP(i).HydroAccel[j] = 0;
        }

        if(!isfinite(SPHP(i).DelayTime ))
            endrun(6, "Bad DelayTime %g for part %d id %ld\n", SPHP(i).DelayTime, i, P[i].ID);
        SPHP(i).DtEntropy = 0;

        if(RestartSnapNum == -1)
        {
            SPHP(i).Density = -1;
            SPHP(i).EgyWtDensity = -1;
            SPHP(i).DhsmlEgyDensityFactor = -1;
            SPHP(i).Entropy = -1;
            SPHP(i).Ne = 1.0;
            SPHP(i).DivVel = 0;
            SPHP(i).CurlVel = 0;
            SPHP(i).DelayTime = 0;
            SPHP(i).Metallicity = 0;
            memset(SPHP(i).Metals, 0, NMETALS*sizeof(float));
            /* Initialise to primordial abundances for H and He*/
            SPHP(i).Metals[0] = HYDROGEN_MASSFRAC;
            SPHP(i).Metals[1] = 1- HYDROGEN_MASSFRAC;
            SPHP(i).Sfr = 0;
            SPHP(i).MaxSignalVel = 0;
        }
    }

    walltime_measure("/Init");

    domain_decompose_full(ddecomp);	/* do initial domain decomposition (gives equal numbers of particles) */

    if(All.DensityOn)
        setup_smoothinglengths(RestartSnapNum, ddecomp, Ti_Current);

    malloc_permanent_uvbg_grids();
    return Ti_Current;
}


/*! This routine computes the mass content of the box and compares it to the
 * specified value of Omega-matter.  If discrepant, the run is terminated.
 */
void check_omega(int generations)
{
    double mass = 0, masstot, omega;
    int i, badmass = 0;
    int64_t totbad;

    #pragma omp parallel for reduction(+: mass) reduction(+: badmass)
    for(i = 0; i < PartManager->NumPart; i++) {
        /* In case zeros have been written to the saved mass array,
         * recover the true masses*/
        if(P[i].Mass == 0) {
            P[i].Mass = All.MassTable[P[i].Type] * ( 1. - (double)P[i].Generation/generations);
            badmass++;
        }
        mass += P[i].Mass;
    }

    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    sumup_large_ints(1, &badmass, &totbad);
    if(totbad)
        message(0, "Warning: recovering from %ld Mass entries corrupted on disc\n",totbad);

    omega =
        masstot / (All.BoxSize * All.BoxSize * All.BoxSize) / (3 * All.CP.Hubble * All.CP.Hubble / (8 * M_PI * All.G));

    /*Add the density for analytically follows massive neutrinos*/
    if(All.MassiveNuLinRespOn)
        omega += get_omega_nu_nopart(&All.CP.ONu, 1);
    if(fabs(omega - All.CP.Omega0) > 1.0e-3)
    {
        endrun(0, "The mass content accounts only for Omega=%g,\nbut you specified Omega=%g in the parameterfile.\n",
                omega, All.CP.Omega0);
    }
}

/*! This routine checks that the initial positions of the particles are within the box.
 * If not, there is likely a bug in the IC generator and we abort.
 * It also checks for multiple zeros in the positions, guarding against a common fs bug.
 */
void check_positions(void)
{
    int i;
    int numzero = 0;
    int lastzero = -1;
    #pragma omp parallel for reduction(+: numzero) reduction(max:lastzero)
    for(i=0; i< PartManager->NumPart; i++){
        int j;
        for(j=0; j<3; j++) {
            if(P[i].Pos[j] < 0 || P[i].Pos[j] > All.BoxSize || !isfinite(P[i].Pos[j]))
                endrun(0,"Particle %d is outside the box (L=%g) at (%g %g %g)\n",i,All.BoxSize, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
        }
        if((P[i].Pos[0] < 1e-35) && (P[i].Pos[1] < 1e-35) && (P[i].Pos[2] < 1e-35)) {
            numzero++;
            lastzero = i;
        }
    }
    if(numzero > 1)
        endrun(5, "Particle positions contain %d zeros at particle %d. Pos %g %g %g. Likely write corruption!\n",
                numzero, lastzero, P[lastzero].Pos[0], P[lastzero].Pos[1], P[lastzero].Pos[2]);
}

/*! This routine checks that the initial smoothing lengths of the particles
 *  are sensible and resets them to mean interparticle spacing if not.
 *  Guards against a problem writing the snapshot. Matters because
 *  a very large initial smoothing length will cause density() to go crazy.
 */
void check_smoothing_length(double * MeanSpacing, const double BoxSize)
{
    int i;
    int numprob = 0;
    int lastprob = -1;
    #pragma omp parallel for reduction(+: numprob) reduction(max:lastprob)
    for(i=0; i< PartManager->NumPart; i++){
        if(P[i].Type != 5 && P[i].Type != 0)
            continue;
        if(P[i].Hsml > BoxSize || P[i].Hsml <= 0) {
            P[i].Hsml = MeanSpacing[P[i].Type];
            numprob++;
            lastprob = i;
        }
    }
    if(numprob > 0)
        message(5, "Bad smoothing lengths %d last bad %d hsml %g id %ld\n", numprob, lastprob, P[lastprob].Hsml, P[lastprob].ID);
}

/* Initialize the entropy variable in Pressure-Entropy Sph.
 * Initialization of the entropy variable is a little trickier in this version of SPH,
 * since we need to make sure it 'talks to' the density appropriately */
static void
setup_density_indep_entropy(const ActiveParticles * act, ForceTree * Tree, struct sph_pred_data * sph_pred, double u_init, double a3, const inttime_t Ti_Current)
{
    int j;
    int stop = 0;

    message(0, "Converting u -> entropy, with density split sph\n");

    /* This gives better convergence than initializing EgyWtDensity before Density is known*/
    #pragma omp parallel for
    for(j = 0; j < SlotsManager->info[0].size; j++)
        SphP[j].EgyWtDensity = SphP[j].Density;

    MyFloat * olddensity = (MyFloat *)mymalloc("olddensity ", SlotsManager->info[0].size * sizeof(MyFloat));
    for(j = 0; j < 100; j++)
    {
        int i;
        /* since ICs give energies, not entropies, need to iterate get this initialized correctly */
        #pragma omp parallel for
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            SphP[i].Entropy = GAMMA_MINUS1 * u_init / pow(SphP[i].EgyWtDensity / a3 , GAMMA_MINUS1);
            olddensity[i] = SphP[i].EgyWtDensity;
        }
        /* Empty kick factors as we do not move*/
        DriftKickTimes times = init_driftkicktime(Ti_Current);
        /* Update the EgyWtDensity*/
        density(act, 0, DensityIndependentSphOn(), All.BlackHoleOn, 0,  times, &All.CP, sph_pred, NULL, Tree);
        if(stop)
            break;

        double maxdiff = 0;
        #pragma omp parallel for reduction(max: maxdiff)
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            double value = fabs(SphP[i].EgyWtDensity - olddensity[i]) / SphP[i].EgyWtDensity;
            maxdiff = DMAX(maxdiff,value);
        }
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        message(0, "iteration %d, max relative change in EgyWtDensity = %g \n", j, maxdiff);

        /* If maxdiff is small, do one more iteration and stop*/
        if(maxdiff < 1e-3)
            stop = 1;

    }
    myfree(olddensity);
}

/*! This function is used to find an initial smoothing length for each SPH
 *  particle. It guarantees that the number of neighbours will be between
 *  desired_ngb-MAXDEV and desired_ngb+MAXDEV. For simplicity, a first guess
 *  of the smoothing length is provided to the function density(), which will
 *  then iterate if needed to find the right smoothing length.
 */
static void
setup_smoothinglengths(int RestartSnapNum, DomainDecomp * ddecomp, const inttime_t Ti_Current)
{
    int i;
    const double a3 = All.Time * All.Time * All.Time;

    int64_t tot_sph, tot_bh;
    MPI_Allreduce(&SlotsManager->info[0].size, &tot_sph, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&SlotsManager->info[5].size, &tot_bh, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);

    /* Do nothing if we are a pure DM run*/
    if(tot_sph + tot_bh == 0)
        return;
//
    ForceTree Tree = {0};
    /* Need moments because we use them to set Hsml*/
    force_tree_rebuild(&Tree, ddecomp, All.BoxSize, 0, 1, All.OutputDir);

    if(RestartSnapNum == -1)
    {
        /* quick hack to adjust for the baryon fraction
         * only this fraction of mass is of that type.
         * this won't work for non-dm non baryon;
         * ideally each node shall have separate count of
         * ptypes of each type.
         *
         * Eventually the iteration will fix this. */
        const double massfactor = All.CP.OmegaBaryon / All.CP.Omega0;
        const double DesNumNgb = GetNumNgb(GetDensityKernelType());

        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
        {
            /* These initial smoothing lengths are only used for SPH.
             * BH is set elsewhere. */
            if(P[i].Type != 0)
                continue;

            int no = force_get_father(i, &Tree);

            while(10 * DesNumNgb * P[i].Mass > massfactor * Tree.Nodes[no].mom.mass)
            {
                int p = force_get_father(no, &Tree);

                if(p < 0)
                    break;

                no = p;
            }

            P[i].Hsml =
                pow(3.0 / (4 * M_PI) * DesNumNgb * P[i].Mass / (massfactor * Tree.Nodes[no].mom.mass),
                        1.0 / 3) * Tree.Nodes[no].len;

            /* recover from a poor initial guess */
            if(P[i].Hsml > 500.0 * All.MeanSeparation[0])
                P[i].Hsml = All.MeanSeparation[0];
        }
    }
    /* When we restart, validate the SPH properties of the particles.
     * This also allows us to increase MinEgySpec on a restart if we choose.*/
    else
    {
        int bad = 0;
        double meanbar = All.CP.OmegaBaryon * 3 * HUBBLE * All.CP.HubbleParam * HUBBLE * All.CP.HubbleParam/ (8 * M_PI * GRAVITY);
        #pragma omp parallel for reduction(+: bad)
        for(i = 0; i < SlotsManager->info[0].size; i++) {
            /* This allows us to continue gracefully if
             * there was some kind of bug in the run that output the snapshot.
             * density() below will fix this up.*/
            if(SphP[i].Density <= 0 || !isfinite(SphP[i].Density)) {
                SphP[i].Density = meanbar;
                bad++;
            }
            if(SphP[i].EgyWtDensity <= 0 || !isfinite(SphP[i].EgyWtDensity)) {
                SphP[i].EgyWtDensity = SphP[i].Density;
            }
            double minent = GAMMA_MINUS1 * All.MinEgySpec / pow(SphP[i].Density / a3 , GAMMA_MINUS1);
            if(SphP[i].Entropy < minent)
                SphP[i].Entropy = minent;
        }
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(bad > 0)
            message(0, "Detected bad densities in %d particles on disc\n",bad);
    }

    /*Allocate the extra SPH data for transient SPH particle properties.*/
    struct sph_pred_data sph_pred = slots_allocate_sph_pred_data(SlotsManager->info[0].size);

    /*At the first time step all particles should be active*/
    ActiveParticles act = {0};
    act.ActiveParticle = NULL;
    act.NumActiveParticle = PartManager->NumPart;

    /* Empty kick factors as we do not move*/
    DriftKickTimes times = init_driftkicktime(Ti_Current);
    density(&act, 1, 0, All.BlackHoleOn, 0,  times, &All.CP, &sph_pred, NULL, &Tree);

    /* for clean IC with U input only, we need to iterate to find entrpoy */
    if(RestartSnapNum == -1)
    {
        double u_init = (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.InitGasTemp;
        u_init *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;	/* unit conversion */

        double molecular_weight;
        if(All.InitGasTemp > 1.0e4)	/* assuming FULL ionization */
            molecular_weight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
        else				/* assuming NEUTRAL GAS */
            molecular_weight = 4 / (1 + 3 * HYDROGEN_MASSFRAC);

        u_init /= molecular_weight;
        if(u_init < All.MinEgySpec)
            u_init = All.MinEgySpec;
        /* snapshot already has EgyWtDensity; hope it is read in correctly.
         * (need a test on this!) */
        if(DensityIndependentSphOn()) {
            setup_density_indep_entropy(&act, &Tree, &sph_pred, u_init, a3, Ti_Current);
        }
        else {
           /*Initialize to initial energy*/
            #pragma omp parallel for
            for(i = 0; i < SlotsManager->info[0].size; i++)
                SphP[i].Entropy = GAMMA_MINUS1 * u_init / pow(SphP[i].Density / a3 , GAMMA_MINUS1);
        }
    }
    slots_free_sph_pred_data(&sph_pred);
    force_tree_free(&Tree);
}
