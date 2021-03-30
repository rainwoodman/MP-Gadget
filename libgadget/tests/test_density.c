/*Simple test for the tree building functions*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#include <libgadget/partmanager.h>
#include <libgadget/walltime.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/density.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/timestep.h>
#include <libgadget/gravity.h>

#include "stub.h"

static double BoxSize;
/* The true struct for the state variable*/
struct density_testdata
{
    struct sph_pred_data sph_pred;
    DomainDecomp ddecomp;
    struct density_params dp;
    gsl_rng * r;
};

static void set_init_hsml(ForceTree * Tree)
{
    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int no = force_get_father(i, Tree);

        double DesNumNgb = GetNumNgb(GetDensityKernelType());
        while(10 * DesNumNgb * P[i].Mass > Tree->Nodes[no].mom.mass)
        {
            int p = force_get_father(no, Tree);

            if(p < 0)
                break;

            no = p;
        }

        P[i].Hsml =
            pow(3.0 / (4 * M_PI) * DesNumNgb * P[i].Mass / (Tree->Nodes[no].mom.mass),
                    1.0 / 3) * Tree->Nodes[no].len;
    }
}

/* Perform some simple checks on the densities*/
static void check_densities(double MinGasHsml)
{
    int i;
    double maxHsml=P[0].Hsml, minHsml= P[0].Hsml;
    #pragma omp parallel for reduction(min:minHsml) reduction(max:maxHsml)
    for(i=0; i<PartManager->NumPart; i++) {
        assert_true(isfinite(P[i].Hsml));
        assert_true(isfinite(SPHP(i).Density));
        assert_true(SPHP(i).Density > 0);
        if(P[i].Hsml < minHsml)
            minHsml = P[i].Hsml;
        if(P[i].Hsml > maxHsml)
            maxHsml = P[i].Hsml;
    }
    assert_true(isfinite(minHsml));
    assert_true(minHsml >= MinGasHsml);
    assert_true(maxHsml <= BoxSize);

}

static void do_density_test(void ** state, const int numpart, double expectedhsml, double hsmlerr)
{
    int i, npbh=0;
    #pragma omp parallel for reduction(+: npbh)
    for(i=0; i<numpart; i++) {
        int j;
        P[i].Key = PEANO(P[i].Pos, BoxSize);
        P[i].Mass = 1;
        P[i].TimeBin = 0;
        P[i].Ti_drift = 0;
        for(j=0; j<3; j++)
            P[i].Vel[j] = 1.5;
        if(P[i].Type == 0) {
            SPHP(i).Entropy = 1;
            SPHP(i).DtEntropy = 0;
            SPHP(i).Density = 1;
        }
        if(P[i].Type == 5)
            npbh++;
    }

    SlotsManager->info[0].size = numpart-npbh;
    SlotsManager->info[5].size = npbh;
    PartManager->NumPart = numpart;
    ActiveParticles act = {0};
    act.NumActiveParticle = numpart;
    act.ActiveParticle = NULL;
    struct density_testdata * data = * (struct density_testdata **) state;
    DomainDecomp ddecomp = data->ddecomp;
    ddecomp.TopLeaves[0].topnode = PartManager->MaxPart;

    ForceTree tree = {0};
    force_tree_rebuild(&tree, &ddecomp, BoxSize, 0, 1, NULL);
    set_init_hsml(&tree);
    /* Rebuild without moments to check it works*/
    force_tree_rebuild(&tree, &ddecomp, BoxSize, 0, 0, NULL);
    /*Time doing the density finding*/
    double start, end;
    start = MPI_Wtime();
    /*Find the density*/
    DriftKickTimes kick = {0};
    Cosmology CP = {0};
    CP.CMBTemperature = 2.7255;
    CP.Omega0 = 0.3;
    CP.OmegaLambda = 1- CP.Omega0;
    CP.OmegaBaryon = 0.045;
    CP.HubbleParam = 0.7;
    CP.RadiationOn = 0;
    CP.w0_fld = -1; /*Dark energy equation of state parameter*/
    /*Should be 0.1*/
    CP.Hubble = 0.1;
    /*Default value for L=kpc v=km/s*/
    double UnitTime_in_s = 3.08568e+16;
    double UnitLength_in_cm = 3.085678e+21;
    double UnitMass_in_g = 1.989e+43;
    /*Do the main cosmology initialisation*/
    init_cosmology(&CP,0.01,UnitLength_in_cm,UnitMass_in_g,UnitTime_in_s);

    density(&act, 1, 0, 0, 0, kick, &CP, &data->sph_pred, NULL, &tree);
    end = MPI_Wtime();
    double ms = (end - start)*1000;
    message(0, "Found densities in %.3g ms\n", ms);
    check_densities(data->dp.MinGasHsmlFractional);

    double avghsml = 0;
    #pragma omp parallel for reduction(+:avghsml)
    for(i=0; i<numpart; i++) {
        avghsml += P[i].Hsml;
    }
    message(0, "Average Hsml: %g Expected %g +- %g\n",avghsml/numpart, expectedhsml, hsmlerr);
    assert_true(fabs(avghsml/numpart - expectedhsml) < hsmlerr);
    /* Make MaxNumNgbDeviation smaller and check we get a consistent result.*/
    double * Hsml = mymalloc2("Hsml", numpart * sizeof(double));
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        Hsml[i] = P[i].Hsml;
    }
    data->dp.MaxNumNgbDeviation = 0.5;
    set_densitypar(data->dp);

    start = MPI_Wtime();
    /*Find the density*/
    density(&act, 1, 0, 0, 0, kick, &CP, &data->sph_pred, NULL, &tree);
    end = MPI_Wtime();
    ms = (end - start)*1000;
    message(0, "Found 1 dev densities in %.3g ms\n", ms);
    double diff = 0;
    double DesNumNgb = GetNumNgb(GetDensityKernelType());
    /* Free tree before checks so that we still recover if checks fail*/
    force_tree_free(&tree);

    #pragma omp parallel for reduction(max:diff)
    for(i=0; i<numpart; i++) {
        assert_true(fabs(Hsml[i]/P[i].Hsml-1) < data->dp.MaxNumNgbDeviation / DesNumNgb);
        if(fabs(Hsml[i] - P[i].Hsml) > diff)
            diff = fabs(Hsml[i] - P[i].Hsml);
    }
    message(0, "Max diff between Hsml: %g\n",diff);
    myfree(Hsml);

    check_densities(data->dp.MinGasHsmlFractional);
}

static void test_density_flat(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    /* Create a regular grid of particles, 8x8x8, all of type 1,
     * in a box 8 kpc across.*/
    int i;
    #pragma omp parallel for
    for(i=0; i<numpart; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = 1.5*BoxSize/cbrt(numpart);
        P[i].Pos[0] = (BoxSize/ncbrt) * (i/ncbrt/ncbrt);
        P[i].Pos[1] = (BoxSize/ncbrt) * ((i/ncbrt) % ncbrt);
        P[i].Pos[2] = (BoxSize/ncbrt) * (i % ncbrt);
    }
    do_density_test(state, numpart, 0.501747, 1e-4);
}

static void test_density_close(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 32;
    int numpart = ncbrt*ncbrt*ncbrt;
    double close = 500.;
    int i;
    /* A few particles scattered about the place so the tree is not sparse*/
    #pragma omp parallel for
    for(i=0; i<numpart/4; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = 4*BoxSize/cbrt(numpart/8);
        P[i].Pos[0] = (BoxSize/ncbrt) * (i/(ncbrt/2.)/(ncbrt/2.));
        P[i].Pos[1] = (BoxSize/ncbrt) * ((i*2/ncbrt) % (ncbrt/2));
        P[i].Pos[2] = (BoxSize/ncbrt) * (i % (ncbrt/2));
    }

    /* Create particles clustered in one place, all of type 0.*/
    #pragma omp parallel for
    for(i=numpart/4; i<numpart; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = 2*ncbrt/close;
        P[i].Pos[0] = 4.1 + (i/ncbrt/ncbrt)/close;
        P[i].Pos[1] = 4.1 + ((i/ncbrt) % ncbrt) /close;
        P[i].Pos[2] = 4.1 + (i % ncbrt)/close;
    }
    P[numpart-1].Type = 5;
    P[numpart-1].PI = 0;

    do_density_test(state, numpart, 0.131726, 1e-4);
}

void do_random_test(void **state, gsl_rng * r, const int numpart)
{
    /* Create a randomly space set of particles, 8x8x8, all of type 0. */
    int i;
    for(i=0; i<numpart/4; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = BoxSize/cbrt(numpart);

        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize * gsl_rng_uniform(r);
    }
    for(i=numpart/4; i<3*numpart/4; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize/2 + BoxSize/8 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    for(i=3*numpart/4; i<numpart; i++) {
        P[i].Type = 0;
        P[i].PI = i;
        P[i].Hsml = BoxSize/cbrt(numpart);
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize*0.1 + BoxSize/32 * exp(pow(gsl_rng_uniform(r)-0.5,2));
    }
    do_density_test(state, numpart, 0.187515, 1e-3);
}

static void test_density_random(void ** state) {
    /*Set up the particle data*/
    int ncbrt = 32;
    struct density_testdata * data = * (struct density_testdata **) state;
    gsl_rng * r = (gsl_rng *) data->r;
    int numpart = ncbrt*ncbrt*ncbrt;
    /*Allocate tree*/
    /*Base pointer*/
    int i;
    for(i=0; i<2; i++) {
        do_random_test(state, r, numpart);
    }
}


/*Make a simple trivial domain for all data on a single processor*/
void trivial_domain(DomainDecomp * ddecomp)
{
    /* The whole tree goes into one topnode.
     * Set up just enough of the TopNode structure that
     * domain_get_topleaf works*/
    ddecomp->domain_allocated_flag = 1;
    ddecomp->NTopNodes = 1;
    ddecomp->NTopLeaves = 1;
    ddecomp->TopNodes = mymalloc("topnode", sizeof(struct topnode_data));
    ddecomp->TopNodes[0].Daughter = -1;
    ddecomp->TopNodes[0].Leaf = 0;
    ddecomp->TopLeaves = mymalloc("topleaf",sizeof(struct topleaf_data));
    ddecomp->TopLeaves[0].Task = 0;
    ddecomp->TopLeaves[0].topnode = 0;
    /*These are not used*/
    ddecomp->TopNodes[0].StartKey = 0;
    ddecomp->TopNodes[0].Shift = BITS_PER_DIMENSION * 3;
    /*To tell the code we are in serial*/
    ddecomp->Tasks = mymalloc("task",sizeof(struct task_data));
    ddecomp->Tasks[0].StartLeaf = 0;
    ddecomp->Tasks[0].EndLeaf = 1;
}

static int teardown_density(void **state) {
    struct density_testdata * data = (struct density_testdata * ) *state;
    slots_free_sph_pred_data(&data->sph_pred);
    myfree(data->ddecomp.Tasks);
    myfree(data->ddecomp.TopLeaves);
    myfree(data->ddecomp.TopNodes);
    free(data->r);
    myfree(data);
    return 0;
}

static struct ClockTable CT;

static int setup_density(void **state) {
    /* Needed so the integer timeline works*/
    setup_sync_points(NULL,0.01, 0.1, 0, 0.0, 0.0, 0.0, 0.0, 0);

    /*Reserve space for the slots*/
    slots_init(0.01 * PartManager->MaxPart, SlotsManager);
    slots_set_enabled(0, sizeof(struct sph_particle_data), SlotsManager);
    int64_t atleast[6] = {0};
    atleast[0] = pow(32,3);
    atleast[5] = 2;
    int64_t maxpart = 0;
    int i;
    for(i = 0; i < 6; i++)
        maxpart+=atleast[i];
    particle_alloc_memory(maxpart);
    slots_reserve(1, atleast, SlotsManager);
    walltime_init(&CT);
    init_forcetree_params(2);
    struct density_testdata *data = mymalloc("data", sizeof(struct density_testdata));
    data->sph_pred = slots_allocate_sph_pred_data(maxpart);
    /*Set up the top-level domain grid*/
    trivial_domain(&data->ddecomp);
    data->dp.DensityResolutionEta = 1.;
    data->dp.BlackHoleNgbFactor = 2;
    data->dp.MaxNumNgbDeviation = 2;
    data->dp.DensityKernelType = DENSITY_KERNEL_CUBIC_SPLINE;
    BoxSize = 8;
    data->dp.MinGasHsmlFractional = 0.006;
    struct gravshort_tree_params tree_params = {0};
    tree_params.FractionalGravitySoftening = 1;
    set_gravshort_treepar(tree_params);

    gravshort_set_softenings(1);
    data->dp.BlackHoleMaxAccretionRadius = 99999.;

    set_densitypar(data->dp);
    data->r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(data->r, 0);
    *state = (void *) data;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_density_flat),
        cmocka_unit_test(test_density_close),
        cmocka_unit_test(test_density_random),
    };
    return cmocka_run_group_tests_mpi(tests, setup_density, teardown_density);
}
