/*Simple test for the exchange function*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#define qsort_openmp qsort

#include <libgadget/fof.h>
#include <libgadget/walltime.h>
#include <libgadget/domain.h>
#include <libgadget/forcetree.h>
#include <libgadget/partmanager.h>
#include "stub.h"

static struct ClockTable CT;

#define NUMPART1 8
static int
setup_particles(int NumPart, double BoxSize)
{

    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, 0);

    particle_alloc_memory(1.5 * NumPart);
    PartManager->NumPart = NumPart;

    int i;
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i ++) {
        P[i].ID = i + PartManager->NumPart * ThisTask;
        /* DM only*/
        P[i].Type = 1;
        P[i].Mass = 1;
        P[i].TimeBin = 0;
        P[i].IsGarbage = 0;
        int j;
        for(j=0; j<3; j++)
            P[i].Pos[j] = BoxSize * gsl_rng_uniform(r);
        P[i].Key = PEANO(P[i].Pos, BoxSize);

    }

    gsl_rng_free(r);
    /* TODO: Here create particles in some halo-like configuration*/

    return 0;
}

static int
teardown_particles(void **state)
{
    myfree(P);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}


static void
test_fof(void **state)
{
    walltime_init(&CT);

    struct DomainParams dp = {0};
    dp.DomainOverDecompositionFactor = 1;
    dp.DomainUseGlobalSorting = 0;
    dp.TopNodeAllocFactor = 1.;
    dp.SetAsideFactor = 1;
    set_domain_par(dp);
    init_forcetree_params(2);

    int NumPart = 1024;
    /* 20000 kpc*/
    double BoxSize = 20000;
    setup_particles(NumPart, BoxSize);

    int i, NTask, ThisTask;
    /* Build a tree and domain decomposition*/
    DomainDecomp ddecomp = {0};
    domain_decompose_full(&ddecomp);
    ForceTree Tree = {0};
    force_tree_rebuild(&Tree, &ddecomp, BoxSize, 0, 0, NULL);

    FOFGroups fof = fof_fof(&Tree, MPI_COMM_WORLD);

    /* Example assertion: this checks that the address of a struct is not NULL,
     * which will always pass! */
    assert_all_true(&fof);

    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

    /* Assert some more things about the particles,
     * maybe checking the halo properties*/
    for(i = 0; i < PartManager->NumPart; i ++) {
        assert_true(P[i].ID % NTask == ThisTask);
    }

    fof_finish(&fof);

    force_tree_free(&Tree);

    teardown_particles(state);
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fof),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
