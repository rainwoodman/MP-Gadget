#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <gsl/gsl_math.h>
#include <inttypes.h>
#include <omp.h>

#include "utils.h"
#include "utils/mpsort.h"

#include "walltime.h"
#include "sfr_eff.h"
#include "blackhole.h"
#include "domain.h"
#include "winds.h"

#include "forcetree.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "partmanager.h"
#include "densitykernel.h"

/*! \file fof.c
 *  \brief parallel FoF group finder
 */

#include "fof.h"

/* Never change the primary link it is always DM. */
#define FOF_PRIMARY_LINK_TYPES 2

/* FIXME: convert this to a parameter */
#define FOF_SECONDARY_LINK_TYPES (1+16+32)    // 2^type for the types linked to nearest primaries
#define LARGE 1e29
#define MAXITER 400

struct FOFParams
{
    int FOFSaveParticles ; /* saving particles in the fof group */
    double MinFoFMassForNewSeed;	/* Halo mass required before new seed is put in */
    double MinMStarForNewSeed; /* Minimum stellar mass required before new seed */
    double FOFHaloLinkingLength;
    double FOFHaloComovingLinkingLength; /* in code units */
    int FOFHaloMinLength;
} fof_params;

/*Set the parameters of the BH module*/
void set_fof_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        fof_params.FOFSaveParticles = param_get_int(ps, "FOFSaveParticles");
        fof_params.FOFHaloLinkingLength = param_get_double(ps, "FOFHaloLinkingLength");
        fof_params.FOFHaloMinLength = param_get_int(ps, "FOFHaloMinLength");
        fof_params.MinFoFMassForNewSeed = param_get_double(ps, "MinFoFMassForNewSeed");
        fof_params.MinMStarForNewSeed = param_get_double(ps, "MinMStarForNewSeed");
    }
    MPI_Bcast(&fof_params, sizeof(struct FOFParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void fof_init(double DMMeanSeparation)
{
    fof_params.FOFHaloComovingLinkingLength = fof_params.FOFHaloLinkingLength * DMMeanSeparation;
}

static double fof_periodic(double x, double BoxSize)
{
    if(x >= 0.5 * BoxSize)
        x -= BoxSize;
    if(x < -0.5 * BoxSize)
        x += BoxSize;
    return x;
}


static double fof_periodic_wrap(double x, double BoxSize)
{
    while(x >= BoxSize)
        x -= BoxSize;
    while(x < 0)
        x += BoxSize;
    return x;
}

static void fof_label_secondary(ForceTree * tree);
static int fof_compare_HaloLabel_MinID(const void *a, const void *b);
static int _fof_compare_Group_MinIDTask_ThisTask;
static int fof_compare_Group_MinIDTask(const void *a, const void *b);
static int fof_compare_Group_OriginalIndex(const void *a, const void *b);
static int fof_compare_Group_MinID(const void *a, const void *b);
static void fof_reduce_groups(
    void * groups,
    int nmemb,
    size_t elsize,
    void (*reduce_group)(void * gdst, void * gsrc), MPI_Comm Comm);

static void fof_finish_group_properties(FOFGroups * fof, double BoxSize);

static int fof_compile_base(struct BaseGroup * base, int NgroupsExt, MPI_Comm Comm);
static void fof_compile_catalogue(FOFGroups * fof, const int NgroupsExt, double BoxSize, MPI_Comm Comm);

static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm);

void fof_label_primary(ForceTree * tree, MPI_Comm Comm);
extern void fof_save_particles(FOFGroups * fof, int num, int SaveParticles, MPI_Comm Comm);

typedef struct {
    TreeWalkQueryBase base;
    MyFloat Hsml;
    MyIDType MinID;
    int MinIDTask;
    int pad;
} TreeWalkQueryFOF;

typedef struct {
    TreeWalkResultBase base;
    MyFloat Distance;
    MyIDType MinID;
    int MinIDTask;
    int pad;
} TreeWalkResultFOF;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterFOF;

static struct fof_particle_list
{
    MyIDType MinID;
    int MinIDTask;
    int Pindex;
}
*HaloLabel;

static MPI_Datatype MPI_TYPE_GROUP;

/*
 * The FOF finder will produce Group[], which is allocated to the top side of the
 * main heap.
 *
 **/

FOFGroups
fof_fof(ForceTree * tree, MPI_Comm Comm)
{
    int i;

    message(0, "Begin to compute FoF group catalogues. (allocated: %g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/Misc");

    message(0, "Comoving linking length: %g\n", fof_params.FOFHaloComovingLinkingLength);

    HaloLabel = (struct fof_particle_list *) mymalloc("HaloLabel", PartManager->NumPart * sizeof(struct fof_particle_list));

    /* HaloLabel stores the MinID and MinIDTask of particles, this pair serves as a halo label. */
    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++) {
        HaloLabel[i].Pindex = i;
    }
    /* Fill FOFP_List of primary */
    fof_label_primary(tree, Comm);

    MPIU_Barrier(Comm);
    message(0, "Group finding done.\n");
    walltime_measure("/FOF/Primary");

    /* Fill FOFP_List of secondary */
    fof_label_secondary(tree);
    MPIU_Barrier(Comm);
    message(0, "Attached gas and star particles to nearest dm particles.\n");

    walltime_measure("/FOF/Secondary");

    /* sort HaloLabel according to MinID, because we need that for compiling catalogues */
    qsort_openmp(HaloLabel, PartManager->NumPart, sizeof(struct fof_particle_list), fof_compare_HaloLabel_MinID);

    int NgroupsExt = 0;

    for(i = 0; i < PartManager->NumPart; i ++) {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) NgroupsExt ++;
    }

    /* The first round is to eliminate groups that are too short. */
    /* We create the smaller 'BaseGroup' data set for this. */
    struct BaseGroup * base = (struct BaseGroup *) mymalloc("BaseGroup", sizeof(struct BaseGroup) * NgroupsExt);

    NgroupsExt = fof_compile_base(base, NgroupsExt, Comm);

    MPIU_Barrier(Comm);
    message(0, "Compiled local group data and catalogue.\n");

    walltime_measure("/FOF/Compile");

    fof_assign_grnr(base, NgroupsExt, Comm);

    /*Initialise the Group object from the BaseGroup*/
    FOFGroups fof;
    MPI_Type_contiguous(sizeof(fof.Group[0]), MPI_BYTE, &MPI_TYPE_GROUP);
    MPI_Type_commit(&MPI_TYPE_GROUP);

    fof.Group = fof_alloc_group(base, NgroupsExt);

    myfree(base);

    fof_compile_catalogue(&fof, NgroupsExt, tree->BoxSize, Comm);

    MPIU_Barrier(Comm);
    message(0, "Finished FoF. Group properties are now allocated.. (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/FOF/Prop");

    myfree(HaloLabel);

    return fof;
}

void
fof_finish(FOFGroups * fof)
{
    myfree(fof->Group);

    message(0, "Finished computing FoF groups.  (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    walltime_measure("/FOF/MISC");

    MPI_Type_free(&MPI_TYPE_GROUP);
}

struct FOFPrimaryPriv {
    int * Head;
    struct SpinLocks * spin;
    char * PrimaryActive;
    MyIDType * OldMinID;
};
#define FOF_PRIMARY_GET_PRIV(tw) ((struct FOFPrimaryPriv *) (tw->priv))

/* This function walks the particle tree starting at particle i until it reaches
 * a particle which has Head[i] = i, the root node (particles are initialised in
 * this state, so this is equivalent to finding a particle which has yet to be merged).
 * Once it reaches a root, it returns that particle number.
 * Arguments:
 *
 * stop: When this particle is reached, return -1. We use this to find an already merged tree.
 *
 * Returns:
 *      root particle if found
 *      -1 if stop particle reached
 */
static int
HEADl(int stop, int i, const int * const Head)
{
    int next = i;

    do {
        i = next;
        /* Reached stop, return*/
        if(i == stop)
            return -1;
        /* atomic read because we may change
         * this in update_root: not necessary on x86_64, but avoids tears elsewhere*/
        #pragma omp atomic read
        next = Head[i];
    } while(next != i);

    /* return unmerged particle*/
    return i;
}

/* Rewrite a tree so that all values in it point directly to the true root.
 * This means that the trees are O(1) deep and speeds up future accesses.
 * See https://arxiv.org/abs/1607.03224 */
static void
update_root(int i, const int r, int * Head)
{
    int t = i;
    do {
        i = t;
        #pragma omp atomic capture
        {
            t = Head[i];
            Head[i]= r;
        }
        /* Stop if we reached the top (new head is the same as the old)
         * or if the new head is less than or equal to the desired head, indicating
         * another thread changed us*/
    } while(t != i && (t > r));
}

/* Find the current head particle by walking the tree. No updates are done
 * so this can be performed from a threaded context. */
static int
HEAD(int i, const int * const Head)
{
    int r = i;
    while(Head[r] != r) {
        r = Head[r];
    }
    return r;
}

static void fof_primary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {
    /* The copied data is *only* used for the
     * secondary treewalk, so fill up garbage for the primary treewalk.
     * The copy is a technical race otherwise. */
    if(I->base.NodeList[0] == tw->tree->firstnode) {
        I->MinID = -1;
        I->MinIDTask = -1;
        return;
    }
    /* Secondary treewalk, no need for locking here*/
    int head = HEAD(place, FOF_PRIMARY_GET_PRIV(tw)->Head);
    I->MinID = HaloLabel[head].MinID;
    I->MinIDTask = HaloLabel[head].MinIDTask;
}

static int fof_primary_haswork(int n, TreeWalk * tw) {
    if(P[n].IsGarbage || P[n].Swallowed)
        return 0;
    return (((1 << P[n].Type) & (FOF_PRIMARY_LINK_TYPES))) && FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[n];
}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv);

void fof_label_primary(ForceTree * tree, MPI_Comm Comm)
{
    int i;
    int64_t link_across;
    int64_t link_across_tot;
    double t0, t1;
    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    message(0, "Start linking particles (presently allocated=%g MB)\n", mymalloc_usedbytes() / (1024.0 * 1024.0));

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "FOF_FIND_GROUPS";
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_primary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);

    tw->haswork = fof_primary_haswork;
    tw->fill = (TreeWalkFillQueryFunction) fof_primary_copy;
    tw->reduce = NULL;
    tw->type = TREEWALK_ALL;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);
    tw->tree = tree;
    struct FOFPrimaryPriv priv[1];
    tw->priv = priv;

    FOF_PRIMARY_GET_PRIV(tw)->Head = (int*) mymalloc("FOF_Links", PartManager->NumPart * sizeof(int));
    FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive = (char*) mymalloc("FOFActive", PartManager->NumPart * sizeof(char));
    FOF_PRIMARY_GET_PRIV(tw)->OldMinID = (MyIDType *) mymalloc("FOFActive", PartManager->NumPart * sizeof(MyIDType));

    /* allocate buffers to arrange communication */

    t0 = second();

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        FOF_PRIMARY_GET_PRIV(tw)->Head[i] = i;
        FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i]= P[i].ID;
        FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 1;

        HaloLabel[i].MinID = P[i].ID;
        HaloLabel[i].MinIDTask = ThisTask;
    }

    /* The lock is used to protect MinID*/
    priv[0].spin = init_spinlocks(PartManager->NumPart);
    do
    {
        t0 = second();

        treewalk_run(tw, NULL, PartManager->NumPart);

        t1 = second();
        /* This sets the MinID of the head particle to the minimum ID
         * of the child particles. We set this inside the treewalk,
         * but the locking allows a race, where the particle with MinID set
         * is no longer the one which is the true Head of the group.
         * So we must check it again here.*/
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++) {
            int head = HEAD(i, FOF_PRIMARY_GET_PRIV(tw)->Head);
            /* Don't check against ourself*/
            if(head == i)
                continue;
            MyIDType headminid;
            #pragma omp atomic read
            headminid = HaloLabel[head].MinID;
            /* No atomic needed for i as this is not a head*/
            if(headminid > HaloLabel[i].MinID) {
                lock_spinlock(head, priv->spin);
                if(HaloLabel[head].MinID > HaloLabel[i].MinID) {
                    #pragma omp atomic write
                    HaloLabel[head].MinID = HaloLabel[i].MinID;
                    HaloLabel[head].MinIDTask = HaloLabel[i].MinIDTask;
                }
                unlock_spinlock(head, priv->spin);
            }
        }
        /* let's check out which particles have changed their MinID,
         * mark them for next round. */
        link_across = 0;
#pragma omp parallel for reduction(+: link_across)
        for(i = 0; i < PartManager->NumPart; i++) {
            int head = HEAD(i, FOF_PRIMARY_GET_PRIV(tw)->Head);
            /* This loop sets the MinID of the children to the minID of the head.
             * The minID of the head is set above and is stable at this point.*/
            if(i != head) {
                HaloLabel[i].MinID = HaloLabel[head].MinID;
                HaloLabel[i].MinIDTask = HaloLabel[head].MinIDTask;
            }
            MyIDType newMinID = HaloLabel[head].MinID;
            if(newMinID != FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i]) {
                FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 1;
                link_across ++;
            } else {
                FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive[i] = 0;
            }
            FOF_PRIMARY_GET_PRIV(tw)->OldMinID[i] = newMinID;
        }
        MPI_Allreduce(&link_across, &link_across_tot, 1, MPI_INT64, MPI_SUM, Comm);
        message(0, "Linked %ld particles %g seconds\n", link_across_tot, t1 - t0);
    }
    while(link_across_tot > 0);

    free_spinlocks(priv[0].spin);

    message(0, "Local groups found.\n");

    myfree(FOF_PRIMARY_GET_PRIV(tw)->OldMinID);
    myfree(FOF_PRIMARY_GET_PRIV(tw)->PrimaryActive);
    myfree(FOF_PRIMARY_GET_PRIV(tw)->Head);
}

static void
fofp_merge(int target, int other, TreeWalk * tw)
{
    /* this will lock h1 */
    int * Head = FOF_PRIMARY_GET_PRIV(tw)->Head;
    int h1, h2;
    do {
        h1 = HEADl(-1, target, Head);
        /* Done if we find h1 along the path
         * (because other is already in the same halo) */
        h2 = HEADl(h1, other, Head);
        if(h2 < 0)
            return;
        /* Ensure that we always merge to the lower entry.
         * This avoids circular loops in the Head entries:
         * a -> b -> a */
        if(h1 > h2) {
            int tmp = h2;
            h2 = h1;
            h1 = tmp;
        }
     /* Atomic compare exchange to make h2 a subtree of h1.
      * Set Head[h2] = h1 iff Head[h2] is still h2. Otherwise loop.*/
    } while(!__atomic_compare_exchange(&Head[h2], &h2, &h1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED));

    struct SpinLocks * spin = FOF_PRIMARY_GET_PRIV(tw)->spin;

    /* update MinID of h1: h2 is now just another child of h1
     * so we don't need to check that h2 changes its head.
     * It might happen that h1 is added to another halo at this point
     * and the addition gets the wrong MinID.
     * For this reason we recompute the MinIDs after the main treewalk.
     * We also lock h2 for a copy in case it is the h1 in another thread,
     * and may have inconsistent MinID and MinIDTask.*/

    /* Get a copy of h2 under the lock, which ensures
     * that MinID and MinIDTask do not change independently. */
    struct fof_particle_list h2label;
    lock_spinlock(h2, spin);
    h2label.MinID = HaloLabel[h2].MinID;
    h2label.MinIDTask = HaloLabel[h2].MinIDTask;
    unlock_spinlock(h2, spin);

    /* Now lock h1 so we don't change MinID but not MinIDTask.*/
    lock_spinlock(h1, spin);
    if(HaloLabel[h1].MinID > h2label.MinID)
    {
        HaloLabel[h1].MinID = h2label.MinID;
        HaloLabel[h1].MinIDTask = h2label.MinIDTask;
    }
    unlock_spinlock(h1, spin);

    /* h1 must be the root of other and target both:
     * do the splay to speed up future accesses.
     * We do not need to have h2 locked, because h2 is
     * now just another child of h1: these do not change the root,
     * they make the tree shallow.*/
    update_root(target, h1, Head);
    update_root(other, h1, Head);

}

static void
fof_primary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    TreeWalk * tw = lv->tw;
    if(iter->base.other == -1) {
        iter->base.Hsml = fof_params.FOFHaloComovingLinkingLength;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.mask = FOF_PRIMARY_LINK_TYPES;
        return;
    }
    int other = iter->base.other;

    if(lv->mode == 0) {
        /* Local FOF */
        if(lv->target <= other) {
            // printf("locked merge %d %d by %d\n", lv->target, other, omp_get_thread_num());
            fofp_merge(lv->target, other, tw);
        }
    }
    else /* mode is 1, target is a ghost */
    {
        int head = HEAD(other, FOF_PRIMARY_GET_PRIV(tw)->Head);
        struct SpinLocks * spin = FOF_PRIMARY_GET_PRIV(tw)->spin;
//        printf("locking %d by %d in ngbiter\n", other, omp_get_thread_num());
        lock_spinlock(head, spin);
        if(HaloLabel[head].MinID > I->MinID)
        {
            HaloLabel[head].MinID = I->MinID;
            HaloLabel[head].MinIDTask = I->MinIDTask;
        }
//        printf("unlocking %d by %d in ngbiter\n", other, omp_get_thread_num());
        unlock_spinlock(head, spin);
    }
}

static void fof_reduce_base_group(void * pdst, void * psrc) {
    struct BaseGroup * gdst = pdst;
    struct BaseGroup * gsrc = psrc;
    gdst->Length += gsrc->Length;
    /* preserve the dst FirstPos so all other base group gets the same FirstPos */
}

static void fof_reduce_group(void * pdst, void * psrc) {
    struct Group * gdst = pdst;
    struct Group * gsrc = psrc;
    int j;
    gdst->Length += gsrc->Length;
    gdst->Mass += gsrc->Mass;

    for(j = 0; j < 6; j++)
    {
        gdst->LenType[j] += gsrc->LenType[j];
        gdst->MassType[j] += gsrc->MassType[j];
    }

    gdst->Sfr += gsrc->Sfr;
    gdst->GasMetalMass += gsrc->GasMetalMass;
    gdst->StellarMetalMass += gsrc->StellarMetalMass;
    for(j = 0; j < NMETALS; j++) {
        gdst->GasMetalElemMass[j] += gsrc->GasMetalElemMass[j];
        gdst->StellarMetalElemMass[j] += gsrc->StellarMetalElemMass[j];
    }
    gdst->BH_Mdot += gsrc->BH_Mdot;
    gdst->BH_Mass += gsrc->BH_Mass;
    if(gsrc->MaxDens > gdst->MaxDens)
    {
        gdst->MaxDens = gsrc->MaxDens;
        gdst->seed_index = gsrc->seed_index;
        gdst->seed_task = gsrc->seed_task;
    }

    int d1, d2;
    for(d1 = 0; d1 < 3; d1++)
    {
        gdst->CM[d1] += gsrc->CM[d1];
        gdst->Vel[d1] += gsrc->Vel[d1];
        gdst->Jmom[d1] += gsrc->Jmom[d1];
        for(d2 = 0; d2 < 3; d2 ++) {
            gdst->Imom[d1][d2] += gsrc->Imom[d1][d2];
        }
    }

}

static void add_particle_to_group(struct Group * gdst, int i, double BoxSize, int ThisTask) {

    /* My local number of particles contributing to the full catalogue. */
    const int index = i;
    if(gdst->Length == 0) {
        struct BaseGroup base = gdst->base;
        memset(gdst, 0, sizeof(gdst[0]));
        gdst->base = base;
        gdst->seed_index = gdst->seed_task = -1;
    }

    gdst->Length ++;
    gdst->Mass += P[index].Mass;
    gdst->LenType[P[index].Type]++;
    gdst->MassType[P[index].Type] += P[index].Mass;

    if(P[index].Type == 0) {
        gdst->Sfr += SPHP(index).Sfr;
        gdst->GasMetalMass += SPHP(index).Metallicity * P[index].Mass;
        int j;
        for(j = 0; j < NMETALS; j++)
            gdst->GasMetalElemMass[j] += SPHP(index).Metals[j] * P[index].Mass;
    }
    if(P[index].Type == 4) {
        int j;
        gdst->StellarMetalMass += STARP(index).Metallicity * P[index].Mass;
        for(j = 0; j < NMETALS; j++)
            gdst->StellarMetalElemMass[j] += STARP(index).Metals[j] * P[index].Mass;
    }

    if(P[index].Type == 5)
    {
        gdst->BH_Mdot += BHP(index).Mdot;
        gdst->BH_Mass += BHP(index).Mass;
    }
    /*This used to depend on black holes being enabled, but I do not see why.
     * I think because it is only useful for seeding*/
    /* Don't make bh in wind.*/
    if(P[index].Type == 0 && !winds_is_particle_decoupled(index))
        if(SPHP(index).Density > gdst->MaxDens)
        {
            gdst->MaxDens = SPHP(index).Density;
            gdst->seed_index = index;
            gdst->seed_task = ThisTask;
        }

    int d1, d2;
    double xyz[3];
    double rel[3];
    double vel[3];
    double jmom[3];

    for(d1 = 0; d1 < 3; d1++)
    {
        double first = gdst->base.FirstPos[d1];
        rel[d1] = fof_periodic(P[index].Pos[d1] - first, BoxSize) ;
        xyz[d1] = rel[d1] + first;
        vel[d1] = P[index].Vel[d1];
    }

    crossproduct(rel, vel, jmom);

    for(d1 = 0; d1 < 3; d1++) {
        gdst->CM[d1] += P[index].Mass * xyz[d1];
        gdst->Vel[d1] += P[index].Mass * vel[d1];
        gdst->Jmom[d1] += P[index].Mass * jmom[d1];

        for(d2 = 0; d2 < 3; d2++) {
            gdst->Imom[d1][d2] += P[index].Mass * rel[d1] * rel[d2];
        }
    }
}

static void
fof_finish_group_properties(struct FOFGroups * fof, double BoxSize)
{
    int i;

    for(i = 0; i < fof->Ngroups; i++)
    {
        int d1, d2;
        double cm[3];
        double rel[3];
        double jcm[3];
        double vcm[3];

        struct Group * gdst = &fof->Group[i];
        for(d1 = 0; d1 < 3; d1++)
        {
            gdst->Vel[d1] /= gdst->Mass;
            vcm[d1] = gdst->Vel[d1];
            cm[d1] = gdst->CM[d1] / gdst->Mass;

            rel[d1] = fof_periodic(cm[d1] - gdst->base.FirstPos[d1], BoxSize);

            cm[d1] = fof_periodic_wrap(cm[d1], BoxSize);
            gdst->CM[d1] = cm[d1];

        }
        crossproduct(rel, vcm, jcm);

        for(d1 = 0; d1 < 3; d1 ++) {
            gdst->Jmom[d1] -= jcm[d1] * gdst->Mass;
        }

        for(d1 = 0; d1 < 3; d1 ++) {
            for(d2 = 0; d2 < 3; d2++) {
                /* Parallel Axis theorem:
                 * https://en.wikipedia.org/wiki/Parallel_axis_theorem ;
                 * J was relative to FirstPos, I is relative to CM.
                 *
                 * Note that our definition of Imom follows the astronomy one,
                 *
                 * I_ij = sum x_i x_j (where x_i x_j is relative displacement)
                 * */

                double diff = rel[d1] * rel[d2];

                gdst->Imom[d1][d2] -= gdst->Mass * diff;
            }
        }
    }

}

static int
fof_compile_base(struct BaseGroup * base, int NgroupsExt, MPI_Comm Comm)
{
    memset(base, 0, sizeof(base[0]) * NgroupsExt);

    int i;
    int start;

    start = 0;
    for(i = 0; i < PartManager->NumPart; i++)
    {
        if(i == 0 || HaloLabel[i].MinID != HaloLabel[i - 1].MinID) {
            base[start].MinID = HaloLabel[i].MinID;
            base[start].MinIDTask = HaloLabel[i].MinIDTask;
            int d;
            for(d = 0; d < 3; d ++) {
                base[start].FirstPos[d] = P[HaloLabel[i].Pindex].Pos[d];
            }
            start ++;
        }
    }

    /* count local lengths */
    /* This works because base is sorted by MinID by construction. */
    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= base[i].MinID) break;
        }
        /* count particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != base[i].MinID) {
                break;
            }
            base[i].Length ++;
        }
    }

    /* update global attributes */
    fof_reduce_groups(base, NgroupsExt, sizeof(base[0]), fof_reduce_base_group, Comm);

    /* eliminate all groups that are too small */
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].Length < fof_params.FOFHaloMinLength)
        {
            base[i] = base[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
    }
    return NgroupsExt;
}

/* Allocate memory for and initialise a Group object
 * from a BaseGroup object.*/
static struct Group *
fof_alloc_group(const struct BaseGroup * base, const int NgroupsExt)
{
    int i;
    struct Group * Group = (struct Group *) mymalloc2("Group", sizeof(struct Group) * NgroupsExt);
    memset(Group, 0, sizeof(Group[0]) * NgroupsExt);

    /* copy in the base properties */
    /* at this point base group shall be sorted by MinID */
    #pragma omp parallel for
    for(i = 0; i < NgroupsExt; i ++) {
        Group[i].base = base[i];
    }
    return Group;
}

/* TODO: It would be a good idea to generalise this to arbitrary fof/particle properties */
static void fof_set_escapefraction(struct FOFGroups * fof, const int NgroupsExt)
{
    int i = 0;
    for(i = 0; i < PartManager->NumPart; i++){
        if(P[i].Type == 4){
            STARP(i).EscapeFraction = 0.;	/* will mark particles that are not in any group */
        }
    }

    int start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            int pi = HaloLabel[start].Pindex;

            if(P[pi].GrNr != fof->Group[i].base.GrNr) {
                endrun(3333, "GrNr mismatch\n");
            }

            /* only do stars */
            if(P[pi].Type != 4) continue;

            /* putting halo mass in escape fraction for now, converted before uvbg calculation */ 
            STARP(pi).EscapeFraction = fof->Group[i].Mass;
        }
    }
}

static void
fof_compile_catalogue(struct FOFGroups * fof, const int NgroupsExt, double BoxSize, MPI_Comm Comm)
{
    int i, start, ThisTask;

    MPI_Comm_rank(Comm, &ThisTask);

    start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        /* find the first particle */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID >= fof->Group[i].base.MinID) break;
        }
        /* add particles */
        for(;start < PartManager->NumPart; start++) {
            if(HaloLabel[start].MinID != fof->Group[i].base.MinID) {
                break;
            }
            add_particle_to_group(&fof->Group[i], HaloLabel[start].Pindex, BoxSize, ThisTask);
        }
    }

    /* collect global properties */
    fof_reduce_groups(fof->Group, NgroupsExt, sizeof(fof->Group[0]), fof_reduce_group, Comm);

    /* count Groups and number of particles hosted by me */
    fof->Ngroups = 0;
    int64_t Nids = 0;
    for(i = 0; i < NgroupsExt; i ++) {
        if(fof->Group[i].base.MinIDTask != ThisTask) continue;

        fof->Ngroups++;
        Nids += fof->Group[i].base.Length;

        if(fof->Group[i].base.Length != fof->Group[i].Length) {
            /* These two shall be consistent */
            endrun(3333, "i=%d Group base Length %d != Group Length %d\n", i, fof->Group[i].base.Length, fof->Group[i].Length);
        }
    }

    fof_finish_group_properties(fof, BoxSize);

    /* feed group property back to each particle. */
    /* TODO: put a flag here to only run if we need it */
    fof_set_escapefraction(fof, NgroupsExt);

    int64_t TotNids;
    sumup_large_ints(1, &fof->Ngroups, &fof->TotNgroups);
    MPI_Allreduce(&Nids, &TotNids, 1, MPI_INT64, MPI_SUM, Comm);

    /* report some statistics */
    int largestloc_tot = 0;
    double largestmass_tot= 0;
    if(fof->TotNgroups > 0)
    {
        double largestmass = 0;
        int largestlength = 0;

        for(i = 0; i < NgroupsExt; i++)
            if(fof->Group[i].Length > largestlength) {
                largestlength = fof->Group[i].Length;
                largestmass = fof->Group[i].Mass;
            }
        MPI_Allreduce(&largestlength, &largestloc_tot, 1, MPI_INT, MPI_MAX, Comm);
        MPI_Allreduce(&largestmass, &largestmass_tot, 1, MPI_DOUBLE, MPI_MAX, Comm);
    }

    message(0, "Total number of groups with at least %d particles: %ld\n", fof_params.FOFHaloMinLength, fof->TotNgroups);
    if(fof->TotNgroups > 0)
    {
        message(0, "Largest group has %d particles, mass %g.\n", largestloc_tot, largestmass_tot);
        message(0, "Total number of particles in groups: %012ld\n", TotNids);
    }
}


static void fof_reduce_groups(
    void * groups,
    int nmemb,
    size_t elsize,
    void (*reduce_group)(void * gdst, void * gsrc), MPI_Comm Comm)
{

    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);
    /* slangs:
     *   prime: groups hosted by ThisTask
     *   ghosts: groups that spans into ThisTask but not hosted by ThisTask;
     *           part of the local catalogue
     *   images: ghosts that are sent from another rank.
     *           images are reduced to prime, then the prime attributes
     *           are copied to images, and sent back to the ghosts.
     *
     *   in the begining, prime and ghosts contains local group attributes.
     *   in the end, prime and ghosts all contain full group attributes.
     **/
    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    void * images = NULL;
    void * ghosts = NULL;
    int i;
    int start;

    MPI_Datatype dtype;

    MPI_Type_contiguous(elsize, MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);

    /*Set global data for the comparison*/
    _fof_compare_Group_MinIDTask_ThisTask = ThisTask;
    /* local groups will be moved to the beginning, we skip them with offset */
    qsort_openmp(groups, nmemb, elsize, fof_compare_Group_MinIDTask);
    /* count how many we have of each task */
    memset(Send_count, 0, sizeof(int) * NTask);

    for(i = 0; i < nmemb; i++) {
        struct BaseGroup * gi = (struct BaseGroup *) (((char*) groups) + i * elsize);
        Send_count[gi->MinIDTask]++;
    }

    /* Skip local groups */
    int Nmine = Send_count[ThisTask];
    Send_count[ThisTask] = 0;

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int nimport = 0;
    for(i = 0; i < NTask; i ++) {
        nimport += Recv_count[i];
    }

    images = mymalloc("images", nimport * elsize);
    ghosts = ((char*) groups) + elsize * Nmine;

    MPI_Alltoallv_smart(ghosts, Send_count, NULL, dtype,
                        images, Recv_count, NULL, dtype, Comm);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        gi->OriginalIndex = i;
    }

    /* sort the groups according to MinID */
    qsort_openmp(groups, Nmine, elsize, fof_compare_Group_MinID);
    qsort_openmp(images, nimport, elsize, fof_compare_Group_MinID);

    /* merge the imported ones with the local ones */
    start = 0;
    for(i = 0; i < Nmine; i++) {
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start  * elsize);
            if(image->MinID >= prime->MinID) {
                break;
            }
        }
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID != prime->MinID) {
                break;
            }
            reduce_group(prime, image);
        }
    }

    /* update the images, such that they can be send back to the ghosts */
    start = 0;
    for(i = 0; i < Nmine; i++)
    {
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID >= prime->MinID) {
                break;
            }
        }
        for(;start < nimport; start++) {
            struct BaseGroup * prime = (struct BaseGroup*) ((char*) groups + i * elsize);
            struct BaseGroup * image = (struct BaseGroup*) ((char*) images + start * elsize);
            if(image->MinID != prime->MinID) {
                break;
            }
            int save = image->OriginalIndex;
            memcpy(image, prime, elsize);
            image->OriginalIndex = save;
        }
    }

    /* reset the ordering of imported list, such that it can be properly returned */
    qsort_openmp(images, nimport, elsize, fof_compare_Group_OriginalIndex);

    for(i = 0; i < nimport; i++) {
        struct BaseGroup * gi = (struct BaseGroup*) ((char*) images + i * elsize);
        if(gi->MinIDTask != ThisTask) {
            abort();
        }
    }
    void * ghosts2 = mymalloc("TMP", nmemb * elsize);

    MPI_Alltoallv_smart(images, Recv_count, NULL, dtype,
                        ghosts2, Send_count, NULL, dtype,
                        Comm);
    for(i = 0; i < nmemb - Nmine; i ++) {
        struct BaseGroup * g1 = (struct BaseGroup*) ((char*) ghosts + i * elsize);
        struct BaseGroup * g2 = (struct BaseGroup*) ((char*) ghosts2 + i* elsize);
        if(g1->MinID != g2->MinID) {
            endrun(2, "g1 minID %lu, g2 minID %lu\n", g1->MinID, g2->MinID);
        }
        if(g1->MinIDTask != g2->MinIDTask) {
            endrun(2, "g1 minIDTask %d, g2 minIDTask %d\n", g1->MinIDTask, g2->MinIDTask);
        }
    }
    memcpy(ghosts, ghosts2, elsize * (nmemb - Nmine));
    myfree(ghosts2);

    myfree(images);

    MPI_Type_free(&dtype);

    /* At this point, each Group entry has the reduced attribute of the full group */
    /* And the local groups (MinIDTask == ThisTask) are placed at the begining of the list*/
    ta_free(Recv_count);
    ta_free(Send_count);
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg);
static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg);

static void fof_assign_grnr(struct BaseGroup * base, const int NgroupsExt, MPI_Comm Comm)
{
    int i, j, NTask, ThisTask;
    int64_t ngr;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);

    for(i = 0; i < NgroupsExt; i++)
    {
        base[i].OriginalTask = ThisTask;	/* original task */
    }

    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_TotalCountTaskDiffMinID, 24, NULL, Comm);

    /* assign group numbers
     * at this point, both Group are is sorted by length,
     * and the every time OriginalTask == MinIDTask, a list of ghost base is stored.
     * they shall get the same GrNr.
     * */
    ngr = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        if(base[i].OriginalTask == base[i].MinIDTask) {
            ngr++;
        }
        base[i].GrNr = ngr;
    }

    int64_t * ngra = ta_malloc("NGRA", int64_t, NTask);

    MPI_Allgather(&ngr, 1, MPI_INT64, ngra, 1, MPI_INT64, Comm);

    /* shift to the global grnr. */
    int64_t groffset = 0;
    for(j = 0; j < ThisTask; j++)
        groffset += ngra[j];
    for(i = 0; i < NgroupsExt; i++)
        base[i].GrNr += groffset;

    ta_free(ngra);

    /* bring the group list back into the original task, sorted by MinID */
    mpsort_mpi(base, NgroupsExt, sizeof(base[0]),
            fof_radix_Group_OriginalTaskMinID, 16, NULL, Comm);

    for(i = 0; i < PartManager->NumPart; i++)
        P[i].GrNr = -1;	/* will mark particles that are not in any group */

    int start = 0;
    for(i = 0; i < NgroupsExt; i++)
    {
        for(;start < PartManager->NumPart; start++) {
            if (HaloLabel[start].MinID >= base[i].MinID)
                break;
        }

        for(;start < PartManager->NumPart; start++) {
            if (HaloLabel[start].MinID != base[i].MinID)
                break;
            P[HaloLabel[start].Pindex].GrNr = base[i].GrNr;
        }
    }
}


void
fof_save_groups(FOFGroups * fof, int num, MPI_Comm Comm)
{
    message(0, "start global sorting of group catalogues\n");

    fof_save_particles(fof, num, fof_params.FOFSaveParticles, Comm);

    message(0, "Group catalogues saved.\n");
}

/* FIXME: these shall goto the private member of secondary tree walk */
struct FOFSecondaryPriv {
    float *distance;
    float *hsml;
    int *npleft;
};

#define FOF_SECONDARY_GET_PRIV(tw) ((struct FOFSecondaryPriv *) (tw->priv))

static void fof_secondary_copy(int place, TreeWalkQueryFOF * I, TreeWalk * tw) {

    I->Hsml = FOF_SECONDARY_GET_PRIV(tw)->hsml[place];
}
static int fof_secondary_haswork(int n, TreeWalk * tw) {
    if(P[n].IsGarbage || P[n].Swallowed)
        return 0;
    /* Exclude particles where we already found a neighbour*/
    if(FOF_SECONDARY_GET_PRIV(tw)->distance[n] < 0.5 * LARGE)
        return 0;
    return (((1 << P[n].Type) & (FOF_SECONDARY_LINK_TYPES)));
}
static void fof_secondary_reduce(int place, TreeWalkResultFOF * O, enum TreeWalkReduceMode mode, TreeWalk * tw) {
    if(O->Distance < FOF_SECONDARY_GET_PRIV(tw)->distance[place])
    {
        FOF_SECONDARY_GET_PRIV(tw)->distance[place] = O->Distance;
        HaloLabel[place].MinID = O->MinID;
        HaloLabel[place].MinIDTask = O->MinIDTask;
    }
}

static void
fof_secondary_ngbiter(TreeWalkQueryFOF * I,
        TreeWalkResultFOF * O,
        TreeWalkNgbIterFOF * iter,
        LocalTreeWalk * lv)
{
    if(iter->base.other == -1) {
        O->Distance = LARGE;
        iter->base.Hsml = I->Hsml;
        iter->base.mask = FOF_PRIMARY_LINK_TYPES;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;
    if(r < O->Distance)
    {
        O->Distance = r;
        O->MinID = HaloLabel[other].MinID;
        O->MinIDTask = HaloLabel[other].MinIDTask;
    }
    /* No need to search nodes at a greater distance
     * now that we have a neighbour.*/
    iter->base.Hsml = iter->base.r;
}

static void
fof_secondary_postprocess(int p, TreeWalk * tw)
{
    /* More work needed: add this particle to the redo queue*/
    int tid = omp_get_thread_num();

    if(FOF_SECONDARY_GET_PRIV(tw)->distance[p] > 0.5 * LARGE)
    {
        if(FOF_SECONDARY_GET_PRIV(tw)->hsml[p] < 4 * fof_params.FOFHaloComovingLinkingLength)  /* we only search out to a maximum distance */
        {
            /* need to redo this particle */
            FOF_SECONDARY_GET_PRIV(tw)->npleft[tid]++;
            FOF_SECONDARY_GET_PRIV(tw)->hsml[p] *= 2.0;
/*
            if(iter >= MAXITER - 10)
            {
                endrun(1, "i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)\n",
                        p, ThisTask, P[p].ID, FOF_SECONDARY_GET_PRIV(tw)->hsml[p],
                        P[p].Pos[0], P[p].Pos[1], P[p].Pos[2]);
            }
*/
        } else {
            FOF_SECONDARY_GET_PRIV(tw)->distance[p] = 0;  /* we not continue to search for this particle */
        }
    }
}

static void fof_label_secondary(ForceTree * tree)
{
    int n;

    TreeWalk tw[1] = {{0}};
    tw->ev_label = "FOF_FIND_NEAREST";
    tw->visit = treewalk_visit_nolist_ngbiter;
    tw->ngbiter = (TreeWalkNgbIterFunction) fof_secondary_ngbiter;
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterFOF);
    tw->haswork = fof_secondary_haswork;
    tw->fill = (TreeWalkFillQueryFunction) fof_secondary_copy;
    tw->reduce = (TreeWalkReduceResultFunction) fof_secondary_reduce;
    tw->postprocess = (TreeWalkProcessFunction) fof_secondary_postprocess;
    tw->type = TREEWALK_ALL;
    tw->query_type_elsize = sizeof(TreeWalkQueryFOF);
    tw->result_type_elsize = sizeof(TreeWalkResultFOF);
    tw->tree = tree;
    struct FOFSecondaryPriv priv[1];

    tw->priv = priv;

    message(0, "Start finding nearest dm-particle (presently allocated=%g MB)\n",
            mymalloc_usedbytes() / (1024.0 * 1024.0));

    FOF_SECONDARY_GET_PRIV(tw)->distance = (float *) mymalloc("FOF_SECONDARY->distance", sizeof(float) * PartManager->NumPart);
    FOF_SECONDARY_GET_PRIV(tw)->hsml = (float *) mymalloc("FOF_SECONDARY->hsml", sizeof(float) * PartManager->NumPart);

    #pragma omp parallel for
    for(n = 0; n < PartManager->NumPart; n++)
    {
        FOF_SECONDARY_GET_PRIV(tw)->distance[n] = LARGE;
        FOF_SECONDARY_GET_PRIV(tw)->hsml[n] = 0.4 * fof_params.FOFHaloComovingLinkingLength;

        if((P[n].Type == 0 || P[n].Type == 4 || P[n].Type == 5) && FOF_SECONDARY_GET_PRIV(tw)->hsml[n] < 0.5 * P[n].Hsml) {
            /* use gas sml as a hint (faster convergence than 0.1 fof_params.FOFHaloComovingLinkingLength at high-z */
            FOF_SECONDARY_GET_PRIV(tw)->hsml[n] = 0.5 * P[n].Hsml;
        }
    }

    int64_t ntot;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

    message(0, "fof-nearest iteration started\n");
    int NumThreads = omp_get_max_threads();
    FOF_SECONDARY_GET_PRIV(tw)->npleft = ta_malloc("NPLeft", int, NumThreads);

    do
    {
        memset(FOF_SECONDARY_GET_PRIV(tw)->npleft, 0, sizeof(int) * NumThreads);

        treewalk_run(tw, NULL, PartManager->NumPart);

        for(n = 1; n < NumThreads; n++) {
            FOF_SECONDARY_GET_PRIV(tw)->npleft[0] += FOF_SECONDARY_GET_PRIV(tw)->npleft[n];
        }
        sumup_large_ints(1, &FOF_SECONDARY_GET_PRIV(tw)->npleft[0], &ntot);

        if(ntot < 0 || (ntot > 0 && tw->Niteration > MAXITER))
            endrun(1159, "Failed to converge in fof-nearest: ntot %ld", ntot);
    }
    while(ntot > 0);

    ta_free(FOF_SECONDARY_GET_PRIV(tw)->npleft);
    myfree(FOF_SECONDARY_GET_PRIV(tw)->hsml);
    myfree(FOF_SECONDARY_GET_PRIV(tw)->distance);
}

/*
 * Deal with seeding of particles At each FOF stage,
 * if seed_index is >= 0,  then that particle on seed_task
 * will be converted to a seed.
 *
 * */
static int cmp_seed_task(const void * c1, const void * c2) {
    const struct Group * g1 = c1;
    const struct Group * g2 = c2;

    return g1->seed_task - g2->seed_task;
}
static void fof_seed_make_one(struct Group * g, int ThisTask);

void fof_seed(FOFGroups * fof, ForceTree * tree, ActiveParticles * act, MPI_Comm Comm)
{
    int i, j, n, ntot;

    int NTask;
    MPI_Comm_size(Comm, &NTask);

    char * Marked = mymalloc2("SeedMark", fof->Ngroups);

    int Nexport = 0;
    for(i = 0; i < fof->Ngroups; i++)
    {
        Marked[i] =
            (fof->Group[i].Mass >= fof_params.MinFoFMassForNewSeed)
        &&  (fof->Group[i].MassType[4] >= fof_params.MinMStarForNewSeed)
        &&  (fof->Group[i].LenType[5] == 0)
        &&  (fof->Group[i].seed_index >= 0);

        if(Marked[i]) Nexport ++;
    }
    struct Group * ExportGroups = mymalloc("Export", sizeof(fof->Group[0]) * Nexport);
    j = 0;
    for(i = 0; i < fof->Ngroups; i ++) {
        if(Marked[i]) {
            ExportGroups[j] = fof->Group[i];
            j++;
        }
    }
    myfree(Marked);

    qsort_openmp(ExportGroups, Nexport, sizeof(ExportGroups[0]), cmp_seed_task);

    int * Send_count = ta_malloc("Send_count", int, NTask);
    int * Recv_count = ta_malloc("Recv_count", int, NTask);

    memset(Send_count, 0, NTask * sizeof(int));
    for(i = 0; i < Nexport; i++) {
        Send_count[ExportGroups[i].seed_task]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, Comm);

    int Nimport = 0;

    for(j = 0;  j < NTask; j++)
    {
        Nimport += Recv_count[j];
    }

    struct Group * ImportGroups = (struct Group *)
            mymalloc2("ImportGroups", Nimport * sizeof(struct Group));

    MPI_Alltoallv_smart(ExportGroups, Send_count, NULL, MPI_TYPE_GROUP,
                        ImportGroups, Recv_count, NULL, MPI_TYPE_GROUP,
                        Comm);

    myfree(ExportGroups);
    ta_free(Recv_count);
    ta_free(Send_count);

    MPI_Allreduce(&Nimport, &ntot, 1, MPI_INT, MPI_SUM, Comm);

    message(0, "Making %d new black hole particles.\n", ntot);

    /* Do we have enough black hole slots to create this many black holes?
     * If not, allocate more slots. */
    if(Nimport + SlotsManager->info[5].size > SlotsManager->info[5].maxsize)
    {
        struct NODE * nodes_base_tmp=NULL;
        int *Father_tmp=NULL;
        int *ActiveParticle_tmp=NULL;
        if(force_tree_allocated(tree)) {
            nodes_base_tmp = mymalloc2("nodesbasetmp", tree->numnodes * sizeof(struct NODE));
            memmove(nodes_base_tmp, tree->Nodes_base, tree->numnodes * sizeof(struct NODE));
            myfree(tree->Nodes_base);
            Father_tmp = mymalloc2("Father_tmp", PartManager->MaxPart * sizeof(int));
            memmove(Father_tmp, tree->Father, PartManager->MaxPart * sizeof(int));
            myfree(tree->Father);
        }
        /* This is only called on a PM step, so the condition should never be true*/
        if(act->ActiveParticle) {
            ActiveParticle_tmp = mymalloc2("ActiveParticle_tmp", act->NumActiveParticle * sizeof(int));
            memmove(ActiveParticle_tmp, act->ActiveParticle, act->NumActiveParticle * sizeof(int));
            myfree(act->ActiveParticle);
        }

        /*Now we can extend the slots! */
        int64_t atleast[6];
        int64_t i;
        for(i = 0; i < 6; i++)
            atleast[i] = SlotsManager->info[i].maxsize;
        atleast[5] += ntot*1.1;
        slots_reserve(1, atleast, SlotsManager);

        /*And now we need our memory back in the right place*/
        if(ActiveParticle_tmp) {
            act->ActiveParticle = mymalloc("ActiveParticle", sizeof(int)*(act->NumActiveParticle + PartManager->MaxPart - PartManager->NumPart));
            memmove(act->ActiveParticle, ActiveParticle_tmp, act->NumActiveParticle * sizeof(int));
            myfree(ActiveParticle_tmp);
        }
        if(force_tree_allocated(tree)) {
            tree->Father = mymalloc("Father", PartManager->MaxPart * sizeof(int));
            memmove(tree->Father, Father_tmp, PartManager->MaxPart * sizeof(int));
            myfree(Father_tmp);
            tree->Nodes_base = mymalloc("Nodes_base", tree->numnodes * sizeof(struct NODE));
            memmove(tree->Nodes_base, nodes_base_tmp, tree->numnodes * sizeof(struct NODE));
            myfree(nodes_base_tmp);
            /*Don't forget to update the Node pointer as well as Node_base!*/
            tree->Nodes = tree->Nodes_base - tree->firstnode;
        }
    }

    int ThisTask;
    MPI_Comm_rank(Comm, &ThisTask);

    for(n = 0; n < Nimport; n++)
    {
        fof_seed_make_one(&ImportGroups[n], ThisTask);
    }

    myfree(ImportGroups);

    walltime_measure("/FOF/Seeding");
}

static void fof_seed_make_one(struct Group * g, int ThisTask) {
   if(g->seed_task != ThisTask) {
        endrun(7771, "Seed does not belong to the right task");
    }
    int index = g->seed_index;
    blackhole_make_one(index);
}

static int fof_compare_HaloLabel_MinID(const void *a, const void *b)
{
    if(((struct fof_particle_list *) a)->MinID < ((struct fof_particle_list *) b)->MinID)
        return -1;

    if(((struct fof_particle_list *) a)->MinID > ((struct fof_particle_list *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinID(const void *a, const void *b)

{
    if(((struct BaseGroup *) a)->MinID < ((struct BaseGroup *) b)->MinID)
        return -1;

    if(((struct BaseGroup *) a)->MinID > ((struct BaseGroup *) b)->MinID)
        return +1;

    return 0;
}

static int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
    const struct BaseGroup * p1 = a;
    const struct BaseGroup * p2 = b;
    int t1 = p1->MinIDTask;
    int t2 = p2->MinIDTask;
    if(t1 == _fof_compare_Group_MinIDTask_ThisTask) t1 = -1;
    if(t2 == _fof_compare_Group_MinIDTask_ThisTask) t2 = -1;

    if(t1 < t2) return -1;
    if(t1 > t2) return +1;
    return 0;

}

static int fof_compare_Group_OriginalIndex(const void *a, const void *b)

{
    return ((struct BaseGroup *) a)->OriginalIndex - ((struct BaseGroup *) b)->OriginalIndex;
}

static void fof_radix_Group_TotalCountTaskDiffMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = labs(f->OriginalTask - f->MinIDTask);
    u[1] = f->MinID;
    u[2] = UINT64_MAX - (f->Length);
}

static void fof_radix_Group_OriginalTaskMinID(const void * a, void * radix, void * arg) {
    uint64_t * u = (uint64_t *) radix;
    struct BaseGroup * f = (struct BaseGroup *) a;
    u[0] = f->MinID;
    u[1] = f->OriginalTask;
}
