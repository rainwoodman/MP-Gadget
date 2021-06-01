#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

#include <stdint.h>
#include "utils/paramset.h"
#include "forcetree.h"

#define  NODELISTLENGTH      8

enum NgbTreeFindSymmetric {
    NGB_TREEFIND_SYMMETRIC,
    NGB_TREEFIND_ASYMMETRIC,
};

enum TreeWalkReduceMode {
    TREEWALK_PRIMARY,
    TREEWALK_GHOSTS,
};

typedef struct TreeWalk TreeWalk;

typedef struct {
    double Pos[3];
#ifdef DEBUG
    MyIDType ID;
#endif
    int NodeList[NODELISTLENGTH];
} TreeWalkQueryBase;

typedef struct {
#ifdef DEBUG
    MyIDType ID;
#endif
} TreeWalkResultBase;

typedef struct {
    enum NgbTreeFindSymmetric symmetric;
    int mask;
    double Hsml;
    double dist[3];
    double r2;
    double r;
    int other;
} TreeWalkNgbIterBase;

typedef struct {
    TreeWalk * tw;

    int mode; /* 0 for Primary, 1 for Secondary */
    int target; /* defined only for primary (mode == 0) */

    /* Thread local export variables*/
    size_t Nexport;
    size_t BunchSize;
    /* Number of entries in the export table for this particle*/
    size_t NThisParticleExport;
    int *exportflag;
    int *exportnodecount;
    size_t *exportindex;
    size_t DataIndexOffset;

    int * ngblist;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
    int64_t Nlist;
} LocalTreeWalk;

typedef int (*TreeWalkVisitFunction) (TreeWalkQueryBase * input, TreeWalkResultBase * output, LocalTreeWalk * lv);

typedef void (*TreeWalkNgbIterFunction) (TreeWalkQueryBase * input, TreeWalkResultBase * output, TreeWalkNgbIterBase * iter, LocalTreeWalk * lv);

typedef int (*TreeWalkHasWorkFunction) (const int i, TreeWalk * tw);
typedef void (*TreeWalkProcessFunction) (const int i, TreeWalk * tw);

typedef void (*TreeWalkFillQueryFunction)(const int j, TreeWalkQueryBase * query, TreeWalk * tw);
typedef void (*TreeWalkReduceResultFunction)(const int j, TreeWalkResultBase * result, const enum TreeWalkReduceMode mode, TreeWalk * tw);

enum TreeWalkType {
    TREEWALK_ACTIVE = 0,
    TREEWALK_ALL,
    TREEWALK_SPLIT,
};

struct TreeWalk {
    void * priv;

    /* A pointer to the force tree structure to walk.*/
    const ForceTree * tree;

    /* name of the evaluator (used in printing messages) */
    char * ev_label;

    enum TreeWalkType type;

    size_t query_type_elsize;
    size_t result_type_elsize;
    size_t ngbiter_type_elsize;

    TreeWalkVisitFunction visit;                /* Function to be called between a tree node and a particle */
    TreeWalkHasWorkFunction haswork; /* Is the particle part of this interaction? */
    TreeWalkFillQueryFunction fill;       /* Copy the useful attributes of a particle to a query */
    TreeWalkReduceResultFunction reduce;  /* Reduce a partial result to the local particle storage */
    TreeWalkNgbIterFunction ngbiter;     /* called for each pair of particles if visit is set to ngbiter */
    TreeWalkProcessFunction postprocess; /* postprocess finalizes quantities for each particle, e.g. divide the normalization */
    TreeWalkProcessFunction preprocess; /* Preprocess initializes quantities for each particle */
    int NTask; /*Number of MPI tasks*/
    int64_t NThread; /*Number of OpenMP threads*/

    /* Unlike in Gadget-3, when exporting we now always send tree branches.*/
    char * dataget;
    char * dataresult;

    /* The metal return alters neighbours,
     * which cannot be evaluated twice.
     * If repeatdisallowd is true, we allocate memory
     * to keep track of the evaluated particles.*/
    int repeatdisallowed;
    char * evaluated;

    /* performance metrics */
    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecomp3;
    double timecommsumm1;
    double timecommsumm2;
    /* For secondary tree walks this stores the
     * total number of pseudo-particles in all
     * node lists of exported particles.*/
    int64_t Nnodesinlist;
    /* Stores the total number of node lists created for all exported particles.
     * Used to find the average number of nodes in each nodelist.*/
    int64_t Nlist;
    /* Total number of exported particles
     * (Nexport is only the exported particles in the current export buffer). */
    int64_t Nexport_sum;
    /* Number of times we filled up our export buffer*/
    int64_t Nexportfull;
    /* Number of times we needed to re-run the treewalk.
     * Convenience variable for density. */
    int64_t Niteration;

    /* internal flags*/
    /* Number of particles marked for export to another processor*/
    size_t Nexport;
    /* Number of particles exported to this processor*/
    size_t Nimport;
    /* Flags that our export buffer is full*/
    int BufferFullFlag;
    /* Number of particles we can fit into the export buffer*/
    size_t BunchSize;
    /* List of neighbour candidates.*/
    int *Ngblist;
    /* Flag not allocating nighbour list*/
    int NoNgblist;
    /* Index into WorkSet to start iteration.
     * Will be !=0 if the export buffer fills up*/
    int64_t WorkSetStart;
    /* The list of particles to work on. May be NULL, in which case all particles are used.*/
    int * WorkSet;
    /* Size of the workset list*/
    int64_t WorkSetSize;
    /*Did we use the active_set array as the WorkSet?*/
    int work_set_stolen_from_active;
    /* Redo counters and queues*/
    size_t *NPLeft;
    int **NPRedo;
    size_t Redo_thread_alloc;
    /* Max and min arrays for each iteration of the count*/
    double * maxnumngb;
    double * minnumngb;
};

/*Initialise treewalk parameters on first run*/
void set_treewalk_params(ParameterSet * ps);

/* Do the distributed tree walking. Warning: as this is a threaded treewalk,
 * it may call tw->visit on particles more than once and in a noneterministic order.
 * Your module should behave correctly in this case! */
void treewalk_run(TreeWalk * tw, int * active_set, size_t size);

int treewalk_visit_ngbiter(TreeWalkQueryBase * I,
            TreeWalkResultBase * O,
            LocalTreeWalk * lv);

/* used in uvbg.c to find particles in fof groups */
//int cull_node(const TreeWalkQueryBase * const I, const TreeWalkNgbIterBase * const iter, const struct NODE * const current, const double BoxSize);

/*returns -1 if the buffer is full */
int treewalk_export_particle(LocalTreeWalk * lv, int no);
#define TREEWALK_REDUCE(A, B) (A) = (mode==TREEWALK_PRIMARY)?(B):((A) + (B))

/*****
 * Variant of ngbiter that doesn't use the Ngblist.
 * The ngblist is generally preferred for memory locality reasons and
 * to avoid particles being partially evaluated
 * twice if the buffer fills up. Use this variant if the evaluation
 * wants to change the search radius, such as for knn algorithms
 * or some density code. Don't use it if the treewalk modifies other particles.
 * */
int treewalk_visit_nolist_ngbiter(TreeWalkQueryBase * I, TreeWalkResultBase * O, LocalTreeWalk * lv);

#define MAXITER 400

/* This function does treewalk_run in a loop, allocating a queue to allow some particles to be redone.
 * This loop is used primarily in density estimation.*/
void treewalk_do_hsml_loop(TreeWalk * tw, int * queue, int64_t queuesize, int update_hsml);

/* This function find the closest index in the multi-evaluation list of hsml and numNgb, update left and right bound, and return the new hsml */
double ngb_narrow_down(double *right, double *left, const double *radius, const double *numNgb, int maxcmpt, int desnumngb, int *closeidx, double BoxSize);

void treewalk_build_queue(TreeWalk * tw, int * active_set, const size_t size, int may_have_garbage);

#endif
