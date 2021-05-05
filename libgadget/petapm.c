#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */

#include "types.h"
#include "petapm.h"

#include "utils.h"
#include "walltime.h"

static void
layout_prepare(PetaPM * pm,
               struct Layout * L,
               double * meshbuf,
               PetaPMRegion * regions,
               const int Nregions,
               MPI_Comm comm);
static void layout_finish(struct Layout * L);
static void layout_build_and_exchange_cells_to_pfft(PetaPM * pm, struct Layout * L, double * meshbuf, double * real);
static void layout_build_and_exchange_cells_to_local(PetaPM * pm, struct Layout * L, double * meshbuf, double * real);

/* cell_iterator needs to be thread safe !*/
typedef void (* cell_iterator)(double * cell_value, double * comm_buffer);
static void layout_iterate_cells(PetaPM * pm, struct Layout * L, cell_iterator iter, double * real);

struct Pencil { /* a pencil starting at offset, with lenght len */
    int offset[3];
    int len;
    int first;
    int meshbuf_first; /* first pixel in meshbuf */
    int task;
};
static int pencil_cmp_target(const void * v1, const void * v2);
static int pos_get_target(PetaPM * pm, const int pos[2]);

/* FIXME: move this to MPIU_. */
static int64_t reduce_int64(int64_t input, MPI_Comm comm);
#ifdef DEBUG
/* for debugging */
static void verify_density_field(PetaPM * pm, double * real, double * meshbuf, const size_t meshsize);
#endif

static MPI_Datatype MPI_PENCIL;

/*Used only in MP-GenIC*/
pfft_complex *
petapm_alloc_rhok(PetaPM * pm)
{
    pfft_complex * rho_k = (pfft_complex * ) mymalloc("PMrho_k", pm->priv->fftsize * sizeof(double));
    memset(rho_k, 0, pm->priv->fftsize * sizeof(double));
    return rho_k;
}

static void pm_init_regions(PetaPM * pm, PetaPMRegion * regions, const int Nregions);

static PetaPMParticleStruct * CPS; /* stored by petapm_force, how to access the P array */
static PetaPMReionPartStruct * CPS_R; /* stored by petapm_force, how to access other properties in P, SphP, and Fof */
#define POS(i) ((double*)  (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS->offset_pos]))
#define MASS(i) ((float*) (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS->offset_mass]))
#define INACTIVE(i) (CPS->active && !CPS->active(i))

/* (jdavies) reion defs */
#define TYPE(i) ((int*)  (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS_R->offset_type]))
#define PI(i) ((int*)  (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS_R->offset_pi]))
#define FESC(i) ((double*) (&((char*)CPS_R->Starslot)[CPS_R->star_elsize * *PI(i) + CPS_R->offset_fesc]))
/*TODO: this is a MyFloat, also not currently used and possibly broken*/
#define SFR(i) ((double*)  (&((char*)CPS_R->Sphslot)[CPS_R->sph_elsize * *PI(i) + CPS_R->offset_sfr]))

PetaPMRegion * petapm_get_fourier_region(PetaPM * pm) {
    return &pm->fourier_space_region;
}
PetaPMRegion * petapm_get_real_region(PetaPM * pm) {
    return &pm->real_space_region;
}
int petapm_mesh_to_k(PetaPM * pm, int i) {
    /*Return the position of this point on the Fourier mesh*/
    return i<=pm->Nmesh/2 ? i : (i-pm->Nmesh);
}
int *petapm_get_thistask2d(PetaPM * pm) {
    return pm->ThisTask2d;
}
int *petapm_get_ntask2d(PetaPM * pm) {
    return pm->NTask2d;
}

void
petapm_module_init(int Nthreads)
{
    pfft_init();

    pfft_plan_with_nthreads(Nthreads);

    /* initialize the MPI Datatype of pencil */
    MPI_Type_contiguous(sizeof(struct Pencil), MPI_BYTE, &MPI_PENCIL);
    MPI_Type_commit(&MPI_PENCIL);
}

void
petapm_init(PetaPM * pm, double BoxSize, double Asmth, int Nmesh, double G, MPI_Comm comm)
{
    /* define the global long / short range force cut */
    pm->BoxSize = BoxSize;
    pm->Asmth = Asmth;
    pm->Nmesh = Nmesh;
    pm->G = G;
    pm->CellSize = BoxSize / Nmesh;
    pm->comm = comm;

    ptrdiff_t n[3] = {Nmesh, Nmesh, Nmesh};
    ptrdiff_t np[2];

    int ThisTask;
    int NTask;

    pm->Mesh2Task[0] = mymalloc("Mesh2Task", 2*sizeof(int) * Nmesh);
    pm->Mesh2Task[1] = pm->Mesh2Task[0] + Nmesh;

    MPI_Comm_rank(comm, &ThisTask);
    MPI_Comm_size(comm, &NTask);

    /* try to find a square 2d decomposition */
    int i;
    int k;
    for(i = sqrt(NTask) + 1; i >= 0; i --) {
        if(NTask % i == 0) break;
    }
    np[0] = i;
    np[1] = NTask / i;

    message(0, "Using 2D Task mesh %td x %td \n", np[0], np[1]);
    if( pfft_create_procmesh_2d(comm, np[0], np[1], &pm->priv->comm_cart_2d) ){
        endrun(0, "Error: This test file only works with %td processes.\n", np[0]*np[1]);
    }

    int periods_unused[2];
    MPI_Cart_get(pm->priv->comm_cart_2d, 2, pm->NTask2d, periods_unused, pm->ThisTask2d);

    if(pm->NTask2d[0] != np[0]) abort();
    if(pm->NTask2d[1] != np[1]) abort();

    pm->priv->fftsize = 2 * pfft_local_size_dft_r2c_3d(n, pm->priv->comm_cart_2d,
           PFFT_TRANSPOSED_OUT,
           pm->real_space_region.size, pm->real_space_region.offset,
           pm->fourier_space_region.size, pm->fourier_space_region.offset);

    /*
     * In fourier space, the transposed array is ordered in
     * are in (y, z, x). The strides and sizes returned
     * from local size is in (Nx, Ny, Nz), hence we roll them once
     * so that the strides will give correct linear indexing for
     * integer coordinates given in order of (y, z, x).
     * */

#define ROLL(a, N, j) { \
    typeof(a[0]) tmp[N]; \
    ptrdiff_t k; \
    for(k = 0; k < N; k ++) tmp[k] = a[k]; \
    for(k = 0; k < N; k ++) a[k] = tmp[(k + j)% N]; \
    }

    ROLL(pm->fourier_space_region.offset, 3, 1);
    ROLL(pm->fourier_space_region.size, 3, 1);

#undef ROLL

    /* calculate the strides */
    petapm_region_init_strides(&pm->real_space_region);
    petapm_region_init_strides(&pm->fourier_space_region);

    /* planning the fft; need temporary arrays */

    double * real = (double * ) mymalloc("PMreal", pm->priv->fftsize * sizeof(double));
    pfft_complex * rho_k = (pfft_complex * ) mymalloc("PMrho_k", pm->priv->fftsize * sizeof(double));
    pfft_complex * complx = (pfft_complex *) mymalloc("PMcomplex", pm->priv->fftsize * sizeof(double));

    pm->priv->plan_forw = pfft_plan_dft_r2c_3d(
        n, real, rho_k, pm->priv->comm_cart_2d, PFFT_FORWARD,
        PFFT_TRANSPOSED_OUT | PFFT_ESTIMATE | PFFT_TUNE | PFFT_DESTROY_INPUT);
    pm->priv->plan_back = pfft_plan_dft_c2r_3d(
        n, complx, real, pm->priv->comm_cart_2d, PFFT_BACKWARD,
        PFFT_TRANSPOSED_IN | PFFT_ESTIMATE | PFFT_TUNE | PFFT_DESTROY_INPUT);

    myfree(complx);
    myfree(rho_k);
    myfree(real);

    /* now lets fill up the mesh2task arrays */

#if 0
    message(1, "ThisTask = %d (%td %td %td) - (%td %td %td)\n", ThisTask,
            pm->real_space_region.offset[0],
            pm->real_space_region.offset[1],
            pm->real_space_region.offset[2],
            pm->real_space_region.size[0],
            pm->real_space_region.size[1],
            pm->real_space_region.size[2]);
#endif

    int * tmp = mymalloc("tmp", sizeof(int) * Nmesh);
    for(k = 0; k < 2; k ++) {
        for(i = 0; i < Nmesh; i ++) {
            tmp[i] = 0;
        }
        for(i = 0; i < pm->real_space_region.size[k]; i ++) {
            tmp[i + pm->real_space_region.offset[k]] = pm->ThisTask2d[k];
        }
        /* which column / row hosts this tile? */
        /* FIXME: this is very inefficient */
        MPI_Allreduce(tmp, pm->Mesh2Task[k], Nmesh, MPI_INT, MPI_MAX, comm);
        /*
        for(i = 0; i < Nmesh; i ++) {
            message(0, "Mesh2Task[%d][%d] == %d\n", k, i, Mesh2Task[k][i]);
        }
        */
    }
    myfree(tmp);
}

void
petapm_destroy(PetaPM * pm)
{
    pfft_destroy_plan(pm->priv->plan_forw);
    pfft_destroy_plan(pm->priv->plan_back);
    MPI_Comm_free(&pm->priv->comm_cart_2d);
    myfree(pm->Mesh2Task[0]);
}

/*
 * read out field to particle i, with value no need to be thread safe
 * (particle i is never done by same thread)
 * */
typedef void (* pm_iterator)(PetaPM * pm, int i, double * mesh, double weight);
static void pm_iterate(PetaPM * pm, pm_iterator iterator, PetaPMRegion * regions, const int Nregions);
/* apply transfer function to value, kpos array is in x, y, z order */
static void pm_apply_transfer_function(PetaPM * pm,
        pfft_complex * src,
        pfft_complex * dst, petapm_transfer_func H);

static void put_particle_to_mesh(PetaPM * pm, int i, double * mesh, double weight);
static void put_star_to_mesh(PetaPM * pm, int i, double * mesh, double weight);
static void put_sfr_to_mesh(PetaPM * pm, int i, double * mesh, double weight);

/*
 * 1. calls prepare to build the Regions covering particles
 * 2. CIC the particles
 * 3. Transform to rho_k
 * 4. apply global_transfer (if not NULL --
 *       this is the place to fill in gaussian seeds,
 *       the transfer is stacked onto all following transfers.
 * 5. for each transfer, readout in functions
 * 6.    apply transfer from global_transfer -> complex
 * 7.    transform to real
 * 8.    readout
 * 9. free regions
 * */

PetaPMRegion *
petapm_force_init(
        PetaPM * pm,
        petapm_prepare_func prepare,
        PetaPMParticleStruct * pstruct,
        int * Nregions,
        void * userdata) {
    CPS = pstruct;

    *Nregions = 0;
    PetaPMRegion * regions = prepare(pm, pstruct, userdata, Nregions);
    pm_init_regions(pm, regions, *Nregions);

    walltime_measure("/PMgrav/Misc");
    pm_iterate(pm, put_particle_to_mesh, regions, *Nregions);
    walltime_measure("/PMgrav/cic");

    layout_prepare(pm, &pm->priv->layout, pm->priv->meshbuf, regions, *Nregions, pm->comm);

    walltime_measure("/PMgrav/comm");
    return regions;
}

pfft_complex * petapm_force_r2c(PetaPM * pm,
        PetaPMGlobalFunctions * global_functions
        ) {
    /* call pfft rho_k is CFT of rho */

    /* this is because
     *
     * CFT = DFT * dx **3
     * CFT[rho] = DFT [rho * dx **3] = DFT[CIC]
     * */
    double * real = (double * ) mymalloc2("PMreal", pm->priv->fftsize * sizeof(double));
    memset(real, 0, sizeof(double) * pm->priv->fftsize);
    layout_build_and_exchange_cells_to_pfft(pm, &pm->priv->layout, pm->priv->meshbuf, real);
    walltime_measure("/PMgrav/comm2");

#ifdef DEBUG
    verify_density_field(pm, real, pm->priv->meshbuf, pm->priv->meshbufsize);
    walltime_measure("/PMgrav/Misc");
#endif

    pfft_complex * complx = (pfft_complex *) mymalloc("PMcomplex", pm->priv->fftsize * sizeof(double));
    pfft_execute_dft_r2c(pm->priv->plan_forw, real, complx);
    myfree(real);

    pfft_complex * rho_k = (pfft_complex * ) mymalloc2("PMrho_k", pm->priv->fftsize * sizeof(double));

    /*Do any analysis that may be required before the transfer function is applied*/
    petapm_transfer_func global_readout = global_functions->global_readout;
    if(global_readout)
        pm_apply_transfer_function(pm, complx, rho_k, global_readout);
    if(global_functions->global_analysis)
        global_functions->global_analysis(pm);
    /*Apply the transfer function*/
    petapm_transfer_func global_transfer = global_functions->global_transfer;
    pm_apply_transfer_function(pm, complx, rho_k, global_transfer);
    walltime_measure("/PMgrav/r2c");

    report_memory_usage("PetaPM");

    myfree(complx);
    return rho_k;
}

void
petapm_force_c2r(PetaPM * pm,
        pfft_complex * rho_k,
        PetaPMRegion * regions,
        const int Nregions,
        PetaPMFunctions * functions)
{

    PetaPMFunctions * f = functions;
    for (f = functions; f->name; f ++) {
        petapm_transfer_func transfer = f->transfer;
        petapm_readout_func readout = f->readout;

        pfft_complex * complx = (pfft_complex *) mymalloc("PMcomplex", pm->priv->fftsize * sizeof(double));
        /* apply the greens function turn rho_k into potential in fourier space */
        pm_apply_transfer_function(pm, rho_k, complx, transfer);
        walltime_measure("/PMgrav/calc");

        double * real = (double * ) mymalloc2("PMreal", pm->priv->fftsize * sizeof(double));
        pfft_execute_dft_c2r(pm->priv->plan_back, complx, real);
        walltime_measure("/PMgrav/c2r");
        myfree(complx);
        /* read out the potential: this will copy and free real.*/
        layout_build_and_exchange_cells_to_local(pm, &pm->priv->layout, pm->priv->meshbuf, real);
        walltime_measure("/PMgrav/comm");

        pm_iterate(pm, readout, regions, Nregions);
        walltime_measure("/PMgrav/readout");
    }
    walltime_measure("/PMgrav/Misc");

}
void petapm_force_finish(PetaPM * pm) {
    layout_finish(&pm->priv->layout);
    myfree(pm->priv->meshbuf);
}

void petapm_force(PetaPM * pm, petapm_prepare_func prepare,
        PetaPMGlobalFunctions * global_functions, //petapm_transfer_func global_transfer,
        PetaPMFunctions * functions,
        PetaPMParticleStruct * pstruct,
        void * userdata) {
    int Nregions;
    PetaPMRegion * regions = petapm_force_init(pm, prepare, pstruct, &Nregions, userdata);
    pfft_complex * rho_k = petapm_force_r2c(pm, global_functions);
    if(functions)
        petapm_force_c2r(pm, rho_k, regions, Nregions, functions);
    myfree(rho_k);
    if(CPS->RegionInd)
        myfree(CPS->RegionInd);
    myfree(regions);
    petapm_force_finish(pm);
}

/* initialise one set of regions with custom iterator 
 * this is the same as petapm_force_init with a custom iterator 
 * (and no CPS definition since it's called multiple times)*/
PetaPMRegion *
petapm_reion_init(
        PetaPM * pm,
        petapm_prepare_func prepare,
        pm_iterator iterator,
        PetaPMParticleStruct * pstruct,
        int * Nregions,
        void * userdata) {

    *Nregions = 0;
    PetaPMRegion * regions = prepare(pm, pstruct, userdata, Nregions);
    pm_init_regions(pm, regions, *Nregions);

    walltime_measure("/PMreion/Misc");
    pm_iterate(pm, iterator, regions, *Nregions);
    walltime_measure("/PMreion/cic");

    layout_prepare(pm, &pm->priv->layout, pm->priv->meshbuf, regions, *Nregions, pm->comm);

    walltime_measure("/PMreion/comm");
    return regions;
}

/* 30Mpc to 0.5 Mpc with a delta of 1.1 is ~50 iterations, this should be more than enough*/
#define MAX_R_ITERATIONS 10000

/* differences from force c2r (why I think I need this separate)
 * radius loop (could do this with long list of same function + global R)
 * I'm pretty sure I need a third function type (reion loop) with all three grids
 * ,after c2r but iteration over the grid, instead of particles */
void
petapm_reion_c2r(PetaPM * pm_mass, PetaPM * pm_star, PetaPM * pm_sfr,
        pfft_complex * mass_unfiltered, pfft_complex * star_unfiltered, pfft_complex * sfr_unfiltered,
        PetaPMRegion * regions,
        const int Nregions,
        PetaPMFunctions * functions,
        petapm_reion_func reion_loop,
        double R_max, double R_min, double R_delta)
{
    PetaPMFunctions * f = functions;
    double R = fmin(R_max,pm_mass->BoxSize);
    int last_step = 0;
    int f_count = 0;
    petapm_transfer_func transfer = f->transfer;
    petapm_readout_func readout = f->readout;
   
    /* TODO: seriously re-think the allocation ordering in this function */
    double * mass_real = (double * ) mymalloc2("mass_real", pm_mass->priv->fftsize * sizeof(double));

    //TODO: add CellLengthFactor for lowres (>1Mpc, see old find_HII_bubbles function)
    while(!last_step) {
        f_count++;
        //The last step will be unfiltered
        if(R/R_delta < R_min || R/R_delta < (pm_mass->CellSize) || f_count > MAX_R_ITERATIONS)
        {
            last_step = 1;
            R = pm_mass->CellSize;
        }

        //NOTE: The PetaPM structs for reionisation use the G variable for filter radius in order to use
        //the transfer functions correctly
        pm_mass->G = R;
        pm_star->G = R;
        pm_sfr->G = R;

        //TODO: maybe allocate and free these outside the loop
        pfft_complex * mass_filtered = (pfft_complex *) mymalloc("mass_filtered", pm_mass->priv->fftsize * sizeof(double));
        pfft_complex * star_filtered = (pfft_complex *) mymalloc("star_filtered", pm_star->priv->fftsize * sizeof(double));
        pfft_complex * sfr_filtered = (pfft_complex *) mymalloc("sfr_filtered", pm_sfr->priv->fftsize * sizeof(double));

        /* apply the filtering at this radius */
        /*We want the last step to be unfiltered,
         *  calling apply transfer with NULL should just copy the grids */

        transfer = last_step ? NULL : f->transfer;

        pm_apply_transfer_function(pm_mass, mass_unfiltered, mass_filtered, transfer);
        pm_apply_transfer_function(pm_star, star_unfiltered, star_filtered, transfer);
        pm_apply_transfer_function(pm_sfr, sfr_unfiltered, sfr_filtered, transfer);

        walltime_measure("/PMreion/calc");

        double * star_real = (double * ) mymalloc2("star_real", pm_star->priv->fftsize * sizeof(double));
        double * sfr_real = (double * ) mymalloc2("sfr_real", pm_sfr->priv->fftsize * sizeof(double));

        /* back to real space */
        pfft_execute_dft_c2r(pm_mass->priv->plan_back, mass_filtered, mass_real);
        pfft_execute_dft_c2r(pm_star->priv->plan_back, star_filtered, star_real);
        pfft_execute_dft_c2r(pm_sfr->priv->plan_back, sfr_filtered, sfr_real);
        walltime_measure("/PMreion/c2r");
        
        myfree(sfr_filtered);
        myfree(star_filtered);
        myfree(mass_filtered);

        /* the reion loop calculates the J21 and stores it,
         * for now the mass_real grid will be reused to hold J21
         * on the last filtering step*/
        reion_loop(pm_mass,pm_star,pm_sfr,mass_real,star_real,sfr_real,last_step);

        /* since we don't need to readout star and sfr grids...*/
        /* on the last step, the mass grid is populated with J21 and read out*/
        myfree(sfr_real);
        myfree(star_real);
        
        R = R / R_delta;
    }
    //J21 grid is exchanged to pm_mass buffer and freed
    layout_build_and_exchange_cells_to_local(pm_mass, &pm_mass->priv->layout, pm_mass->priv->meshbuf, mass_real);
    walltime_measure("/PMreion/comm");
    //J21 read out to particles
    pm_iterate(pm_mass, readout, regions, Nregions);
    walltime_measure("/PMreion/readout");
}

/* We need a slightly different flow for reionisation, so I 
 * will define these here instead of messing with the force functions.
 * The c2r function is the same, however we need a new function, reion_loop
 * to run over all three filtered grids, after the inverse transform.
 * The c2r function itself is also different since we need to apply the
 * transfer (filter) function on all three grids and run reion_loop before any readout.*/
void petapm_reion(PetaPM * pm_mass, PetaPM * pm_star, PetaPM * pm_sfr,
        petapm_prepare_func prepare,
        PetaPMGlobalFunctions * global_functions, //petapm_transfer_func global_transfer,
        PetaPMFunctions * functions,
        PetaPMParticleStruct * pstruct,
        PetaPMReionPartStruct * rstruct,
        petapm_reion_func reion_loop,
        double R_max, double R_min, double R_delta,
        void * userdata) {
    
    //assigning CPS here due to three sets of regions
    CPS = pstruct;
    CPS_R = rstruct;
    
    int i_star = -10;
    for(int ii = CPS->NumPart; ii >= 0; ii --) {
        if(*TYPE(ii) == 4) {
            i_star = ii;
            break;
        }
    }
   
    if(i_star > 0) {
        //#define PI(i) ((int*)  (&((char*)CPS->Parts)[CPS->elsize * (i) + CPS_R->offset_pi]))
        message(1,"last star idx inside petapm = %d, Type = %d, PI = %d\n",i_star,*TYPE(i_star),*PI(i_star));
        message(1,"has fesc = %e\n",*FESC(i_star));
        message(1,"total indices (PI,fesc) = (%d, %d)\n"
                ,CPS->elsize * (i_star) + CPS_R->offset_pi
                , CPS_R->star_elsize * *PI(i_star) + CPS_R->offset_fesc);
    }

    /* initialise regions for each grid
     * NOTE: these regions should be identical except for the grid buffer */
    int Nregions_mass, Nregions_star, Nregions_sfr;
    PetaPMRegion * regions_mass = petapm_reion_init(pm_mass, prepare, put_particle_to_mesh, pstruct, &Nregions_mass, userdata);
    PetaPMRegion * regions_star = petapm_reion_init(pm_star, prepare, put_star_to_mesh, pstruct, &Nregions_star, userdata);
    PetaPMRegion * regions_sfr = petapm_reion_init(pm_sfr, prepare, put_sfr_to_mesh, pstruct, &Nregions_sfr, userdata);

    walltime_measure("/PMreion/comm2");

    //using force r2c since this part can be done independently
    pfft_complex * mass_unfiltered = petapm_force_r2c(pm_mass, global_functions);
    pfft_complex * star_unfiltered = petapm_force_r2c(pm_star, global_functions);
    pfft_complex * sfr_unfiltered = petapm_force_r2c(pm_sfr, global_functions);

    message(1,"starting c2r\n");
    //need custom reion_c2r to implement the 3 grid c2r and readout
    //the readout is only performed on the mass grid so for now I only pass in regions/Nregions for mass
    if(functions)
        petapm_reion_c2r(pm_mass, pm_star, pm_sfr,
               mass_unfiltered, star_unfiltered, sfr_unfiltered,
               regions_mass, Nregions_mass, functions, reion_loop,
               R_max, R_min, R_delta);
    
    //free everything in the correct order
    myfree(sfr_unfiltered);
    myfree(star_unfiltered);
    myfree(mass_unfiltered);
    
    if(CPS->RegionInd)
        myfree(CPS->RegionInd);
    
    myfree(regions_sfr);
    myfree(regions_star);
    myfree(regions_mass);

    petapm_force_finish(pm_sfr);
    petapm_force_finish(pm_star);
    petapm_force_finish(pm_mass);
}


/* build a communication layout */

static void layout_build_pencils(PetaPM * pm, struct Layout * L, double * meshbuf, PetaPMRegion * regions, const int Nregions);
static void layout_exchange_pencils(struct Layout * L);
static void
layout_prepare (PetaPM * pm,
                struct Layout * L,
                double * meshbuf,
                PetaPMRegion * regions,
                const int Nregions,
                MPI_Comm comm)
{
    int r;
    int i;
    int NTask;
    L->comm = comm;

    MPI_Comm_size(L->comm, &NTask);

    L->ibuffer = mymalloc("PMlayout", sizeof(int) * NTask * 8);

    memset(L->ibuffer, 0, sizeof(int) * NTask * 8);
    L->NpSend = &L->ibuffer[NTask * 0];
    L->NpRecv = &L->ibuffer[NTask * 1];
    L->NcSend = &L->ibuffer[NTask * 2];
    L->NcRecv = &L->ibuffer[NTask * 3];
    L->DcSend = &L->ibuffer[NTask * 4];
    L->DcRecv = &L->ibuffer[NTask * 5];
    L->DpSend = &L->ibuffer[NTask * 6];
    L->DpRecv = &L->ibuffer[NTask * 7];

    L->NpExport = 0;
    L->NcExport = 0;
    L->NpImport = 0;
    L->NcImport = 0;

    int NpAlloc = 0;
    /* count pencils until buffer would run out */
    for (r = 0; r < Nregions; r ++) {
        NpAlloc += regions[r].size[0] * regions[r].size[1];
    }

    L->PencilSend = mymalloc("PencilSend", NpAlloc * sizeof(struct Pencil));

    layout_build_pencils(pm, L, meshbuf, regions, Nregions);

    /* sort the pencils by the target rank for ease of next step */
    qsort_openmp(L->PencilSend, NpAlloc, sizeof(struct Pencil), pencil_cmp_target);
    /* zero length pixels are moved to the tail */

    /* now shrink NpExport*/
    L->NpExport = NpAlloc;
    while(L->NpExport > 0 && L->PencilSend[L->NpExport - 1].len == 0) {
        L->NpExport --;
    }

    /* count total number of cells to be exported */
    int NcExport = 0;
    for(i = 0; i < L->NpExport; i++) {
        int task = L->PencilSend[i].task;
        L->NcSend[task] += L->PencilSend[i].len;
        NcExport += L->PencilSend[i].len;
        L->NpSend[task] ++;
    }
    L->NcExport = NcExport;

    MPI_Alltoall(L->NpSend, 1, MPI_INT, L->NpRecv, 1, MPI_INT, L->comm);
    MPI_Alltoall(L->NcSend, 1, MPI_INT, L->NcRecv, 1, MPI_INT, L->comm);

    /* build the displacement array; why doesn't MPI build these automatically? */
    L->DpSend[0] = 0; L->DpRecv[0] = 0;
    L->DcSend[0] = 0; L->DcRecv[0] = 0;
    for(i = 1; i < NTask; i ++) {
        L->DpSend[i] = L->NpSend[i - 1] + L->DpSend[i - 1];
        L->DpRecv[i] = L->NpRecv[i - 1] + L->DpRecv[i - 1];
        L->DcSend[i] = L->NcSend[i - 1] + L->DcSend[i - 1];
        L->DcRecv[i] = L->NcRecv[i - 1] + L->DcRecv[i - 1];
    }
    L->NpImport = L->DpRecv[NTask -1] + L->NpRecv[NTask -1];
    L->NcImport = L->DcRecv[NTask -1] + L->NcRecv[NTask -1];

    /* some checks */
    if(L->DpSend[NTask - 1] + L->NpSend[NTask -1] != L->NpExport) {
        endrun(1, "NpExport = %d NpSend=%d DpSend=%d\n", L->NpExport, L->NpSend[NTask -1], L->DpSend[NTask - 1]);
    }
    if(L->DcSend[NTask - 1] + L->NcSend[NTask -1] != L->NcExport) {
        endrun(1, "NcExport = %d NcSend=%d DcSend=%d\n", L->NcExport, L->NcSend[NTask -1], L->DcSend[NTask - 1]);
    }
    int64_t totNpAlloc = reduce_int64(NpAlloc, L->comm);
    int64_t totNpExport = reduce_int64(L->NpExport, L->comm);
    int64_t totNcExport = reduce_int64(L->NcExport, L->comm);
    int64_t totNpImport = reduce_int64(L->NpImport, L->comm);
    int64_t totNcImport = reduce_int64(L->NcImport, L->comm);

    if(totNpExport != totNpImport) {
        endrun(1, "totNpExport = %ld\n", totNpExport);
    }
    if(totNcExport != totNcImport) {
        endrun(1, "totNcExport = %ld\n", totNcExport);
    }

    /* exchange the pencils */
    message(0, "PetaPM:  %010ld/%010ld Pencils and %010ld Cells\n", totNpExport, totNpAlloc, totNcExport);
    L->PencilRecv = mymalloc("PencilRecv", L->NpImport * sizeof(struct Pencil));
    memset(L->PencilRecv, 0xfc, L->NpImport * sizeof(struct Pencil));
    layout_exchange_pencils(L);
}

static void
layout_build_pencils(PetaPM * pm,
                     struct Layout * L,
                     double * meshbuf,
                     PetaPMRegion * regions,
                     const int Nregions)
{
    /* now build pencils to be exported */
    int p0 = 0;
    int r;
    for (r = 0; r < Nregions; r++) {
        int ix;
#pragma omp parallel for private(ix)
        for(ix = 0; ix < regions[r].size[0]; ix++) {
            int iy;
            for(iy = 0; iy < regions[r].size[1]; iy++) {
                int poffset = ix * regions[r].size[1] + iy;
                struct Pencil * p = &L->PencilSend[p0 + poffset];

                p->offset[0] = ix + regions[r].offset[0];
                p->offset[1] = iy + regions[r].offset[1];
                p->offset[2] = regions[r].offset[2];
                p->len = regions[r].size[2];
                p->meshbuf_first = (regions[r].buffer - meshbuf) +
                    regions[r].strides[0] * ix +
                    regions[r].strides[1] * iy;
                /* now lets compress the pencil */
                while((p->len > 0) && (meshbuf[p->meshbuf_first + p->len - 1] == 0.0)) {
                    p->len --;
                }
                while((p->len > 0) && (meshbuf[p->meshbuf_first] == 0.0)) {
                    p->len --;
                    p->meshbuf_first++;
                    p->offset[2] ++;
                }

                p->task = pos_get_target(pm, p->offset);
            }
        }
        p0 += regions[r].size[0] * regions[r].size[1];
    }

}

static void layout_exchange_pencils(struct Layout * L) {
    int i;
    int offset;
    int NTask;
    MPI_Comm_size(L->comm, &NTask);
    /* build the first pointers to refer to the correct relative buffer locations */
    /* note that the buffer hasn't bee assembled yet */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        int j;
        struct Pencil * p = &L->PencilSend[offset];
        if(L->NpSend[i] == 0) continue;
        p->first = 0;
        for(j = 1; j < L->NpSend[i]; j++) {
            p[j].first = p[j - 1].first + p[j - 1].len;
        }
        offset += L->NpSend[i];
    }

    MPI_Alltoallv(
            L->PencilSend, L->NpSend, L->DpSend, MPI_PENCIL,
            L->PencilRecv, L->NpRecv, L->DpRecv, MPI_PENCIL,
            L->comm);

    /* set first to point to absolute position in the full import cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilRecv[offset];
        int j;
        for(j = 0; j < L->NpRecv[i]; j++) {
            p[j].first += L->DcRecv[i];
        }
        offset += L->NpRecv[i];
    }

    /* set first to point to absolute position in the full export cell buffer */
    offset = 0;
    for(i = 0; i < NTask; i ++) {
        struct Pencil * p = &L->PencilSend[offset];
        int j;
        for(j = 0; j < L->NpSend[i]; j++) {
            p[j].first += L->DcSend[i];
        }
        offset += L->NpSend[i];
    }
}

static void layout_finish(struct Layout * L) {
    myfree(L->PencilRecv);
    myfree(L->PencilSend);
    myfree(L->ibuffer);
}

/* exchange cells to their pfft host, then reduce the cells to the pfft
 * array */
static void to_pfft(double * cell, double * buf) {
#pragma omp atomic update
            cell[0] += buf[0];
}

static void
layout_build_and_exchange_cells_to_pfft(
        PetaPM * pm,
        struct Layout * L,
        double * meshbuf,
        double * real)
{
    L->BufSend = mymalloc("PMBufSend", L->NcExport * sizeof(double));
    L->BufRecv = mymalloc("PMBufRecv", L->NcImport * sizeof(double));

    int i;
    int offset;

    /* collect all cells into the send buffer */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(L->BufSend + offset, &meshbuf[p->meshbuf_first],
                sizeof(double) * p->len);
        offset += p->len;
    }

    /* receive cells */
    MPI_Alltoallv(
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE,
            L->comm);

#if 0
    double massExport = 0;
    for(i = 0; i < L->NcExport; i ++) {
        massExport += L->BufSend[i];
    }

    double massImport = 0;
    for(i = 0; i < L->NcImport; i ++) {
        massImport += L->BufRecv[i];
    }
    double totmassExport;
    double totmassImport;
    MPI_Allreduce(&massExport, &totmassExport, 1, MPI_DOUBLE, MPI_SUM, L->comm);
    MPI_Allreduce(&massImport, &totmassImport, 1, MPI_DOUBLE, MPI_SUM, L->comm);
    message(0, "totmassExport = %g totmassImport = %g\n", totmassExport, totmassImport);
#endif

    layout_iterate_cells(pm, L, to_pfft, real);
    myfree(L->BufRecv);
    myfree(L->BufSend);
}

/* readout cells on their pfft host, then exchange the cells to the domain
 * host */
static void to_region(double * cell, double * region) {
    *region = *cell;
}

static void
layout_build_and_exchange_cells_to_local(
        PetaPM * pm,
        struct Layout * L,
        double * meshbuf,
        double * real)
{
    L->BufRecv = mymalloc("PMBufRecv", L->NcImport * sizeof(double));
    int i;
    int offset;

    /*layout_iterate_cells transfers real to L->BufRecv*/
    layout_iterate_cells(pm, L, to_region, real);

    /*Real is done now: reuse the memory for BufSend*/
    myfree(real);
    /*Now allocate BufSend, which is confusingly used to receive data*/
    L->BufSend = mymalloc("PMBufSend", L->NcExport * sizeof(double));

    /* exchange cells */
    /* notice the order is reversed from to_pfft */
    MPI_Alltoallv(
            L->BufRecv, L->NcRecv, L->DcRecv, MPI_DOUBLE,
            L->BufSend, L->NcSend, L->DcSend, MPI_DOUBLE,
            L->comm);

    /* distribute BufSend to meshbuf */
    offset = 0;
    for(i = 0; i < L->NpExport; i ++) {
        struct Pencil * p = &L->PencilSend[i];
        memcpy(&meshbuf[p->meshbuf_first],
                L->BufSend + offset,
                sizeof(double) * p->len);
        offset += p->len;
    }
    myfree(L->BufSend);
    myfree(L->BufRecv);
}

/* iterate over the pairs of real field cells and RecvBuf cells
 *
 * !!! iter has to be thread safe. !!!
 * */
static void
layout_iterate_cells(PetaPM * pm,
                     struct Layout * L,
                     cell_iterator iter,
                     double * real)
{
    int i;
#pragma omp parallel for
    for(i = 0; i < L->NpImport; i ++) {
        struct Pencil * p = &L->PencilRecv[i];
        int k;
        ptrdiff_t linear0 = 0;
        for(k = 0; k < 2; k ++) {
            int ix = p->offset[k];
            while(ix < 0) ix += pm->Nmesh;
            while(ix >= pm->Nmesh) ix -= pm->Nmesh;
            ix -= pm->real_space_region.offset[k];
            if(ix >= pm->real_space_region.size[k]) {
                /* serious problem assumption about pfft layout was wrong*/
                endrun(1, "bad pfft: original k: %d ix: %d, cur ix: %d, region: off %d size %d\n", k, p->offset[k], ix, pm->real_space_region.offset[k], pm->real_space_region.size[k]);
            }
            linear0 += ix * pm->real_space_region.strides[k];
        }
        int j;
        for(j = 0; j < p->len; j ++) {
            int iz = p->offset[2] + j;
            while(iz < 0) iz += pm->Nmesh;
            while(iz >= pm->Nmesh) iz -= pm->Nmesh;
            if(iz >= pm->real_space_region.size[2]) {
                /* serious problem assmpution about pfft layout was wrong*/
                endrun(1, "bad pfft: original iz: %d, cur iz: %d, region: off %d size %d\n", p->offset[2], iz, pm->real_space_region.offset[2], pm->real_space_region.size[2]);
            }
            ptrdiff_t linear = iz * pm->real_space_region.strides[2] + linear0;
            /*
             * operate on the pencil, either modifying real or BufRecv
             * */
            iter(&real[linear], &L->BufRecv[p->first + j]);
        }
    }
}

static void
pm_init_regions(PetaPM * pm, PetaPMRegion * regions, const int Nregions)
{
    if(regions) {
        int i;
        size_t size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            size += regions[i].totalsize;
        }
        pm->priv->meshbufsize = size;
        if ( size == 0 ) return;
        pm->priv->meshbuf = (double *) mymalloc("PMmesh", size * sizeof(double));
        /* this takes care of the padding */
        memset(pm->priv->meshbuf, 0, size * sizeof(double));
        size = 0;
        for(i = 0 ; i < Nregions; i ++) {
            regions[i].buffer = pm->priv->meshbuf + size;
            size += regions[i].totalsize;
        }
    }
}


static void
pm_iterate_one(PetaPM * pm,
               int i,
               pm_iterator iterator,
               PetaPMRegion * regions,
               const int Nregions)
{
    int k;
    int iCell[3];  /* integer coordinate on the regional mesh */
    double Res[3]; /* residual*/
    double * Pos = POS(i);
    const int RegionInd = CPS->RegionInd ? CPS->RegionInd[i] : 0;

    /* Asserts that the swallowed particles are not considered (region -2).*/
    if(RegionInd < 0)
        return;
    /* This should never happen: it is pure paranoia and to avoid icc being crazy*/
    if(RegionInd >= Nregions)
        endrun(1, "Particle %d has region %d out of bounds %d\n", i, RegionInd, Nregions);

    PetaPMRegion * region = &regions[RegionInd];
    for(k = 0; k < 3; k++) {
        double tmp = Pos[k] / pm->CellSize;
        iCell[k] = floor(tmp);
        Res[k] = tmp - iCell[k];
        iCell[k] -= region->offset[k];
        /* seriously?! particles are supposed to be contained in cells */
        if(iCell[k] >= region->size[k] - 1 || iCell[k] < 0) {
            endrun(1, "particle out of cell better stop %d (k=%d) %g %g %g region: %td %td\n", iCell[k],k,
                Pos[0], Pos[1], Pos[2],
                region->offset[k], region->size[k]);
        }
    }

    int connection;
    for(connection = 0; connection < 8; connection++) {
        double weight = 1.0;
        size_t linear = 0;
        for(k = 0; k < 3; k++) {
            int offset = (connection >> k) & 1;
            int tmp = iCell[k] + offset;
            linear += tmp * region->strides[k];
            weight *= offset?
                /* offset == 1*/ (Res[k])    :
                /* offset == 0*/ (1 - Res[k]);
        }
        if(linear >= region->totalsize) {
            endrun(1, "particle linear index out of cell better stop\n");
        }
        iterator(pm, i, &region->buffer[linear], weight);
    }
}

/*
 * iterate over all particle / mesh pairs, call iterator
 * function . iterator function shall be aware of thread safety.
 * no threads run on same particle same time but may
 * access one mesh points same time.
 * */
static void pm_iterate(PetaPM * pm, pm_iterator iterator, PetaPMRegion * regions, const int Nregions) {
    int i;
#pragma omp parallel for
    for(i = 0; i < CPS->NumPart; i ++) {
        pm_iterate_one(pm, i, iterator, regions, Nregions);
    }
    MPIU_Barrier(pm->comm);
}

void petapm_region_init_strides(PetaPMRegion * region) {
    int k;
    size_t rt = 1;
    for(k = 2; k >= 0; k --) {
        region->strides[k] = rt;
        rt = region->size[k] * rt;
    }
    region->totalsize = rt;
    region->buffer = NULL;
}

static int pos_get_target(PetaPM * pm, const int pos[2]) {
    int k;
    int task2d[2];
    int rank;
    for(k = 0; k < 2; k ++) {
        int ix = pos[k];
        while(ix < 0) ix += pm->Nmesh;
        while(ix >= pm->Nmesh) ix -= pm->Nmesh;
        task2d[k] = pm->Mesh2Task[k][ix];
    }
    MPI_Cart_rank(pm->priv->comm_cart_2d, task2d, &rank);
    return rank;
}
static int pencil_cmp_target(const void * v1, const void * v2) {
    const struct Pencil * p1 = v1;
    const struct Pencil * p2 = v2;
    /* move zero length pixels to the end */
    if(p2->len == 0) return -1;
    if(p1->len == 0) return 1;
    int t1 = p1->task;
    int t2 = p2->task;
    return ((t2 < t1) - (t1 < t2)) * 2 +
        ((p2->meshbuf_first < p1->meshbuf_first) - (p1->meshbuf_first < p2->meshbuf_first));
}

#ifdef DEBUG
static void verify_density_field(PetaPM * pm, double * real, double * meshbuf, const size_t meshsize) {
    /* verify the density field */
    double mass_Part = 0;
    int j;
#pragma omp parallel for reduction(+: mass_Part)
    for(j = 0; j < CPS->NumPart; j ++) {
        double Mass = *MASS(j);
        mass_Part += Mass;
    }
    double totmass_Part = 0;
    MPI_Allreduce(&mass_Part, &totmass_Part, 1, MPI_DOUBLE, MPI_SUM, pm->comm);

    double mass_Region = 0;
    size_t i;

#pragma omp parallel for reduction(+: mass_Region)
    for(i = 0; i < meshsize; i ++) {
        mass_Region += meshbuf[i];
    }
    double totmass_Region = 0;
    MPI_Allreduce(&mass_Region, &totmass_Region, 1, MPI_DOUBLE, MPI_SUM, pm->comm);
    double mass_CIC = 0;
#pragma omp parallel for reduction(+: mass_CIC)
    for(i = 0; i < pm->real_space_region.totalsize; i ++) {
        mass_CIC += real[i];
    }
    double totmass_CIC = 0;
    MPI_Allreduce(&mass_CIC, &totmass_CIC, 1, MPI_DOUBLE, MPI_SUM, pm->comm);

    message(0, "total Region mass err = %g CIC mass err = %g Particle mass = %g\n", totmass_Region / totmass_Part - 1, totmass_CIC / totmass_Part - 1, totmass_Part);
}
#endif

static void pm_apply_transfer_function(PetaPM * pm,
        pfft_complex * src,
        pfft_complex * dst, petapm_transfer_func H
        ){
    size_t ip = 0;

    PetaPMRegion * region = &pm->fourier_space_region;

#pragma omp parallel for
    for(ip = 0; ip < region->totalsize; ip ++) {
        ptrdiff_t tmp = ip;
        int pos[3];
        int kpos[3];
        int64_t k2 = 0.0;
        int k;
        for(k = 0; k < 3; k ++) {
            pos[k] = tmp / region->strides[k];
            tmp -= pos[k] * region->strides[k];
            /* lets get the abs pos on the grid*/
            pos[k] += region->offset[k];
            /* check */
            if(pos[k] >= pm->Nmesh) {
                endrun(1, "position didn't make sense\n");
            }
            kpos[k] = petapm_mesh_to_k(pm, pos[k]);
            /* Watch out the cast */
            k2 += ((int64_t)kpos[k]) * kpos[k];
        }
        /* swap 0 and 1 because fourier space was transposed */
        /* kpos is y, z, x */
        pos[0] = kpos[2];
        pos[1] = kpos[0];
        pos[2] = kpos[1];
        dst[ip][0] = src[ip][0];
        dst[ip][1] = src[ip][1];
        if(H) {
            H(pm, k2, pos, &dst[ip]);
        }
    }

}


/**************
 * functions iterating over particle / mesh pairs
 ***************/
static void put_particle_to_mesh(PetaPM * pm, int i, double * mesh, double weight) {
    double Mass = *MASS(i);
    if(INACTIVE(i))
        return;
#pragma omp atomic update
    mesh[0] += weight * Mass;
}
//escape fraction scaled GSM
static void put_star_to_mesh(PetaPM * pm, int i, double * mesh, double weight) {
    if(INACTIVE(i) || *TYPE(i) != 4)
        return;
    double Mass = *MASS(i);
    double fesc = *FESC(i);
#pragma omp atomic update
    mesh[0] += weight * Mass * fesc;
}
//escape fraciton scaled SFR
static void put_sfr_to_mesh(PetaPM * pm, int i, double * mesh, double weight) {
    if(INACTIVE(i) || *TYPE(i) != 0)
        return;
    double Sfr = *SFR(i);
    double fesc = 1.;
    //double fesc = *FESC(i);
#pragma omp atomic update
    mesh[0] += weight * Sfr * fesc;
}
static int64_t reduce_int64(int64_t input, MPI_Comm comm) {
    int64_t result = 0;
    MPI_Allreduce(&input, &result, 1, MPI_INT64, MPI_SUM, comm);
    return result;
}

/** Some FFT notes
 *
 *
 * CFT = dx * iDFT (thus CFT has no 2pi factors and iCFT has,
 *           same as wikipedia.)
 *
 * iCFT = dk * DFT
 * iCFT(CFG) = dx * dk * DFT(iDFT)
 *           = L / N * (2pi / L) * N
 *           = 2 pi
 * agreed with the usual def that
 * iCFT(CFT) = 2pi
 *
 * **************************8*/
