#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

#include <bigfile-mpi.h>

#include "allvars.h"
#include "sfr_eff.h"
#include "cooling.h"
#include "timestep.h"

#include "petaio.h"
#include "slotsmanager.h"
#include "hydra.h"
#include "density.h"
#include "partmanager.h"
#include "config.h"
#include "neutrinos_lra.h"

#include "utils.h"
#include "uvbg.h"

/************
 *
 * The IO api , intented to replace io.c and read_ic.c
 * currently we have a function to register the blocks and enumerate the blocks
 *
 */
static struct petaio_params {
    size_t BytesPerFile;   /* Number of bytes per physical file; this decides how many files bigfile creates each block */
    int WritersPerFile;    /* Number of concurrent writers per file; this decides number of writers */
    int NumWriters;        /* Number of concurrent writers, this caps number of writers */
    int MinNumWriters;        /* Min Number of concurrent writers, this caps number of writers */
    int EnableAggregatedIO;  /* Enable aggregated IO policy for small files.*/
    size_t AggregatedIOThreshold; /* bytes per writer above which to use non-aggregated IO (avoid OOM)*/
    /* Changes the comoving factors of the snapshot outputs. Set in the ICs.
     * If UsePeculiarVelocity = 1 then snapshots save to the velocity field the physical peculiar velocity, v = a dx/dt (where x is comoving distance).
     * If UsePeculiarVelocity = 0 then the velocity field is a * v = a^2 dx/dt in snapshots
     * and v / sqrt(a) = sqrt(a) dx/dt in the ICs. Note that snapshots never match Gadget-2, which
     * saves physical peculiar velocity / sqrt(a) in both ICs and snapshots. */
    int UsePeculiarVelocity;
} IO;

/*Set the IO parameters*/
void
set_petaio_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        IO.BytesPerFile = param_get_int(ps, "BytesPerFile");
        IO.UsePeculiarVelocity = 0; /* Will be set by the Initial Condition File */
        IO.NumWriters = param_get_int(ps, "NumWriters");
        IO.MinNumWriters = param_get_int(ps, "MinNumWriters");
        IO.WritersPerFile = param_get_int(ps, "WritersPerFile");
        IO.AggregatedIOThreshold = param_get_int(ps, "AggregatedIOThreshold");
        IO.EnableAggregatedIO = param_get_int(ps, "EnableAggregatedIO");

    }
    MPI_Bcast(&IO, sizeof(struct petaio_params), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int GetUsePeculiarVelocity(void)
{
    return IO.UsePeculiarVelocity;
}

static void petaio_write_header(BigFile * bf, const int64_t * NTotal);
static void petaio_read_header_internal(BigFile * bf);

/* these are only used in reading in */
void petaio_init(void) {
    /* Smaller files will do aggregated IO.*/
    if(IO.EnableAggregatedIO) {
        message(0, "Aggregated IO is enabled\n");
        big_file_mpi_set_aggregated_threshold(IO.AggregatedIOThreshold);
    } else {
        message(0, "Aggregated IO is disabled.\n");
        big_file_mpi_set_aggregated_threshold(0);
    }
    if(IO.NumWriters == 0)
        MPI_Comm_size(MPI_COMM_WORLD, &IO.NumWriters);
}

/* save a snapshot file */
static void petaio_save_internal(char * fname, struct IOTable * IOTable, int verbose);

void
petaio_save_snapshot(struct IOTable * IOTable, int verbose, const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char * fname = fastpm_strdup_vprintf(fmt, va);
    va_end(va);
    message(0, "saving snapshot into %s\n", fname);

    petaio_save_internal(fname, IOTable, verbose);
    myfree(fname);
}

/* Build a list of the first particle of each type on the current processor.
 * This assumes that all particles are sorted!*/
/**
 * Create a Selection array for the buffers. This array indirectly sort
 * the particles by the type.
 *
 * The offset for the starting of each type is stored in ptype_offset.
 *
 * if select_func is provided, it shall return 1 for those that shall be
 * included in the output.
 */
void
petaio_build_selection(int * selection,
    int * ptype_offset,
    int * ptype_count,
    const struct particle_data * Parts,
    const int NumPart,
    int (*select_func)(int i, const struct particle_data * Parts)
    )
{
    int i;
    ptype_offset[0] = 0;
    ptype_count[0] = 0;

    for(i = 0; i < NumPart; i ++) {
        if(P[i].IsGarbage)
            continue;
        if((select_func == NULL) || (select_func(i, Parts) != 0)) {
            int ptype = Parts[i].Type;
            ptype_count[ptype] ++;
        }
    }
    for(i = 1; i < 6; i ++) {
        ptype_offset[i] = ptype_offset[i-1] + ptype_count[i-1];
        ptype_count[i-1] = 0;
    }

    ptype_count[5] = 0;
    for(i = 0; i < NumPart; i ++) {
        int ptype = Parts[i].Type;
        if(P[i].IsGarbage)
            continue;
        if((select_func == NULL) || (select_func(i, Parts) != 0)) {
            selection[ptype_offset[ptype] + ptype_count[ptype]] = i;
            ptype_count[ptype]++;
        }
    }
}

static void petaio_save_internal(char * fname, struct IOTable * IOTable, int verbose) {
    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    int ptype_offset[6]={0};
    int ptype_count[6]={0};
    int64_t NTotal[6]={0};

    int * selection = mymalloc("Selection", sizeof(int) * PartManager->NumPart);

    petaio_build_selection(selection, ptype_offset, ptype_count, P, PartManager->NumPart, NULL);

    sumup_large_ints(6, ptype_count, NTotal);

    petaio_write_header(&bf, NTotal);

    int i;
    for(i = 0; i < IOTable->used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable->ent[i].ptype;
        BigArray array = {0};
        /*This exclude FOF blocks*/
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable->ent[i].name);
        petaio_build_buffer(&array, &IOTable->ent[i], selection + ptype_offset[ptype], ptype_count[ptype], P, SlotsManager);
        petaio_save_block(&bf, blockname, &array, verbose);
        petaio_destroy_buffer(&array);
    }

    if(All.MassiveNuLinRespOn) {
        int ThisTask;
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        petaio_save_neutrinos(&bf, ThisTask);
    }
    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)){
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    myfree(selection);
}

void petaio_read_internal(char * fname, int ic, struct IOTable * IOTable, MPI_Comm Comm) {
    int ptype;
    int i;
    BigFile bf = {0};
    BigBlock bh;
    message(0, "Reading snapshot %s\n", fname);

    int NTask, ThisTask;
    MPI_Comm_size(Comm, &NTask);
    MPI_Comm_rank(Comm, &ThisTask);

    if(0 != big_file_mpi_open(&bf, fname, Comm)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    int64_t NTotal[6];
    if(0 != big_file_mpi_open_block(&bf, &bh, "Header", Comm)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    if ((0 != big_block_get_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
        (0 != big_block_mpi_close(&bh, Comm))) {
        endrun(0, "Failed to close block: %s\n",
                    big_file_get_error_message());
    }
    /*Read neutrinos from the snapshot if necessary*/
    if(All.MassiveNuLinRespOn) {
        size_t nk = All.Nmesh;
        /* Get the nk and do allocation. */
        if(!ic) {
            BigBlock bn;
            if(0 != big_file_mpi_open_block(&bf, &bn, "Neutrino", MPI_COMM_WORLD)) {
                endrun(0, "Failed to open block at %s:%s\n", "Neutrino",
                            big_file_get_error_message());
            }
            if(0 != big_block_get_attr(&bn, "Nkval", &nk, "u8", 1)) {
                endrun(0, "Failed to read attr: %s\n",
                            big_file_get_error_message());
            }
            if(0 != big_block_mpi_close(&bn, MPI_COMM_WORLD)) {
                endrun(0, "Failed to close block %s\n",
                            big_file_get_error_message());
            }
        }
        init_neutrinos_lra(nk, All.TimeIC, All.TimeMax, All.CP.Omega0, &All.CP.ONu, All.UnitTime_in_s, CM_PER_MPC);
        /*Read the neutrino transfer function from the ICs*/
        if(ic)
            petaio_read_icnutransfer(&bf, ThisTask);
        else
            petaio_read_neutrinos(&bf, ThisTask);
    }

    /* sets the maximum number of particles that may reside on a processor */
    int MaxPart = (int) (All.PartAllocFactor * All.TotNumPartInit / NTask);

    /*Allocate the particle memory*/
    particle_alloc_memory(MaxPart);

    int64_t NLocal[6];

    /*Allocate Permanent UV grids*/
    //if(All.ExcursionSetFlag)
    //    malloc_permanent_uvbg_grids();

    int NLocal[6];
    for(ptype = 0; ptype < 6; ptype ++) {
        int64_t start = ThisTask * NTotal[ptype] / NTask;
        int64_t end = (ThisTask + 1) * NTotal[ptype] / NTask;
        NLocal[ptype] = end - start;
        PartManager->NumPart += NLocal[ptype];
    }

    /* Allocate enough memory for stars and black holes.
     * This will be dynamically increased as needed.*/

    if(PartManager->NumPart >= PartManager->MaxPart) {
        endrun(1, "Overwhelmed by part: %ld > %ld\n", PartManager->NumPart, PartManager->MaxPart);
    }

    /* Now allocate memory for the secondary particle data arrays.
     * This may be dynamically resized later!*/

    /*Ensure all processors have initially the same number of particle slots*/
    int64_t newSlots[6] = {0};

    /* Can't use MPI_IN_PLACE, which is broken for arrays and MPI_MAX at least on intel mpi 19.0.5*/
    MPI_Allreduce(NLocal, newSlots, 6, MPI_INT64, MPI_MAX, Comm);

    for(ptype = 0; ptype < 6; ptype ++) {
            newSlots[ptype] *= All.PartAllocFactor;
    }
    /* Boost initial amount of stars allocated, as it is often uneven.
     * The total number of stars is usually small so this doesn't
     * waste that much memory*/
    newSlots[4] *= 2;

    slots_reserve(0, newSlots, SlotsManager);

    /* so we can set up the memory topology of secondary slots */
    slots_setup_topology(PartManager, NLocal, SlotsManager);

    for(i = 0; i < IOTable->used; i ++) {
        /* only process the particle blocks */
        char blockname[128];
        int ptype = IOTable->ent[i].ptype;
        BigArray array = {0};
        if(!(ptype < 6 && ptype >= 0)) {
            continue;
        }
        if(NTotal[ptype] == 0) continue;
        if(ic) {
            /* for IC read in only three blocks */
            int keep = 0;
            keep |= (0 == strcmp(IOTable->ent[i].name, "Position"));
            keep |= (0 == strcmp(IOTable->ent[i].name, "Velocity"));
            keep |= (0 == strcmp(IOTable->ent[i].name, "ID"));
            if (ptype == 5) {
                keep |= (0 == strcmp(IOTable->ent[i].name, "Mass"));
                keep |= (0 == strcmp(IOTable->ent[i].name, "BlackholeMass"));
                keep |= (0 == strcmp(IOTable->ent[i].name, "MinPotPos"));
            }
            if(!keep) continue;
        }
        if(IOTable->ent[i].setter == NULL) {
            /* FIXME: do not know how to read this block; assume the fucker is
             * internally intialized; */
            continue;
        }
        sprintf(blockname, "%d/%s", ptype, IOTable->ent[i].name);
        petaio_alloc_buffer(&array, &IOTable->ent[i], NLocal[ptype]);
        if(0 == petaio_read_block(&bf, blockname, &array, IOTable->ent[i].required))
            petaio_readout_buffer(&array, &IOTable->ent[i]);
        petaio_destroy_buffer(&array);
    }

    if(0 != big_file_mpi_close(&bf, Comm)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
    /* now we have IDs, set up the ID consistency between slots. */
    slots_setup_id(PartManager, SlotsManager);
}

void
petaio_read_header(int num)
{
    BigFile bf = {0};

    char * fname;
    if(num == -1) {
        fname = fastpm_strdup_printf("%s", All.InitCondFile);
    } else {
        fname = fastpm_strdup_printf("%s/%s_%03d", All.OutputDir, All.SnapshotFileBase, num);
    }
    message(0, "Probing Header of snapshot file: %s\n", fname);

    if(0 != big_file_mpi_open(&bf, fname, MPI_COMM_WORLD)) {
        endrun(0, "Failed to open snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }

    petaio_read_header_internal(&bf);

    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close snapshot at %s:%s\n", fname,
                    big_file_get_error_message());
    }
    myfree(fname);
}

void
petaio_read_snapshot(int num, MPI_Comm Comm)
{
    char * fname;
    struct IOTable IOTable = {0};

    register_io_blocks(&IOTable, 0);

    if(num == -1) {
        fname = fastpm_strdup_printf("%s", All.InitCondFile);
        /*
         *  IC doesn't have entropy or energy; always use the
         *  InitTemp in paramfile, then use init.c to convert to
         *  entropy.
         * */
        petaio_read_internal(fname, 1, &IOTable, Comm);

        int i;
        /* touch up the mass -- IC files save mass in header */
        #pragma omp parallel for
        for(i = 0; i < PartManager->NumPart; i++)
        {
            P[i].Mass = All.MassTable[P[i].Type];
        }

        if (!IO.UsePeculiarVelocity ) {
            /* fixing the unit of velocity from Legacy GenIC IC */
            #pragma omp parallel for
            for(i = 0; i < PartManager->NumPart; i++) {
                int k;
                /* for GenIC's Gadget-1 snapshot Unit to Gadget-2 Internal velocity unit */
                for(k = 0; k < 3; k++)
                    P[i].Vel[k] *= sqrt(All.cf.a) * All.cf.a;
            }

        }
    } else {
        fname = fastpm_strdup_printf("%s/%s_%03d", All.OutputDir, All.SnapshotFileBase, num);
        /*
         * we always save the Entropy, init.c will not mess with the entropy
         * */
        petaio_read_internal(fname, 0, &IOTable, Comm);

        /*if we are doing the excursion set, load in the star grid
         * requires the last UV grid to be output
         * */
        //TODO (jdavies) make this not crash if the file doesn't exist:
        //if(All.ExcursionSetFlag)
            //read_star_grids(num);

    }
    myfree(fname);
}


/* write a header block */
static void petaio_write_header(BigFile * bf, const int64_t * NTotal) {
    BigBlock bh;
    if(0 != big_file_mpi_create_block(bf, &bh, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                big_file_get_error_message());
    }

    /* conversion from peculiar velocity to RSD */
    double RSD = 1.0 / (All.cf.a * All.cf.hubble);

    if(!IO.UsePeculiarVelocity) {
        RSD /= All.cf.a; /* Conversion from internal velocity to RSD */
    }

    int dk = GetDensityKernelType();
    if(
    (0 != big_block_set_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "TotNumPartInit", All.NTotalInit, "u8", 6)) ||
    (0 != big_block_set_attr(&bh, "MassTable", All.MassTable, "f8", 6)) ||
    (0 != big_block_set_attr(&bh, "Time", &All.Time, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "TimeIC", &All.TimeIC, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaLambda", &All.CP.OmegaLambda, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "RSDFactor", &RSD, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UsePeculiarVelocity", &IO.UsePeculiarVelocity, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "Omega0", &All.CP.Omega0, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "CMBTemperature", &All.CP.CMBTemperature, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "OmegaBaryon", &All.CP.OmegaBaryon, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitLength_in_cm", &All.UnitLength_in_cm, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitMass_in_g", &All.UnitMass_in_g, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "UnitVelocity_in_cm_per_s", &All.UnitVelocity_in_cm_per_s, "f8", 1)) ||
    (0 != big_block_set_attr(&bh, "CodeVersion", GADGET_VERSION, "S1", strlen(GADGET_VERSION))) ||
    (0 != big_block_set_attr(&bh, "CompilerSettings", GADGET_COMPILER_SETTINGS, "S1", strlen(GADGET_COMPILER_SETTINGS))) ||
    (0 != big_block_set_attr(&bh, "DensityKernel", &dk, "i4", 1)) ||
    (0 != big_block_set_attr(&bh, "HubbleParam", &All.CP.HubbleParam, "f8", 1)) ) {
        endrun(0, "Failed to write attributes %s\n",
                    big_file_get_error_message());
    }

    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block %s\n",
                    big_file_get_error_message());
    }
}
static double
_get_attr_double(BigBlock * bh, char * name, double def)
{
    double foo;
    if(0 != big_block_get_attr(bh, name, &foo, "f8", 1)) {
        foo = def;
    }
    return foo;
}
static int
_get_attr_int(BigBlock * bh, char * name, int def)
{
    int foo;
    if(0 != big_block_get_attr(bh, name, &foo, "i4", 1)) {
        foo = def;
    }
    return foo;
}

static void
petaio_read_header_internal(BigFile * bf) {
    BigBlock bh;
    if(0 != big_file_mpi_open_block(bf, &bh, "Header", MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", "Header",
                    big_file_get_error_message());
    }
    double Time = 0.;
    int ptype;
    int64_t NTotal[6];
    if(
    (0 != big_block_get_attr(&bh, "TotNumPart", NTotal, "u8", 6)) ||
    (0 != big_block_get_attr(&bh, "MassTable", All.MassTable, "f8", 6)) ||
    (0 != big_block_get_attr(&bh, "BoxSize", &All.BoxSize, "f8", 1)) ||
    (0 != big_block_get_attr(&bh, "Time", &Time, "f8", 1))
    ) {
        endrun(0, "Failed to read attr: %s\n",
                    big_file_get_error_message());
    }

    /*Set Nmesh to triple the mean grid spacing of the dark matter by default.*/
    if(All.Nmesh  < 0)
        All.Nmesh = 3*pow(2, (int)(log(NTotal[1])/3./log(2)) );
    All.TimeInit = Time;
    if(0!= big_block_get_attr(&bh, "TimeIC", &All.TimeIC, "f8", 1))
        All.TimeIC = Time;

    /* Set a reasonable MassTable entry for stars and BHs*/
    All.MassTable[4] = All.MassTable[0] / get_generations();
    All.MassTable[5] = All.MassTable[4];

    /* fall back to traditional MP-Gadget Units if not given in the snapshot file. */
    All.UnitVelocity_in_cm_per_s = _get_attr_double(&bh, "UnitVelocity_in_cm_per_s", 1e5); /* 1 km/sec */
    All.UnitLength_in_cm = _get_attr_double(&bh, "UnitLength_in_cm",  3.085678e21); /* 1.0 Kpc /h */
    All.UnitMass_in_g = _get_attr_double(&bh, "UnitMass_in_g", 1.989e43); /* 1e10 Msun/h */

    double OmegaBaryon = All.CP.OmegaBaryon;
    double HubbleParam = All.CP.HubbleParam;
    double OmegaLambda = All.CP.OmegaLambda;

    if(0 == big_block_get_attr(&bh, "OmegaBaryon", &All.CP.OmegaBaryon, "f8", 1) &&
            OmegaBaryon > 0 && fabs(All.CP.OmegaBaryon - OmegaBaryon) > 1e-3)
        message(0,"IC Header has Omega_b = %g but paramfile wants %g\n", All.CP.OmegaBaryon, OmegaBaryon);
    if(0 == big_block_get_attr(&bh, "HubbleParam", &All.CP.HubbleParam, "f8", 1) &&
            HubbleParam > 0 && fabs(All.CP.HubbleParam - HubbleParam) > 1e-3)
        message(0,"IC Header has HubbleParam = %g but paramfile wants %g\n", All.CP.HubbleParam, HubbleParam);
    if(0 == big_block_get_attr(&bh, "OmegaLambda", &All.CP.OmegaLambda, "f8", 1) &&
            OmegaLambda > 0 && fabs(All.CP.OmegaLambda - OmegaLambda) > 1e-3)
        message(0,"IC Header has Omega_L = %g but paramfile wants %g\n", All.CP.OmegaLambda, OmegaLambda);

    /* If UsePeculiarVelocity = 1 then snapshots save to the velocity field the physical peculiar velocity, v = a dx/dt (where x is comoving distance).
     * If UsePeculiarVelocity = 0 then the velocity field is a * v = a^2 dx/dt in snapshots
     * and v / sqrt(a) = sqrt(a) dx/dt in the ICs. Note that snapshots never match Gadget-2, which
     * saves physical peculiar velocity / sqrt(a) in both ICs and snapshots. */
    IO.UsePeculiarVelocity = _get_attr_int(&bh, "UsePeculiarVelocity", 0);

    if(0 != big_block_get_attr(&bh, "TotNumPartInit", All.NTotalInit, "u8", 6)) {
        int ptype;
        for(ptype = 0; ptype < 6; ptype ++) {
            All.NTotalInit[ptype] = NTotal[ptype];
        }
    }

    int64_t TotNumPart = 0;
    All.TotNumPartInit = 0;
    for(ptype = 0; ptype < 6; ptype ++) {
        TotNumPart += NTotal[ptype];
        All.TotNumPartInit += All.NTotalInit[ptype];
        if(All.NTotalInit[ptype] > 0) {
            All.MeanSeparation[ptype] = All.BoxSize / pow(All.NTotalInit[ptype], 1.0 / 3);
        } else {
            All.MeanSeparation[ptype] = 0;
        }
    }

    message(0, "Total number of particles: %018ld\n", TotNumPart);

    const char * PARTICLE_TYPE_NAMES [] = {"Gas", "DarkMatter", "Neutrino", "Unknown", "Star", "BlackHole"};

    for(ptype = 0; ptype < 6; ptype ++) {
        message(0, "% 11s: Total: %018ld Init: %018ld Mean-Sep %g \n",
                PARTICLE_TYPE_NAMES[ptype], NTotal[ptype], All.NTotalInit[ptype], All.MeanSeparation[ptype]);
    }

    /*FIXME: check others as well */
    /*
    big_block_get_attr(&bh, "OmegaLambda", &All.OmegaLambda, "f8", 1);
    big_block_get_attr(&bh, "Omega0", &All.Omega0, "f8", 1);
    big_block_get_attr(&bh, "HubbleParam", &All.HubbleParam, "f8", 1);
    */
    if(0 != big_block_mpi_close(&bh, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block: %s\n",
                    big_file_get_error_message());
    }
}

void petaio_alloc_buffer(BigArray * array, IOTableEntry * ent, int64_t localsize) {
    size_t dims[2];
    ptrdiff_t strides[2];
    int elsize = dtype_itemsize(ent->dtype);

    dims[0] = localsize;
    dims[1] = ent->items;
    strides[1] = elsize;
    strides[0] = elsize * ent->items;
    char * buffer = mymalloc("IOBUFFER", dims[0] * dims[1] * elsize);

    big_array_init(array, buffer, ent->dtype, 2, dims, strides);
}

/* readout array into P struct with setters */
void petaio_readout_buffer(BigArray * array, IOTableEntry * ent) {
    int i;
    /* fill the buffer */
    char * p = array->data;
    for(i = 0; i < PartManager->NumPart; i ++) {
        if(P[i].Type != ent->ptype) continue;
        ent->setter(i, p, P, SlotsManager);
        p += array->strides[0];
    }
}
/* build an IO buffer for block, based on selection
 * only check P[ selection[i]]. If selection is NULL, just use P[i].
 * NOTE: selected range should contain only one particle type!
*/
void
petaio_build_buffer(BigArray * array, IOTableEntry * ent, const int * selection, const int NumSelection, struct particle_data * Parts, struct slots_manager_type * SlotsManager)
{
    if(selection == NULL) {
        endrun(-1, "NULL selection is not supported\n");
    }

    /* don't forget to free buffer after its done*/
    petaio_alloc_buffer(array, ent, NumSelection);

    /* Fast code path if there are no such particles */
    if(NumSelection == 0) {
        return;
    }

#pragma omp parallel
    {
        int i;
        const int tid = omp_get_thread_num();
        const int NT = omp_get_num_threads();
        const int start = NumSelection * (size_t) tid / NT;
        const int end = NumSelection * ((size_t) tid + 1) / NT;
        /* fill the buffer */
        char * p = array->data;
        p += array->strides[0] * start;
        for(i = start; i < end; i ++) {
            const int j = selection[i];
            if(Parts[j].Type != ent->ptype) {
                endrun(2, "Selection %d has type = %d != %d\n", j, Parts[j].Type, ent->ptype);
            }
            ent->getter(j, p, Parts, SlotsManager);
            p += array->strides[0];
        }
    }
}

/* destroy a buffer, freeing its memory */
void petaio_destroy_buffer(BigArray * array) {
    myfree(array->data);
}

/* read a block from disk, spread the values to memory with setters  */
int petaio_read_block(BigFile * bf, char * blockname, BigArray * array, int required) {
    BigBlock bb;
    BigBlockPtr ptr;

    /* open the block */
    if(0 != big_file_mpi_open_block(bf, &bb, blockname, MPI_COMM_WORLD)) {
        if(required)
            endrun(0, "Failed to open block at %s:%s\n", blockname, big_file_get_error_message());
        else
            return 1;
    }
    if(0 != big_block_seek(&bb, &ptr, 0)) {
            endrun(1, "Failed to seek block %s: %s\n", blockname, big_file_get_error_message());
    }
    if(0 != big_block_mpi_read(&bb, &ptr, array, IO.NumWriters, MPI_COMM_WORLD)) {
        endrun(1, "Failed to read from block %s: %s\n", blockname, big_file_get_error_message());
    }
    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block at %s:%s\n", blockname,
                    big_file_get_error_message());
    }
    return 0;
}

/* save a block to disk */
void petaio_save_block(BigFile * bf, char * blockname, BigArray * array, int verbose)
{

    BigBlock bb;
    BigBlockPtr ptr;

    int elsize = big_file_dtype_itemsize(array->dtype);

    int NumWriters = IO.NumWriters;

    size_t size = count_sum(array->dims[0]);
    int NumFiles;

    if(IO.EnableAggregatedIO) {
        NumFiles = (size * elsize + IO.BytesPerFile - 1) / IO.BytesPerFile;
        if(NumWriters > NumFiles * IO.WritersPerFile) {
            NumWriters = NumFiles * IO.WritersPerFile;
            message(0, "Throttling NumWriters to %d.\n", NumWriters);
        }
        if(NumWriters < IO.MinNumWriters) {
            NumWriters = IO.MinNumWriters;
            NumFiles = (NumWriters + IO.WritersPerFile - 1) / IO.WritersPerFile ;
            message(0, "Throttling NumWriters to %d.\n", NumWriters);
        }
    } else {
        NumFiles = NumWriters;
    }
    /*Do not write empty files*/
    if(size == 0) {
        NumFiles = 0;
    }

    if(verbose && size > 0) {
        message(0, "Will write %td particles to %d Files for %s\n", size, NumFiles, blockname);
    }
    /* create the block */
    /* dims[1] is the number of members per item */
    if(0 != big_file_mpi_create_block(bf, &bb, blockname, array->dtype, array->dims[1], NumFiles, size, MPI_COMM_WORLD)) {
        endrun(0, "Failed to create block at %s:%s\n", blockname,
                    big_file_get_error_message());
    }
    if(0 != big_block_seek(&bb, &ptr, 0)) {
        endrun(0, "Failed to seek:%s\n", big_file_get_error_message());
    }
    if(0 != big_block_mpi_write(&bb, &ptr, array, NumWriters, MPI_COMM_WORLD)) {
        endrun(0, "Failed to write :%s\n", big_file_get_error_message());
    }

    if(verbose && size > 0)
        message(0, "Done writing %td particles to %d Files\n", size, NumFiles);

    if(0 != big_block_mpi_close(&bb, MPI_COMM_WORLD)) {
        endrun(0, "Failed to close block at %s:%s\n", blockname,
                big_file_get_error_message());
    }
}

/*
 * register an IO block of name for particle type ptype.
 *
 * use IO_REG wrapper.
 *
 * with getter function getter
 * getter(i, output)
 * will fill the property of particle i to output.
 *
 * NOTE: dtype shall match the format of output of getter
 *
 * NOTE: currently there is a hard limit (4096 blocks ).
 *
 * */
void io_register_io_block(char * name,
        char * dtype,
        int items,
        int ptype,
        property_getter getter,
        property_setter setter,
        int required,
        struct IOTable * IOTable
        ) {
    if (IOTable->used == IOTable->allocated) {
        IOTable->ent = myrealloc(IOTable->ent, 2*IOTable->allocated*sizeof(IOTableEntry));
        IOTable->allocated *= 2;
    }
    IOTableEntry * ent = &IOTable->ent[IOTable->used];
    strncpy(ent->name, name, 63);
    ent->name[63] = '\0';
    ent->zorder = IOTable->used;
    ent->ptype = ptype;
    strncpy(ent->dtype, dtype, 7);
    ent->dtype[7] = '\0';
    ent->getter = getter;
    ent->setter = setter;
    ent->items = items;
    ent->required = required;
    IOTable->used ++;
}

static void GTPosition(int i, double * out, void * baseptr, void * smanptr) {
    /* Remove the particle offset before saving*/
    struct particle_data * part = (struct particle_data *) baseptr;
    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = part[i].Pos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > All.BoxSize) out[d] -= All.BoxSize;
        while(out[d] <= 0) out[d] += All.BoxSize;
    }
}

static void STPosition(int i, double * out, void * baseptr, void * smanptr) {
    int d;
    struct particle_data * part = (struct particle_data *) baseptr;
    for(d = 0; d < 3; d ++) {
        part[i].Pos[d] = out[d];
    }
}

#define SIMPLE_PROPERTY(name, field, type, items) \
    SIMPLE_GETTER(GT ## name , field, type, items, struct particle_data) \
    SIMPLE_SETTER(ST ## name , field, type, items, struct particle_data)
/*A property with getters and setters that are type specific*/
#define SIMPLE_PROPERTY_TYPE(name, ptype, field, type, items) \
    SIMPLE_GETTER(GT ## ptype ## name , field, type, items, struct particle_data) \
    SIMPLE_SETTER(ST ## ptype ## name , field, type, items, struct particle_data)

/* A property that uses getters and setters via the PI of a particle data array.*/
#define SIMPLE_GETTER_PI(name, field, dtype, items, slottype) \
static void name(int i, dtype * out, void * baseptr, void * smanptr) { \
    int PI = ((struct particle_data *) baseptr)[i].PI; \
    int ptype = ((struct particle_data *) baseptr)[i].Type; \
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[ptype]); \
    slottype * sl = (slottype *) info->ptr; \
    int k; \
    for(k = 0; k < items; k ++) { \
        out[k] = *(&(sl[PI].field) + k); \
    } \
}

#define SIMPLE_SETTER_PI(name, field, dtype, items, slottype) \
static void name(int i, dtype * out, void * baseptr, void * smanptr) { \
    int PI = ((struct particle_data *) baseptr)[i].PI; \
    int ptype = ((struct particle_data *) baseptr)[i].Type; \
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[ptype]); \
    slottype * sl = (slottype *) info->ptr; \
    int k; \
    for(k = 0; k < items; k ++) { \
        *(&(sl[PI].field) + k) = out[k]; \
    } \
}

#define SIMPLE_PROPERTY_PI(name, field, type, items, slottype) \
    SIMPLE_GETTER_PI(GT ## name , field, type, items, slottype) \
    SIMPLE_SETTER_PI(ST ## name , field, type, items, slottype)
#define SIMPLE_PROPERTY_TYPE_PI(name, ptype, field, type, items, slottype) \
    SIMPLE_GETTER_PI(GT ## ptype ## name , field, type, items, slottype) \
    SIMPLE_SETTER_PI(ST ## ptype ## name , field, type, items, slottype)

static void GTVelocity(int i, float * out, void * baseptr, void * smanptr) {
    /* Convert to Peculiar Velocity if UsePeculiarVelocity is set */
    double fac;
    struct particle_data * part = (struct particle_data *) baseptr;
    if (IO.UsePeculiarVelocity) {
        fac = 1.0 / All.cf.a;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = fac * part[i].Vel[d];
    }
}
static void STVelocity(int i, float * out, void * baseptr, void * smanptr) {
    double fac;
    struct particle_data * part = (struct particle_data *) baseptr;
    if (IO.UsePeculiarVelocity) {
        fac = All.cf.a;
    } else {
        fac = 1.0;
    }

    int d;
    for(d = 0; d < 3; d ++) {
        part[i].Vel[d] = out[d] * fac;
    }
}
SIMPLE_PROPERTY(Mass, Mass, float, 1)
SIMPLE_PROPERTY(ID, ID, uint64_t, 1)
SIMPLE_PROPERTY(Generation, Generation, unsigned char, 1)
SIMPLE_GETTER(GTPotential, Potential, float, 1, struct particle_data)
SIMPLE_GETTER(GTTimeBin, TimeBin, int, 1, struct particle_data)
SIMPLE_PROPERTY(SmoothingLength, Hsml, float, 1)
SIMPLE_PROPERTY_PI(Density, Density, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(EgyWtDensity, EgyWtDensity, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(ElectronAbundance, Ne, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_PI(DelayTime, DelayTime, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_TYPE_PI(StarFormationTime, 4, FormationTime, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_PI(BirthDensity, BirthDensity, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metallicity, 4, Metallicity, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(LastEnrichmentMyr, 4, LastEnrichmentMyr, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(TotalMassReturned, 4, TotalMassReturned, float, 1, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metallicity, 0, Metallicity, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metals, 4, Metals[0], float, NMETALS, struct star_particle_data)
SIMPLE_PROPERTY_TYPE_PI(Metals, 0, Metals[0], float, NMETALS, struct sph_particle_data)

SIMPLE_GETTER_PI(GTStarFormationRate, Sfr, float, 1, struct sph_particle_data)
SIMPLE_PROPERTY_TYPE_PI(StarFormationTime, 5, FormationTime, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeMass, Mass, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeDensity, Density, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeAccretionRate, Mdot, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeProgenitors, CountProgs, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeSwallowID, SwallowID, uint64_t, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeSwallowTime, SwallowTime, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeJumpToMinPot, JumpToMinPot, int, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeMtrack, Mtrack, float, 1, struct bh_particle_data)
SIMPLE_PROPERTY_PI(BlackholeMseed, Mseed, float, 1, struct bh_particle_data)

SIMPLE_SETTER_PI(STBlackholeMinPotPos , MinPotPos[0], double, 3, struct bh_particle_data)
static void GTBlackholeMinPotPos(int i, double * out, void * baseptr, void * smanptr) {
    /* Remove the particle offset before saving*/
    struct particle_data * part = (struct particle_data *) baseptr;
    int PI = part[i].PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[5]);
    struct bh_particle_data * sl = (struct bh_particle_data *) info->ptr;
    int d;
    for(d = 0; d < 3; d ++) {
        out[d] = sl[PI].MinPotPos[d] - PartManager->CurrentParticleOffset[d];
        while(out[d] > All.BoxSize) out[d] -= All.BoxSize;
        while(out[d] <= 0) out[d] += All.BoxSize;
    }
}

/*This is only used if FoF is enabled*/
SIMPLE_GETTER(GTGroupID, GrNr, uint32_t, 1, struct particle_data)
static void GTNeutralHydrogenFraction(int i, float * out, void * baseptr, void * smanptr) {
    double redshift = 1./All.Time - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_neutral_fraction_sfreff(redshift, pl, sl+PI);
}

static void GTHeliumIFraction(int i, float * out, void * baseptr, void * smanptr) {
    double redshift = 1./All.Time - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_helium_neutral_fraction_sfreff(0, redshift, pl, sl+PI);
}
static void GTHeliumIIFraction(int i, float * out, void * baseptr, void * smanptr) {
    double redshift = 1./All.Time - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_helium_neutral_fraction_sfreff(1, redshift, pl, sl+PI);
}
static void GTHeliumIIIFraction(int i, float * out, void * baseptr, void * smanptr) {
    double redshift = 1./All.Time - 1;
    struct particle_data * pl = ((struct particle_data *) baseptr)+i;
    int PI = pl->PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = get_helium_neutral_fraction_sfreff(2, redshift, pl, sl+PI);
}
static void GTInternalEnergy(int i, float * out, void * baseptr, void * smanptr) {
    int PI = ((struct particle_data *) baseptr)[i].PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    *out = sl[PI].Entropy / GAMMA_MINUS1 * pow(SPH_EOMDensity(&sl[PI]) * All.cf.a3inv, GAMMA_MINUS1);
}

static void STInternalEnergy(int i, float * out, void * baseptr, void * smanptr) {
    float u = *out;
    int PI = ((struct particle_data *) baseptr)[i].PI;
    struct slot_info * info = &(((struct slots_manager_type *) smanptr)->info[0]);
    struct sph_particle_data * sl = (struct sph_particle_data *) info->ptr;
    sl[PI].Entropy  = GAMMA_MINUS1 * u / pow(SPH_EOMDensity(&sl[PI]) * All.cf.a3inv , GAMMA_MINUS1);
}

/* Can't use the macros because cannot take address of a bitfield*/
static void GTHeIIIIonized(int i, unsigned char * out, void * baseptr, void * smanptr) {
    struct particle_data * part = (struct particle_data *) baseptr;
    *out = part[i].HeIIIionized;
}

static void STHeIIIIonized(int i, unsigned char * out, void * baseptr, void * smanptr) {
    struct particle_data * part = (struct particle_data *) baseptr;
    part[i].HeIIIionized = *out;
}
static void GTSwallowed(int i, unsigned char * out, void * baseptr, void * smanptr) {
    struct particle_data * part = (struct particle_data *) baseptr;
    *out = part[i].Swallowed;
}

static void STSwallowed(int i, unsigned char * out, void * baseptr, void * smanptr) {
    struct particle_data * part = (struct particle_data *) baseptr;
    part[i].Swallowed = *out;
}

static int order_by_type(const void *a, const void *b)
{
    const struct IOTableEntry * pa  = (const struct IOTableEntry *) a;
    const struct IOTableEntry * pb  = (const struct IOTableEntry *) b;

    if(pa->ptype < pb->ptype)
        return -1;
    if(pa->ptype > pb->ptype)
        return +1;
    if(pa->zorder < pb->zorder)
        return -1;
    if(pa->zorder > pb->zorder)
        return 1;

    return 0;
}

void register_io_blocks(struct IOTable * IOTable, int WriteGroupID) {
    int i;
    IOTable->used = 0;
    IOTable->allocated = 100;
    IOTable->ent = mymalloc2("IOTable", IOTable->allocated * sizeof(IOTableEntry));
    /* Bare Bone Gravity*/
    for(i = 0; i < 6; i ++) {
        /* We put Mass first because sometimes there is
         * corruption in the first array and we can recover from Mass corruption*/
        IO_REG(Mass,     "f4", 1, i, IOTable);
        IO_REG(Position, "f8", 3, i, IOTable);
        IO_REG(Velocity, "f4", 3, i, IOTable);
        IO_REG(ID,       "u8", 1, i, IOTable);
        if(All.OutputPotential)
            IO_REG_WRONLY(Potential, "f4", 1, i, IOTable);
        if(WriteGroupID)
            IO_REG_WRONLY(GroupID, "u4", 1, i, IOTable);
        if(All.OutputTimebins)
            IO_REG_WRONLY(TimeBin,       "u4", 1, i, IOTable);
    }

    IO_REG(Generation,       "u1", 1, 0, IOTable);
    IO_REG(Generation,       "u1", 1, 4, IOTable);
    IO_REG(Generation,       "u1", 1, 5, IOTable);
    /* Bare Bone SPH*/
    IO_REG(SmoothingLength,  "f4", 1, 0, IOTable);
    IO_REG(Density,          "f4", 1, 0, IOTable);

    if(DensityIndependentSphOn())
        IO_REG(EgyWtDensity,          "f4", 1, 0, IOTable);

    /* On reload this sets the Entropy variable, need the densities.
     * Register this after Density and EgyWtDensity will ensure density is read
     * before this. */
    IO_REG(InternalEnergy,   "f4", 1, 0, IOTable);

    /* Cooling */
    IO_REG(ElectronAbundance,       "f4", 1, 0, IOTable);
    if(All.CoolingOn) {
        IO_REG_WRONLY(NeutralHydrogenFraction, "f4", 1, 0, IOTable);
    }
    if(All.OutputHeliumFractions) {
        IO_REG_WRONLY(HeliumIFraction, "f4", 1, 0, IOTable);
        IO_REG_WRONLY(HeliumIIFraction, "f4", 1, 0, IOTable);
        IO_REG_WRONLY(HeliumIIIFraction, "f4", 1, 0, IOTable);
    }
    /* Marks whether a particle has been HeIII ionized yet*/
    IO_REG_NONFATAL(HeIIIIonized, "u1", 1, 0, IOTable);

    /* SF */
    if(All.StarformationOn) {
        IO_REG_WRONLY(StarFormationRate, "f4", 1, 0, IOTable);
        /* Another new addition: save the DelayTime for wind particles*/
        IO_REG_NONFATAL(DelayTime,  "f4", 1, 0, IOTable);
    }
    IO_REG_NONFATAL(BirthDensity, "f4", 1, 4, IOTable);
    IO_REG_TYPE(StarFormationTime, "f4", 1, 4, IOTable);
    IO_REG_TYPE(Metallicity,       "f4", 1, 0, IOTable);
    IO_REG_TYPE(Metallicity,       "f4", 1, 4, IOTable);
    if(All.MetalReturnOn) {
        IO_REG_TYPE(Metals,       "f4", NMETALS, 0, IOTable);
        IO_REG_TYPE(Metals,       "f4", NMETALS, 4, IOTable);
        IO_REG_TYPE(LastEnrichmentMyr, "f4", 1, 4, IOTable);
        IO_REG_TYPE(TotalMassReturned, "f4", 1, 4, IOTable);
        IO_REG_NONFATAL(SmoothingLength,  "f4", 1, 4, IOTable);
    }
    /* end SF */

    /* Black hole */
    IO_REG_TYPE(StarFormationTime, "f4", 1, 5, IOTable);
    IO_REG(BlackholeMass,          "f4", 1, 5, IOTable);
    IO_REG(BlackholeDensity,          "f4", 1, 5, IOTable);
    IO_REG(BlackholeAccretionRate, "f4", 1, 5, IOTable);
    IO_REG(BlackholeProgenitors,   "i4", 1, 5, IOTable);
    IO_REG(BlackholeMinPotPos, "f8", 3, 5, IOTable);
    IO_REG(BlackholeJumpToMinPot,   "i4", 1, 5, IOTable);
    IO_REG(BlackholeMtrack,         "f4", 1, 5, IOTable);
    IO_REG_NONFATAL(BlackholeMseed,         "f4", 1, 5, IOTable);

    /* Smoothing lengths for black hole: this is a new addition*/
    IO_REG_NONFATAL(SmoothingLength,  "f4", 1, 5, IOTable);
    /* Marks whether a BH particle has been swallowed*/
    IO_REG_NONFATAL(Swallowed, "u1", 1, 5, IOTable);
    /* ID of the swallowing black hole particle. If == -1, then particle is live*/
    IO_REG_NONFATAL(BlackholeSwallowID, "u8", 1, 5, IOTable);
    /* Time the BH was swallowed*/
    IO_REG_NONFATAL(BlackholeSwallowTime, "f4", 1, 5, IOTable);

    /*Sort IO blocks so similar types are together; then ordered by the sequence they are declared. */
    qsort_openmp(IOTable->ent, IOTable->used, sizeof(struct IOTableEntry), order_by_type);
}

/* Add extra debug blocks to the output*/
/* Write (but don't read) them, only useful for debugging the particle structures.
 * Warning: future code versions may change the units!*/
SIMPLE_GETTER(GTGravAccel, GravAccel[0], float, 3, struct particle_data)
SIMPLE_GETTER(GTGravPM, GravPM[0], float, 3, struct particle_data)
SIMPLE_GETTER_PI(GTHydroAccel, HydroAccel[0], float, 3, struct sph_particle_data)
SIMPLE_GETTER_PI(GTMaxSignalVel, MaxSignalVel, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTEntropy, Entropy, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTDtEntropy, DtEntropy, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTDhsmlEgyDensityFactor, DhsmlEgyDensityFactor, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTDivVel, DivVel, float, 1, struct sph_particle_data)
SIMPLE_GETTER_PI(GTCurlVel, CurlVel, float, 1, struct sph_particle_data)

void register_debug_io_blocks(struct IOTable * IOTable)
{
    int ptype;
    for(ptype = 0; ptype < 6; ptype++) {
        IO_REG_WRONLY(GravAccel,       "f4", 3, ptype, IOTable);
        IO_REG_WRONLY(GravPM,       "f4", 3, ptype, IOTable);
        if(!All.OutputTimebins) /* Otherwise it is output in the regular blocks*/
            IO_REG_WRONLY(TimeBin,       "u4", 1, ptype, IOTable);
    }
    IO_REG_WRONLY(HydroAccel,       "f4", 3, 0, IOTable);
    IO_REG_WRONLY(MaxSignalVel,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(Entropy,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(DtEntropy,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(DhsmlEgyDensityFactor,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(DivVel,       "f4", 1, 0, IOTable);
    IO_REG_WRONLY(CurlVel,       "f4", 1, 0, IOTable);
    /*Sort IO blocks so similar types are together; then ordered by the sequence they are declared. */
    qsort_openmp(IOTable->ent, IOTable->used, sizeof(struct IOTableEntry), order_by_type);
}

void destroy_io_blocks(struct IOTable * IOTable) {
    myfree(IOTable->ent);
    IOTable->allocated = 0;
}
