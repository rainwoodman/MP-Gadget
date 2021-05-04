#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* do NOT use complex.h it breaks the code */
#include <pfft.h>
#include "allvars.h"
#include "proto.h"
#include "power.h"
/* Using fastpm's gaussian_fill for ngenic agreement. */
#include "pmesh.h"

#include <libgadget/petapm.h>
#include <libgadget/walltime.h>
#include <libgadget/utils.h>

#define MESH2K(i) petapm_mesh_to_k(i)
static void density_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void vel_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void vel_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void vel_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void disp_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void disp_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void disp_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value);
static void readout_density(PetaPM * pm, int i, double * mesh, double weight);
static void readout_vel_x(PetaPM * pm, int i, double * mesh, double weight);
static void readout_vel_y(PetaPM * pm, int i, double * mesh, double weight);
static void readout_vel_z(PetaPM * pm, int i, double * mesh, double weight);
static void readout_disp_x(PetaPM * pm, int i, double * mesh, double weight);
static void readout_disp_y(PetaPM * pm, int i, double * mesh, double weight);
static void readout_disp_z(PetaPM * pm, int i, double * mesh, double weight);
static void gaussian_fill(int Nmesh, PetaPMRegion * region, pfft_complex * rho_k, int UnitaryAmplitude, int InvertPhase, const int Seed);

static inline double periodic_wrap(double x, const double BoxSize)
{
  while(x >= BoxSize)
    x -= BoxSize;

  while(x < 0)
    x += BoxSize;

  return x;
}

/* Watch out: only works for pencils along-z !*/
void
idgen_init(IDGenerator * idgen, PetaPM * pm, int Ngrid, double BoxSize)
{

    int * ThisTask2d = petapm_get_thistask2d(pm);
    int * NTask2d = petapm_get_ntask2d(pm);
    idgen->NumPart = 1;
    int k;
    for(k = 0; k < 2; k ++) {
        idgen->offset[k] = (ThisTask2d[k]) * Ngrid / NTask2d[k];
        idgen->size[k] = (ThisTask2d[k] + 1) * Ngrid / NTask2d[k];
        idgen->size[k] -= idgen->offset[k];
        idgen->NumPart *= idgen->size[k];
    }
    idgen->offset[2] = 0;
    idgen->size[2] = Ngrid;
    idgen->NumPart *= idgen->size[2];
    idgen->Ngrid = Ngrid;
    idgen->BoxSize = BoxSize;
}

uint64_t
idgen_create_id_from_index(IDGenerator * idgen, int index)
{
    int i = index / (idgen->size[2] * idgen->size[1]) + idgen->offset[0];
    int j = (index % (idgen->size[1] * idgen->size[2])) / idgen->size[2] + idgen->offset[1];
    int k = (index % idgen->size[2]) + idgen->offset[2];
    uint64_t id = ((uint64_t) i) * idgen->Ngrid * idgen->Ngrid + ((uint64_t)j) * idgen->Ngrid + k + 1;
    return id;
}

void
idgen_create_pos_from_index(IDGenerator * idgen, int index, double pos[3])
{
    int x = index / (idgen->size[2] * idgen->size[1]) + idgen->offset[0];
    int y = (index % (idgen->size[1] * idgen->size[2])) / idgen->size[2] + idgen->offset[1];
    int z = (index % idgen->size[2]) + idgen->offset[2];

    pos[0] = x * idgen->BoxSize / idgen->Ngrid;
    pos[1] = y * idgen->BoxSize / idgen->Ngrid;
    pos[2] = z * idgen->BoxSize / idgen->Ngrid;
}

int
setup_grid(IDGenerator * idgen, double shift, double mass, struct ic_part_data * ICP)
{
    memset(ICP, 0, idgen->NumPart*sizeof(struct ic_part_data));

    int i;
    #pragma omp parallel for
    for(i = 0; i < idgen->NumPart; i ++) {
        idgen_create_pos_from_index(idgen, i, &ICP[i].Pos[0]);
        ICP[i].Pos[0] += shift;
        ICP[i].Pos[1] += shift;
        ICP[i].Pos[2] +=  shift;
        ICP[i].Mass = mass;
    }
    return idgen->NumPart;
}

struct ic_prep_data
{
    struct ic_part_data * curICP;
    int NumPart;
};

static PetaPMRegion * makeregion(PetaPM * pm, PetaPMParticleStruct * pstruct, void * userdata, int * Nregions) {
    PetaPMRegion * regions = mymalloc2("Regions", sizeof(PetaPMRegion));
    struct ic_prep_data * icprep = (struct ic_prep_data *) userdata;
    int NumPart = icprep->NumPart;
    struct ic_part_data * ICP = icprep->curICP;
    int k;
    int r = 0;
    int i;
    double min[3] = {pm->BoxSize, pm->BoxSize, pm->BoxSize};
    double max[3] = {0, 0, 0.};

    for(i = 0; i < NumPart; i ++) {
        for(k = 0; k < 3; k ++) {
            if(min[k] > ICP[i].Pos[k])
                min[k] = ICP[i].Pos[k];
            if(max[k] < ICP[i].Pos[k])
                max[k] = ICP[i].Pos[k];
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

/*Global to pass type to *_transfer functions*/
static enum TransferType ptype;
/*Global to pass the particle data to the readout functions*/
static struct ic_part_data * curICP;

void displacement_fields(PetaPM * pm, enum TransferType Type, struct ic_part_data * dispICP, const int NumPart, Cosmology * CP, const struct genic_config GenicConfig) {

    /*MUST set this before doing force.*/
    ptype = Type;
    curICP = dispICP;
    PetaPMParticleStruct pstruct = {
        curICP,
        sizeof(curICP[0]),
        ((char*) &curICP[0].Pos[0]) - (char*) curICP,
        ((char*) &curICP[0].Mass) - (char*) curICP,
        NULL,
        NULL,
        NumPart,
    };

    int i;
    #pragma omp parallel for
    for(i = 0; i < NumPart; i++)
    {
        memset(&curICP[i].Disp[0], 0, sizeof(curICP[i].Disp));
        memset(&curICP[i].Vel[0], 0, sizeof(curICP[i].Vel));
        curICP[i].Density = 0;
    }


    /* This reads out the displacements into P.Disp and the velocities into P.Vel.
     * Disp is used to avoid changing the particle positions mid-way through.
     * Note that for the velocities we do NOT just use the velocity transfer functions.
     * The reason is because of the gauge: velocity transfer is in synchronous gauge for CLASS,
     * newtonian gauge for CAMB. But we want N-body gauge, which we get by taking the time derivative
     * of the synchronous gauge density perturbations. See arxiv:1505.04756*/
    PetaPMFunctions functions[] = {
        {"Density", density_transfer, readout_density},
        {"DispX", disp_x_transfer, readout_disp_x},
        {"DispY", disp_y_transfer, readout_disp_y},
        {"DispZ", disp_z_transfer, readout_disp_z},
        {"VelX", vel_x_transfer, readout_vel_x},
        {"VelY", vel_y_transfer, readout_vel_y},
        {"VelZ", vel_z_transfer, readout_vel_z},
        {NULL, NULL, NULL },
    };

    /*Set up the velocity pre-factors*/
    const double hubble_a = hubble_function(CP, GenicConfig.TimeIC);

    double vel_prefac = GenicConfig.TimeIC * hubble_a;

    if(GenicConfig.UsePeculiarVelocity) {
        /* already for peculiar velocity */
        message(0, "Producing Peculiar Velocity in the output.\n");
    } else {
        vel_prefac /= sqrt(GenicConfig.TimeIC);	/* converts to Gadget velocity */
    }

    if(!GenicConfig.PowerP.ScaleDepVelocity) {
        vel_prefac *= F_Omega(CP, GenicConfig.TimeIC);
        /* If different transfer functions are disabled, we can copy displacements to velocities
         * and we don't need the extra transfers.*/
        functions[4].name = NULL;
    }

    int Nregions;
    struct ic_prep_data icprep = {dispICP, NumPart};
    PetaPMRegion * regions = petapm_force_init(pm,
           makeregion,
           &pstruct,
           &Nregions,
           &icprep);

    /*This allocates the memory*/
    pfft_complex * rho_k = petapm_alloc_rhok(pm);

    gaussian_fill(pm->Nmesh, petapm_get_fourier_region(pm),
		  rho_k, GenicConfig.UnitaryAmplitude, GenicConfig.InvertPhase, GenicConfig.Seed);

    petapm_force_c2r(pm, rho_k, regions, Nregions, functions);

    myfree(rho_k);
    myfree(regions);
    petapm_force_finish(pm);

    double maxdisp = 0, maxvel = 0;

    #pragma omp parallel for reduction(max:maxdisp, maxvel)
    for(i = 0; i < NumPart; i++)
    {
        int k;
        double absv = 0;
        for(k = 0; k < 3; k++)
        {
            double dis = curICP[i].Disp[k];
            if(dis > maxdisp)
                maxdisp = dis;
            /*Copy displacements to positions.*/
            curICP[i].Pos[k] += curICP[i].Disp[k];
            /*Copy displacements to velocities if not done already*/
            if(!GenicConfig.PowerP.ScaleDepVelocity)
                curICP[i].Vel[k] = curICP[i].Disp[k];
            curICP[i].Vel[k] *= vel_prefac;
            absv += curICP[i].Vel[k] * curICP[i].Vel[k];
            curICP[i].Pos[k] = periodic_wrap(curICP[i].Pos[k], pm->BoxSize);
        }
        if(absv > maxvel)
            maxvel = absv;
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxdisp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    message(0, "Type = %d max disp = %g in units of cell sep %g \n", ptype, maxdisp, maxdisp / (pm->BoxSize / pm->Nmesh) );

    MPI_Allreduce(MPI_IN_PLACE, &maxvel, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    message(0, "Max vel=%g km/s, vel_prefac= %g  hubble_a=%g fom=%g \n", sqrt(maxvel), vel_prefac, hubble_a, F_Omega(CP, GenicConfig.TimeIC));

    walltime_measure("/Disp/Finalize");
    MPIU_Barrier(MPI_COMM_WORLD);
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

static void density_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    if(k2) {
        /* density is smoothed in k space by a gaussian kernel of 1 mesh grid */
        double r2 = 1.0 / pm->Nmesh;
        r2 *= r2;
        double fac = exp(- k2 * r2);

        double kmag = sqrt(k2) * 2 * M_PI / pm->BoxSize;
        fac *= DeltaSpec(kmag, ptype) / sqrt(pm->BoxSize * pm->BoxSize * pm->BoxSize);

        value[0][0] *= fac;
        value[0][1] *= fac;
    }
}

static void disp_transfer(PetaPM * pm, int64_t k2, int kaxis, pfft_complex * value, int include_growth) {
    if(k2) {
        double fac = 1./ (2 * M_PI) / sqrt(pm->BoxSize) * kaxis / k2;
        /*
         We avoid high precision kernels to maintain compatibility with N-GenIC.
         The following formular shall cross check with fac in the limit of
         native diff_kernel (disp_y, disp_z shall match too!)

        double fac1 = (2 * M_PI) / pm->BoxSize;
        double fac = diff_kernel(kaxis * (2 * M_PI / pm->Nmesh)) * (pm->Nmesh / pm->BoxSize) / (
                    k2 * fac1 * fac1);
                    */
        double kmag = sqrt(k2) * 2 * M_PI / pm->BoxSize;
        /*Multiply by derivative of scale-dependent growth function*/
        if(include_growth)
            fac *= dlogGrowth(kmag, ptype);
        else
            fac *= DeltaSpec(kmag, ptype);
        double tmp = value[0][0];
        value[0][0] = - value[0][1] * fac;
        value[0][1] = tmp * fac;
    }
}

static void vel_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(pm, k2, kpos[0], value, 1);
}
static void vel_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(pm, k2, kpos[1], value, 1);
}
static void vel_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(pm, k2, kpos[2], value, 1);
}

static void disp_x_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(pm, k2, kpos[0], value, 0);
}
static void disp_y_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(pm, k2, kpos[1], value, 0);
}
static void disp_z_transfer(PetaPM * pm, int64_t k2, int kpos[3], pfft_complex * value) {
    disp_transfer(pm, k2, kpos[2], value, 0);
}

/**************
 * functions iterating over particle / mesh pairs
 ***************/
static void readout_density(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Density += weight * mesh[0];
}
static void readout_vel_x(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Vel[0] += weight * mesh[0];
}
static void readout_vel_y(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Vel[1] += weight * mesh[0];
}
static void readout_vel_z(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Vel[2] += weight * mesh[0];
}

static void readout_disp_x(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Disp[0] += weight * mesh[0];
}
static void readout_disp_y(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Disp[1] += weight * mesh[0];
}
static void readout_disp_z(PetaPM * pm, int i, double * mesh, double weight) {
    curICP[i].Disp[2] += weight * mesh[0];
}

static void
gaussian_fill(int Nmesh, PetaPMRegion * region, pfft_complex * rho_k, int setUnitaryAmplitude, int setInvertPhase, const int Seed)
{
    /* fastpm deals with strides properly; petapm not. So we translate it here. */
    PMDesc pm[1];
    int d;
    for (d = 0; d < 3; d ++) {
        pm->Nmesh[d] = Nmesh;
    }

    pm->ORegion.start[0] = region->offset[2];
    pm->ORegion.size[0] = region->size[2];
    pm->ORegion.strides[0] = region->strides[2];
    pm->ORegion.start[1] = region->offset[0];
    pm->ORegion.size[1] = region->size[0];
    pm->ORegion.strides[1] = region->strides[0];
    pm->ORegion.start[2] = region->offset[1];
    pm->ORegion.size[2] = region->size[1];
    pm->ORegion.strides[2] = region->strides[1];

    pm->ORegion.total = region->totalsize;
    pmic_fill_gaussian_gadget(pm, (double*) rho_k, Seed, setUnitaryAmplitude, setInvertPhase);

#if 0
    /* dump the gaussian field for debugging
     *
     * the z directioin is in axis 1.
     *
     * */
    FILE * rhokf = fopen("rhok", "w");
    printf("strides %td %td %td\n",
            pm->ORegion.strides[0],
            pm->ORegion.strides[1],
            pm->ORegion.strides[2]);
    printf("size %td %td %td\n",
            pm->ORegion.size[0],
            pm->ORegion.size[1],
            pm->ORegion.size[2]);
    printf("offset %td %td %td\n",
            pm->ORegion.start[0],
            pm->ORegion.start[1],
            pm->ORegion.start[2]);
    fwrite(rho_k, sizeof(double) * region->totalsize, 1, rhokf);
    fclose(rhokf);
#endif
}
