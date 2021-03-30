#ifndef __GARBAGE_H
#define __GARBAGE_H
#include <mpi.h>
#include "utils.h"

#include "types.h"
#include "partmanager.h"

struct slot_info {
    char * ptr; /* aliasing ptr for this slot */
    int64_t maxsize; /* max number of supported slots */
    int64_t size; /* currently used slots*/
    size_t elsize; /* itemsize */
    int enabled;
};

/* Slot particle data structures: first the base extension slot, then black holes,
 * then stars, then SPH. SPH still has some compile-time optional elements.
 * Each particle also has the base data, stored in particle_data.*/
struct particle_data_ext {
    /* Used at GC for reverse link to P.
     * Garbage slots have this impossibly large. */
    int ReverseLink;
    MyIDType ID; /* for data consistency check, same as particle ID */
};

/* Data stored for each black hole in addition to collisionless data*/
struct bh_particle_data {
    struct particle_data_ext base;

    int CountProgs;

    MyFloat Mass;
    MyFloat Mdot;
    MyFloat Density;
    MyFloat FormationTime;  /*!< formation time of black hole. */
    /* Merger time of the black hole.
     * After this, all values are fixed. */
    MyFloat SwallowTime;
    int JumpToMinPot;
    double MinPotPos[3];
    MyFloat MinPotVel[3];
    /* After a merger, this gives the ID of the particle which swallowed the BH. Used to keep track of merger trees.*/
    MyIDType SwallowID;

    /*******************************************************/
    double DragAccel[3];
    double DFAccel[3];
    /*******************************************************/

    /* Stores the minimum timebins of all black hole neighbours.
     * The black hole timebin is then set to this.*/
    int minTimeBin;
    int encounter; /* mark the event when BH encounters another BH */
    double Mtrack; /*Swallow gas particle when BHP.Mass accretes from SeedBHMass to SeedDynMass for mass conservation */
    double Mseed; /*Log the seed mass of BH, would be useful in case of the powerlaw seeding*/
};

#define NMETALS 9

/*Data for each star particle*/
struct star_particle_data
{
    struct particle_data_ext base;
    MyFloat FormationTime;      /*!< formation time of star particle */
    MyFloat LastEnrichmentMyr;  /* Last time the star particle had an enrichment event, in Myr since FormationTime.*/
    MyFloat TotalMassReturned; /* The total mass returned from this star since formation.
                                  The initial mass of the SSP in this star is STARP.TotalMassReturned + P.Mass.
                                  It is stored like this to retain compatibility with older snapshots. */
    MyFloat BirthDensity;       /*!< Density of gas particle at star formation. */
    MyFloat Metallicity;        /*!< Total metallicity of star particle */
    float Metals[NMETALS];      /* Metal mass of each species in star particle*/
};

/* the following structure holds data that is stored for each SPH particle in addition to the collisionless
 * variables.
 */
struct sph_particle_data
{
    struct particle_data_ext base;

    /*This is only used if DensityIndependentSph is on.
     * If DensityIndependentSph is off then Density is used instead.*/
    MyFloat EgyWtDensity;           /*!< 'effective' rho to use in hydro equations */

    MyFloat Entropy;		/*!< Entropy (actually entropic function) at kick time of particle.
                                 * Defined as: P_i = A(s) rho_i^gamma. See Springel & Hernquist 2002.*/
    MyFloat MaxSignalVel;           /*!< maximum signal velocity */
    MyFloat       Density;		/*!< current baryonic mass density of particle */
    MyFloat       DtEntropy;		/*!< rate of change of entropy */
    MyFloat       HydroAccel[3];	/*!< acceleration due to hydrodynamical force */
    /*!< correction factor for density-independent entropy formulation. If DensityIndependentSph = 0
     then this is set to the DhsmlDensityFactor appropriate for the entropy formulation of SPH. */
    MyFloat DhsmlEgyDensityFactor;
    MyFloat       DivVel;		/*!< local velocity divergence */
    /* CurlVel has to be here and not in scratch because we re-use the
     * CurlVel of inactive particles inside the artificial viscosity calculation.*/
    MyFloat       CurlVel;     	        /*!< local velocity curl */
    MyFloat Ne;  /*!< electron fraction, expressed as local electron number
                   density normalized to the hydrogen number density. Gives
                   indirectly ionization state and mean molecular weight. */
    MyFloat DelayTime;		/*!< SH03: remaining maximum decoupling time of wind particle */
                            /*!< VS08: remaining waiting for wind particle to be eligible to form winds again */
    MyFloat Sfr; /* Star formation rate in Msun/year. Stored here because, if the H2 dependent star formation is used,
                    it depends on the scratch variable GradRho and thus cannot be recomputed after a fof-exchange. */
    MyFloat local_J21; /* local J21 ionising background calculated from the excursion set */
    MyFloat Metallicity;        /*!< metallicity of gas particle */
    float Metals[NMETALS];
};

extern struct slots_manager_type {
    struct slot_info info[6];
    char * Base; /* memory ptr that holds of all slots */
    double increase; /* Percentage amount to increase
                      * slot reservation by when requested.*/
} SlotsManager[1];

/* shortcuts for accessing different slots directly by the index */
#define SphP ((struct sph_particle_data*) SlotsManager->info[0].ptr)
#define StarP ((struct star_particle_data*) SlotsManager->info[4].ptr)
#define BhP ((struct bh_particle_data*) SlotsManager->info[5].ptr)

/* shortcuts for accessing slots from base particle index */
#define SPHP(i) SphP[P[i].PI]
#define BHP(i) BhP[P[i].PI]
#define STARP(i) StarP[P[i].PI]

extern MPI_Datatype MPI_TYPE_PARTICLE;
extern MPI_Datatype MPI_TYPE_SLOT[6];

/* shortcuts to access base slot attributes */
#define BASESLOT_PI(PI, ptype, sman) ((struct particle_data_ext *)(sman->info[ptype].ptr + sman->info[ptype].elsize * (PI)))

void slots_init(double increase, struct slots_manager_type * sman);
/*Enable a slot on type ptype. All slots are disabled after slots_init().*/
void slots_set_enabled(int ptype, size_t elsize, struct slots_manager_type * sman);
void slots_free(struct slots_manager_type * sman);
void slots_mark_garbage(int i, struct part_manager_type * pman, struct slots_manager_type * sman);
void slots_setup_topology(struct part_manager_type * pman, int64_t * NLocal, struct slots_manager_type * sman);
void slots_setup_id(const struct part_manager_type * pman, struct slots_manager_type * sman);
int slots_split_particle(int parent, double childmass, struct part_manager_type * pman);
int slots_convert(int parent, int ptype, int placement, struct part_manager_type * pman, struct slots_manager_type * sman);
int slots_gc(int * compact_slots, struct part_manager_type * pman, struct slots_manager_type * sman);
void slots_gc_sorted(struct part_manager_type * pman, struct slots_manager_type * sman);
size_t slots_reserve(int where, int64_t atleast[6], struct slots_manager_type * sman);
void slots_check_id_consistency(struct part_manager_type * pman, struct slots_manager_type * sman);

typedef struct {
    EIBase base;
    int parent;
    int child;
} EISlotsFork;

#endif
