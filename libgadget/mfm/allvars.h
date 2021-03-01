
/*! \file allvars.h
 *  \brief declares global variables.
 *
 *  This file declares all global variables. Further variables should be added here, and declared as
 *  'extern'. The actual existence of these variables is provided by the file 'allvars.c'. To produce
 *  'allvars.c' from 'allvars.h', do the following:
 *
 *     - Erase all #define statements
 *     - add #include "allvars.h"
 *     - delete all keywords 'extern'
 *     - delete all struct definitions enclosed in {...}, e.g.
 *        "extern struct global_data_all_processes {....} All;"
 *        becomes "struct global_data_all_processes All;"
 */

/*
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel (volker.springel@h-its.org). The code has been modified
 * in part by Phil Hopkins (phopkins@caltech.edu) for GIZMO (new variables, 
 * and different naming conventions for some old variables)
 */


#ifndef ALLVARS_H
#define ALLVARS_H

#include <mpi.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/*------- Things that are always recommended (this must follow loading GIZMO_config.h!) -------*/
#define DOUBLEPRECISION         /* using double (not floating-point) precision */
#define INHOMOG_GASDISTR_HINT   /* if the gas is distributed very different from collisionless particles, this can helps to avoid problems in the domain decomposition */

/* lock the 'default' hydro mode */
#define HYDRO_MESHLESS_FINITE_MASS   /* otherwise default to MFM if nothing is specified */

#if (defined(HYDRO_DENSITY_SPH) || defined(HYDRO_PRESSURE_SPH)) && !defined(HYDRO_SPH)
#define HYDRO_SPH               /* master flag for SPH: must be enabled if any SPH method is used */
#endif
#ifdef HYDRO_SPH
#define SPHAV_CD10_VISCOSITY_SWITCH 0.05   /* Enables Cullen & Dehnen 2010 'inviscid sph' (viscosity suppression outside shocks) */
#define SPHAV_ARTIFICIAL_CONDUCTIVITY      /* Enables mixing entropy (J.Read's improved Price-Monaghan conductivity with Cullen-Dehnen switches) */
#endif

#ifdef PERIODIC
#define BOX_PERIODIC
#endif

/* options for direct/exact Jiang et al. method for direct evolution on an intensity grid */

//#define ENERGY_ENTROPY_SWITCH_IS_ACTIVE
/* this is a ryu+jones type energy/entropy switch. it can help with some problems, but can also generate significant 
 errors in other types of problems. in general, even for pure hydro, this isn't recommended; use it for special problems if you know what you are doing. */


typedef  int integertime;
static MPI_Datatype MPI_TYPE_TIME = MPI_INT;
#define  TIMEBINS        29

#define  TIMEBASE        (((integertime) 1)<<TIMEBINS)  /*!< The simulated timespan is mapped onto the integer interval [0,TIMESPAN],
                                         *   where TIMESPAN needs to be a power of 2. Note that (1<<28) corresponds
                                         *   to 2^29
                                         */

#define  terminate(x) {char termbuf[2000]; sprintf(termbuf, "code termination on task=%d, function '%s()', file '%s', line %d: '%s'\n", ThisTask, __FUNCTION__, __FILE__, __LINE__, x); printf("%s", termbuf); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1); exit(0);}

#if !defined(GAMMA) /* this allows for either term to be defined, for backwards-compatibility */
#define  GAMMA         (5.0/3.0)	/*!< adiabatic index of simulated gas */
#endif

#define  GAMMA_MINUS1  (GAMMA-1)
#define  GAMMA_MINUS1_INV  (1./(GAMMA-1))

#define  HYDROGEN_MASSFRAC 0.76 /*!< mass fraction of hydrogen, relevant only for radiative cooling */

#define  MAX_REAL_NUMBER  1e37
#define  MIN_REAL_NUMBER  1e-37

#define  CONDITION_NUMBER_DANGER  1.0e3 /*!< condition number above which we will not trust matrix-based gradients */


/* ... often used physical constants (cgs units) */

#define  GRAVITY     6.672e-8
#define  SOLAR_MASS  1.989e33
#define  SOLAR_LUM   3.826e33
#define  RAD_CONST   7.565e-15
#define  AVOGADRO    6.0222e23
#define  BOLTZMANN   1.38066e-16
#define  GAS_CONST   8.31425e7
#define  C           2.9979e10
#define  PLANCK      6.6262e-27
#define  CM_PER_MPC  3.085678e24
#define  PROTONMASS  1.6726e-24
#define  ELECTRONMASS 9.10953e-28
#define  THOMPSON     6.65245e-25
#define  ELECTRONCHARGE  4.8032e-10
#define  HUBBLE          3.2407789e-18	/* in h/sec */
#define  LYMAN_ALPHA      1215.6e-8	/* 1215.6 Angstroem */
#define  LYMAN_ALPHA_HeII  303.8e-8	/* 303.8 Angstroem */
#define  OSCILLATOR_STRENGTH       0.41615
#define  OSCILLATOR_STRENGTH_HeII  0.41615
#define  ELECTRONVOLT_IN_ERGS      1.60217733e-12

#define KAPPA_IR 10.0   /* in cm^2/g for solar abundances */
#define KAPPA_OP 180.0
#define KAPPA_UV 1800.0

#define  SEC_PER_MEGAYEAR   3.155e13
#define  SEC_PER_YEAR       3.155e7

#define COND_TIMESTEP_PARAMETER 0.25
#define VISC_TIMESTEP_PARAMETER 0.25

#define MAXLEN_OUTPUTLIST 1200	/*!< maxmimum number of entries in output list */

#define MAXITER 150

#define MINRESTFAC 0.05

typedef unsigned long long MyIDType;


typedef double  MyFloat;
typedef double  MyDouble;

#define FLT(x) (x)
typedef MyFloat MyLongDouble;
typedef MyDouble MyBigFloat;

#if (BOX_SPATIAL_DIMENSION==1) || defined(ONEDIM)
#define NUMDIMS 1           /* define number of dimensions and volume normalization */
#define NORM_COEFF 2.0
#elif (BOX_SPATIAL_DIMENSION==2) || defined(TWODIMS)
#define NUMDIMS 2
#define NORM_COEFF M_PI
#else
#define NORM_COEFF 4.188790204786  /* 4pi/3 */
#define NUMDIMS 3
#endif


#define PPP P
#define PPPZ SphP

#ifdef BOX_PERIODIC
extern MyDouble boxSize, boxHalf, inverse_boxSize;
#define boxSize_X boxSize
#define boxHalf_X boxHalf
#define inverse_boxSize_X inverse_boxSize
#define boxSize_Y boxSize
#define boxHalf_Y boxHalf
#define inverse_boxSize_Y inverse_boxSize
#define boxSize_Z boxSize
#define boxHalf_Z boxHalf
#define inverse_boxSize_Z inverse_boxSize
#endif


/****************************************************************************************************************************/
/* Here we define the box-wrapping macros NEAREST_XYZ and NGB_PERIODIC_BOX_LONG_X,NGB_PERIODIC_BOX_LONG_Y,NGB_PERIODIC_BOX_LONG_Z. 
 *   The inputs to these functions are (dx_position, dy_position, dz_position, sign), where 
 *     'sign' = -1 if dx_position = x_test_point - x_reference (reference = particle from which we are doing a calculation), 
 *     'sign' = +1 if dx_position = x_reference - x_test_point
 *
 *   For non-periodic cases these functions are trivial (do nothing, or just take absolute values).
 *
 *   For standard periodic cases it will wrap in each dimension, allowing for a different box length in X/Y/Z.
 *      here the "sign" term is irrelevant. Also NGB_PERIODIC_BOX_LONG_X, NGB_PERIODIC_BOX_LONG_Y, NGB_PERIODIC_BOX_LONG_Z will each 
 *      compile to only use the x,y, or z information, but all four inputs are required for the sake of completeness 
 *      and consistency.
 *
 *   The reason for the added complexity is for shearing boxes. In this case, the Y(phi)-coordinate for particles being
 *      wrapped in the X(r)-direction must be modified by a time-dependent term. It also matters for the sign of that 
 *      term "which side" of the box we are wrapping across (i.e. does the 'virtual particle' -- the test point which is
 *      not the particle for which we are currently calculating forces, etc -- lie on the '-x' side or the '+x' side)
 *      (note after all that: if very careful, sign -cancels- within the respective convention, for the type of wrapping below)
 */
/****************************************************************************************************************************/

#define NGB_PERIODIC_BOX_LONG_X(x,y,z,sign) (xtmp=fabs(x),(xtmp>boxHalf_X)?(boxSize_X-xtmp):xtmp) // normal periodic wrap //
#define NGB_PERIODIC_BOX_LONG_Z(x,y,z,sign) (xtmp=fabs(z),(xtmp>boxHalf_Z)?(boxSize_Z-xtmp):xtmp) // normal periodic wrap //

/* Standard Periodic Box:: 
    this box-wraps all three (x,y,z) separation variables when taking position differences */
#define NEAREST_XYZ(x,y,z,sign) (\
x=((x)>boxHalf_X)?((x)-boxSize_X):(((x)<-boxHalf_X)?((x)+boxSize_X):(x)),\
y=((y)>boxHalf_Y)?((y)-boxSize_Y):(((y)<-boxHalf_Y)?((y)+boxSize_Y):(y)),\
z=((z)>boxHalf_Z)?((z)-boxSize_Z):(((z)<-boxHalf_Z)?((z)+boxSize_Z):(z)))
#define NGB_PERIODIC_BOX_LONG_Y(x,y,z,sign) (xtmp=fabs(y),(xtmp>boxHalf_Y)?(boxSize_Y-xtmp):xtmp) // normal periodic wrap //

#define FACT1 0.366025403785	/* FACT1 = 0.5 * (sqrt(3)-1) */
#define FACT2 0.86602540        /* FACT2 = 0.5 * sqrt(3) */


/*********************************************************/
/*  Global variables                                     */
/*********************************************************/


extern int FirstActiveParticle;
extern int *NextActiveParticle;
extern unsigned char *ProcessedFlag;

extern int TimeBinCount[TIMEBINS];
extern int TimeBinCountSph[TIMEBINS];
extern int TimeBinActive[TIMEBINS];

extern int FirstInTimeBin[TIMEBINS];
extern int LastInTimeBin[TIMEBINS];
extern int *NextInTimeBin;
extern int *PrevInTimeBin;

extern int ThisTask;		/*!< the number of the local processor  */
extern int NTask;		/*!< number of processors */
extern int PTask;		/*!< note: NTask = 2^PTask */

extern double CPUThisRun;	/*!< Sums CPU time of current process */

extern int NumForceUpdate;	/*!< number of active particles on local processor in current timestep  */
extern long long GlobNumForceUpdate;

extern int NumSphUpdate;	/*!< number of active SPH particles on local processor in current timestep  */

extern int MaxTopNodes;	        /*!< Maximum number of nodes in the top-level tree used for domain decomposition */

extern int RestartFlag;		/*!< taken from command line used to start code. 0 is normal start-up from
				   initial conditions, 1 is resuming a run from a set of restart files, while 2
				   marks a restart from a snapshot file. */
extern int RestartSnapNum;
extern int SelRnd;

extern int TakeLevel;

extern int *Exportflag;	        /*!< Buffer used for flagging whether a particle needs to be exported to another process */
extern int *Exportnodecount;
extern int *Exportindex;

extern int *Send_offset, *Send_count, *Recv_count, *Recv_offset;

extern size_t AllocatedBytes;
extern size_t HighMarkBytes;
extern size_t FreeBytes;

extern double CPU_Step[CPU_PARTS];
extern char CPU_Symbol[CPU_PARTS];
extern char CPU_SymbolImbalance[CPU_PARTS];
extern char CPU_String[CPU_STRING_LEN + 1];

extern double WallclockTime;    /*!< This holds the last wallclock time measurement for timings measurements */

extern int Flag_FullStep;	/*!< Flag used to signal that the current step involves all particles */

extern size_t HighMark_run,  HighMark_domain, HighMark_gravtree, HighMark_pmperiodic,
  HighMark_pmnonperiodic,  HighMark_sphdensity, HighMark_sphhydro, HighMark_GasGrad;

extern int TreeReconstructFlag;
extern int GlobFlag;

extern char DumpFlag;

extern int NumPart;		/*!< number of particles on the LOCAL processor */
extern int N_gas;		/*!< number of gas particles on the LOCAL processor  */
extern long long Ntype[6];	/*!< total number of particles of each type */
extern int NtypeLocal[6];	/*!< local number of particles of each type */

extern gsl_rng *random_generator;	/*!< the random number generator used */


extern int Gas_split;           /*!< current number of newly-spawned gas particles outside block */
/* variables for input/output , usually only used on process 0 */


extern char ParameterFile[100];	/*!< file name of parameterfile used for starting the simulation */

extern FILE
 *FdTimebin,    /*!< file handle for timebin.txt log-file. */
 *FdInfo,       /*!< file handle for info.txt log-file. */
 *FdEnergy,     /*!< file handle for energy.txt log-file. */
 *FdTimings,    /*!< file handle for timings.txt log-file. */
 *FdBalance,    /*!< file handle for balance.txt log-file. */
 *FdCPU;        /*!< file handle for cpu.txt log-file. */
/*! table for the cosmological drift factors */
extern double DriftTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for gravitational forces */
extern double GravKickTable[DRIFT_TABLE_LENGTH];

extern void *CommBuffer;	/*!< points to communication buffer, which is used at a few places */

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
extern struct global_data_all_processes
{
  long long TotNumPart;		/*!<  total particle numbers (global value) */
  long long TotN_gas;		/*!<  total gas particle number (global value) */
  int MaxPart;			/*!< This gives the maxmimum number of particles that can be stored on one processor. */
  int MaxPartSph;		/*!< This gives the maxmimum number of SPH particles that can be stored on one processor. */
  int ICFormat;			/*!< selects different versions of IC file-format */
  int SnapFormat;		/*!< selects different versions of snapshot file-formats */
  int DoDynamicUpdate;
  int NumFilesPerSnapshot;	/*!< number of files in multi-file snapshot dumps */
  int NumFilesWrittenInParallel;	/*!< maximum number of files that may be written simultaneously when
                                     writing/reading restart-files, or when writing snapshot files */
  double BufferSize;		/*!< size of communication buffer in MB */
  int BunchSize;     	        /*!< number of particles fitting into the buffer in the parallel tree algorithm  */

  double PartAllocFactor;	/*!< in order to maintain work-load balance, the particle load will usually
				   NOT be balanced.  Each processor allocates memory for PartAllocFactor times
				   the average number of particles to allow for that */

  double TreeAllocFactor;	/*!< Each processor allocates a number of nodes which is TreeAllocFactor times
				   the maximum(!) number of particles.  Note: A typical local tree for N
				   particles needs usually about ~0.65*N nodes. */

  double TopNodeAllocFactor;	/*!< Each processor allocates a number of nodes which is TreeAllocFactor times
				   the maximum(!) number of particles.  Note: A typical local tree for N
				   particles needs usually about ~0.65*N nodes. */

  /* some SPH parameters */

  double DesNumNgb;		/*!< Desired number of SPH neighbours */

  double MaxNumNgbDeviation;	/*!< Maximum allowed deviation neighbour number */

  double ArtBulkViscConst;	/*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
  double InitGasTemp;		/*!< may be used to set the temperature in the IC's */
  double InitGasU;		/*!< the same, but converted to thermal energy per unit mass */
  double MinGasTemp;		/*!< may be used to set a floor for the gas temperature */

  double MinEgySpec;		/*!< the minimum allowed temperature expressed as energy per unit mass */
    
    double MinMassForParticleMerger; /*!< the minimum mass of a gas particle below which it will be merged into a neighbor */
    double MaxMassForParticleSplit; /*!< the maximum mass of a gas particle above which it will be split into a pair */

  /* some force counters  */

  long long TotNumOfForces;	/*!< counts total number of force computations  */

  long long NumForcesSinceLastDomainDecomp;	/*!< count particle updates since last domain decomposition */

  /* some variable for dynamic work-load adjustment based on CPU measurements */

  double cf_atime, cf_a2inv, cf_a3inv, cf_afac1, cf_afac2, cf_afac3, cf_hubble_a, cf_hubble_a2;   /* various cosmological factors that are only a function of the current scale factor, and in Newtonian runs are set to 1 */

  /* system of units  */

  double UnitTime_in_s,		/*!< factor to convert internal time unit to seconds/h */
    UnitMass_in_g,		/*!< factor to convert internal mass unit to grams/h */
    UnitVelocity_in_cm_per_s,	/*!< factor to convert intqernal velocity unit to cm/sec */
    UnitLength_in_cm,		/*!< factor to convert internal length unit to cm/h */
    UnitPressure_in_cgs,	/*!< factor to convert internal pressure unit to cgs units (little 'h' still around!) */
    UnitDensity_in_cgs,		/*!< factor to convert internal length unit to g/cm^3*h^2 */
    UnitEnergy_in_cgs,		/*!< factor to convert internal energy to cgs units */
    UnitTime_in_Megayears,	/*!< factor to convert internal time to megayears/h */
    GravityConstantInternal,	/*!< If set to zero in the parameterfile, the internal value of the
				   gravitational constant is set to the Newtonian value based on the system of
				   units specified. Otherwise the value provided is taken as internal gravity
				   constant G. */
    G;				/*!< Gravity-constant in internal units */
    /* Cosmology */

  double Hubble_H0_CodeUnits;		/*!< Hubble-constant (unit-ed version: 100 km/s/Mpc) in internal units */
  double Omega0,		/*!< matter density in units of the critical density (at z=0) */
    OmegaLambda,		/*!< vaccum energy density relative to crictical density (at z=0) */
    OmegaBaryon,		/*!< baryon density in units of the critical density (at z=0) */
    HubbleParam;		/*!< little `h', i.e. Hubble constant in units of 100 km/s/Mpc.  Only needed to get absolute
				 * physical values for cooling physics
				 */

  double BoxSize;		/*!< Boxsize in case periodic boundary conditions are used */

    
  /* Code options */

  int ComovingIntegrationOn;	/*!< flags that comoving integration is enabled */
  int ResubmitOn;		/*!< flags that automatic resubmission of job to queue system is enabled */
  int TypeOfOpeningCriterion;	/*!< determines tree cell-opening criterion: 0 for Barnes-Hut, 1 for relative criterion */
  int OutputListOn;		/*!< flags that output times are listed in a specified file */

  int HighestActiveTimeBin;
  int HighestOccupiedTimeBin;

  /* parameters determining output frequency */

  int SnapshotFileCount;	/*!< number of snapshot that is written next */
  double TimeBetSnapshot,	/*!< simulation time interval between snapshot files */
    TimeOfFirstSnapshot,	/*!< simulation time of first snapshot files */
    CpuTimeBetRestartFile,	/*!< cpu-time between regularly generated restart files */
    TimeLastRestartFile,	/*!< cpu-time when last restart-file was written */
    TimeBetStatistics,		/*!< simulation time interval between computations of energy statistics */
    TimeLastStatistics;		/*!< simulation time when the energy statistics was computed the last time */
  integertime NumCurrentTiStep;		/*!< counts the number of system steps taken up to this point */

  /* Current time of the simulation, global step, and end of simulation */

  double Time,			/*!< current time of the simulation */
    TimeBegin,			/*!< time of initial conditions of the simulation */
    TimeStep,			/*!< difference between current times of previous and current timestep */
    TimeMax;			/*!< marks the point of time until the simulation is to be evolved */

  /* variables for organizing discrete timeline */

  double Timebase_interval;	/*!< factor to convert from floating point time interval to integer timeline */
  integertime Ti_Current;		/*!< current time on integer timeline */
  integertime Previous_Ti_Current;
  integertime Ti_nextoutput;		/*!< next output time on integer timeline */
  integertime Ti_lastoutput;

#ifdef PMGRID
  integertime PM_Ti_endstep, PM_Ti_begstep;
  double Asmth[2], Rcut[2];
  double Corner[2][3], UpperCorner[2][3], Xmintot[2][3], Xmaxtot[2][3];
  double TotalMeshSize[2];
#endif


  integertime Ti_nextlineofsight;
  int    CPU_TimeBinCountMeasurements[TIMEBINS];
  double CPU_TimeBinMeasurements[TIMEBINS][NUMBER_OF_MEASUREMENTS_TO_RECORD];

  int LevelToTimeBin[GRAVCOSTLEVELS];

  /* variables that keep track of cumulative CPU consumption */

  double TimeLimitCPU;
  double CPU_Sum[CPU_PARTS];    /*!< sums wallclock time/CPU consumption in whole run */

  /* tree code opening criterion */

  double ErrTolTheta;		/*!< BH tree opening angle */
  double ErrTolForceAcc;	/*!< parameter for relative opening criterion in tree walk */


  /* adjusts accuracy of time-integration */

  double ErrTolIntAccuracy;	/*!< accuracy tolerance parameter \f$ \eta \f$ for timestep criterion. The
				   timesteps is \f$ \Delta t = \sqrt{\frac{2 \eta eps}{a}} \f$ */

  double MinSizeTimestep,	/*!< minimum allowed timestep. Normally, the simulation terminates if the
				   timestep determined by the timestep criteria falls below this limit. */
    MaxSizeTimestep;		/*!< maximum allowed timestep */

  double MaxRMSDisplacementFac;	/*!< this determines a global timestep criterion for cosmological simulations
				   in comoving coordinates.  To this end, the code computes the rms velocity
				   of all particles, and limits the timestep such that the rms displacement
				   is a fraction of the mean particle separation (determined from the
				   particle mass and the cosmological parameters). This parameter specifies
				   this fraction. */

  int MaxMemSize;

  double CourantFac;		/*!< SPH-Courant factor */


  /* frequency of tree reconstruction/domain decomposition */


  double TreeDomainUpdateFrequency;	/*!< controls frequency of domain decompositions  */


  /* gravitational and hydrodynamical softening lengths (given in terms of an `equivalent' Plummer softening length)
   *
   * five groups of particles are supported 0=gas,1=halo,2=disk,3=bulge,4=stars
   */
    double MinGasHsmlFractional; /*!< minimim allowed gas kernel length relative to force softening (what you actually set) */
    double MinHsml;			/*!< minimum allowed gas kernel length */
    double MaxHsml;           /*!< minimum allowed gas kernel length */

  double SofteningGas,		/*!< for type 0 */
    SofteningHalo,		/*!< for type 1 */
    SofteningDisk,		/*!< for type 2 */
    SofteningBulge,		/*!< for type 3 */
    SofteningStars,		/*!< for type 4 */
    SofteningBndry;		/*!< for type 5 */

  double SofteningGasMaxPhys,	/*!< for type 0 */
    SofteningHaloMaxPhys,	/*!< for type 1 */
    SofteningDiskMaxPhys,	/*!< for type 2 */
    SofteningBulgeMaxPhys,	/*!< for type 3 */
    SofteningStarsMaxPhys,	/*!< for type 4 */
    SofteningBndryMaxPhys;	/*!< for type 5 */

  double SofteningTable[6];	/*!< current (comoving) gravitational softening lengths for each particle type */
  double ForceSoftening[6];	/*!< the same, but multiplied by a factor 2.8 - at that scale the force is Newtonian */


  /*! If particle masses are all equal for one type, the corresponding entry in MassTable is set to this
   *  value, * allowing the size of the snapshot files to be reduced
   */
  double MassTable[6];


  /* some filenames */
  char InitCondFile[100],
    OutputDir[100],
    SnapshotFileBase[100],
    RestartFile[100], ResubmitCommand[100], OutputListFilename[100];
    /* EnergyFile[100], CpuFile[100], InfoFile[100], TimingsFile[100], TimebinFile[100], */

  /*! table with desired output times */
  double OutputListTimes[MAXLEN_OUTPUTLIST];
  char OutputListFlag[MAXLEN_OUTPUTLIST];
  int OutputListLength;		/*!< number of times stored in table of desired output times */

}
All;



/*! This structure holds all the information that is
 * stored for each particle of the simulation.
 */
extern ALIGN(32) struct particle_data
{
    short int Type;                 /*!< flags particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
    short int TimeBin;
    MyIDType ID;                    /*! < unique ID of particle (assigned at beginning of the simulation) */
    MyIDType ID_child_number;       /*! < child number for particles 'split' from main (retain ID, get new child number) */
    int ID_generation;              /*! < generation (need to track for particle-splitting to ensure each 'child' gets a unique child number */
    
    integertime Ti_begstep;         /*!< marks start of current timestep of particle on integer timeline */
    integertime Ti_current;         /*!< current time of the particle */
    
    ALIGN(32) MyDouble Pos[3];      /*!< particle position at its current time */
    MyDouble Mass;                  /*!< particle mass */
    
    MyDouble Vel[3];                /*!< particle velocity at its current time */
    MyDouble dp[3];
    MyFloat Particle_DivVel;        /*!< velocity divergence of neighbors (for predict step) */
    
    MyDouble GravAccel[3];          /*!< particle acceleration due to gravity */
#ifdef PMGRID
    MyFloat GravPM[3];		/*!< particle acceleration due to long-range PM gravity force */
#endif
    MyFloat OldAcc;			/*!< magnitude of old gravitational force. Used in relative opening criterion */
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUT_POTENTIAL)
    MyFloat Potential;		/*!< gravitational potential */
#if defined(EVALPOTENTIAL) && defined(PMGRID)
    MyFloat PM_Potential;
#endif
#endif
#ifdef METALS
    MyFloat Metallicity[NUM_METAL_SPECIES]; /*!< metallicity (species-by-species) of gas or star particle */
#endif
    MyFloat Hsml;                   /*!< search radius around particle for neighbors/interactions */
    MyFloat NumNgb;                 /*!< neighbor number around particle */
    MyFloat DhsmlNgbFactor;        /*!< correction factor needed for varying kernel lengths */
    float GravCost[GRAVCOSTLEVELS];   /*!< weight factor used for balancing the work-load */
    
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
    MyFloat NV_T[3][3];                                           /*!< holds the tensor used for gradient estimation */
#endif
}
 *P,				/*!< holds particle data on local processor */
 *DomainPartBuf;		/*!< buffer for particle data used in domain decomposition */


/* the following struture holds data that is stored for each SPH particle in addition to the collisionless
 * variables.
 */
extern struct sph_particle_data
{
    /* the PRIMITIVE and CONSERVED hydro variables used in STATE reconstruction */
    MyDouble Density;               /*!< current baryonic mass density of particle */

    MyDouble Pressure;              /*!< current pressure */
    MyDouble InternalEnergy;        /*!< internal energy of particle */
    MyDouble InternalEnergyPred;    /*!< predicted value of the internal energy at the current time */
    //MyDouble dInternalEnergy;     /*!< change in internal energy from hydro step */ //manifest-indiv-timestep-debug//
    MyDouble DtInternalEnergy;      /*!< rate of change of internal energy */

    MyDouble VelPred[3];            /*!< predicted SPH particle velocity at the current time */
    //MyDouble dMomentum[3];        /*!< change in momentum from hydro step (conserved variable) */ //manifest-indiv-timestep-debug//
    MyDouble HydroAccel[3];         /*!< acceleration due to hydrodynamical force (for drifting) */
    
    /* matrix of the primitive variable gradients: rho, P, vx, vy, vz, B, phi */
    struct
    {
        MyDouble Density[3];
        MyDouble Pressure[3];
        MyDouble Velocity[3][3];
    } Gradients;
    MyFloat NV_T[3][3];             /*!< holds the tensor used for gradient estimation */
    MyDouble ConditionNumber;       /*!< condition number of the gradient matrix: needed to ensure stability */
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    MyDouble MaxKineticEnergyNgb;   /*!< maximum kinetic energy (with respect to neighbors): use for entropy 'switch' */
#endif

#ifdef HYDRO_SPH
    MyDouble DhsmlHydroSumFactor;   /* for 'traditional' SPH, we need the SPH hydro-element volume estimator */
#endif
    
#ifdef HYDRO_PRESSURE_SPH
    MyDouble EgyWtDensity;          /*!< 'effective' rho to use in hydro equations */
#endif
    MyFloat MaxSignalVel;           /*!< maximum signal velocity (needed for time-stepping) */
#ifdef HYDRO_SPH
  MyFloat alpha_limiter;                /*!< artificial viscosity limiter (Balsara-like) */
#endif
}
  *SphP,				/*!< holds SPH particle data on local processor */
  *DomainSphBuf;			/*!< buffer for SPH particle data in domain decomposition */

