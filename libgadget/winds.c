/*Prototypes and structures for the wind model*/

#include <math.h>
#include <string.h>
#include <omp.h>
#include "winds.h"
#include "physconst.h"
#include "treewalk.h"
#include "slotsmanager.h"
#include "timebinmgr.h"
#include "walltime.h"
#include "density.h"
#include "hydra.h"

/*Parameters of the wind model*/
static struct WindParams
{
    enum WindModel WindModel;  /*!< Which wind model is in use? */
    double WindFreeTravelLength;
    double WindFreeTravelDensFac;
    /*Density threshold at which to recouple wind particles.*/
    double WindFreeTravelDensThresh;
    /* Maximum time in internal time units to allow the wind to be free-streaming.*/
    double MaxWindFreeTravelTime;
    /* used in VS08 and SH03*/
    double WindEfficiency;
    double WindSpeed;
    double WindEnergyFraction;
    /* used in OFJT10*/
    double WindSigma0;
    double WindSpeedFactor;
    /* Minimum wind velocity for kicked particles, in internal velocity units*/
    double MinWindVelocity;
    /* Fraction of wind energy in thermal energy*/
    double WindThermalFactor;
} wind_params;

#define NWINDHSML 5 /* Number of densities to evaluate for wind weight ngbiter*/
#define NUMDMNGB 40 /*Number of DM ngb to evaluate vel dispersion */
#define MAXDMDEVIATION 2

typedef struct {
    TreeWalkQueryBase base;
    MyIDType ID;
    double Dt;
    double Mass;
    double Hsml;
    double TotalWeight;
    double DMRadius[NWINDHSML];
    double Vdisp;
    double Vel[3];
} TreeWalkQueryWind;

typedef struct {
    TreeWalkResultBase base;
    double TotalWeight;
    double V1sum[NWINDHSML][3];
    double V2sum[NWINDHSML];
    double Ngb[NWINDHSML];
    int alignment; /* Ensure alignment*/
    int maxcmpte;
} TreeWalkResultWind;

typedef struct {
    TreeWalkNgbIterBase base;
} TreeWalkNgbIterWind;

/*Set the parameters of the wind module.
 ofjt10 is Okamoto, Frenk, Jenkins and Theuns 2010 https://arxiv.org/abs/0909.0265
 VS08 is Dalla Vecchia & Schaye 2008 https://arxiv.org/abs/0801.2770
 SH03 is Springel & Hernquist 2003 https://arxiv.org/abs/astro-ph/0206395*/
void set_winds_params(ParameterSet * ps)
{
    int ThisTask;
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0) {
        /*Wind model parameters*/
        wind_params.WindModel = param_get_enum(ps, "WindModel");
        /* The following two are for VS08 and SH03*/
        wind_params.WindEfficiency = param_get_double(ps, "WindEfficiency");
        wind_params.WindEnergyFraction = param_get_double(ps, "WindEnergyFraction");

        /* The following two are for OFJT10*/
        wind_params.WindSigma0 = param_get_double(ps, "WindSigma0");
        wind_params.WindSpeedFactor = param_get_double(ps, "WindSpeedFactor");

        wind_params.WindThermalFactor = param_get_double(ps, "WindThermalFactor");
        wind_params.MinWindVelocity = param_get_double(ps, "MinWindVelocity");
        wind_params.MaxWindFreeTravelTime = param_get_double(ps, "MaxWindFreeTravelTime");
        wind_params.WindFreeTravelLength = param_get_double(ps, "WindFreeTravelLength");
        wind_params.WindFreeTravelDensFac = param_get_double(ps, "WindFreeTravelDensFac");
    }
    MPI_Bcast(&wind_params, sizeof(struct WindParams), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void
init_winds(double FactorSN, double EgySpecSN, double PhysDensThresh, double UnitTime_in_s)
{
    wind_params.WindSpeed = sqrt(2 * wind_params.WindEnergyFraction * FactorSN * EgySpecSN / (1 - FactorSN));
    /* Convert wind free travel time from Myr to internal units*/
    wind_params.MaxWindFreeTravelTime = wind_params.MaxWindFreeTravelTime * SEC_PER_MEGAYEAR / UnitTime_in_s;
    wind_params.WindFreeTravelDensThresh = wind_params.WindFreeTravelDensFac * PhysDensThresh;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        wind_params.WindSpeed /= sqrt(wind_params.WindEfficiency);
        message(0, "Windspeed: %g MaxDelay %g\n", wind_params.WindSpeed, wind_params.MaxWindFreeTravelTime);
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        message(0, "Reference Windspeed: %g, MaxDelay %g\n", wind_params.WindSigma0 * wind_params.WindSpeedFactor, wind_params.MaxWindFreeTravelTime);
    } else {
        /* Check for undefined wind models*/
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }
}

int
winds_is_particle_decoupled(int i)
{
    if(HAS(wind_params.WindModel, WIND_DECOUPLE_SPH)
        && P[i].Type == 0 && SPHP(i).DelayTime > 0)
            return 1;
    return 0;
}

void
winds_decoupled_hydro(int i, double atime)
{
    int k;
    for(k = 0; k < 3; k++)
        SPHP(i).HydroAccel[k] = 0;

    SPHP(i).DtEntropy = 0;

    double windspeed = wind_params.WindSpeed * atime;
    const double fac_mu = pow(atime, 3 * (GAMMA - 1) / 2) / atime;
    windspeed *= fac_mu;
    double hsml_c = cbrt(wind_params.WindFreeTravelDensThresh /SPHP(i).Density) * atime;
    SPHP(i).MaxSignalVel = hsml_c * DMAX((2 * windspeed), SPHP(i).MaxSignalVel);
}

static int
get_wind_dir(int i, double dir[3]);

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * remote, enum TreeWalkReduceMode mode, TreeWalk * tw);

static void
sfr_wind_copy(int place, TreeWalkQueryWind * input, TreeWalk * tw);

static void
sfr_wind_weight_postprocess(const int i, TreeWalk * tw);

static void
sfr_wind_weight_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv);

static void
sfr_wind_feedback_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv);

struct winddata {
    double DMRadius;
    double Left;
    double Right;
    double TotalWeight;

    double Vdisp;
    double V2sum[NWINDHSML];
    double V1sum[NWINDHSML][3];
    double Ngb[NWINDHSML];
    int maxcmpte;
};

/* Returns 1 if the winds ever decouple, 0 otherwise*/
int winds_ever_decouple(void)
{
    if(wind_params.MaxWindFreeTravelTime > 0)
        return 1;
    else
        return 0;
}

/* Structure to store a potential kick
 * to a gas particle from a newly formed star.
 * We add a queue of these and resolve them
 * after the treewalk. Note this star may
 * be on another processor.*/
struct StarKick
{
    /* Index of the kicked particle*/
    int part_index;
    /* Distance to the star. The closest star does the kick.*/
    double StarDistance;
    /* Star ID, for resolving ties.*/
    MyIDType StarID;
    /* Kick velocity if this kick is the one used*/
    double StarKickVelocity;
    /* Thermal energy included in the kick*/
    double StarTherm;
};

struct WindPriv {
    double Time;
    double hubble;
    struct winddata * Winddata;
    struct StarKick * kicks;
    int64_t nkicks;
    int64_t maxkicks;
    int * nvisited;
};

/* Comparison function to sort the StarKicks by particle id, distance and star ID.
 * The closest star is used. */
int cmp_by_part_id(const void * a, const void * b)
{
    const struct StarKick * stara = a;
    const struct StarKick * starb = b;
    if(stara->part_index > starb->part_index)
        return 1;
    if(stara->part_index < starb->part_index)
        return -1;
    if(stara->StarDistance > starb->StarDistance)
        return 1;
    if(stara->StarDistance < starb->StarDistance)
        return -1;
    if(stara->StarID > starb->StarID)
        return 1;
    if(stara->StarID < starb->StarID)
        return -1;
    return 0;
}

#define WIND_GET_PRIV(tw) ((struct WindPriv *) (tw->priv))
#define WINDP(i, wind) wind[P[i].PI]

/*Do a treewalk for the wind model. This only changes newly created star particles.*/
void
winds_and_feedback(int * NewStars, int NumNewStars, const double Time, const double hubble, ForceTree * tree)
{
    /*The subgrid model does nothing here*/
    if(HAS(wind_params.WindModel, WIND_SUBGRID))
        return;

    if(!MPIU_Any(NumNewStars > 0, MPI_COMM_WORLD))
        return;

    TreeWalk tw[1] = {{0}};

    tw->ev_label = "WIND_WEIGHT";
    tw->fill = (TreeWalkFillQueryFunction) sfr_wind_copy;
    tw->reduce = (TreeWalkReduceResultFunction) sfr_wind_reduce_weight;
    tw->query_type_elsize = sizeof(TreeWalkQueryWind);
    tw->result_type_elsize = sizeof(TreeWalkResultWind);
    tw->tree = tree;

    /* sum the total weight of surrounding gas */
    tw->ngbiter_type_elsize = sizeof(TreeWalkNgbIterWind);
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_weight_ngbiter;

    tw->haswork = NULL;
    tw->visit = (TreeWalkVisitFunction) treewalk_visit_nolist_ngbiter;
    tw->postprocess = (TreeWalkProcessFunction) sfr_wind_weight_postprocess;
    struct WindPriv priv[1];
    priv[0].Time = Time;
    priv[0].hubble = hubble;
    tw->priv = priv;

    int64_t totalleft = 0;
    sumup_large_ints(1, &NumNewStars, &totalleft);
    priv->Winddata = (struct winddata * ) mymalloc("WindExtraData", SlotsManager->info[4].size * sizeof(struct winddata));
    /* Note that this will be an over-count because each loop will add more.*/
    priv->nvisited = ta_malloc("nvisited", int, omp_get_max_threads());
    memset(WIND_GET_PRIV(tw)->nvisited, 0, omp_get_max_threads()* sizeof(int));

    int i;
    /*Initialise the WINDP array*/
    #pragma omp parallel for
    for (i = 0; i < NumNewStars; i++) {
        int n = NewStars[i];
        WINDP(n, priv->Winddata).DMRadius = 2 * P[n].Hsml;
        WINDP(n, priv->Winddata).Left = 0;
        WINDP(n, priv->Winddata).Right = tree->BoxSize;
        WINDP(n, priv->Winddata).maxcmpte = NUMDMNGB;
    }

    /* Find densities*/
    treewalk_do_hsml_loop(tw, NewStars, NumNewStars, 1);

    for (i = 1; i < omp_get_max_threads(); i++)
        priv->nvisited[0] += priv->nvisited[i];
    priv->maxkicks = priv->nvisited[0]+2;

    /* Some particles may be kicked by multiple stars on the same timestep.
     * To ensure this happens only once and does not depend on the order in
     * which the loops are executed, particles are kicked by the nearest new star.
     * This struct stores all such possible kicks, and we sort it out after the treewalk.*/
    priv->kicks = (struct StarKick * ) mymalloc("StarKicks", priv->maxkicks * sizeof(struct StarKick));
    priv->nkicks = 0;
    ta_free(priv->nvisited);

    /* Then run feedback */
    tw->haswork = NULL;
    tw->ngbiter = (TreeWalkNgbIterFunction) sfr_wind_feedback_ngbiter;
    tw->postprocess = NULL;
    tw->reduce = NULL;
    tw->ev_label = "WIND_KICK";
    tw->Niteration = 0;

    treewalk_run(tw, NewStars, NumNewStars);

    /* Sort the possible kicks*/
    qsort_openmp(priv->kicks, priv->nkicks, sizeof(struct StarKick), cmp_by_part_id);
    /* Not parallel as the number of kicked particles should be pretty small*/
    int64_t last_part = -1;
    int64_t nkicked = 0;
    for(i = 0; i < priv->nkicks; i++) {
        /* Only do the kick for the first particle, which is the closest*/
        if(priv->kicks[i].part_index == last_part)
            continue;
        int other = priv->kicks[i].part_index;
        last_part = other;
        nkicked++;
        /* Kick the gas particle*/
        double dir[3];
        get_wind_dir(other, dir);
        double v = priv->kicks[i].StarKickVelocity;
        if(v > 0 && Time > 0) {
            int j;
            for(j = 0; j < 3; j++)
            {
                P[other].Vel[j] += v * dir[j];
            }
            /* StarTherm is internal energy per unit mass. Need to convert to entropy*/
            const double enttou = pow(SPH_EOMDensity(&SPHP(other)) / pow(Time, 3), GAMMA_MINUS1) / GAMMA_MINUS1;
            SPHP(other).Entropy += priv->kicks[i].StarTherm/enttou;
            if(winds_ever_decouple()) {
                double delay = wind_params.WindFreeTravelLength / (v / Time);
                if(delay > wind_params.MaxWindFreeTravelTime)
                    delay = wind_params.MaxWindFreeTravelTime;
                SPHP(other).DelayTime = delay;
            }
        }
        if(v <= 0 || !isfinite(v) || !isfinite(SPHP(other).DelayTime))
        {
            endrun(5, "Odd v: other = %d, DT = %g v = %g i = %d, nkicks %d maxkicks %d dist %g id %ld\n",
                   other, SPHP(other).DelayTime, v, i, priv->nkicks, priv->maxkicks, priv->kicks[i].StarDistance, priv->kicks[i].StarID);
        }
    }
    /* Get total number of potential new stars to allocate memory.*/
    int64_t tot_newstars, tot_kicks, tot_applied;
    sumup_large_ints(1, &NumNewStars, &tot_newstars);
    MPI_Allreduce(&priv->nkicks, &tot_kicks, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nkicked, &tot_applied, 1, MPI_INT64, MPI_SUM, MPI_COMM_WORLD);
    message(0, "Made %ld gas wind, discarded %ld kicks from %d stars\n", tot_applied, tot_kicks - tot_applied, tot_newstars);

    myfree(priv->kicks);
    myfree(priv->Winddata);
    walltime_measure("/Cooling/Wind");
}

/*Evolve a wind particle, reducing its DelayTime*/
void
winds_evolve(int i, double a3inv, double hubble)
{
    /*Remove a wind particle from the delay mode if the (physical) density has dropped sufficiently.*/
    if(SPHP(i).DelayTime > 0 && SPHP(i).Density * a3inv < wind_params.WindFreeTravelDensThresh) {
        SPHP(i).DelayTime = 0;
    }
    /*Reduce the time until the particle can form stars again by the current timestep*/
    if(SPHP(i).DelayTime > 0) {
        /* Enforce the maximum in case of restarts*/
        if(SPHP(i).DelayTime > wind_params.MaxWindFreeTravelTime)
            SPHP(i).DelayTime = wind_params.MaxWindFreeTravelTime;
        const double dloga = get_dloga_for_bin(P[i].TimeBin, P[i].Ti_drift);
        /*  the proper time duration of the step */
        const double dtime = dloga / hubble;
        SPHP(i).DelayTime = DMAX(SPHP(i).DelayTime - dtime, 0);
    }
}

static inline double
effdmradius(int place, int i, TreeWalk * tw)
{
    struct winddata * Windd = WIND_GET_PRIV(tw)->Winddata;
    double left = WINDP(place, Windd).Left;
    double right = WINDP(place, Windd).Right;
    /*The asymmetry is because it is free to compute extra densities for h < Hsml, but not for h > Hsml*/
    if (right > 0.99*tw->tree->BoxSize){
        right = WINDP(place, Windd).DMRadius;
    }
    if(left == 0)
        left = 0.1 * WINDP(place, Windd).DMRadius;
    /*Evenly split in volume*/
    double rvol = pow(right, 3);
    double lvol = pow(left, 3);
    return pow((1.0*i+1)/(1.0*NWINDHSML+1) * (rvol - lvol) + lvol, 1./3);
}


static void
sfr_wind_weight_postprocess(const int i, TreeWalk * tw)
{
    if(P[i].Type != 4)
        endrun(23, "Wind called on something not a star particle: (i=%d, t=%d, id = %ld)\n", i, P[i].Type, P[i].ID);
    struct winddata * Windd = WIND_GET_PRIV(tw)->Winddata;

    const int maxcmpt = WINDP(i, Windd).maxcmpte;
    int j;
    double evaldmradius[NWINDHSML];
    for(j = 0; j < maxcmpt; j++){
        evaldmradius[j] = effdmradius(i,j,tw);
    }
    int close = 0;
    WINDP(i, Windd).DMRadius = ngb_narrow_down(&WINDP(i, Windd).Right, &WINDP(i, Windd).Left, evaldmradius, WINDP(i, Windd).Ngb, maxcmpt, NUMDMNGB, &close, tw->tree->BoxSize);
    double numngb = WINDP(i, Windd).Ngb[close];

    int tid = omp_get_thread_num();
    /*  If we have 40 neighbours, or if DMRadius is narrow, set vdisp. Otherwise add to redo queue */
    if((numngb < (NUMDMNGB - MAXDMDEVIATION) || numngb > (NUMDMNGB + MAXDMDEVIATION)) &&
        (WINDP(i, Windd).Right - WINDP(i, Windd).Left > 1e-2)) {
        /* More work needed: add this particle to the redo queue*/
        tw->NPRedo[tid][tw->NPLeft[tid]] = i;
        tw->NPLeft[tid] ++;
    }
    else{
        double vdisp = WINDP(i, Windd).V2sum[close] / numngb;
        int d;
        for(d = 0; d<3; d++){
            vdisp -= pow(WINDP(i, Windd).V1sum[close][d] / numngb,2);
        }
        if(vdisp > 0)
            WINDP(i, Windd).Vdisp = sqrt(vdisp / 3);
    }

    if(tw->maxnumngb[tid] < numngb)
        tw->maxnumngb[tid] = numngb;
    if(tw->minnumngb[tid] > numngb)
        tw->minnumngb[tid] = numngb;
}

static void
sfr_wind_reduce_weight(int place, TreeWalkResultWind * O, enum TreeWalkReduceMode mode, TreeWalk * tw)
{
    struct winddata * Windd = WIND_GET_PRIV(tw)->Winddata;
    TREEWALK_REDUCE(WINDP(place, Windd).TotalWeight, O->TotalWeight);

    int i;
    if(mode == 0 || WINDP(place, Windd).maxcmpte > O->maxcmpte)
        WINDP(place, Windd).maxcmpte = O->maxcmpte;
    int k;
    for (i = 0; i < O->maxcmpte; i++){
        TREEWALK_REDUCE(WINDP(place, Windd).Ngb[i], O->Ngb[i]);
        TREEWALK_REDUCE(WINDP(place, Windd).V2sum[i], O->V2sum[i]);
        for(k = 0; k < 3; k ++) {
            TREEWALK_REDUCE(WINDP(place, Windd).V1sum[i][k], O->V1sum[i][k]);
        }
    }
//     message(1, "Reduce ID=%ld, NGB_first=%d NGB_last=%d maxcmpte = %d, left = %g, right = %g\n",
//             P[place].ID, O->Ngb[0],O->Ngb[O->maxcmpte-1],WINDP(place, Windd).maxcmpte,WINDP(place, Windd).Left,WINDP(place, Windd).Right);
}

static void
sfr_wind_copy(int place, TreeWalkQueryWind * input, TreeWalk * tw)
{
    double dtime = get_dloga_for_bin(P[place].TimeBin, P[place].Ti_drift) / WIND_GET_PRIV(tw)->hubble;
    struct winddata * Windd = WIND_GET_PRIV(tw)->Winddata;

    input->ID = P[place].ID;
    input->Dt = dtime;
    input->Mass = P[place].Mass;
    input->Hsml = P[place].Hsml;
    input->TotalWeight = WINDP(place, Windd).TotalWeight;
    input->Vdisp = WINDP(place, Windd).Vdisp;
    int i;
    for(i=0; i<3; i++)
        input->Vel[i] = P[place].Vel[i];
    for(i = 0; i<NWINDHSML; i++){
        input->DMRadius[i]=effdmradius(place,i,tw);
    }
}

static void
sfr_wind_weight_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{
    /* this evaluator walks the tree and sums the total mass of surrounding gas
     * particles as described in VS08. */
    /* it also calculates the DM dispersion of the nearest 40 DM particles */
    if(iter->base.other == -1) {
        double hsearch = DMAX(I->Hsml, I->DMRadius[NWINDHSML-1]);
        iter->base.Hsml = hsearch;
        iter->base.mask = 1 + 2; /* gas and dm */
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        O->maxcmpte = NWINDHSML;
        return;
    }

    int other = iter->base.other;
    double r = iter->base.r;
    double * dist = iter->base.dist;

    if(P[other].Type == 0) {
        if(r > I->Hsml) return;
        /* skip earlier wind particles, which receive
         * no feedback energy */
        if(SPHP(other).DelayTime > 0) return;

        /* NOTE: think twice if we want a symmetric tree walk when wk is used. */
        //double wk = density_kernel_wk(&kernel, r);
        double wk = 1.0;
        O->TotalWeight += wk * P[other].Mass;
        /* Sum up all particles visited on this processor*/
        WIND_GET_PRIV(lv->tw)->nvisited[omp_get_thread_num()]++;
    }

    int i;
    if(P[other].Type == 1) {
        const double atime = WIND_GET_PRIV(lv->tw)->Time;
        for(i = 0; i < O->maxcmpte; i++){
            if(r < I->DMRadius[i]){
                O->Ngb[i] += 1;
                int d;
                for(d = 0; d < 3; d ++) {
                    /* Add hubble flow to relative velocity. */
                    double vel = P[other].Vel[d] - I->Vel[d] + WIND_GET_PRIV(lv->tw)->hubble * atime * atime * dist[d];
                    O->V1sum[i][d] += vel;
                    O->V2sum[i] += vel * vel;
                }
            }
        }
    }

    for(i = 0; i<NWINDHSML; i++){
        if(O->Ngb[i] > NUMDMNGB){
            O->maxcmpte = i+1;
            iter->base.Hsml = I->DMRadius[i];
            break;
        }
    }

    /*
    message(1, "ThisTask = %d %ld ngb=%d NGB=%d TotalWeight=%g V2sum=%g V1sum=%g %g %g\n",
    ThisTask, I->ID, numngb, O->Ngb, O->TotalWeight, O->V2sum,
    O->V1sum[0], O->V1sum[1], O->V1sum[2]);
    */
}


static int
get_wind_dir(int i, double dir[3]) {
    /* v and vmean are in internal units (km/s *a ), not km/s !*/
    /* returns 0 if particle i is converted to wind. */
    // message(1, "%ld Making ID=%ld (%g %g %g) to wind with v= %g\n", ID, P[i].ID, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2], v);
    /* ok, make the particle go into the wind */
    double theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
    double phi = 2 * M_PI * get_random_number(P[i].ID + 4);

    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
    return 0;
}

static void
sfr_wind_feedback_ngbiter(TreeWalkQueryWind * I,
        TreeWalkResultWind * O,
        TreeWalkNgbIterWind * iter,
        LocalTreeWalk * lv)
{

    /* this evaluator walks the tree and blows wind. */

    if(iter->base.other == -1) {
        iter->base.mask = 1;
        iter->base.symmetric = NGB_TREEFIND_ASYMMETRIC;
        iter->base.Hsml = I->Hsml;
        return;
    }
    int other = iter->base.other;
    double r = iter->base.r;

    /* this is radius cut is redundant because the tree walk is asymmetric
     * we may want to use fancier weighting that requires symmetric in the future. */
    if(r > I->Hsml) return;

    /* skip earlier wind particles */
    if(SPHP(other).DelayTime > 0) return;

    /* No eligible gas particles not in wind*/
    if(I->TotalWeight == 0 || I->Vdisp <= 0) return;

    /* Paranoia*/
    if(P[other].Type != 0 || P[other].IsGarbage || P[other].Swallowed)
        return;

    double utherm = wind_params.WindThermalFactor * 1.5 * I->Vdisp * I->Vdisp;
    double windeff=0;
    double v=0;
    if(HAS(wind_params.WindModel, WIND_FIXED_EFFICIENCY)) {
        windeff = wind_params.WindEfficiency;
        v = wind_params.WindSpeed * WIND_GET_PRIV(lv->tw)->Time;
    } else if(HAS(wind_params.WindModel, WIND_USE_HALO)) {
        windeff = 1.0 / (pow(I->Vdisp / WIND_GET_PRIV(lv->tw)->Time / wind_params.WindSigma0,2));
        v = wind_params.WindSpeedFactor * I->Vdisp;
    } else {
        endrun(1, "WindModel = 0x%X is strange. This shall not happen.\n", wind_params.WindModel);
    }
    /* Minimum wind velocity. This ensures particles do not remain in the wind forever*/
    if(v < wind_params.MinWindVelocity * WIND_GET_PRIV(lv->tw)->Time)
        v = wind_params.MinWindVelocity * WIND_GET_PRIV(lv->tw)->Time;

    double p = windeff * I->Mass / I->TotalWeight;
    double random = get_random_number(I->ID + P[other].ID);

    if (random < p && v > 0) {
        /* Store a potential kick. This might not be the kick actually used,
         * because another star particle may be closer, but we can resolve
         * that after the treewalk*/
        int64_t * nkicks = &WIND_GET_PRIV(lv->tw)->nkicks;
        /* Use a single global kick list.*/
        int64_t ikick = atomic_fetch_and_add_64(nkicks, 1);
        if(ikick >= WIND_GET_PRIV(lv->tw)->maxkicks)
            endrun(5, "Not enough room in kick queue: %ld > %ld for particle %d starid %ld distance %g\n",
                   ikick, WIND_GET_PRIV(lv->tw)->maxkicks, other, I->ID, r);
        struct StarKick * kick = &WIND_GET_PRIV(lv->tw)->kicks[ikick];
        kick->StarDistance = r;
        kick->StarID = I->ID;
        kick->StarKickVelocity = v;
        kick->StarTherm = utherm;
        kick->part_index = other;
    }
}

int
winds_make_after_sf(int i, double sm, double atime)
{
    if(!HAS(wind_params.WindModel, WIND_SUBGRID) || !winds_ever_decouple())
        return 0;
    /* Here comes the Springel Hernquist 03 wind model */
    /* Notice that this is the mass of the gas particle after forking a star, 1/GENERATIONS
        * what it was before.*/
    double pw = wind_params.WindEfficiency * sm / P[i].Mass;
    double prob = 1 - exp(-pw);
    if(get_random_number(P[i].ID + 2) < prob) {
        double dir[3];
        get_wind_dir(i, dir);
        int j;
        for(j = 0; j < 3; j++)
        {
            P[i].Vel[j] += wind_params.WindSpeed * atime * dir[j];
        }

        SPHP(i).DelayTime = wind_params.WindFreeTravelLength / wind_params.WindSpeed;
    }

    return 0;
}
