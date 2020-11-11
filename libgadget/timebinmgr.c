#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "timebinmgr.h"
#include "utils.h"
#include "uvbg.h"

/*! table with desired sync points. All forces and phase space variables are synchonized to the same order. */
static SyncPoint * SyncPoints;
static int NSyncPoints;    /* number of times stored in table of desired sync points */
static struct sync_params
{
    int OutputListLength;
    double OutputListTimes[1024];
} Sync;

int cmp_double(const void * a, const void * b)
{
    return ( *(double*)a - *(double*)b );
}

/*! This function parses a string containing a comma-separated list of variables,
 *  each of which is interpreted as a double.
 *  The purpose is to read an array of output times into the code.
 *  So specifying the output list now looks like:
 *  OutputList  0.1,0.3,0.5,1.0
 *
 *  We sort the input after reading it, so that the initial list need not be sorted.
 *  This function could be repurposed for reading generic arrays in future.
 */
int
OutputListAction(ParameterSet * ps, char * name, void * data)
{
    char * outputlist = param_get_string(ps, name);
    char * strtmp = fastpm_strdup(outputlist);
    char * token;
    int count;

    /* Note TimeInit and TimeMax not yet initialised here*/

    /*First parse the string to get the number of outputs*/
    for(count=0, token=strtok(strtmp,","); token; count++, token=strtok(NULL, ","))
    {}
/*     message(1, "Found %d times in output list.\n", count); */

    /*Allocate enough memory*/
    Sync.OutputListLength = count;
    int maxcount = sizeof(Sync.OutputListTimes) / sizeof(Sync.OutputListTimes[0]);
    if(maxcount > (int) MAXSNAPSHOTS)
        maxcount = MAXSNAPSHOTS;
    if(Sync.OutputListLength > maxcount) {
        message(1, "Too many entries (%d) in the OutputList, can take no more than %d.\n", Sync.OutputListLength, maxcount);
        return 1;
    }
    /*Now read in the values*/
    for(count=0,token=strtok(outputlist,","); count < Sync.OutputListLength && token; count++, token=strtok(NULL,","))
    {
        /* Skip a leading quote if one exists.
         * Extra characters are ignored by atof, so
         * no need to skip matching char.*/
        if(token[0] == '"')
            token+=1;

        double a = atof(token);

        if(a < 0.0) {
            endrun(1, "Requesting a negative output scaling factor a = %g\n", a);
        }
        Sync.OutputListTimes[count] = a;
/*         message(1, "Output at: %g\n", Sync.OutputListTimes[count]); */
    }
    myfree(strtmp);
    return 0;
}

/* For the tests*/
void set_sync_params(int OutputListLength, double * OutputListTimes)
{
    int i;
    Sync.OutputListLength = OutputListLength;
    for(i = 0; i < OutputListLength; i++)
        Sync.OutputListTimes[i] = OutputListTimes[i];
}

/* This function compiles
 *
 * Sync.OutputListTimes, All.TimeIC, All.TimeMax
 *
 * into a list of SyncPoint objects.
 *
 * A SyncPoint is a time step where all state variables are at the same time on the
 * KkdkK timeline.
 *
 * TimeIC and TimeMax are used to ensure restarting from snapshot obtains exactly identical
 * integer stamps.
 **/
void
setup_sync_points(double TimeIC, double TimeMax, double UVBGTimestep, double no_snapshot_until_time, int SnapshotWithFOF)
{
    int i;

    qsort_openmp(Sync.OutputListTimes, Sync.OutputListLength, sizeof(double), cmp_double);

    if(NSyncPoints > 0)
        myfree(SyncPoints);
    //TODO(Jdavies): don't use syncpoints for uvbg calculation, or figure out how many are there beforehand
    //z=20 to z=4 is ~150 syncpoints at 10 Myr spaces
    //
    SyncPoints = mymalloc("SyncPoints", sizeof(SyncPoint) * (Sync.OutputListLength+2+400)); 

    /* Set up first and last entry to SyncPoints; TODO we can insert many more! */
    //NOTE(jdavies): these first syncpoints need to be in order
   
    SyncPoints[0].a = TimeIC;
    SyncPoints[0].loga = log(TimeIC);
    SyncPoints[0].write_snapshot = 0; /* by default no output here. */
    SyncPoints[0].write_fof = 0;
    SyncPoints[0].calc_uvbg = 0;
    NSyncPoints = 1;

    // UVBG calculation every 10 Myr from z=20
    {
        double z_start = 20.;
        double uv_a = 1.0 / (1.0 + z_start);
        while (uv_a <= TimeMax) {
            SyncPoints[NSyncPoints].a = uv_a;
            SyncPoints[NSyncPoints].loga = log(uv_a);
            SyncPoints[NSyncPoints].write_snapshot = 0;
            SyncPoints[NSyncPoints].write_fof = 0;
            SyncPoints[NSyncPoints].calc_uvbg = 1;
            NSyncPoints++;
            //message(0,"added UVBG syncpoint at a = %.3f, Nsync = %d\n",uv_a,NSyncPoints);

            // TODO(smutch): OK - this is ridiculous (sorry!), but I just wanted to quickly hack something...
            // TODO(jdavies): fix low-z where delta_a > 10Myr
            double delta_a = 0.0001;
            double lbt = time_to_present(uv_a);
            double delta_lbt = 0.0;
            while ((delta_lbt <= UVBGTimestep) && (uv_a <= TimeMax)) {
                uv_a += delta_a;
                delta_lbt = lbt - time_to_present(uv_a);
                //message(0,"trying UVBG syncpoint at a = %.3e, z = %.3e, delta_lbt = %.3e\n",uv_a,1/uv_a - 1,delta_lbt);
            }
        }
    }
    
    SyncPoints[NSyncPoints].a = TimeMax;
    SyncPoints[NSyncPoints].loga = log(TimeMax);
    SyncPoints[NSyncPoints].write_snapshot = 1;
    SyncPoints[NSyncPoints].calc_uvbg = 1;
    SyncPoints[NSyncPoints].write_fof = 0;
    NSyncPoints++;

    /* we do an insertion sort here. A heap is faster but who cares the speed for this? */
    for(i = 0; i < Sync.OutputListLength; i ++) {
        int j = 0;
        double a = Sync.OutputListTimes[i];
        double loga = log(a);

        for(j = 0; j < NSyncPoints; j ++) {
            if(a <= SyncPoints[j].a) {
                break;
            }
        }
        if(j == NSyncPoints) {
            /* beyond TimeMax, skip */
            continue;
        }
        /* found, so loga >= SyncPoints[j].loga */
        if(a == SyncPoints[j].a) {
            /* requesting output on an existing entry, e.g. TimeInit or duplicated entry */
        } else {
            /* insert the item; */
            memmove(&SyncPoints[j + 1], &SyncPoints[j],
                sizeof(SyncPoints[0]) * (NSyncPoints - j));
            SyncPoints[j].a = a;
            SyncPoints[j].loga = loga;
            NSyncPoints ++;
            //message(0,"added outlist syncpoint at a = %.3f, j = %d, Ns = %d\n",a,j,NSyncPoints);
        }
        if(SyncPoints[j].a > no_snapshot_until_time) {
            SyncPoints[j].write_snapshot = 1;
            SyncPoints[j].calc_uvbg = 0;
            if(SnapshotWithFOF) {
                SyncPoints[j].write_fof = 1;
            }
            else
                SyncPoints[j].write_fof = 0;
        } else {
            SyncPoints[j].write_snapshot = 0;
            SyncPoints[j].write_fof = 0;
            SyncPoints[j].calc_uvbg = 0;
        }
    }

    for(i = 0; i < NSyncPoints; i++) {
        SyncPoints[i].ti = (i * 1L) << (TIMEBINS);
    }

    //message(1,"NSyncPoints = %d, OutputListLength = %d , timemax = %.3f\n",NSyncPoints,Sync.OutputListLength,TimeMax);
    /*for(i = 0; i < NSyncPoints; i++) {
        message(1,"Out: %g %ld\n", exp(SyncPoints[i].loga), SyncPoints[i].ti);
    }*/
}

/*! this function returns the next output time that is in the future of
 *  ti_curr; if none is find it return NULL, indication the run shall terminate.
 */
SyncPoint *
find_next_sync_point(inttime_t ti)
{
    int i;
    for(i = 0; i < NSyncPoints; i ++) {
        if(SyncPoints[i].ti > ti) {
            return &SyncPoints[i];
        }
    }
    return NULL;
}

/* This function finds if ti is a sync point; if so returns the sync point;
 * otherwise, NULL. We check if we shall write a snapshot with this. */
SyncPoint *
find_current_sync_point(inttime_t ti)
{
    int i;
    for(i = 0; i < NSyncPoints; i ++) {
        if(SyncPoints[i].ti == ti) {
            return &SyncPoints[i];
        }
    }
    return NULL;
}

/* Each integer time stores in the first 10 bits the snapshot number.
 * Then the rest of the bits are the standard integer timeline,
 * which should be a power-of-two hierarchy. We use this bit trick to speed up
 * the dloga look up. But the additional math makes this quite fragile. */

/*Gets Dloga / ti for the current integer timeline.
 * Valid up to the next snapshot, after which it will change*/
static double
Dloga_interval_ti(inttime_t ti)
{
    /* FIXME: This uses the bit tricks because it has to be fast
     * -- till we clean up the calls to loga_from_ti; then we can avoid bit tricks. */

    inttime_t lastsnap = ti >> TIMEBINS;

    if(lastsnap >= NSyncPoints - 1) {
        /* stop advancing loga after the last sync point. */
        return 0;
    }
    double lastoutput = SyncPoints[lastsnap].loga;
    return (SyncPoints[lastsnap+1].loga - lastoutput)/TIMEBASE;
}

double
loga_from_ti(inttime_t ti)
{
    inttime_t lastsnap = ti >> TIMEBINS;
    if(lastsnap > NSyncPoints) {
        endrun(1, "Requesting snap %d, from ti %d, beyond last sync point %d\n", lastsnap, ti, NSyncPoints);
    }
    double last = SyncPoints[lastsnap].loga;
    inttime_t dti = ti & (TIMEBASE - 1);
    double logDTime = Dloga_interval_ti(ti);
    return last + dti * logDTime;
}

inttime_t
ti_from_loga(double loga)
{
    int i;
    int ti;
    for(i = 0; i < NSyncPoints - 1; i++)
    {
        if(SyncPoints[i].loga > loga)
            break;
    }
    /*If loop didn't trigger, i == All.NSyncPointTimes-1*/
    double logDTime = (SyncPoints[i].loga - SyncPoints[i-1].loga)/TIMEBASE;
    ti = (i-1) << TIMEBINS;
    /* Note this means if we overrun the end of the timeline,
     * we still get something reasonable*/
    ti += (loga - SyncPoints[i-1].loga)/logDTime;
    return ti;
}

double
dloga_from_dti(inttime_t dti, const inttime_t Ti_Current)
{
    double Dloga = Dloga_interval_ti(Ti_Current);
    int sign = 1;
    if(dti < 0) {
        dti = -dti;
        sign = -1;
    }
    if((unsigned int) dti > TIMEBASE) {
        endrun(1, "Requesting dti %d larger than TIMEBASE %u\n", sign*dti, TIMEBASE);
    }
    return Dloga * dti * sign;
}

inttime_t
dti_from_dloga(double loga, const inttime_t Ti_Current)
{
    inttime_t ti = ti_from_loga(loga_from_ti(Ti_Current));
    inttime_t tip = ti_from_loga(loga+loga_from_ti(Ti_Current));
    return tip - ti;
}

double
get_dloga_for_bin(int timebin, const inttime_t Ti_Current)
{
    double logDTime = Dloga_interval_ti(Ti_Current);
    return (timebin > 0 ? (1u << (unsigned) timebin) : 0 ) * logDTime;
}

inttime_t
round_down_power_of_two(inttime_t dti)
{
    /* make dti a power 2 subdivision */
    inttime_t ti_min = TIMEBASE;
    int sign = 1;
    if(dti < 0) {
        dti = -dti;
        sign = -1;
    }
    while(ti_min > dti)
        ti_min >>= 1;
    return ti_min * sign;
}

