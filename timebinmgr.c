#include <math.h>
#include <string.h>

#include "allvars.h"
#include "timebinmgr.h"

/*! table with desired sync points. All forces and phase space variables are synchonized to the same order. */
static SyncPoint SyncPoints[8192];
static SyncPoint PMSyncPoint[1];
static int NSyncPoints;    /* number of times stored in table of desired sync points */

/*Make sure the OutputList runs from TimeInit to TimeMax, inclusive.*/
void
setup_sync_points(void)
{
    int i;

    /* Set up first and last entry to SyncPoints; TODO we can insert many more! */

    SyncPoints[0].loga = log(All.TimeInit); 
    SyncPoints[0].write_snapshot = 0; /* by default no output here. */
    SyncPoints[1].loga = log(All.TimeMax);
    SyncPoints[1].write_snapshot = 1;

    NSyncPoints = 2;

    /* we do an insertion sort here. A heap is faster but who cares the speed for this? */
    for(i = 0; i < All.OutputListLength; i ++) {
        int j = 0;
        double loga = log(All.OutputListTimes[i]);

        for(j = 0; j < NSyncPoints; j ++) {
            if(loga <= SyncPoints[j].loga) {
                break;
            }
        }
        if(j == NSyncPoints) {
            /* beyond TimeMax, skip */
            continue;
        }
        /* found, so loga >= SyncPoints[j].loga */
        if(loga == SyncPoints[j].loga) {
            /* requesting output on an existing entry, e.g. TimeInit or duplicated entry */
        } else {
            /* insert the item; */
            memmove(&SyncPoints[j], &SyncPoints[j + 1],
                sizeof(SyncPoints[0]) * (NSyncPoints - j));
            SyncPoints[j].loga = loga;
            NSyncPoints ++;
        }
        SyncPoints[j].write_snapshot = 1;
        if(All.SnapshotWithFOF) {
            SyncPoints[j].write_fof = 1;
        }
    }

    for(i = 0; i < NSyncPoints; i++) {
        SyncPoints[i].ti = (i * 1L) << (TIMEBINS);
    }

    for(i = 0; i < NSyncPoints; i++) {
        message(1,"Out: %g %ld\n", exp(SyncPoints[i].loga), SyncPoints[i].ti);
    }
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

SyncPoint *
get_pm_sync_point(inttime_t ti)
{
    PMSyncPoint[0].ti = ti;
    PMSyncPoint[0].loga = loga_from_ti(ti);
    PMSyncPoint[0].write_fof = 0;
    PMSyncPoint[0].write_snapshot = 1;
    return &PMSyncPoint[0];
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
    inttime_t lastsnap = ti >> TIMEBINS;
    /*Use logDTime from the last valid interval*/
    if(lastsnap >= NSyncPoints - 1)
        lastsnap = NSyncPoints - 2;
    double lastoutput = SyncPoints[lastsnap].loga;
    return (SyncPoints[lastsnap+1].loga - lastoutput)/TIMEBASE;
}

double
loga_from_ti(inttime_t ti)
{
    inttime_t lastsnap = ti >> TIMEBINS;
    if(lastsnap >= NSyncPoints)
        lastsnap = NSyncPoints - 1;
    double lastoutput = SyncPoints[lastsnap].loga;
    double logDTime = Dloga_interval_ti(ti);
    return lastoutput + (ti & (TIMEBASE-1)) * logDTime;
}

inttime_t
ti_from_loga(double loga)
{
    int i;
    int ti;
    for(i = 1; i < NSyncPoints - 1; i++)
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
dloga_from_dti(inttime_t dti)
{
    double loga = loga_from_ti(All.Ti_Current);
    double logap = loga_from_ti(All.Ti_Current+dti);
    return logap - loga;
}

inttime_t
dti_from_dloga(double loga)
{
    int ti = ti_from_loga(loga_from_ti(All.Ti_Current));
    int tip = ti_from_loga(loga+loga_from_ti(All.Ti_Current));
    return tip - ti;
}

double
get_dloga_for_bin(int timebin)
{
    double logDTime = Dloga_interval_ti(All.Ti_Current);
    return (timebin ? (1u << timebin) : 0 ) * logDTime;
}

inttime_t
round_down_power_of_two(inttime_t dti)
{
    /* make dti a power 2 subdivision */
    inttime_t ti_min = TIMEBASE;
    while(ti_min > dti)
        ti_min >>= 1;
    return ti_min;
}

