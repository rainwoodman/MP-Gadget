/**************
 * This file is based on the Gadget-3 cooling.c.
 *
 * The main atomic rates have been rewritten from scratch and now exist in cooling_rates.c.
 * There is also support for metal cooling and a fluctuating UV background in cooling_uvfluc.c
 *
 * There is no public version of original cooling.c freely available.
 *
 * Only reference is 2004 version by Brian O'Shea in ENZO repo,
 * which contains a deferred freedom claimer depending on the action
 * of Volker Springel.
 *
 * The LICENSE of the DoCooling function is therefore still in limbo,
 * although everything else is now unencumbered.
 *
 ************** */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <bigfile.h>
#include "utils/endrun.h"
#include "utils/mymalloc.h"
#include "utils/interp.h"
#include "physconst.h"
#include "cooling.h"
#include "cooling_rates.h"
#include "cooling_qso_lightup.h"
#include "cosmology.h"
#include "uvbg.h"

static struct cooling_units coolunits;

/*Do initialisation for the cooling module*/
void init_cooling(const char * TreeCoolFile, const char * MetalCoolFile, char * reion_hist_file, struct cooling_units cu, Cosmology * CP)
{
    coolunits = cu;
    /* Get mean cosmic baryon density for photoheating rate from long mean free path photons */
    coolunits.rho_crit_baryon =  3 * pow(CP->HubbleParam * HUBBLE,2) * CP->OmegaBaryon / (8 * M_PI * GRAVITY);
    /*Initialize the cooling rates*/
    if(coolunits.CoolingOn)
        init_cooling_rates(TreeCoolFile, MetalCoolFile, CP);
    /* Initialize the helium reionization model*/
    init_qso_lightup(reion_hist_file);
}

#define MAXITER 1000

/* Wrapper function which returns the rate of change of internal energy in units of
 * erg/s/g. Arguments:
 * rho: density in protons/cm^3 (physical)
 * u: internal energy in units of erg/g
 * Z: metallicity
 * redshift: redshift
 * isHeIIIionized: flags whether the particle has been HeII reionized.
 */
static double
get_lambdanet(double rho, double u, double redshift, double Z, struct UVBG * uvbg, double * ne_guess, int isHeIIIionized)
{
    double LambdaNet = get_heatingcooling_rate(rho, u, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);
    if(!isHeIIIionized) {
        /* get_long_mean_free_path_heating returns the heating in units of erg/s/cm^3,
         * the factor of the mean density converts from erg/s/cm^3 to erg/s/g */
        LambdaNet += get_long_mean_free_path_heating(redshift) / (coolunits.rho_crit_baryon * pow(1 + redshift,3));
    }
    return LambdaNet;
}

/* returns new internal energy per unit mass.
 * Arguments are passed in code units, density is proper density.
 */
double DoCooling(double redshift, double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z, double MinEgySpec, int isHeIIIionized)
{
    if(!coolunits.CoolingOn) return 0;

    double u, du;
    double u_lower, u_upper;
    double LambdaNet;
    int iter = 0;

    rho *= coolunits.density_in_phys_cgs / PROTONMASS;	/* convert to (physical) protons/cm^3 */
    u_old *= coolunits.uu_in_cgs;
    MinEgySpec *= coolunits.uu_in_cgs;
    if(u_old < MinEgySpec)
        u_old = MinEgySpec;
    dt *= coolunits.tt_in_s;

    u = u_old;
    u_lower = u;
    u_upper = u;

    LambdaNet = get_lambdanet(rho, u, redshift, Z, uvbg, ne_guess, isHeIIIionized);

    /* bracketing */
    if(u - u_old - LambdaNet * dt < 0)	/* heating */
    {
        do
        {
            u_lower = u_upper;
            u_upper *= 1.1;
        } while(u_upper - u_old - get_lambdanet(rho, u_upper, redshift, Z, uvbg, ne_guess, isHeIIIionized) * dt < 0);
    }
    else
    {
        do {
            u_upper = u_lower;
            u_lower /= 1.1;
            /* This means that we don't need an initial bracket*/
            if(u_upper <= MinEgySpec) {
                break;
            }
        } while(u_lower - u_old - get_lambdanet(rho, u_lower, redshift, Z, uvbg, ne_guess, isHeIIIionized) * dt > 0);
    }

    do
    {
        u = 0.5 * (u_lower + u_upper);
        /* If we know that the new energy
         * is below the minimum gas internal energy, we are done here.*/
        if(u_upper <= MinEgySpec) {
            u = MinEgySpec;
            break;
        }

        LambdaNet = get_lambdanet(rho, u, redshift, Z, uvbg, ne_guess, isHeIIIionized);

        if(u - u_old - LambdaNet * dt > 0)
        {
            u_upper = u;
        }
        else
        {
            u_lower = u;
        }

        du = u_upper - u_lower;

        iter++;

        if(iter >= (MAXITER - 10))
            message(1, "u= %g\n", u);
    }
    while(fabs(du / u) > 1.0e-6 && iter < MAXITER);

    if(iter >= MAXITER)
    {
        endrun(10, "failed to converge in DoCooling()\n");
    }

    u /= coolunits.uu_in_cgs;   /*convert back to internal units */

    return u;
}

/* returns cooling time.
 * NOTE: If we actually have heating, a cooling time of 0 is returned.
 */
double GetCoolingTime(double redshift, double u_old, double rho, struct UVBG * uvbg, double *ne_guess, double Z)
{
    if(!coolunits.CoolingOn) return 0;

    /* convert to physical cgs units */
    rho *= coolunits.density_in_phys_cgs / PROTONMASS;
    u_old *= coolunits.uu_in_cgs;

    /* Note: this does not include the long mean free path heating from helium reionization*/
    double LambdaNet = get_heatingcooling_rate(rho, u_old, 1 - HYDROGEN_MASSFRAC, redshift, Z, uvbg, ne_guess);

    if(LambdaNet >= 0)		/* ups, we have actually heating due to UV background */
        return 0;

    double coolingtime = u_old / (- LambdaNet);

    /*Convert back to internal units*/
    coolingtime /= coolunits.tt_in_s;

    return coolingtime;
}

/*Gets the neutral fraction from density and internal energy in internal units*/
double
GetNeutralFraction(double u_old, double rho, const struct UVBG * uvbg, double ne_init)
{
    /* convert to physical cgs units */
    rho *= coolunits.density_in_phys_cgs / PROTONMASS;
    u_old *= coolunits.uu_in_cgs;
    double nh0 = get_neutral_fraction_phys_cgs(rho, u_old, 1 - HYDROGEN_MASSFRAC, uvbg, &ne_init);
    return nh0;
}

/*Gets the helium ion fraction from density and internal energy in internal units*/
double
GetHeliumIonFraction(int ion, double u_old, double rho, const struct UVBG * uvbg, double ne_init)
{
    /* convert to physical cgs units */
    rho *= coolunits.density_in_phys_cgs / PROTONMASS;
    u_old *= coolunits.uu_in_cgs;
    double helium_ion = get_helium_ion_phys_cgs(ion, rho, u_old, 1 - HYDROGEN_MASSFRAC, uvbg, ne_init);
    return helium_ion;
}
/* 
 * returns the spatial dependent UVBG if UV fluctuation is enabled. 
 *
 * */
// void GetParticleUVBG(int i, struct UVBG * uvbg) {
//     double z = 1 / All.cf.a - 1;
//     if(All.UVRedshiftThreshold >= 0.0 && z > All.UVRedshiftThreshold) {
//         [> if a threshold is set, disable UV bg above that redshift <]
//         memset(uvbg, 0, sizeof(struct UVBG));
//         return;
//     }
//     if(UVF.disabled) {
//         [> directly use the TREECOOL table if UVF is disabled <]
//         memcpy(uvbg, &GlobalUVBG, sizeof(struct UVBG));
//         return;
//     }
//     double pos[3];
//     int k;
//     for(k = 0; k < 3; k ++) {
//         pos[k] = P[i].Pos[k];
//     }
//     double zreion = interp_eval_periodic(&UVF.interp, pos, UVF.Table);
//     if(zreion < z) {
//         memset(uvbg, 0, sizeof(struct UVBG));
//     } else {
//         memcpy(uvbg, &GlobalUVBG, sizeof(struct UVBG));
//     }
// }

void GetParticleUVBG(int i, struct UVBG * uvbg) {
    int ind[3] = {-1};
    for (int ii = 0; ii<3; ii++) {
        ind[ii] = pos_to_ngp(P[i].Pos[ii], All.BoxSize, UVBG_DIM);
    }

    // N.B. J21 must be in units of 1e-21 erg s-1 Hz-1 (proper cm)-2 sr-1
    double J21 = UVBGgrids.J21[grid_index(ind[0], ind[1], ind[2], UVBG_DIM, INDEX_REAL)];
    uvbg->J_UV = J21;

    // ionisation rate
    uvbg->gJH0   = 2.090e-12 * J21; // s-1
    uvbg->gJHep  = 5.049e-13 * J21; // s-1
    uvbg->gJHe0  = 6.118e-15 * J21; // s-1

    // photoheating rate
    uvbg->epsH0  = 5.951e-12 * J21 * 1.60218e-12; // erg s-1
    uvbg->epsHep = 3.180e-12 * J21 * 1.60218e-12; // erg s-1
    uvbg->epsHe0 = 6.883e-14 * J21 * 1.60218e-12; // erg s-1
}
