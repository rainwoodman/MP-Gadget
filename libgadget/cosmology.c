#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

#include "cosmology.h"
#include "physconst.h"
#include "utils.h"

/*Stefan-Boltzmann constant in cgs units*/
#define  STEFAN_BOLTZMANN 5.670373e-5

static inline double OmegaFLD(const Cosmology * CP, const double a);

void init_cosmology(Cosmology * CP, const double TimeBegin)
{
    /*With slightly relativistic massive neutrinos, for consistency we need to include radiation.
     * A note on normalisation (as of 08/02/2012):
     * CAMB appears to set Omega_Lambda + Omega_Matter+Omega_K = 1,
     * calculating Omega_K in the code and specifying Omega_Lambda and Omega_Matter in the paramfile.
     * This means that Omega_tot = 1+ Omega_r + Omega_g, effectively
     * making h0 (very) slightly larger than specified, and the Universe is no longer flat!
     */
    CP->OmegaCDM = CP->Omega0 - CP->OmegaBaryon;
    CP->OmegaK = 1.0 - CP->Omega0 - CP->OmegaLambda;

    CP->RhoCrit = 3.0 * CP->Hubble * CP->Hubble / (8.0 * M_PI * GRAVITY);

    /* Omega_g = 4 \sigma_B T_{CMB}^4 8 \pi G / (3 c^3 H^2) */

    CP->OmegaG = 4 * STEFAN_BOLTZMANN
                  * pow(CP->CMBTemperature, 4)
                  * (8 * M_PI * GRAVITY)
                  / (3*pow(LIGHTCGS, 3)*HUBBLE*HUBBLE)
                  / (CP->HubbleParam*CP->HubbleParam);

    init_omega_nu(&CP->ONu, CP->MNu, TimeBegin, CP->HubbleParam, CP->CMBTemperature);
    /* Neutrinos will be included in Omega0, if massive.
     * This ensures that OmegaCDM contains only non-relativistic species.*/
    if(CP->MNu[0] + CP->MNu[1] + CP->MNu[2] > 0) {
        CP->OmegaCDM -= get_omega_nu(&CP->ONu, 1);
    }
}

/*Hubble function at scale factor a, in dimensions of CP.Hubble*/
double hubble_function(const Cosmology * CP, double a)
{

    double hubble_a;

    /* first do the terms in SQRT */
    hubble_a = CP->OmegaLambda;

    hubble_a += OmegaFLD(CP, a);
    hubble_a += CP->OmegaK / (a * a);
    hubble_a += (CP->OmegaCDM + CP->OmegaBaryon) / (a * a * a);

    if(CP->RadiationOn) {
        hubble_a += CP->OmegaG / (a * a * a * a);
        hubble_a += get_omega_nu(&CP->ONu, a);
    }
    hubble_a += CP->Omega_ur/(a*a*a*a);
    /* Now finish it up. */
    hubble_a = CP->Hubble * sqrt(hubble_a);
    return (hubble_a);
}

static double growth(Cosmology * CP, double a, double *dDda);

double GrowthFactor(Cosmology * CP, double astart, double aend)
{
    return growth(CP, astart, NULL) / growth(CP, aend, NULL);
}

int growth_ode(double a, const double yy[], double dyda[], void * params)
{
    Cosmology * CP = (Cosmology *) params;
    const double hub = hubble_function(CP, a)/CP->Hubble;
    dyda[0] = yy[1]/pow(a,3)/hub;
    /*Only use gravitating part*/
    /* Note: we do not include neutrinos
     * here as they are free-streaming at the initial time.
     * This is not right if our box is very large and thus overlaps
     * with their free-streaming scale. In that case the growth factor will be scale-dependent
     * and we need to numerically differentiate. In practice the box will either be larger
     * than the horizon, and so need radiation perturbations, or the neutrino
     * mass will be larger than current constraints allow, so we just warn for now.*/
    dyda[1] = yy[0] * 1.5 * a * (CP->OmegaCDM + CP->OmegaBaryon)/(a*a*a) / hub;
    return GSL_SUCCESS;
}

/** The growth function is given as a 2nd order DE in Peacock 1999, Cosmological Physics.
 * D'' + a'/a D' - 1.5 * (a'/a)^2 D = 0
 * 1/a (a D')' - 1.5 (a'/a)^2 D
 * where ' is d/d tau = a^2 H d/da
 * Define F = a^3 H dD/da
 * and we have: dF/da = 1.5 a H D
 */
double growth(Cosmology * CP, double a, double * dDda)
{
  gsl_odeiv2_system FF;
  FF.function = &growth_ode;
  FF.jacobian = NULL;
  FF.params = CP;
  FF.dimension = 2;
  gsl_odeiv2_driver * drive = gsl_odeiv2_driver_alloc_standard_new(&FF,gsl_odeiv2_step_rkf45, 1e-5, 1e-8,1e-8,1,1);
   /* We start early to avoid lambda.*/
  double curtime = 1e-5;
  /* Handle even earlier times*/
  if(a < curtime)
      curtime = a / 10;
  /* Initial velocity chosen so that D = Omegar + 3/2 Omega_m a,
   * the solution for a matter/radiation universe.*
   * Note the normalisation of D is arbitrary
   * and never seen outside this function.*/
  double yinit[2] = {1.5 * (CP->OmegaCDM + CP->OmegaBaryon)/(curtime*curtime), pow(curtime,3)*hubble_function(CP, curtime)/CP->Hubble * 1.5 * (CP->OmegaCDM + CP->OmegaBaryon)/(curtime*curtime*curtime)};
  if(CP->RadiationOn)
      yinit[0] += CP->OmegaG/pow(curtime, 4)+get_omega_nu(&CP->ONu, curtime);

  int stat = gsl_odeiv2_driver_apply(drive, &curtime,a, yinit);
  if (stat != GSL_SUCCESS) {
      endrun(1,"gsl_odeiv in growth: %d. Result at %g is %g %g\n",stat, curtime, yinit[0], yinit[1]);
  }
  gsl_odeiv2_driver_free(drive);
  /*Store derivative of D if needed.*/
  if(dDda) {
      *dDda = yinit[1]/pow(a,3)/(hubble_function(CP, a)/CP->Hubble);
  }
  return yinit[0];
}

/*
 * This is the Zeldovich approximation prefactor,
 * f1 = d ln D1 / dlna = a / D (dD/da)
 */
double F_Omega(Cosmology * CP, double a)
{
    double dD1da=0;
    double D1 = growth(CP, a, &dD1da);
    return a / D1 * dD1da;
}

/*Dark energy density as a function of time:
 * OmegaFLD(a)  ~ exp(-3 int((1+w(a))/a da)_a^1
 * and w(a) = w0 + (1-a) wa*/
static inline double OmegaFLD(const Cosmology * CP, const double a)
{
    if(CP->Omega_fld == 0.)
        return 0;
    return CP->Omega_fld * pow(a, 3 * (1 + CP->w0_fld + CP->wa_fld))*exp(3*CP->wa_fld*(1-a));
}

static double sigma2_int(double k, void * p)
{
    void ** params = p;
    FunctionOfK * fk = params[0];
    double * R = params[1];
    double kr, kr3, kr2, w, x;

    kr = *R * k;
    kr2 = kr * kr;
    kr3 = kr2 * kr;

    if(kr < 1e-8)
        return 0;

    w = 3 * (sin(kr) / kr3 - cos(kr) / kr2);
    x = 4 * M_PI * k * k * w * w * function_of_k_eval(fk, k);

    return x;
}

double function_of_k_eval(FunctionOfK * fk, double k)
{
    /* ignore the 0 mode */

    if(k == 0) return 1;

    int l = 0;
    int r = fk->size - 1;

    while(r - l > 1) {
        int m = (r + l) / 2;
        if(k < fk->table[m].k)
            r = m;
        else
            l = m;
    }
    double k2 = fk->table[r].k,
           k1 = fk->table[l].k;
    double p2 = fk->table[r].Pk,
           p1 = fk->table[l].Pk;

    if(l == r) {
        return fk->table[l].Pk;
    }

    if(p1 == 0 || p2 == 0 || k1 == 0 || k2 == 0) {
        /* if any of the p is zero, use linear interpolation */
        double p = (k - k1) * p2 + (k2 - k) * p1;
        p /= (k2 - k1);
        return p;
    } else {
        k = log(k);
        p1 = log(p1);
        p2 = log(p2);
        k1 = log(k1);
        k2 = log(k2);
        double p = (k - k1) * p2 + (k2 - k) * p1;
        p /= (k2 - k1);
        return exp(p);
    }
}

double function_of_k_tophat_sigma(FunctionOfK * fk, double R)
{
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    void * params[] = {fk, &R};
    double result,abserr;
    gsl_function F;
    F.function = &sigma2_int;
    F.params = params;

    /* note: 500/R is here chosen as integration boundary (infinity) */
    gsl_integration_qags (&F, 0, 500. / R, 0, 1e-4,1000,w,&result, &abserr);
    //   printf("gsl_integration_qng in TopHatSigma2. Result %g, error: %g, intervals: %lu\n",result, abserr,w->size);
    gsl_integration_workspace_free (w);
    return sqrt(result);
}

void function_of_k_normalize_sigma(FunctionOfK * fk, double R, double sigma) {
    double old = function_of_k_tophat_sigma(fk, R);
    size_t i;
    for(i = 0; i < fk->size; i ++) {
        fk->table[i].Pk *= sigma / old;
    };
}
