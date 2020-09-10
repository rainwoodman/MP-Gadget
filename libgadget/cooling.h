#ifndef _COOLING_H_
#define _COOLING_H_

#include "cosmology.h"

/* Ultra-violet background structure.
 * Can be changed on a particle-by-particle basis*/
struct UVBG {
    double J_UV;
    double gJH0;
    double gJHep;
    double gJHe0;
    double epsH0;
    double epsHep;
    double epsHe0;
    double self_shield_dens;
};

/*rates for excursion set, for J21 == 1*/
struct J21_coeffs {
    double gJH0;
    double gJHep;
    double gJHe0;
    double epsH0;
    double epsHep;
    double epsHe0;
};

/*Global unit system for the cooling module*/
struct cooling_units
{
    /*Flag to enable or disable all cooling.*/
    int CoolingOn;
    /*Factor to convert internal density to cgs. By default should be UnitDensity_in_cgs * h^2 */
    double density_in_phys_cgs; //All.UnitDensity_in_cgs * All.CP.HubbleParam * All.CP.HubbleParam
    /*Factor to convert internal internal energy to cgs. By default should be UnitEnergy_in_cgs / UnitMass_in_cgs */
    double uu_in_cgs; //All.UnitEnergy_in_cgs / All.UnitMass_in_cgs
    /*Factor to convert time to s. By default should be UnitTime_in_s / h */
    double tt_in_s; //All.UnitTime_in_s / All.CP.HubbleParam
    /* Baryonic critial density in g cm^-3 at z=0 */
    double rho_crit_baryon;
};

/*Initialise the cooling module.*/
void init_cooling(const char * TreeCoolFile, const char * MetalCoolFile, char * reion_hist_file, struct cooling_units cu, Cosmology * CP);

/*Reads and initialises the tables for a spatially varying redshift of reionization*/
void init_uvf_table(const char * UVFluctuationFile, const double BoxSize, const double UnitLength_in_cm);

/* Get the cooling time for a particle from the internal energy and density, specifying a UVB appropriately.
 * Sets ne_guess to the equilibrium electron density.*/
double GetCoolingTime(double redshift, double u_old, double rho, struct UVBG * uvbg,  double *ne_guess, double Z);

/*Get the new internal energy per unit mass. ne_guess is set to the new internal equilibrium electron density*/
double DoCooling(double redshift, double u_old, double rho, double dt, struct UVBG * uvbg, double *ne_guess, double Z, double MinEgySpec, int isHeIIIionized);

/*Interpolates the ultra-violet background tables to the desired redshift and returns a cooling rate table*/
struct UVBG get_global_UVBG(double redshift);

/* Change the ultra-violet background table according to a pre-computed table of UV fluctuations.
 * This zeros the UVBG if this particular particle has not reionized yet*/
struct UVBG get_local_UVBG(double redshift, const struct UVBG * const GlobalUVBG, const double * const Pos, const double * const PosOffset, int heiiionized);

/*Get the equilibrium temperature at given internal energy.
    density is total gas density in protons/cm^3
    Internal energy is in J/kg == 10^-10 ergs/g.
    helium is a mass fraction*/
double get_temp(double density, double ienergy, double helium, const struct UVBG * uvbg, double * ne_init);

/*Gets the neutral fraction from density and internal energy in internal units.
  u_old is in internal units. rho is in internal physical density units and is converted to
  physical protons/cm^3 inside the function. */
double GetNeutralFraction(double u_old, double rho, const struct UVBG * uvbg, double ne);

/*Gets the helium ionic fractions from density and internal energy in internal units.
  u_old is in internal units. rho is in internal physical density units and is converted to
  physical protons/cm^3 inside the function. */
double GetHeliumIonFraction(int ion, double u_old, double rho, const struct UVBG * uvbg, double ne_init);

#endif
