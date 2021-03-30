/*Support header for functions used internally in the cooling module*/

#ifndef _COOLING_RATES_H_
#define _COOLING_RATES_H_

#include "cooling.h"
#include "utils/paramset.h"

/* Definitions for the cooling rates code*/
enum RecombType {
    Cen92 = 0, // Recombination from Cen 92
    Verner96 = 1, //Verner 96 recombination rates. Basically accurate, used by Sherwood and Nyx by default.
    Badnell06 = 2, //Even more up to date rates from Badnell 2006, cloudy's current default.
};

enum CoolingType {
    KWH92 = 0, //Cooling from KWH 92
    Enzo2Nyx = 1, //Updated cooling rates from Avery Meiksin, used in Nyx and Enzo 2.
    Sherwood =2 ,  //Same as KWH92, but with an improved large temperature correction factor.
};

/*Parameters for the cooling rate network*/
struct cooling_params
{
    /*Default: Verner96*/
    enum RecombType recomb;
    /*Default: Sherwood*/
    enum CoolingType cooling;

    /*Enable a self-shielding cooling and ionization correction from Rahmati & Schaye 2013. Default: on.*/
    int SelfShieldingOn;
    /*Disable using photo-ionization and heating table, for testing.*/
    int PhotoIonizationOn;
    /*Global baryon fraction, Omega_b/Omega_cdm, used for the self-shielding formula.*/
    double fBar;

    /*Normalization factor to apply to the UVB: Default: 1.*/
    double PhotoIonizeFactor;

    /*CMB temperature in K*/
    double CMBTemperature;

    /*Minimum gas temperature in K*/
    double MinGasTemp;

    /*Threshold redshift above which the UVB is set to zero*/
    double UVRedshiftThreshold;

    /* Hydrogen heating modifier */
    double HydrogenHeatAmp;

    /*Parameters for the 'extra heating' Helium photoionization model.*/
    int HeliumHeatOn;
    double HeliumHeatThresh;
    double HeliumHeatAmp;
    double HeliumHeatExp;
    double rho_crit_baryon;
};

/*Set the parameters for the cooling module from the parameter file.*/
void set_cooling_params(ParameterSet * ps);
/*Set cooling module parameters from a cooling_params struct for the tests*/
void set_coolpar(struct cooling_params cp);

/*Initialize the cooling rate module. This builds a lot of interpolation tables.
 * Defaults: TCMB 2.7255, recomb = Verner96, cooling = Sherwood.*/
void init_cooling_rates(const char * TreeCoolFile, const char * J21CoeffFile, const char * MetalCoolFile, Cosmology * CP);

/* Reads and initializes the cloudy metal cooling table. Called in init_cooling_rates. No need to call it separately.*/
void InitMetalCooling(const char * MetalCoolFile);

/*Get the metal cooling rate from the table.*/
double TableMetalCoolingRate(double redshift, double temp, double nHcgs);

/*Solve the system of equations for photo-ionization equilibrium,
  starting with ne = nH and continuing until convergence.
  density is gas density in protons/cm^3
  Internal energy is in J/kg == 10^-10 ergs/g.
  helium is a mass fraction.
*/
double get_equilib_ne(double density, double ienergy, double helium, double *logt, const struct UVBG * uvbg, double ne_init);

/*Same as above, but get electrons per proton.*/
double get_ne_by_nh(double density, double ienergy, double helium, const struct UVBG * uvbg, double ne_init);

/*Get the total (photo) heating and cooling rate for a given temperature (internal energy) and density.
  density is total gas density in protons/cm^3
  Internal energy is in J/kg == 10^-10 ergs/g.
  helium is a mass fraction, 1 - HYDROGEN_MASSFRAC = 0.24 for primordial gas.
  Returns heating - cooling.
 */
double get_heatingcooling_rate(double density, double ienergy, double helium, double redshift, double metallicity, const struct UVBG * uvbg, double * ne_equilib);

enum CoolProcess {
    RECOMB,
    COLLIS,
    FREEFREE,
    HEAT
};

/* As above but returns only the specified heating rate. */
double get_individual_cooling(enum CoolProcess process, double density, double ienergy, double helium, const struct UVBG * uvbg, double *ne_equilib);

/* As above but returns only the Compton cooling. */
double get_compton_cooling(double density, double ienergy, double helium, double redshift, double nebynh);

/*Get the neutral hydrogen fraction at a given temperature and density.
density is gas density in protons/cm^3
Internal energy is in J/kg == 10^-10 ergs/g.
helium is a mass fraction.*/
double get_neutral_fraction_phys_cgs(double density, double ienergy, double helium, const struct UVBG * uvbg, double * ne_init);

/* Get the helium ionic fractions at a temperature and density. Same conventions as above*/
double get_helium_ion_phys_cgs(int ion, double density, double ienergy, double helium, const struct UVBG * uvbg, double ne_init);

/* get self_shielding density from uvbg (added to .h by jdavies or uvfluc) */
double get_self_shield_dens(double redshift, const struct UVBG * uvbg);

/* get ionrate coefficients for excursion set*/
struct J21_coeffs get_J21_coeffs(double alpha);

#endif
