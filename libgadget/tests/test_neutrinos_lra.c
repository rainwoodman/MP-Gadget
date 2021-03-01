#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../neutrinos_lra.h"
#include "../omega_nu_single.h"
#include "../physconst.h"
#include "../utils/endrun.h"
#include "stub.h"

#define T_CMB0 2.7255
/** Fit to the special function J(x) that is accurate to better than 3% relative and 0.07% absolute*/
double specialJ(const double x, const double vcmnubylight, const double nufrac_low);
double fslength(Cosmology * CP, const double logai, const double logaf, const double light);

void petaio_save_block(BigFile * bf, char * blockname, BigArray * array, int verbose) {};
int petaio_read_block(BigFile * bf, char * blockname, BigArray * array, int required)
{
    return 0;
}


extern _delta_tot_table delta_tot_table;


void setup_cosmology(Cosmology * CP, double MNu[])
{
    CP->CMBTemperature = 2.7255;
    CP->Omega0 = 0.2793;
    CP->OmegaLambda = 1- CP->Omega0;
    CP->OmegaBaryon = 0.0464;
    CP->HubbleParam = 0.7;
    CP->RadiationOn = 1;
    CP->Omega_fld = 0;
    CP->Omega_ur = 0;
    int i;
    for(i = 0; i<3; i++)
        CP->MNu[i] = MNu[i];
    /*Default value for L=kpc v=km/s*/
    double UnitTime_in_s = 3.08568e+16;
    double UnitLength_in_cm = 3.085678e+21;
    double UnitMass_in_g = 1.989e+43;
    /*Should be 0.1*/
    CP->Hubble = HUBBLE * UnitTime_in_s;
    /*Do the main cosmology initialisation*/
    init_cosmology(CP,0.01,UnitLength_in_cm,UnitMass_in_g,UnitTime_in_s);
}

/* Test that the allocations are done correctly.
 * delta_tot is still empty (but allocated) after this.*/
static void test_allocate_delta_tot_table(void **state)
{
    _omega_nu omnu;
    double MNu[3] = {0, 0, 0};
    int i;
    init_omega_nu(&omnu, MNu, 0.01, 0.7,T_CMB0);
    init_neutrinos_lra(300, 0.01, 1, 0.2793, &omnu, 1, 1);
    assert_true(delta_tot_table.ia == 0);
    assert_true(delta_tot_table.namax > 10);
    assert_true(delta_tot_table.scalefact);
    assert_true(delta_tot_table.delta_nu_init);
    assert_true(delta_tot_table.delta_tot);
    for(i=0; i<delta_tot_table.nk_allocated; i++){
        assert_true(delta_tot_table.delta_tot[i]);
    }
}

/*Check that the fits to the special J function are accurate, by doing explicit integration.*/
static void test_specialJ(void **state)
{
    /*Check against mathematica computed values:
    Integrate[(Sinc[q*x])*(q^2/(Exp[q] + 1)), {q, 0, Infinity}]*/
    assert_true(specialJ(0,-1,0) == 1);
    assert_true(fabs(specialJ(1,-1, 0) - 0.2117) < 1e-3);
    assert_true(fabs(specialJ(2,-1, 0) - 0.0223807) < 1e-3);
    assert_true(fabs(specialJ(0.5,-1, 0) - 0.614729) < 1e-3);
    assert_true(fabs(specialJ(0.3,-1, 0) - 0.829763) < 1e-3);
    /*Test that it is ok when truncated*/
    /*Mathematica: Jfrac[x_, qc_] := NIntegrate[(Sinc[q*x])*(q^2/(Exp[q] + 1)), {q, qc, Infinity}]/(3*Zeta[3]/2) */
    assert_true(fabs(specialJ(0,1, 0) - 0.940437) < 1e-4);
    assert_true(fabs(specialJ(0.5,1e-2, 0.5) - 0.614729/0.5) < 1e-3);
    assert_true(fabs(specialJ(0.5,1, 0.5) - 0.556557/0.5) < 1e-4);
    assert_true(fabs(specialJ(1,0.1, 0.5) - 0.211662/0.5) < 1e-4);
}

/* Check that we accurately work out the free-streaming length.
 * Free-streaming length for a non-relativistic particle of momentum q = T0, from scale factor ai to af.
 * The 'light' argument defines the units.
 * Test values use the following mathematica snippet:
 kB = 8.61734*10^(-5);
 Tnu = 2.7255*(4/11.)^(1/3.)*1.00328;
 omegar = 5.04672*10^(-5);
 Hubble[a_] := 3.085678*10^(21)/10^5*3.24077929*10^(-18)*Sqrt[0.2793/a^3 + (1 - 0.2793) + omegar/a^4]
  fs[a_, Mnu_] := kB*Tnu/(a*Mnu)/(a*Hubble[a])
  fslength[ai_, af_, Mnu_] := 299792*NIntegrate[fs[Exp[loga], Mnu], {loga, Log[ai], Log[af]}]
 */
static void test_fslength(void **state)
{
    Cosmology CP;
    double MNu[3] = {0.15, 0.15, 0.15};
    setup_cosmology(&CP, MNu);
    /*Note that MNu is the mass of a single neutrino species:
     *we use large masses so that we don't have to compute omega_nu in mathematica.*/
    double kT = BOLEVK*TNUCMB*T_CMB0;
    /*fslength function returns fslength * (MNu / kT)*/
    assert_true(fabs(fslength(&CP, log(0.5), log(1), 299792.)/ 1272.92/(0.45/kT) -1 ) < 1e-5);
    double MNu2[3] = {0.2, 0.2, 0.2};
    setup_cosmology(&CP, MNu2);
    assert_true(fabs(fslength(&CP, log(0.1), log(0.5),299792.)/ 5427.8/(0.6/kT) -1 ) < 1e-5);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_allocate_delta_tot_table),
        cmocka_unit_test(test_specialJ),
        cmocka_unit_test(test_fslength),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
