#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "timestep.h"
#include "physconst.h"
#include "cooling.h"
#include "slotsmanager.h"
#include "hydra.h"
#include "utils.h"

/* global state of system
*/
struct state_of_system
{
    double Mass;
    double EnergyKin;
    double EnergyPot;
    double EnergyInt;
    double EnergyTot;

    double Momentum[4];
    double AngMomentum[4];
    double CenterOfMass[4];
    double MassComp[6];
    /* Only Gas is used */
    double TemperatureComp[6];

    double EnergyKinComp[6];
    double EnergyPotComp[6];
    double EnergyIntComp[6];
    double EnergyTotComp[6];
    double MomentumComp[6][4];
    double AngMomentumComp[6][4];
    double CenterOfMassComp[6][4];
};

/* This routine computes various global properties of the particle
 * distribution and stores the result in the struct `SysState'.
 * Currently, not all the information that's computed here is
 * actually used (e.g. momentum is not really used anywhere),
 * just the energies are written to a log-file every once in a while.
 */
struct state_of_system compute_global_quantities_of_system(const double Time,  struct part_manager_type * PartManager)
{
    int i, j;
    struct state_of_system sys;
    struct state_of_system SysState;
    double a1, a2, a3;
    int ThisTask;

    a1 = Time;
    a2 = Time * Time;
    a3 = Time * Time * Time;

    double redshift = 1. / Time - 1;
    memset(&sys, 0, sizeof(sys));
    struct UVBG GlobalUVBG = get_global_UVBG(redshift);

    #pragma omp parallel for
    for(i = 0; i < PartManager->NumPart; i++)
    {
        int j;
        double entr = 0, egyspec;

        sys.MassComp[P[i].Type] += P[i].Mass;

        sys.EnergyPotComp[P[i].Type] += 0.5 * P[i].Mass * P[i].Potential / a1;

        sys.EnergyKinComp[P[i].Type] +=
            0.5 * P[i].Mass * (P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2]) / a2;

        if(P[i].Type == 0)
        {
            struct UVBG uvbg = get_local_UVBG(redshift, &GlobalUVBG, P[i].Pos, PartManager->CurrentParticleOffset, P[i].HeIIIionized);
            entr = SPHP(i).Entropy;
            egyspec = entr / (GAMMA_MINUS1) * pow(SPH_EOMDensity(&SPHP(i)) / a3, GAMMA_MINUS1);
            sys.EnergyIntComp[0] += P[i].Mass * egyspec;
            double ne = SPHP(i).Ne;
            sys.TemperatureComp[0] += P[i].Mass * get_temp(SPH_EOMDensity(&SPHP(i)), egyspec, (1 - HYDROGEN_MASSFRAC), &uvbg, &ne);
        }

        for(j = 0; j < 3; j++)
        {
            sys.MomentumComp[P[i].Type][j] += P[i].Mass * P[i].Vel[j];
            sys.CenterOfMassComp[P[i].Type][j] += P[i].Mass * P[i].Pos[j];
        }

        sys.AngMomentumComp[P[i].Type][0] += P[i].Mass * (P[i].Pos[1] * P[i].Vel[2] - P[i].Pos[2] * P[i].Vel[1]);
        sys.AngMomentumComp[P[i].Type][1] += P[i].Mass * (P[i].Pos[2] * P[i].Vel[0] - P[i].Pos[0] * P[i].Vel[2]);
        sys.AngMomentumComp[P[i].Type][2] += P[i].Mass * (P[i].Pos[0] * P[i].Vel[1] - P[i].Pos[1] * P[i].Vel[0]);
    }


    /* some the stuff over all processors */
    MPI_Reduce(&sys.MassComp[0], &SysState.MassComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyPotComp[0], &SysState.EnergyPotComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyIntComp[0], &SysState.EnergyIntComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyKinComp[0], &SysState.EnergyKinComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.TemperatureComp[0], &SysState.TemperatureComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sys.MomentumComp[0][0], &SysState.MomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&sys.AngMomentumComp[0][0], &SysState.AngMomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&sys.CenterOfMassComp[0][0], &SysState.CenterOfMassComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);

    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    if(ThisTask == 0)
    {
        for(i = 0; i < 6; i++) {
            SysState.EnergyTotComp[i] = SysState.EnergyKinComp[i] +
                SysState.EnergyPotComp[i] + SysState.EnergyIntComp[i];
        }

        SysState.Mass = SysState.EnergyKin = SysState.EnergyPot = SysState.EnergyInt = SysState.EnergyTot = 0;

        for(i = 0; i < 6; i++)
        {
            if(SysState.MassComp[i] > 0) {
                SysState.TemperatureComp[i] /= SysState.MassComp[i];
            }
        }

        for(j = 0; j < 3; j++)
            SysState.Momentum[j] = SysState.AngMomentum[j] = SysState.CenterOfMass[j] = 0;

        for(i = 0; i < 6; i++)
        {
            SysState.Mass += SysState.MassComp[i];
            SysState.EnergyKin += SysState.EnergyKinComp[i];
            SysState.EnergyPot += SysState.EnergyPotComp[i];
            SysState.EnergyInt += SysState.EnergyIntComp[i];
            SysState.EnergyTot += SysState.EnergyTotComp[i];

            for(j = 0; j < 3; j++)
            {
                SysState.Momentum[j] += SysState.MomentumComp[i][j];
                SysState.AngMomentum[j] += SysState.AngMomentumComp[i][j];
                SysState.CenterOfMass[j] += SysState.CenterOfMassComp[i][j];
            }
        }

        for(i = 0; i < 6; i++)
            for(j = 0; j < 3; j++)
                if(SysState.MassComp[i] > 0)
                    SysState.CenterOfMassComp[i][j] /= SysState.MassComp[i];

        for(j = 0; j < 3; j++)
            if(SysState.Mass > 0)
                SysState.CenterOfMass[j] /= SysState.Mass;

        for(i = 0; i < 6; i++)
        {
            SysState.CenterOfMassComp[i][3] = SysState.MomentumComp[i][3] = SysState.AngMomentumComp[i][3] = 0;
            for(j = 0; j < 3; j++)
            {
                SysState.CenterOfMassComp[i][3] +=
                    SysState.CenterOfMassComp[i][j] * SysState.CenterOfMassComp[i][j];
                SysState.MomentumComp[i][3] += SysState.MomentumComp[i][j] * SysState.MomentumComp[i][j];
                SysState.AngMomentumComp[i][3] +=
                    SysState.AngMomentumComp[i][j] * SysState.AngMomentumComp[i][j];
            }
            SysState.CenterOfMassComp[i][3] = sqrt(SysState.CenterOfMassComp[i][3]);
            SysState.MomentumComp[i][3] = sqrt(SysState.MomentumComp[i][3]);
            SysState.AngMomentumComp[i][3] = sqrt(SysState.AngMomentumComp[i][3]);
        }

        SysState.CenterOfMass[3] = SysState.Momentum[3] = SysState.AngMomentum[3] = 0;

        for(j = 0; j < 3; j++)
        {
            SysState.CenterOfMass[3] += SysState.CenterOfMass[j] * SysState.CenterOfMass[j];
            SysState.Momentum[3] += SysState.Momentum[j] * SysState.Momentum[j];
            SysState.AngMomentum[3] += SysState.AngMomentum[j] * SysState.AngMomentum[j];
        }

        SysState.CenterOfMass[3] = sqrt(SysState.CenterOfMass[3]);
        SysState.Momentum[3] = sqrt(SysState.Momentum[3]);
        SysState.AngMomentum[3] = sqrt(SysState.AngMomentum[3]);
    }

    /* give everyone the result, maybe the want to do something with it */
    MPI_Bcast(&SysState, sizeof(struct state_of_system), MPI_BYTE, 0, MPI_COMM_WORLD);
    return SysState;
}

/*! This routine first calls a computation of various global
 * quantities of the particle distribution, and then writes some
 * statistics about the energies in the various particle components to
 * the file FdEnergy.
 */
void energy_statistics(FILE * FdEnergy, const double Time, struct part_manager_type * PartManager)
{
    struct state_of_system SysState = compute_global_quantities_of_system(Time, PartManager);

    message(0, "Time %g Mean Temperature of Gas %g\n",
                Time, SysState.TemperatureComp[0]);

    if(FdEnergy)
    {
        fprintf(FdEnergy,
                "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
                Time, SysState.TemperatureComp[0], SysState.EnergyInt, SysState.EnergyPot, SysState.EnergyKin, SysState.EnergyIntComp[0],
                SysState.EnergyPotComp[0], SysState.EnergyKinComp[0], SysState.EnergyIntComp[1],
                SysState.EnergyPotComp[1], SysState.EnergyKinComp[1], SysState.EnergyIntComp[2],
                SysState.EnergyPotComp[2], SysState.EnergyKinComp[2], SysState.EnergyIntComp[3],
                SysState.EnergyPotComp[3], SysState.EnergyKinComp[3], SysState.EnergyIntComp[4],
                SysState.EnergyPotComp[4], SysState.EnergyKinComp[4], SysState.EnergyIntComp[5],
                SysState.EnergyPotComp[5], SysState.EnergyKinComp[5], SysState.MassComp[0],
                SysState.MassComp[1], SysState.MassComp[2], SysState.MassComp[3], SysState.MassComp[4],
                SysState.MassComp[5]);

        fflush(FdEnergy);
    }
}
