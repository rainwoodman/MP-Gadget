#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"
#define NDEBUG
#ifdef PTHREADS_NUM_THREADS
#include <pthread.h>
#endif

#ifdef PTHREADS_NUM_THREADS
extern pthread_mutex_t mutex_nexport;
extern pthread_mutex_t mutex_partnodedrift;
#define LOCK_NEXPORT     pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT   pthread_mutex_unlock(&mutex_nexport);
#else
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#endif

/*! \file hydra_master.c
 *  \brief This contains the "second hydro loop", where the hydro fluxes are computed.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */


/* some very useful notes on the hydro variables in comoving integrations:
 
 v_code = a * v_peculiar/physical (canonical momentum)
 r_code = r_physical / a (comoving coordinates)
 m_code = m_physical
 rho_code = rho_physical * a^3 (from length/mass scaling)
 InternalEnergy_code = InternalEnergy_physical
 Pressure_code =
    InternalEnergy_code * rho_code * (gamma-1) = Pressure_physical * a^3 (energy SPH)
    -- the distinction between these cases and e.g. entropy sph (now depricated) 
        should be taken care of in the factors
        All.cf_afac1/2/3, which will correctly assign between the two --
 B_code = a*a * B_physical (comoving magnetic fields)
 Phi_code = B_code*v_code = a^3 * Phi_physical (damping field for Dedner divergence cleaning)
    (note: spec egy of phi field is: phi*phi/(2*mu0*rho*ch*ch); compare Bfield is B*B/(mu0*rho);
    so [phi]~[B]*[ch], where ch is the signal velocity used in the damping equation);

 -- Time derivatives (rate of change from hydro forces) here are all
        assumed to end up in *physical* units ---
 HydroAccel, dMomentum are assumed to end up in *physical* units
    (note, this is different from GADGET's convention, where 
     HydroAccel is in units of (Pcode/rhocode)/rcode)
 DtInternalEnergy and dInternalEnergy are assumed to end up in *physical* units
 DtMass and dMass are assumed to end up in *physical* units

 -----------------------------------------
 
 // All.cf_atime = a = 1/(1+z), the cosmological scale factor //
 All.cf_atime = All.Time;
 // All.cf_a2inv is just handy //
 All.cf_a2inv = 1 / (All.Time * All.Time);
 // All.cf_a3inv * Density_code = Density_physical //
 All.cf_a3inv = 1 / (All.Time * All.Time * All.Time);
 // Pressure_code/Density_code = All.cf_afac1 * Pressure_physical/Density_physical //
 All.cf_afac1 = 1;
 // All.cf_afac2 * Pressure_code/Density_code * 1/r_code = Pressure_physical/Density_physical * 1/r_physical //
 All.cf_afac2 = 1 / (All.Time * All.cf_afac1);
 // All.cf_afac3 * cs_code = All.cf_afac3 * sqrt(Pressure_code/Density_code) = sqrt(Pressure_phys/Density_phys) = cs_physical //
 All.cf_afac3 = 1 / sqrt(All.cf_afac1);
 // time units: proper time dt_phys = 1/hubble_function(a) * dz/(1+z) = dlna / hubble_function(a)
 code time unit in comoving is dlna, so dt_phys = dt_code / All.cf_hubble_a   //
 All.cf_hubble_a = hubble_function(All.Time); // hubble_function(a) = H(a) = H(z) //
 // dt_code * v_code/r_code = All.cf_hubble_a2 * dt_phys * v_phys/r_phys //
 All.cf_hubble_a2 = All.Time * All.Time * hubble_function(All.Time);
 
 
 -----------------------------------------
 A REMINDER ABOUT GIZMO/GADGET VELOCITY UNITS:: (direct quote from Volker)
 
 The IC file should contain the *peculiar* velocity divided by sqrt(a),
 not the *physical* velocity. Let "x" denote comoving
 coordinates and "r=a*x" physical coordinates. Then I call
 
 comoving velocity: dx/dt
 physical velocity: dr/dt = H(a)*r + a*dx/dt
 peculiar velocity: v = a * dx/dt
 
 The physical velocity is hence the peculiar velocity plus the Hubble flow.
 
 The internal velocity variable is not given by dx/d(ln a). Rather, it is given by
 the canonical momentum p = a^2 * dx/dt.
 The IC-file and snapshot files of gadget/GIZMO don't
 contain the variable "p" directly because of historical reasons.
 Instead, they contain the velocity variable
 u = v/sqrt(a) = sqrt(a) * dx/dt = p / a^(3/2), which is just what the
 manual says. (The conversion between u and p is done on the fly when
 reading or writing snapshot files.)
 
 Also note that d(ln a)/dt is equal to the
 Hubble rate, i.e.: d(ln a)/dt = H(a) = H_0 * sqrt(omega_m/a^3 + omega_v
 + (1 - omega_m - omega_v)/a^2).
 
 Best wishes,
 Volker
 
 -----------------------------------------
*/


static double fac_mu, fac_vsic_fix;

/* --------------------------------------------------------------------------------- */
/* define the kernel structure -- purely for handy purposes to clean up notation */
/* --------------------------------------------------------------------------------- */
/* structure to hold fluxes being passed from the hydro sub-routine */
struct Conserved_var_Riemann
{
    MyDouble rho;
    MyDouble p;
    MyDouble v[3];
    MyDouble u;
    MyDouble cs;
};


struct kernel_hydra
{
    double dp[3];
    double r, vsig, sound_i, sound_j;
    double dv[3], vdotr2;
    double wk_i, wk_j, dwk_i, dwk_j;
    double h_i, h_j, dwk_ij, rho_ij_inv;
    double spec_egy_u_i;
#ifdef HYDRO_SPH
    double p_over_rho2_i;
#endif
};
#ifndef HYDRO_SPH
#include "reimann.h"
#endif


/* --------------------------------------------------------------------------------- */
/* inputs to the routine: put here what's needed to do the calculation! */
/* --------------------------------------------------------------------------------- */
struct hydrodata_in
{
    /* basic hydro variables */
    MyDouble Pos[3];
    MyFloat Vel[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyFloat Density;
    MyFloat Pressure;
    MyFloat ConditionNumber;
    MyFloat InternalEnergyPred;
    MyFloat SoundSpeed;
    integertime Timestep;
    MyFloat DhsmlNgbFactor;
#ifdef HYDRO_SPH
    MyFloat DhsmlHydroSumFactor;
    MyFloat alpha;
#endif
    
    /* matrix of the conserved variable gradients: rho, u, vx, vy, vz */
    struct
    {
        MyDouble Density[3];
        MyDouble Pressure[3];
        MyDouble Velocity[3][3];
    } Gradients;
    MyFloat NV_T[3][3];
}
*HydroDataIn, *HydroDataGet;



/* --------------------------------------------------------------------------------- */
/* outputs: this is what the routine needs to return to the particles to set their final values */
/* --------------------------------------------------------------------------------- */
struct hydrodata_out
{
    MyLongDouble Acc[3];
    //MyLongDouble dMomentum[3]; //manifest-indiv-timestep-debug//
    MyLongDouble DtInternalEnergy;
    //MyLongDouble dInternalEnergy; //manifest-indiv-timestep-debug//
    MyFloat MaxSignalVel;
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    MyFloat MaxKineticEnergyNgb;
#endif
    
}
*HydroDataResult, *HydroDataOut;




/* --------------------------------------------------------------------------------- */
/* this subroutine actually loads the particle data into the structure to share between nodes */
/* --------------------------------------------------------------------------------- */
static inline void particle2in_hydra(struct hydrodata_in *in, int i);
static inline void out2particle_hydra(struct hydrodata_out *out, int i, int mode);
static inline void particle2in_hydra(struct hydrodata_in *in, int i)
{
    int k;
    for(k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = SphP[i].VelPred[k];
    }
    in->Hsml = PPP[i].Hsml;
    in->Mass = P[i].Mass;
    in->Density = SphP[i].Density;
    in->Pressure = SphP[i].Pressure;
    in->InternalEnergyPred = SphP[i].InternalEnergyPred;
    in->SoundSpeed = Particle_effective_soundspeed_i(i);
    in->Timestep = (P[i].TimeBin ? (((integertime) 1) << P[i].TimeBin) : 0);
    in->ConditionNumber = SphP[i].ConditionNumber;
    in->DhsmlNgbFactor = PPP[i].DhsmlNgbFactor;
#ifdef HYDRO_SPH
    in->DhsmlHydroSumFactor = SphP[i].DhsmlHydroSumFactor;
    in->alpha = SphP[i].alpha_limiter;
#endif
    
#ifdef HYDRO_PRESSURE_SPH
    in->EgyWtRho = SphP[i].EgyWtDensity;
#endif
    int j;
    for(j=0;j<3;j++) {for(k=0;k<3;k++) {in->NV_T[j][k] = SphP[i].NV_T[j][k];}}

    
    /* matrix of the conserved variable gradients: rho, u, vx, vy, vz */
    for(k=0;k<3;k++)
    {
        in->Gradients.Density[k] = SphP[i].Gradients.Density[k];
        in->Gradients.Pressure[k] = SphP[i].Gradients.Pressure[k];
        for(j=0;j<3;j++) {in->Gradients.Velocity[j][k] = SphP[i].Gradients.Velocity[j][k];}
    }

}



/* --------------------------------------------------------------------------------- */
/* this subroutine adds the output variables back to the particle values */
/* --------------------------------------------------------------------------------- */
static inline void out2particle_hydra(struct hydrodata_out *out, int i, int mode)
{
    int k;
    /* these are zero-d out at beginning of hydro loop so should always be added */
    for(k = 0; k < 3; k++)
    {
        SphP[i].HydroAccel[k] += out->Acc[k];
        //SphP[i].dMomentum[k] += out->dMomentum[k]; //manifest-indiv-timestep-debug//
    }
    SphP[i].DtInternalEnergy += out->DtInternalEnergy;
    //SphP[i].dInternalEnergy += out->dInternalEnergy; //manifest-indiv-timestep-debug//

    if(SphP[i].MaxSignalVel < out->MaxSignalVel)
        SphP[i].MaxSignalVel = out->MaxSignalVel;
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
    if(SphP[i].MaxKineticEnergyNgb < out->MaxKineticEnergyNgb)
        SphP[i].MaxKineticEnergyNgb = out->MaxKineticEnergyNgb;
#endif
}


/* --------------------------------------------------------------------------------- */
/* need to link to the file "hydra_evaluate" which actually contains the computation part of the loop! */
/* --------------------------------------------------------------------------------- */
#include "hydra_evaluate.h"

/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/* This will perform final operations and corrections on the output from the 
    hydro routines, AFTER the neighbors have all been checked and summed */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
void hydro_final_operations_and_cleanup(void)
{
    int i,k;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type == 0 && P[i].Mass > 0)
        {
            double dt;
            dt = (P[i].TimeBin ? (((integertime) 1) << P[i].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
            
            /* we calculated the flux of conserved variables: these are used in the kick operation. But for
             intermediate drift operations, we need the primive variables, so reduce to those here 
             (remembering that v_phys = v_code/All.cf_atime, for the sake of doing the unit conversions to physical) */
            for(k=0;k<3;k++)
            {
                SphP[i].DtInternalEnergy -= (SphP[i].VelPred[k]/All.cf_atime) * SphP[i].HydroAccel[k];
                /* we solved for total energy flux (and remember, HydroAccel is still momentum -- keep units straight here!) */
                SphP[i].HydroAccel[k] /= P[i].Mass; /* we solved for momentum flux */
            }
            
            SphP[i].DtInternalEnergy /= P[i].Mass;
            /* ok, now: HydroAccel = dv/dt, DtInternalEnergy = du/dt (energy per unit mass) */
            
            /* zero out hydrodynamic PdV work terms if the particle is at the maximum smoothing, these will be incorrect */
            if(PPP[i].Hsml >= 0.99*All.MaxHsml) {SphP[i].DtInternalEnergy = 0;}
            
            // need to explicitly include adiabatic correction from the hubble-flow (for drifting) here //
            if(All.ComovingIntegrationOn) SphP[i].DtInternalEnergy -= 3*GAMMA_MINUS1 * SphP[i].InternalEnergyPred * All.cf_hubble_a;
            // = du/dlna -3*(gamma-1)*u ; then dlna/dt = H(z) =  All.cf_hubble_a //
            
        } // closes P[i].Type==0 check and so closes loop over particles i
    } // for (loop over active particles) //
    
    
    
}




/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/*! This function is the driver routine for the calculation of hydrodynamical
 *  force, fluxes, etc. */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
void hydro_force(void)
{
    int i, j, k, ngrp, ndone, ndone_flag;
    int recvTask, place;
    double timeall=0, timecomp1=0, timecomp2=0, timecommsumm1=0, timecommsumm2=0, timewait1=0, timewait2=0, timenetwork=0;
    double timecomp, timecomm, timewait, tstart, tend, t0, t1;
    int save_NextParticle;
    long long n_exported = 0;
    /* need to zero out all numbers that can be set -EITHER- by an active particle in the domain, or by one of the neighbors we will get sent */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        if(P[i].Type==0)
        {
            SphP[i].MaxSignalVel = -1.e10;
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
            SphP[i].MaxKineticEnergyNgb = -1.e10;
#endif
            SphP[i].DtInternalEnergy = 0;//SphP[i].dInternalEnergy = 0;//manifest-indiv-timestep-debug//
            for(k=0;k<3;k++)
            {
                SphP[i].HydroAccel[k] = 0;//SphP[i].dMomentum[k] = 0;//manifest-indiv-timestep-debug//
            }
        }
    
    /* --------------------------------------------------------------------------------- */
    // Global factors for comoving integration of hydro //
    fac_mu = 1 / (All.cf_afac3 * All.cf_atime);
    // code_vel * fac_mu = sqrt[code_pressure/code_density] = code_soundspeed //
    // note also that signal_vel in forms below should be in units of code_soundspeed //
    fac_vsic_fix = All.cf_hubble_a * All.cf_afac1;
    /* --------------------------------------------------------------------------------- */
    /* allocate buffers to arrange communication */
    long long NTaskTimesNumPart;
    NTaskTimesNumPart = maxThreads * NumPart;
    Ngblist = (int *) mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    size_t MyBufferSize = All.BufferSize;
    All.BunchSize = (int) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                                             sizeof(struct hydrodata_in) +
                                                             sizeof(struct hydrodata_out) +
                                                             sizemax(sizeof(struct hydrodata_in),sizeof(struct hydrodata_out))));
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    CPU_Step[CPU_HYDMISC] += measure_time();
    t0 = my_second();
    NextParticle = FirstActiveParticle;	/* begin with this index */
    
    do
    {
        BufferFullFlag = 0;
        Nexport = 0;
        save_NextParticle = NextParticle;
        for(j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }
        /* do local particles and prepare export list */
        tstart = my_second();
        
#ifdef PTHREADS_NUM_THREADS
        pthread_t mythreads[PTHREADS_NUM_THREADS - 1];
        int threadid[PTHREADS_NUM_THREADS - 1];
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_mutex_init(&mutex_nexport, NULL);
        pthread_mutex_init(&mutex_partnodedrift, NULL);
        TimerFlag = 0;
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
        {
            threadid[j] = j + 1;
            pthread_create(&mythreads[j], &attr, hydro_evaluate_primary, &threadid[j]);
        }
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
#else
            int mainthreadid = 0;
#endif
            hydro_evaluate_primary(&mainthreadid);	/* do local particles and prepare export list */
        }
        
#ifdef PTHREADS_NUM_THREADS
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            pthread_join(mythreads[j], NULL);
#endif
        tend = my_second();
        timecomp1 += timediff(tstart, tend);
        
        if(BufferFullFlag)
        {
            int last_nextparticle = NextParticle;
            NextParticle = save_NextParticle;
            while(NextParticle >= 0)
            {
                if(NextParticle == last_nextparticle)
                    break;
                
                if(ProcessedFlag[NextParticle] != 1)
                    break;
                
                ProcessedFlag[NextParticle] = 2;
                NextParticle = NextActiveParticle[NextParticle];
            }
            if(NextParticle == save_NextParticle)
            {
                /* in this case, the buffer is too small to process even a single particle */
                endrun(115508);
            }
            int new_export = 0;
            for(j = 0, k = 0; j < Nexport; j++)
                if(ProcessedFlag[DataIndexTable[j].Index] != 2)
                {
                    if(k < j + 1)
                        k = j + 1;
                    
                    for(; k < Nexport; k++)
                        if(ProcessedFlag[DataIndexTable[k].Index] == 2)
                        {
                            int old_index = DataIndexTable[j].Index;
                            
                            DataIndexTable[j] = DataIndexTable[k];
                            DataNodeList[j] = DataNodeList[k];
                            DataIndexTable[j].IndexGet = j;
                            new_export++;
                            
                            DataIndexTable[k].Index = old_index;
                            k++;
                            break;
                        }
                }
                else
                    new_export++;
            
            Nexport = new_export;
        }
        
        n_exported += Nexport;
        for(j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Recv_count[j] = 0;
        }
        for(j = 0; j < Nexport; j++)
            Send_count[DataIndexTable[j].Task]++;
        
        MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
        tstart = my_second();
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
        tend = my_second();
        timewait1 += timediff(tstart, tend);
        
        for(j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
        {
            Nimport += Recv_count[j];
            if(j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }
        HydroDataGet = (struct hydrodata_in *) mymalloc("HydroDataGet", Nimport * sizeof(struct hydrodata_in));
        HydroDataIn = (struct hydrodata_in *) mymalloc("HydroDataIn", Nexport * sizeof(struct hydrodata_in));
        
        /* prepare particle data for export */
        for(j = 0; j < Nexport; j++)
        {
            place = DataIndexTable[j].Index;
            particle2in_hydra(&HydroDataIn[j], place);		// MADE D_IND CHANGE IN HERE
        }
        
        /* exchange particle data */
        tstart = my_second();
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* get the particles */
                    MPI_Sendrecv(&HydroDataIn[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct hydrodata_in), MPI_BYTE,
                                 recvTask, TAG_HYDRO_A,
                                 &HydroDataGet[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct hydrodata_in), MPI_BYTE,
                                 recvTask, TAG_HYDRO_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        tend = my_second();
        timecommsumm1 += timediff(tstart, tend);
        
        myfree(HydroDataIn);
        HydroDataResult = (struct hydrodata_out *) mymalloc("HydroDataResult", Nimport * sizeof(struct hydrodata_out));
        HydroDataOut = (struct hydrodata_out *) mymalloc("HydroDataOut", Nexport * sizeof(struct hydrodata_out));
        report_memory_usage(&HighMark_sphhydro, "SPH_HYDRO");
        
        /* now do the particles that were sent to us */
        tstart = my_second();
        NextJ = 0;
#ifdef PTHREADS_NUM_THREADS
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            pthread_create(&mythreads[j], &attr, hydro_evaluate_secondary, &threadid[j]);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
#else
            int mainthreadid = 0;
#endif
            hydro_evaluate_secondary(&mainthreadid);
        }
        
#ifdef PTHREADS_NUM_THREADS
        for(j = 0; j < PTHREADS_NUM_THREADS - 1; j++)
            pthread_join(mythreads[j], NULL);
        
        pthread_mutex_destroy(&mutex_partnodedrift);
        pthread_mutex_destroy(&mutex_nexport);
        pthread_attr_destroy(&attr);
#endif
        tend = my_second();
        timecomp2 += timediff(tstart, tend);
        
        if(NextParticle < 0)
            ndone_flag = 1;
        else
            ndone_flag = 0;
        
        tstart = my_second();
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        tend = my_second();
        timewait2 += timediff(tstart, tend);
        
        /* get the result */
        tstart = my_second();
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* send the results */
                    MPI_Sendrecv(&HydroDataResult[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct hydrodata_out),
                                 MPI_BYTE, recvTask, TAG_HYDRO_B,
                                 &HydroDataOut[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct hydrodata_out),
                                 MPI_BYTE, recvTask, TAG_HYDRO_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        tend = my_second();
        timecommsumm2 += timediff(tstart, tend);
        
        /* add the result to the local particles */
        tstart = my_second();
        for(j = 0; j < Nexport; j++)
        {
            place = DataIndexTable[j].Index;
            out2particle_hydra(&HydroDataOut[j], place, 1);
        }
        tend = my_second();
        timecomp1 += timediff(tstart, tend);
        
        myfree(HydroDataOut);
        myfree(HydroDataResult);
        myfree(HydroDataGet);
    }
    while(ndone < NTask);
    
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);
    
    
    /* --------------------------------------------------------------------------------- */
    /* do final operations on results */
    /* --------------------------------------------------------------------------------- */
    hydro_final_operations_and_cleanup();
    
    
    /* --------------------------------------------------------------------------------- */
    /* collect some timing information */
    t1 = WallclockTime = my_second();
    timeall += timediff(t0, t1);
    timecomp = timecomp1 + timecomp2;
    timewait = timewait1 + timewait2;
    timecomm = timecommsumm1 + timecommsumm2;
    CPU_Step[CPU_HYDCOMPUTE] += timecomp;
    CPU_Step[CPU_HYDWAIT] += timewait;
    CPU_Step[CPU_HYDCOMM] += timecomm;
    CPU_Step[CPU_HYDNETWORK] += timenetwork;
    CPU_Step[CPU_HYDMISC] += timeall - (timecomp + timewait + timecomm + timenetwork);
}



/* --------------------------------------------------------------------------------- */
/* one of the core sub-routines used to do the MPI version of the hydro evaluation
 (don't put actual operations here!!!) */
/* --------------------------------------------------------------------------------- */
void *hydro_evaluate_primary(void *p)
{
#define CONDITION_FOR_EVALUATION if((P[i].Type==0)&&(P[i].Mass>0))
#define EVALUATION_CALL hydro_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist)
#include "../system/code_block_primary_loop_evaluation.h"
#undef CONDITION_FOR_EVALUATION
#undef EVALUATION_CALL
}
void *hydro_evaluate_secondary(void *p)
{
#define EVALUATION_CALL hydro_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist);
#include "../system/code_block_secondary_loop_evaluation.h"
#undef EVALUATION_CALL
}

