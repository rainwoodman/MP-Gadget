/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/*! This function is the 'core' of the hydro force computation. A target
*  particle is specified which may either be local, or reside in the
*  communication buffer.
*   In this routine, we find the gas particle neighbors, and do the loop over 
*  neighbors to calculate the hydro fluxes. The actual flux calculation, 
*  and the returned values, should be in PHYSICAL (not comoving) units */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
int hydro_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
    int j, k, n, startnode, numngb, kernel_mode, listindex;
    double hinv_i,hinv3_i,hinv4_i,hinv_j,hinv3_j,hinv4_j,V_i,V_j,dt_hydrostep,r2,rinv,rinv_soft,u,Particle_Size_i;
    double v_hll,k_hll,b_hll; v_hll=k_hll=0,b_hll=1;
    struct kernel_hydra kernel;
    struct hydrodata_in local;
    struct hydrodata_out out;
    struct Conserved_var_Riemann Fluxes;
    listindex = 0;
    memset(&out, 0, sizeof(struct hydrodata_out));
    memset(&kernel, 0, sizeof(struct kernel_hydra));
    memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
#ifndef HYDRO_SPH
    struct Input_vec_Riemann Riemann_vec;
    struct Riemann_outputs Riemann_out;
    double face_area_dot_vel;
    face_area_dot_vel = 0;
#endif
    double face_vel_i=0, face_vel_j=0, Face_Area_Norm=0, Face_Area_Vec[3];

#ifdef HYDRO_MESHLESS_FINITE_MASS
    double epsilon_entropic_eos_big = 0.5; // can be anything from (small number=more diffusive, less accurate entropy conservation) to ~1.1-1.3 (least diffusive, most noisy)
    double epsilon_entropic_eos_small = 1.e-3; // should be << epsilon_entropic_eos_big
    epsilon_entropic_eos_small = 1.e-2; epsilon_entropic_eos_big = 0.6; // with gravity larger tolerance behaves better on hydrostatic equilibrium problems //
#endif

    if(mode == 0)
    {
        particle2in_hydra(&local, target); // this setup allows for all the fields we need to define (don't hard-code here)
    }
    else
    {
        local = HydroDataGet[target]; // this setup allows for all the fields we need to define (don't hard-code here)
    }
    
    /* certain particles should never enter the loop: check for these */
    if(local.Mass <= 0) return 0;
    if(local.DelayTime > 0) {return 0;}
    
    /* --------------------------------------------------------------------------------- */
    /* pre-define Particle-i based variables (so we save time in the loop below) */
    /* --------------------------------------------------------------------------------- */
    kernel.sound_i = local.SoundSpeed;
    kernel.spec_egy_u_i = local.InternalEnergyPred;
    kernel.h_i = local.Hsml;
    kernel_hinv(kernel.h_i, &hinv_i, &hinv3_i, &hinv4_i);
    hinv_j=hinv3_j=hinv4_j=0;
    V_i = local.Mass / local.Density;
    Particle_Size_i = pow(V_i,1./NUMDIMS) * All.cf_atime; // in physical, used below in some routines //
    double Amax_i = MAX_REAL_NUMBER;
#if (NUMDIMS==2)
    Amax_i = 2. * sqrt(V_i/M_PI);
#endif
#if (NUMDIMS==3)
    Amax_i = M_PI * pow((3.*V_i)/(4.*M_PI), 2./3.);
#endif    
    dt_hydrostep = local.Timestep * All.Timebase_interval / All.cf_hubble_a; /* (physical) timestep */
    out.MaxSignalVel = kernel.sound_i;
    kernel_mode = 0; /* need dwk and wk */
    double cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - local.ConditionNumber*local.ConditionNumber;
#if defined(HYDRO_SPH)
#ifdef HYDRO_PRESSURE_SPH
    kernel.p_over_rho2_i = local.Pressure / (local.EgyWtRho*local.EgyWtRho);
#else 
    kernel.p_over_rho2_i = local.Pressure / (local.Density*local.Density);
#endif
#endif
    
    /* --------------------------------------------------------------------------------- */
    /* Now start the actual SPH computation for this particle */
    /* --------------------------------------------------------------------------------- */
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = HydroDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }
    
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            /* --------------------------------------------------------------------------------- */
            /* get the neighbor list */
            /* --------------------------------------------------------------------------------- */
            numngb = ngb_treefind_pairs_threads(local.Pos, kernel.h_i, target, &startnode, mode, exportflag,
                                       exportnodecount, exportindex, ngblist);
            if(numngb < 0) return -1;
            
            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n];
                
                /* check if I need to compute this pair-wise interaction from "i" to "j", or skip it and 
                    let it be computed from "j" to "i" */
                integertime TimeStep_J = (P[j].TimeBin ? (((integertime) 1) << P[j].TimeBin) : 0);
                int j_is_active_for_fluxes = 0;
                if(local.Timestep > TimeStep_J) continue; /* compute from particle with smaller timestep */
                /* use relative positions to break degeneracy */
                if(local.Timestep == TimeStep_J)
                {
                    int n0=0; if(local.Pos[n0] == P[j].Pos[n0]) {n0++; if(local.Pos[n0] == P[j].Pos[n0]) n0++;}
                    if(local.Pos[n0] < P[j].Pos[n0]) continue;
                }
                if(TimeBinActive[P[j].TimeBin]) {j_is_active_for_fluxes = 1;}
                if(P[j].Mass <= 0) continue;
                if(SphP[j].Density <= 0) continue;
                if(SphP[j].DelayTime > 0) continue; /* no hydro forces for decoupled wind particles */
                kernel.dp[0] = local.Pos[0] - P[j].Pos[0];
                kernel.dp[1] = local.Pos[1] - P[j].Pos[1];
                kernel.dp[2] = local.Pos[2] - P[j].Pos[2];
 /* find the closest image in the given box size  */
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1);
                r2 = kernel.dp[0] * kernel.dp[0] + kernel.dp[1] * kernel.dp[1] + kernel.dp[2] * kernel.dp[2];
                kernel.h_j = PPP[j].Hsml;
                
                /* force applied for all particles inside each-others kernels! */
                if((r2 >= kernel.h_i * kernel.h_i) && (r2 >= kernel.h_j * kernel.h_j)) continue;
                if(r2 <= 0) continue;
                
                /* --------------------------------------------------------------------------------- */
                /* ok, now we definitely have two interacting particles */
                /* --------------------------------------------------------------------------------- */
                
                /* --------------------------------------------------------------------------------- */
                /* calculate a couple basic properties needed: separation, velocity difference (needed for timestepping) */
                kernel.r = sqrt(r2);
                rinv = 1 / kernel.r;
                /* we require a 'softener' to prevent numerical madness in interpolating functions */
                rinv_soft = 1.0 / sqrt(r2 + 0.0001*kernel.h_i*kernel.h_i);
                /* faster to just set a pointer directly */
                MyDouble *VelPred_j = SphP[j].VelPred;
                kernel.dv[0] = local.Vel[0] - VelPred_j[0];
                kernel.dv[1] = local.Vel[1] - VelPred_j[1];
                kernel.dv[2] = local.Vel[2] - VelPred_j[2];
                kernel.rho_ij_inv = 2.0 / (local.Density + SphP[j].Density);
                
                /* --------------------------------------------------------------------------------- */
                /* sound speed, relative velocity, and signal velocity computation */
                kernel.sound_j = Particle_effective_soundspeed_i(j);
                kernel.vsig = kernel.sound_i + kernel.sound_j;
                kernel.vdotr2 = kernel.dp[0] * kernel.dv[0] + kernel.dp[1] * kernel.dv[1] + kernel.dp[2] * kernel.dv[2];
                // hubble-flow correction: need in -code- units, hence extra a2 appearing here //
                if(All.ComovingIntegrationOn) kernel.vdotr2 += All.cf_hubble_a2 * r2;
                if(kernel.vdotr2 < 0)
                {
#if defined(HYDRO_SPH)
                    kernel.vsig -= 3 * fac_mu * kernel.vdotr2 * rinv;
#else
                    kernel.vsig -= fac_mu * kernel.vdotr2 * rinv;
#endif
                }
                /* --------------------------------------------------------------------------------- */
                /* calculate the kernel functions (centered on both 'i' and 'j') */
                if(kernel.r < kernel.h_i)
                {
                    u = kernel.r * hinv_i;
                    kernel_main(u, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, kernel_mode);
                }
                else
                {
                    kernel.dwk_i = 0;
                    kernel.wk_i = 0;
                }
                if(kernel.r < kernel.h_j)
                {
                    kernel_hinv(kernel.h_j, &hinv_j, &hinv3_j, &hinv4_j);
                    u = kernel.r * hinv_j;
                    kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, kernel_mode);
                }
                else
                {
                    kernel.dwk_j = 0;
                    kernel.wk_j = 0;
                }
                
                /* --------------------------------------------------------------------------------- */
                /* with the overhead numbers above calculated, we now 'feed into' the "core" 
                    hydro computation (SPH, meshless godunov, etc -- doesn't matter, should all take the same inputs) 
                    the core code is -inserted- here from the appropriate .h file, depending on the mode 
                    the code has been compiled in */
                /* --------------------------------------------------------------------------------- */
#ifdef HYDRO_SPH
#include "hydra_core_sph.h"
#else
#include "hydra_core_meshless.h"
#endif
                
#ifdef HYDRO_SPH
        face_vel_i = face_vel_j = 0;
        for(k=0;k<3;k++) 
        {
        face_vel_i += local.Vel[k] * kernel.dp[k] / (kernel.r * All.cf_atime); 
        face_vel_j += SphP[j].VelPred[k] * kernel.dp[k] / (kernel.r * All.cf_atime);
        }
        // SPH: use the sph 'effective areas' oriented along the lines between particles and direct-difference gradients
        Face_Area_Norm = local.Mass * P[j].Mass * fabs(kernel.dwk_i+kernel.dwk_j) / (local.Density * SphP[j].Density);
        for(k=0;k<3;k++) {Face_Area_Vec[k] = Face_Area_Norm * kernel.dp[k]/kernel.r;}
#endif
                v_hll = 0.5*fabs(face_vel_i-face_vel_j) + DMAX(kernel.sound_i,kernel.sound_j);
                /* --------------------------------------------------------------------------------- */
                /* now we will actually assign the hydro variables for the evolution step */
                /* --------------------------------------------------------------------------------- */
                for(k=0;k<3;k++) {out.Acc[k] += Fluxes.v[k];}
                out.DtInternalEnergy += Fluxes.p;                
                
                /* if this is particle j's active timestep, you should sent them the time-derivative information as well, for their subsequent drift operations */
                if(j_is_active_for_fluxes)
                {
                    for(k=0;k<3;k++) {SphP[j].HydroAccel[k] -= Fluxes.v[k];}
                    SphP[j].DtInternalEnergy -= Fluxes.p;
                }

                /* if we have mass fluxes, we need to have metal fluxes if we're using them (or any other passive scalars) */

                /* --------------------------------------------------------------------------------- */
                /* don't forget to save the signal velocity for time-stepping! */
                /* --------------------------------------------------------------------------------- */
                if(kernel.vsig > out.MaxSignalVel) out.MaxSignalVel = kernel.vsig;
                if(j_is_active_for_fluxes) {if(kernel.vsig > SphP[j].MaxSignalVel) SphP[j].MaxSignalVel = kernel.vsig;}
                
            } // for(n = 0; n < numngb; n++) //
        } // while(startnode >= 0) //
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = HydroDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        } // if(mode == 1) //
    } // while(startnode >= 0) //
    
    /* Now collect the result at the right place */
    if(mode == 0)
        out2particle_hydra(&out, target, 0);
    else
        HydroDataResult[target] = out;
    
    return 0;
}

