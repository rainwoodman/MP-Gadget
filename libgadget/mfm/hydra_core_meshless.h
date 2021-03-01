/* --------------------------------------------------------------------------------- */
/* this is the sub-routine where we actually extrapolate quantities to the faces 
    and set up, then solve the pair-wise Riemann problem for the method */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
{
    double s_star_ij,s_i,s_j,v_frame[3],n_unit[3],dummy_pressure;
    double distance_from_i[3],distance_from_j[3];
    dummy_pressure=face_area_dot_vel=face_vel_i=face_vel_j=Face_Area_Norm=0;
    double Pressure_i = local.Pressure, Pressure_j = SphP[j].Pressure;
    
    /* --------------------------------------------------------------------------------- */
    /* define volume elements and interface position */
    /* --------------------------------------------------------------------------------- */
    V_j = P[j].Mass / SphP[j].Density;
    s_star_ij = 0;
    //
    /* ------------------------------------------------------------------------------------------------------------------- */
    /* now we're ready to compute the volume integral of the fluxes (or equivalently an 'effective area'/face orientation) */
    /* ------------------------------------------------------------------------------------------------------------------- */
    double wt_i,wt_j;
    wt_i=V_i; wt_j=V_j;
    //wt_i=wt_j = 2.*V_i*V_j / (V_i + V_j); // more conservatively, could use DMIN(V_i,V_j), but that is less accurate
    if((fabs(V_i-V_j)/DMIN(V_i,V_j))/NUMDIMS > 1.25) {wt_i=wt_j=2.*V_i*V_j/(V_i+V_j);} else {wt_i=V_i; wt_j=V_j;}
    /* the effective gradient matrix is well-conditioned: we can safely use the consistent EOM */
    // note the 'default' formulation from Lanson and Vila takes wt_i=V_i, wt_j=V_j; but this assumes negligible variation in h between particles;
    //      it is more accurate to use a centered wt (centered face area), which we get by linear interpolation //
    double facenormal_dot_dp = 0;
    for(k=0;k<3;k++)
    {
        Face_Area_Vec[k] = kernel.wk_i * wt_i * (local.NV_T[k][0]*kernel.dp[0] + local.NV_T[k][1]*kernel.dp[1] + local.NV_T[k][2]*kernel.dp[2])
                         + kernel.wk_j * wt_j * (SphP[j].NV_T[k][0]*kernel.dp[0] + SphP[j].NV_T[k][1]*kernel.dp[1] + SphP[j].NV_T[k][2]*kernel.dp[2]);
        Face_Area_Vec[k] *= All.cf_atime*All.cf_atime; /* Face_Area_Norm has units of area, need to convert to physical */
        Face_Area_Norm += Face_Area_Vec[k]*Face_Area_Vec[k];
        facenormal_dot_dp += Face_Area_Vec[k] * kernel.dp[k]; /* check that face points same direction as vector normal: should be true for positive-definite (well-conditioned) NV_T */
    }
    
    if((SphP[j].ConditionNumber*SphP[j].ConditionNumber > 1.0e12 + cnumcrit2) || (facenormal_dot_dp < 0))
    {
        /* the effective gradient matrix is ill-conditioned (or not positive-definite!): for stability, we revert to the "RSPH" EOM */
        Face_Area_Norm = -(wt_i*V_i*kernel.dwk_i + wt_j*V_j*kernel.dwk_j) / kernel.r;
        Face_Area_Norm *= All.cf_atime*All.cf_atime; /* Face_Area_Norm has units of area, need to convert to physical */
        Face_Area_Vec[0] = Face_Area_Norm * kernel.dp[0];
        Face_Area_Vec[1] = Face_Area_Norm * kernel.dp[1];
        Face_Area_Vec[2] = Face_Area_Norm * kernel.dp[2];
        Face_Area_Norm = Face_Area_Norm * Face_Area_Norm * r2;
    }
    if(Face_Area_Norm == 0)
    {
        memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
    } else {
        
        if((Face_Area_Norm<=0)||(isnan(Face_Area_Norm)))
        {
            printf("PANIC! Face_Area_Norm=%g Mij=%g/%g wk_ij=%g/%g Vij=%g/%g dx/dy/dz=%g/%g/%g NVT=%g/%g/%g NVT_j=%g/%g/%g \n",Face_Area_Norm,local.Mass,P[j].Mass,kernel.wk_i,
                   kernel.wk_j,V_i,V_j,kernel.dp[0],kernel.dp[1],kernel.dp[2],local.NV_T[0][0],local.NV_T[0][1],local.NV_T[0][2],SphP[j].NV_T[0][0],SphP[j].NV_T[0][1],
                   SphP[j].NV_T[0][2]);
            fflush(stdout);
        }
        Face_Area_Norm = sqrt(Face_Area_Norm);
        
        /* below, if we are using fixed-grid mode for the code, we manually set the areas to the correct geometric areas */
        
        for(k=0;k<3;k++) {n_unit[k] = Face_Area_Vec[k] / Face_Area_Norm;} /* define useful unit vector for below */
        /* --------------------------------------------------------------------------------- */
        /* extrapolate the conserved quantities to the interaction face between the particles */
        /* first we define some useful variables for the extrapolation */
        /* --------------------------------------------------------------------------------- */
        s_i =  0.5 * kernel.r;
        s_j = -0.5 * kernel.r;
        s_i = s_star_ij - s_i; /* projection element for gradients */
        s_j = s_star_ij - s_j;
        distance_from_i[0]=kernel.dp[0]*rinv; distance_from_i[1]=kernel.dp[1]*rinv; distance_from_i[2]=kernel.dp[2]*rinv;
        for(k=0;k<3;k++) {distance_from_j[k] = distance_from_i[k] * s_j; distance_from_i[k] *= s_i;}
        //for(k=0;k<3;k++) {v_frame[k] = 0.5 * (VelPred_j[k] + local.Vel[k]);}
        for(k=0;k<3;k++) {v_frame[k] = rinv * (-s_i*VelPred_j[k] + s_j*local.Vel[k]);} // allows for face to be off-center (to second-order)
        // (note that in the above, the s_i/s_j terms are crossed with the opposing velocity terms: this is because the face is closer to the
        //   particle with the smaller smoothing length; so it's values are slightly up-weighted //
        
        /* we need the face velocities, dotted into the face vector, for correction back to the lab frame */
        for(k=0;k<3;k++) {face_vel_i+=local.Vel[k]*n_unit[k]; face_vel_j+=VelPred_j[k]*n_unit[k];}
        face_vel_i /= All.cf_atime; face_vel_j /= All.cf_atime;
        face_area_dot_vel = rinv*(-s_i*face_vel_j + s_j*face_vel_i);
        
        /* also will need approach velocities to determine maximum upwind pressure */
        double v2_approach = 0;
        double vdotr2_phys = kernel.vdotr2;
        if(All.ComovingIntegrationOn) {vdotr2_phys -= All.cf_hubble_a2 * r2;}
        vdotr2_phys *= 1/(kernel.r * All.cf_atime);
        if(vdotr2_phys < 0) {v2_approach = vdotr2_phys*vdotr2_phys;}
        double vdotf2_phys = face_vel_i - face_vel_j; // need to be careful of sign here //
        if(vdotf2_phys < 0) {v2_approach = DMAX( v2_approach , vdotf2_phys*vdotf2_phys );}
        
        
        /* now we do the reconstruction (second-order reconstruction at the face) */
        int recon_mode = 1; // default to 'normal' reconstruction: some special physics will set this to zero for low-order reconstructions
        if(fabs(vdotr2_phys)*All.UnitVelocity_in_cm_per_s > 1.0e8) {recon_mode = 0;} // particle approach/recession velocity > 1000 km/s: be extra careful here!
        reconstruct_face_states(local.Density, local.Gradients.Density, SphP[j].Density, SphP[j].Gradients.Density,
                                distance_from_i, distance_from_j, &Riemann_vec.L.rho, &Riemann_vec.R.rho, recon_mode);
        reconstruct_face_states(Pressure_i, local.Gradients.Pressure, Pressure_j, SphP[j].Gradients.Pressure,
                                distance_from_i, distance_from_j, &Riemann_vec.L.p, &Riemann_vec.R.p, recon_mode);
        for(k=0;k<3;k++)
        {
            reconstruct_face_states(local.Vel[k], local.Gradients.Velocity[k], VelPred_j[k], SphP[j].Gradients.Velocity[k],
                                    distance_from_i, distance_from_j, &Riemann_vec.L.v[k], &Riemann_vec.R.v[k], recon_mode);
            Riemann_vec.L.v[k] -= v_frame[k]; Riemann_vec.R.v[k] -= v_frame[k];
        }

        /* estimate maximum upwind pressure */
        double press_i_tot = Pressure_i + local.Density * v2_approach;
        double press_j_tot = Pressure_j + SphP[j].Density * v2_approach;
        double press_tot_limiter;
        press_tot_limiter = 1.1 * All.cf_a3inv * DMAX( press_i_tot , press_j_tot );
        if(recon_mode==0) {press_tot_limiter = DMAX(press_tot_limiter , DMAX(DMAX(Pressure_i,Pressure_j),2.*DMAX(local.Density,SphP[j].Density)*v2_approach));}
        
        /* --------------------------------------------------------------------------------- */
        /* Alright! Now we're actually ready to solve the Riemann problem at the particle interface */
        /* --------------------------------------------------------------------------------- */
        Riemann_solver(Riemann_vec, &Riemann_out, n_unit, press_tot_limiter);
        /* before going on, check to make sure we have a valid Riemann solution */
        if((Riemann_out.P_M<0)||(isnan(Riemann_out.P_M))||(Riemann_out.P_M>1.4*press_tot_limiter))
        {
            /* go to a linear reconstruction of P, rho, and v, and re-try */
            Riemann_vec.R.p = Pressure_i; Riemann_vec.L.p = Pressure_j;
            Riemann_vec.R.rho = local.Density; Riemann_vec.L.rho = SphP[j].Density;
            for(k=0;k<3;k++) {Riemann_vec.R.v[k]=local.Vel[k]-v_frame[k]; Riemann_vec.L.v[k]=VelPred_j[k]-v_frame[k];}
            Riemann_solver(Riemann_vec, &Riemann_out, n_unit, 1.4*press_tot_limiter);
            if((Riemann_out.P_M<0)||(isnan(Riemann_out.P_M)))
            {
                /* ignore any velocity difference between the particles: this should gaurantee we have a positive pressure! */
                Riemann_vec.R.p = Pressure_i; Riemann_vec.L.p = Pressure_j;
                Riemann_vec.R.rho = local.Density; Riemann_vec.L.rho = SphP[j].Density;
                for(k=0;k<3;k++) {Riemann_vec.R.v[k]=0; Riemann_vec.L.v[k]=0;}
                Riemann_solver(Riemann_vec, &Riemann_out, n_unit, 2.0*press_tot_limiter);
                if((Riemann_out.P_M<0)||(isnan(Riemann_out.P_M)))
                {
                    printf("Riemann Solver Failed to Find Positive Pressure!: Pmax=%g PL/M/R=%g/%g/%g Mi/j=%g/%g rhoL/R=%g/%g vL=%g/%g/%g vR=%g/%g/%g n_unit=%g/%g/%g \n",
                           press_tot_limiter,Riemann_vec.L.p,Riemann_out.P_M,Riemann_vec.R.p,local.Mass,P[j].Mass,Riemann_vec.L.rho,Riemann_vec.R.rho,
                           Riemann_vec.L.v[0],Riemann_vec.L.v[1],Riemann_vec.L.v[2],
                           Riemann_vec.R.v[0],Riemann_vec.R.v[1],Riemann_vec.R.v[2],n_unit[0],n_unit[1],n_unit[2]);
                    exit(1234);
                }
            }
        } // closes loop of alternative reconstructions if invalid pressures are found //
        
        /* --------------------------------------------------------------------------------- */
        /* Calculate the fluxes (EQUATION OF MOTION) -- all in physical units -- */
        /* --------------------------------------------------------------------------------- */
        if((Riemann_out.P_M>0)&&(!isnan(Riemann_out.P_M)))
        {
            if(All.ComovingIntegrationOn) {for(k=0;k<3;k++) v_frame[k] /= All.cf_atime;}
            
#if defined(HYDRO_MESHLESS_FINITE_MASS)
            Riemann_out.P_M -= dummy_pressure; // correct back to (allowed) negative pressures //
            double facenorm_pm = Face_Area_Norm * Riemann_out.P_M;
            for(k=0;k<3;k++) {Fluxes.v[k] = facenorm_pm * n_unit[k];} /* total momentum flux */
            Fluxes.p = facenorm_pm * (Riemann_out.S_M + face_area_dot_vel); // default: total energy flux = v_frame.dot.mom_flux //
            
// below is defined for adiabatic ideal fluids, don't use for materials
            /* for MFM, do the face correction for adiabatic flows here */
            int use_entropic_energy_equation = 0;
            double du_new = 0;
            double SM_over_ceff = fabs(Riemann_out.S_M) / DMIN(kernel.sound_i,kernel.sound_j);
            if(SM_over_ceff < epsilon_entropic_eos_big)
            {
                use_entropic_energy_equation = 1;
                double PdV_fac = Riemann_out.P_M * vdotr2_phys / All.cf_a2inv;
                double PdV_i = kernel.dwk_i * V_i*V_i * local.DhsmlNgbFactor * PdV_fac;
                double PdV_j = kernel.dwk_j * V_j*V_j * PPP[j].DhsmlNgbFactor * PdV_fac;
                du_new = 0.5 * (PdV_i - PdV_j + facenorm_pm * (face_vel_i+face_vel_j));
                // check if, for the (weakly) diffusive case, heat is (correctly) flowing from hot to cold after particle averaging (flux-limit) //
                double cnum2 = SphP[j].ConditionNumber*SphP[j].ConditionNumber;
                if(SM_over_ceff > epsilon_entropic_eos_small && cnum2 < cnumcrit2)
                {
                    double du_old = facenorm_pm * (Riemann_out.S_M + face_area_dot_vel);
                    if(Pressure_i/local.Density > Pressure_j/SphP[j].Density)
                    {
                        double dtoj = -du_old + facenorm_pm * face_vel_j;
                        if(dtoj > 0) {use_entropic_energy_equation=0;} else {
                            if(dtoj > -du_new+facenorm_pm*face_vel_j) {use_entropic_energy_equation=0;}}
                    } else {
                        double dtoi = du_old - facenorm_pm * face_vel_i;
                        if(dtoi > 0) {use_entropic_energy_equation=0;} else {
                            if(dtoi > du_new-facenorm_pm*face_vel_i) {use_entropic_energy_equation=0;}}
                    }
                }
                if(cnum2 >= cnumcrit2) {use_entropic_energy_equation=1;}
                // alright, if we've come this far, we need to subtract -off- the thermal energy part of the flux, and replace it //
                if(use_entropic_energy_equation) {Fluxes.p = du_new;}
            }
#endif // endif for clause opening full fluxes (mfv or magnetic)
        } else {
            /* nothing but bad riemann solutions found! */
            memset(&Fluxes, 0, sizeof(struct Conserved_var_Riemann));
        }
    } // Face_Area_Norm != 0
}
