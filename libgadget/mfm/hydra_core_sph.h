/* --------------------------------------------------------------------------------- */
/* this is the sub-routine where we actually evaluate the SPH equations of motion */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
/* --------------------------------------------------------------------------------- */
{
    /* basic overhead variables and zero-ing fluxes for the computation */
    Fluxes.rho = Fluxes.p = Fluxes.v[0] = Fluxes.v[1] = Fluxes.v[2] = 0;
    double du_ij;
    kernel.dwk_ij = 0.5 * (kernel.dwk_i + kernel.dwk_j);
    cnumcrit2 *= 1.0;
    double vdotr2_phys = kernel.vdotr2;
    if(All.ComovingIntegrationOn) vdotr2_phys -= All.cf_hubble_a2 * r2;
    V_j = P[j].Mass / SphP[j].Density;
    
    /* --------------------------------------------------------------------------------- */
    /* --------------------------------------------------------------------------------- */
    /* ... EQUATION OF MOTION (HYDRO) ... */
    /* --------------------------------------------------------------------------------- */
    /* --------------------------------------------------------------------------------- */
    double vi_dot_r,hfc,hfc_visc,hfc_i,hfc_j,hfc_dwk_i,hfc_dwk_j,hfc_egy=0,wt_corr_i=1,wt_corr_j=1;;
    /* 'Standard' (Lagrangian) Density Formulation: the acceleration term is identical whether we use 'entropy' or 'energy' sph */
    /* (this step is the same in both 'Lagrangian' and 'traditional' SPH */

#ifdef HYDRO_PRESSURE_SPH
    /* Pressure-Energy and/or Pressure-Entropy form of SPH (using 'constant mass in kernel' h-constraint */
    /* -- note that, using appropriate definitions, both forms have an identical EOM in appearance here -- */
    double p_over_rho2_j = SphP[j].Pressure / (SphP[j].EgyWtDensity * SphP[j].EgyWtDensity);
    hfc_i = kernel.p_over_rho2_i * (SphP[j].InternalEnergyPred/local.InternalEnergyPred) *
        (1 + local.DhsmlHydroSumFactor / (P[j].Mass * SphP[j].InternalEnergyPred));
    hfc_j = p_over_rho2_j * (local.InternalEnergyPred/SphP[j].InternalEnergyPred) *
        (1 + SphP[j].DhsmlHydroSumFactor / (local.Mass * local.InternalEnergyPred));
#else
    /* Density-Entropy (or Density-Energy) formulation: x_tilde=1, x=mass */
    double p_over_rho2_j = SphP[j].Pressure / (SphP[j].Density * SphP[j].Density);
    hfc_i = kernel.p_over_rho2_i * (1 + local.DhsmlHydroSumFactor / P[j].Mass);
    hfc_j = p_over_rho2_j * (1 + SphP[j].DhsmlHydroSumFactor / local.Mass);
#endif
    hfc_i *= wt_corr_i; hfc_j *= wt_corr_j; // apply tensile instability suppression //

    hfc_egy = hfc_i; /* needed to follow the internal energy explicitly; note this is the same for any of the formulations above */
        
    /* use the traditional 'kernel derivative' (dwk) to compute derivatives */
    hfc_dwk_i = hfc_dwk_j = local.Mass * P[j].Mass / kernel.r;
    hfc_dwk_i *= kernel.dwk_i; /* grad-h terms have already been multiplied in here */
    hfc_dwk_j *= kernel.dwk_j;
    hfc = hfc_i*hfc_dwk_i + hfc_j*hfc_dwk_j;
        
    /* GASOLINE-like equation of motion: */
    /* hfc_dwk_i = 0.5 * (hfc_dwk_i + hfc_dwk_j); */
    /* hfc = (p_over_rho2_j + kernel.p_over_rho2_i) * hfc_dwk_i */
        
    /* RSPH equation of motion */
    /* hfc_dwk_i = 0.5 * (hfc_dwk_i + hfc_dwk_j); */
    /* hfc = (local.Pressure-SphP[j].Pressure)/(local.Density*SphP[j].Density) * hfc_dwk_i */
        
    Fluxes.v[0] += -hfc * kernel.dp[0]; /* momentum */
    Fluxes.v[1] += -hfc * kernel.dp[1];
    Fluxes.v[2] += -hfc * kernel.dp[2];
    vi_dot_r = local.Vel[0]*kernel.dp[0] + local.Vel[1]*kernel.dp[1] + local.Vel[2]*kernel.dp[2];
    Fluxes.p += hfc_egy * hfc_dwk_i * vdotr2_phys - hfc * vi_dot_r; /* total energy */
    
    
    /* --------------------------------------------------------------------------------- */
    /* ... artificial viscosity evaluation ... */
    /* --------------------------------------------------------------------------------- */
    if(kernel.vdotr2 < 0) // no viscosity applied if particles are moving away from each other //
    {
        double c_ij = 0.5 * (kernel.sound_i + kernel.sound_j);
        double BulkVisc_ij = 0.5 * All.ArtBulkViscConst * (local.alpha + SphP[j].alpha_limiter);
        double h_ij = KERNEL_CORE_SIZE * 0.5 * (kernel.h_i + kernel.h_j);
        double mu_ij = fac_mu * h_ij * kernel.vdotr2 / (r2 + 0.0001 * h_ij * h_ij); /* note: this is negative! */
        double visc = -BulkVisc_ij * mu_ij * (c_ij - 2*mu_ij) * kernel.rho_ij_inv; /* this method should use beta/alpha=2 */
#ifndef NOVISCOSITYLIMITER
        double dt = 2 * TIMAX(local.Timestep, (P[j].TimeBin ? (((integertime) 1) << P[j].TimeBin) : 0)) * All.Timebase_interval;
        if(dt > 0 && kernel.dwk_ij < 0)
            visc = DMIN(visc, 0.5 * fac_vsic_fix * kernel.vdotr2 / ((local.Mass + P[j].Mass) * kernel.dwk_ij * kernel.r * dt));
#endif
        hfc_visc = -local.Mass * P[j].Mass * visc * kernel.dwk_ij / kernel.r;
        Fluxes.v[0] += hfc_visc * kernel.dp[0]; /* this is momentum */
        Fluxes.v[1] += hfc_visc * kernel.dp[1];
        Fluxes.v[2] += hfc_visc * kernel.dp[2];
        Fluxes.p += hfc_visc * (vi_dot_r - 0.5*vdotr2_phys); /* remember, this is -total- energy now */
    } // kernel.vdotr2 < 0 -- triggers artificial viscosity
    
    /* --------------------------------------------------------------------------------- */
    /* convert everything to PHYSICAL units! */
    /* --------------------------------------------------------------------------------- */
    Fluxes.p *= All.cf_afac2 / All.cf_atime;
    for(k=0;k<3;k++)
        Fluxes.v[k] *= All.cf_afac2;
}
