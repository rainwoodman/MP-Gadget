#define GAMMA_G1 ((GAMMA-1.0)/(2.0*GAMMA))
#define GAMMA_G2 ((GAMMA+1.0)/(2.0*GAMMA))
#define GAMMA_G3 ((2.0*GAMMA/(GAMMA-1.0)))
#define GAMMA_G4 (2.0/(GAMMA-1.0))
#define GAMMA_G5 (2.0/(GAMMA+1.0))
#define GAMMA_G6 ((GAMMA-1.0)/(GAMMA+1.0))
#define GAMMA_G7 (0.5*(GAMMA-1.0))
#define GAMMA_G8 (1.0/GAMMA)
#define GAMMA_G9 (GAMMA-1.0)

#define TOL_ITER 1.e-6
#define NMAX_ITER 1000

/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. 
 *   However some of the sub-routines here are adopted from other codes, in particular
 *   AREPO by Volker Springel (volker.springel@h-its.org) and 
 *   ATHENA by Jim Stone (jstone@astro.princeton.edu). These sections should be 
 *   identified explicitly in the code below.
 */


/* --------------------------------------------------------------------------------- */
/* some structures with the conserved variables to pass to/from the Riemann solver */
/* --------------------------------------------------------------------------------- */
struct Input_vec_Riemann
{
    struct Conserved_var_Riemann L;
    struct Conserved_var_Riemann R;
};
struct Riemann_outputs
{
    MyDouble P_M;
    MyDouble S_M;
    struct Conserved_var_Riemann Fluxes;
};
struct rotation_matrix
{
    MyDouble n[3];
    MyDouble m[3];
    MyDouble p[3];
};


/* --------------------------------------------------------------------------------- */
/* function definitions */
/* --------------------------------------------------------------------------------- */
void Riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3], double press_tot_limiter);
double guess_for_pressure(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                          double v_line_L, double v_line_R, double cs_L, double cs_R);
void sample_reimann_standard(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                             double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R);
void sample_reimann_vaccum_internal(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                                    double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R);
void sample_reimann_vaccum_right(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                                 double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R);
void sample_reimann_vaccum_left(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                                double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R);
void Riemann_solver_exact(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3],
                            double v_line_L, double v_line_R, double cs_L, double cs_R, double h_L, double h_R);
void get_wavespeeds_and_pressure_star(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3],
                                      double v_line_L, double v_line_R, double cs_L, double cs_R, double h_L, double h_R,
                                      double *S_L_out, double *S_R_out, double press_tot_limiter);
void HLLC_Riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3],
                         double v_line_L, double v_line_R, double cs_L, double cs_R, double h_L, double h_R, double press_tot_limiter);
void convert_face_to_flux(struct Riemann_outputs *Riemann_out, double n_unit[3]);
int iterative_Riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                              double v_line_L, double v_line_R, double cs_L, double cs_R);
void reconstruct_face_states(double Q_i, MyFloat Grad_Q_i[3], double Q_j, MyFloat Grad_Q_j[3],
                             double distance_from_i[3], double distance_from_j[3], double *Q_L, double *Q_R, int mode);
/* --------------------------------------------------------------------------------- */
/* reconstruction procedure (use to extrapolate from cell/particle centered quantities to faces) */
/*  (reconstruction and slope-limiter from P. Hopkins) */
/* --------------------------------------------------------------------------------- */
void reconstruct_face_states(double Q_i, MyFloat Grad_Q_i[3], double Q_j, MyFloat Grad_Q_j[3],
                             double distance_from_i[3], double distance_from_j[3], double *Q_L, double *Q_R, int mode)
{
    if(mode == 0)
    {
        /* zeroth order reconstruction: this case is trivial */
        *Q_R = Q_i;
        *Q_L = Q_j;
        return;
    }
    /* check for the (also) trivial case of equal values on both sides */
    if(Q_i==Q_j) {*Q_L=*Q_R=Q_i; return;}
    
    /* first order reconstruction */
    *Q_R = Q_i + Grad_Q_i[0]*distance_from_i[0] + Grad_Q_i[1]*distance_from_i[1] + Grad_Q_i[2]*distance_from_i[2];
    *Q_L = Q_j + Grad_Q_j[0]*distance_from_j[0] + Grad_Q_j[1]*distance_from_j[1] + Grad_Q_j[2]*distance_from_j[2];

    /* here we do our slightly-fancy slope-limiting */
    double Qmin,Qmax,Qmed,Qmax_eff,Qmin_eff,fac,Qmed_max,Qmed_min;
    double fac_minmax = 0.5; /* 0.5, 0.1 works; 1.0 unstable; 0.75 is stable but begins to 'creep' */
    double fac_meddev = 0.375; /* 0.25,0.375 work well; 0.5 unstable; 0.44 is on-edge */
    /* get the max/min vals, difference, and midpoint value */
    Qmed = 0.5*(Q_i+Q_j);
    if(Q_i<Q_j) {Qmax=Q_j; Qmin=Q_i;} else {Qmax=Q_i; Qmin=Q_j;}
    fac = fac_minmax * (Qmax-Qmin);
    if(mode == -1) {fac += MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM * fabs(Qmed);}
    Qmax_eff = Qmax + fac; /* 'overshoot tolerance' */
    Qmin_eff = Qmin - fac; /* 'undershoot tolerance' */
    /* check if this implies a sign from the min/max values: if so, we re-interpret the derivative as a
     logarithmic derivative to prevent sign changes from occurring */
    if(mode > 0)
    {
        if(Qmax<0) {if(Qmax_eff>0) Qmax_eff=Qmax*Qmax/(Qmax-(Qmax_eff-Qmax));} // works with 0.5,0.1 //
        if(Qmin>0) {if(Qmin_eff<0) Qmin_eff=Qmin*Qmin/(Qmin+(Qmin-Qmin_eff));}
    }
    /* also allow tolerance to over/undershoot the exact midpoint value in the reconstruction */
    fac = fac_meddev * (Qmax-Qmin);
    if(mode == -1) {fac += MHD_CONSTRAINED_GRADIENT_FAC_MED_PM * fabs(Qmed);}
    Qmed_max = Qmed + fac;
    Qmed_min = Qmed - fac;
    if(Qmed_max>Qmax_eff) Qmed_max=Qmax_eff;
    if(Qmed_min<Qmin_eff) Qmed_min=Qmin_eff;
    /* now check which side is which and apply these limiters */
    if(Q_i<Q_j)
    {
        if(*Q_R<Qmin_eff) *Q_R=Qmin_eff;
        if(*Q_R>Qmed_max) *Q_R=Qmed_max;
        if(*Q_L>Qmax_eff) *Q_L=Qmax_eff;
        if(*Q_L<Qmed_min) *Q_L=Qmed_min;
    } else {
        if(*Q_R>Qmax_eff) *Q_R=Qmax_eff;
        if(*Q_R<Qmed_min) *Q_R=Qmed_min;
        if(*Q_L<Qmin_eff) *Q_L=Qmin_eff;
        if(*Q_L>Qmed_max) *Q_L=Qmed_max;
    }
    /* done! */
}


/* --------------------------------------------------------------------------------- */
/* Master Riemann solver routine: call this, it will call sub-routines */
/*  (written by P. Hopkins, this is just a wrapper though for the various sub-routines) */
/* --------------------------------------------------------------------------------- */
void Riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3], double press_tot_limiter)
{
    if((Riemann_vec.L.p < 0 && Riemann_vec.R.p < 0)||(Riemann_vec.L.rho < 0)||(Riemann_vec.R.rho < 0))
    {
        printf("FAILURE: Unphysical Inputs to Reimann Solver: Left P/rho=%g/%g, Right P/rho=%g/%g \n",
               Riemann_vec.L.p,Riemann_vec.L.rho,Riemann_vec.R.p,Riemann_vec.R.rho); fflush(stdout);
        Riemann_out->P_M = 0;
        return;
    }
    
    if(All.ComovingIntegrationOn)
    {
        /* first convert the input variables to -PHYSICAL- units so the answer makes sense:
         note that we don't require a Hubble-flow correction, because we're solving at the face */
        int k;
        for(k=0;k<3;k++)
        {
            Riemann_vec.L.v[k] /= All.cf_atime;
            Riemann_vec.R.v[k] /= All.cf_atime;
        }
        Riemann_vec.L.rho *= All.cf_a3inv;
        Riemann_vec.R.rho *= All.cf_a3inv;
        Riemann_vec.L.p *= All.cf_a3inv / All.cf_afac1;
        Riemann_vec.R.p *= All.cf_a3inv / All.cf_afac1;
    }
    /* here we haven't reconstructed the sound speeds and internal energies explicitly, so need to do it from pressure, density */
    Riemann_vec.L.cs = sqrt(GAMMA * Riemann_vec.L.p / Riemann_vec.L.rho);
    Riemann_vec.R.cs = sqrt(GAMMA * Riemann_vec.R.p / Riemann_vec.R.rho);
    Riemann_vec.L.u  = Riemann_vec.L.p / (GAMMA_MINUS1 * Riemann_vec.L.rho);
    Riemann_vec.R.u  = Riemann_vec.R.p / (GAMMA_MINUS1 * Riemann_vec.R.rho);
    
    double cs_L = Riemann_vec.L.cs;
    double cs_R = Riemann_vec.R.cs;
    double h_L = Riemann_vec.L.p/Riemann_vec.L.rho + Riemann_vec.L.u + 0.5*(Riemann_vec.L.v[0]*Riemann_vec.L.v[0]+Riemann_vec.L.v[1]*Riemann_vec.L.v[1]+Riemann_vec.L.v[2]*Riemann_vec.L.v[2]);
    double h_R = Riemann_vec.R.p/Riemann_vec.R.rho + Riemann_vec.R.u + 0.5*(Riemann_vec.R.v[0]*Riemann_vec.R.v[0]+Riemann_vec.R.v[1]*Riemann_vec.R.v[1]+Riemann_vec.R.v[2]*Riemann_vec.R.v[2]);
    double v_line_L = Riemann_vec.L.v[0]*n_unit[0] + Riemann_vec.L.v[1]*n_unit[1] + Riemann_vec.L.v[2]*n_unit[2];
    double v_line_R = Riemann_vec.R.v[0]*n_unit[0] + Riemann_vec.R.v[1]*n_unit[1] + Riemann_vec.R.v[2]*n_unit[2];
    
    HLLC_Riemann_solver(Riemann_vec, Riemann_out, n_unit, v_line_L, v_line_R, cs_L, cs_R, h_L, h_R, press_tot_limiter);
}

/* -------------------------------------------------------------------------------------------------------------- */
/* the HLLC Riemann solver: try this first - it's approximate, but fast, and accurate for our purposes */
/*  (wrapper for sub-routines to evaluate hydro reimann problem) */
/* HLLC: hydro (no MHD) */
void HLLC_Riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3],
                        double v_line_L, double v_line_R, double cs_L, double cs_R, double h_L, double h_R, double press_tot_limiter)
{
    double S_L,S_R;
    get_wavespeeds_and_pressure_star(Riemann_vec, Riemann_out, n_unit,  v_line_L, v_line_R, cs_L, cs_R, h_L, h_R, &S_L, &S_R, press_tot_limiter);
}

/* here we obtain wave-speed and pressure estimates for the 'star' region for the HLLC solver or the 
    Lagrangian (contact-wave) method; note we keep trying several methods here in the hopes of eventually getting a
    valid (positive-pressure) solution */
/*  (written by P. Hopkins) */
void get_wavespeeds_and_pressure_star(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3],
                                      double v_line_L, double v_line_R, double cs_L, double cs_R, double h_L, double h_R,
                                      double *S_L_out, double *S_R_out, double press_tot_limiter)
{
    double S_L, S_R;
    
    /* first, check for vacuum conditions, not accounted for in the standard HLLC scheme */
    if((v_line_R - v_line_L) > DMAX(cs_L,cs_R))
    {
        Riemann_out->P_M = MIN_REAL_NUMBER; Riemann_out->S_M = S_L = S_R = 0;
    } else {
        /* Gaburov: 'simplest' HLLC discretization with weighting scheme */
        double PT_L = Riemann_vec.L.p;
        double PT_R = Riemann_vec.R.p;
        S_L = DMIN(v_line_L,v_line_R) - DMAX(cs_L,cs_R);
        S_R = DMAX(v_line_L,v_line_R) + DMAX(cs_L,cs_R);
        double rho_wt_L = Riemann_vec.L.rho*(S_L-v_line_L);
        double rho_wt_R = Riemann_vec.R.rho*(S_R-v_line_R);
        Riemann_out->S_M = ((PT_R-PT_L) + rho_wt_L*v_line_L - rho_wt_R*v_line_R) / (rho_wt_L - rho_wt_R);
        Riemann_out->P_M = (PT_L*rho_wt_R - PT_R*rho_wt_L + rho_wt_L*rho_wt_R*(v_line_R - v_line_L)) / (rho_wt_R - rho_wt_L);
        if(Riemann_out->P_M <= MIN_REAL_NUMBER) {Riemann_out->P_M = MIN_REAL_NUMBER; Riemann_out->S_M = S_L = S_R = 0;}
        
        if((Riemann_out->P_M <= 0)||(isnan(Riemann_out->P_M))||(Riemann_out->P_M>press_tot_limiter))
        {
            /* failed: compute Roe-averaged values (Roe 1981) [roe-averaging not strictly necessary for HLLC, though it improves accuracy */
            /* note that enthalpy H=(Etotal+P)/d is averaged, with Etotal=Ekinetic+Einternal (Einternal=P/(gamma-1)) */
            double sqrt_rho_L = sqrt(Riemann_vec.L.rho);
            double sqrt_rho_R = sqrt(Riemann_vec.R.rho);
            double sqrt_rho_inv = 1 / (sqrt_rho_L + sqrt_rho_R);
            double vx_roe = (sqrt_rho_L*Riemann_vec.L.v[0] + sqrt_rho_R*Riemann_vec.R.v[0]) * sqrt_rho_inv;
            double vy_roe = (sqrt_rho_L*Riemann_vec.L.v[1] + sqrt_rho_R*Riemann_vec.R.v[1]) * sqrt_rho_inv;
            double vz_roe = (sqrt_rho_L*Riemann_vec.L.v[2] + sqrt_rho_R*Riemann_vec.R.v[2]) * sqrt_rho_inv;
            /* compute velocity along the line connecting the nodes, and max/min wave speeds */
            double v_line_roe = vx_roe*n_unit[0] + vy_roe*n_unit[1] + vz_roe*n_unit[2];
            double h_roe  = (sqrt_rho_L*h_L  + sqrt_rho_R*h_R) * sqrt_rho_inv;
            double cs_roe = sqrt(DMAX(1.e-30, GAMMA_MINUS1*(h_roe - 0.5*(vx_roe*vx_roe+vy_roe*vy_roe+vz_roe*vz_roe))));
            S_R = DMAX(v_line_R + cs_R , v_line_roe + cs_roe);
            S_L = DMIN(v_line_L - cs_L , v_line_roe - cs_roe);
            rho_wt_R =  Riemann_vec.R.rho * (S_R - v_line_R);
            rho_wt_L = -Riemann_vec.L.rho * (S_L - v_line_L); /* note the sign */
            /* contact wave speed (speed at contact surface): */
            Riemann_out->S_M = ((rho_wt_R*v_line_R + rho_wt_L*v_line_L) + (PT_L - PT_R)) / (rho_wt_R + rho_wt_L);
            /* S_M = v_line_L* = v_line_R* = v_line_M --- this is the speed at interface */
            /* contact pressure (pressure at contact surface): */
            Riemann_out->P_M = Riemann_vec.L.rho * (v_line_L-S_L)*(v_line_L-Riemann_out->S_M) + PT_L;
            if(Riemann_out->P_M <= MIN_REAL_NUMBER) {Riemann_out->P_M = MIN_REAL_NUMBER; Riemann_out->S_M = S_L = S_R = 0;}
            /* p_M = p_L* = p_R*  */
            
            if((Riemann_out->P_M <= 0)||(isnan(Riemann_out->P_M))||(Riemann_out->P_M>press_tot_limiter))
            {
                /* failed again! try the simple primitive-variable estimate (as we would for Rusanov) */
                Riemann_out->P_M = 0.5*((PT_L + PT_R) + (v_line_L-v_line_R)*0.25*(Riemann_vec.L.rho+Riemann_vec.R.rho)*(cs_L+cs_R));
                /* compute the new wave speeds from it */
                Riemann_out->S_M = 0.5*(v_line_R+v_line_L) + 2.0*(PT_L-PT_R)/((Riemann_vec.L.rho+Riemann_vec.R.rho)*(cs_L+cs_R));
                double S_plus = DMAX(DMAX(fabs(v_line_L - cs_L), fabs(v_line_R - cs_R)), DMAX(fabs(v_line_L + cs_L), fabs(v_line_R + cs_R)));
                S_L=-S_plus; S_R=S_plus; if(Riemann_out->S_M<S_L) Riemann_out->S_M=S_L; if(Riemann_out->S_M>S_R) Riemann_out->S_M=S_R;
                if(Riemann_out->P_M <= MIN_REAL_NUMBER) {Riemann_out->P_M = MIN_REAL_NUMBER; Riemann_out->S_M = S_L = S_R = 0;}
            }
        }
    }
    *S_L_out = S_L; *S_R_out = S_R;
    return;
} // if((Riemann_out->P_M <= 0)||(isnan(Riemann_out->P_M))) //

/* --------------------------------------------------------------------------------- */
/*  exact Riemann solver here -- deals with all the problematic states! */
/*  (written by V. Springel for AREPO; as are the extensions to the exact solver below) */
/* --------------------------------------------------------------------------------- */
void Riemann_solver_exact(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double n_unit[3],
                       double v_line_L, double v_line_R, double cs_L, double cs_R, double h_L, double h_R)
{
    /* first, we need to check for all the special/exceptional cases that will cause things to go haywire */
    if((Riemann_vec.L.p == 0 && Riemann_vec.L.p == 0) || (Riemann_vec.L.rho==0 && Riemann_vec.R.rho==0))
    {
        /* we're in a Vaccuum! */
        Riemann_out->P_M = Riemann_out->S_M = 0;
        return;
    }
    /* the usual situation is here:: */
    if((Riemann_vec.L.rho > 0) && (Riemann_vec.R.rho > 0))
    {
        if(iterative_Riemann_solver(Riemann_vec, Riemann_out, v_line_L, v_line_R, cs_L, cs_R))
        {
            /* this is the 'normal' Reimann solution */
            sample_reimann_standard(0.0,Riemann_vec,Riemann_out,n_unit,v_line_L,v_line_R,cs_L,cs_R);
        }
        else
        {
            /* ICs lead to vacuum, need to sample vacuum solution */
            sample_reimann_vaccum_internal(0.0,Riemann_vec,Riemann_out,n_unit,v_line_L,v_line_R,cs_L,cs_R);
        }
    } else {
        /* one of the densities is zero or negative */
        if((Riemann_vec.L.rho<0)||(Riemann_vec.R.rho<0))
            exit(1234);
        if(Riemann_vec.L.rho>0)
            sample_reimann_vaccum_right(0.0,Riemann_vec,Riemann_out,n_unit,v_line_L,v_line_R,cs_L,cs_R);
        if(Riemann_vec.R.rho>0)
            sample_reimann_vaccum_left(0.0,Riemann_vec,Riemann_out,n_unit,v_line_L,v_line_R,cs_L,cs_R);
    }
}


/* --------------------------------------------------------------------------------- */
/* part of exact Riemann solver: */
/* left state is a vacuum, but right state is not: sample the fan appropriately */
/*  (written by V. Springel for AREPO) */
/* --------------------------------------------------------------------------------- */
void sample_reimann_vaccum_left(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                                double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R)
{
    double S_R = v_line_R - GAMMA_G4 * cs_R;
    /* in this code mode, we are -always- moving with the contact discontinuity so density flux = 0; 
     this constrains where we reside in the solution fan */
    Riemann_out->P_M = 0;
    Riemann_out->S_M = S_R;
    return;
}


/* --------------------------------------------------------------------------------- */
/* Part of exact Riemann solver: */
/* right state is a vacuum, but left state is not: sample the fan appropriately */
/*  (written by V. Springel for AREPO) */
/* --------------------------------------------------------------------------------- */
void sample_reimann_vaccum_right(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                                double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R)
{
    //double S_L = v_line_L - GAMMA_G4 * cs_L;
    double S_L = v_line_L + GAMMA_G4 * cs_L; // above line was a sign error, caught by Bert Vandenbroucke
    /* in this code mode, we are -always- moving with the contact discontinuity so density flux = 0;
     this constrains where we reside in the solution fan */
    Riemann_out->P_M = 0;
    Riemann_out->S_M = S_L;
    return;
}



/* --------------------------------------------------------------------------------- */
/* Part of exact Riemann solver: */
/* solution generations a vacuum inside the fan: sample the vacuum appropriately */
/*   (note that these solutions are identical to the left/right solutions above) */
/*  (written by V. Springel for AREPO) */
/* --------------------------------------------------------------------------------- */
void sample_reimann_vaccum_internal(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                                    double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R)
{
    double S_L = v_line_L + GAMMA_G4 * cs_L;
    double S_R = v_line_R - GAMMA_G4 * cs_R;
    if(S <= S_L)
    {
        /* left fan */
        sample_reimann_vaccum_right(S,Riemann_vec,Riemann_out,n_unit,v_line_L,v_line_R,cs_L,cs_R);
    }
    else if(S >= S_R)
    {
        /* right fan */
        sample_reimann_vaccum_left(S,Riemann_vec,Riemann_out,n_unit,v_line_L,v_line_R,cs_L,cs_R);
    }
    else
    {
        /* vacuum in between */
        Riemann_out->P_M = 0;
        Riemann_out->S_M = S;
    }
}




/* --------------------------------------------------------------------------------- */
/* Part of exact Riemann solver: */
/*  This is the "normal" Riemann fan, with no vacuum on L or R state! */
/*  (written by V. Springel for AREPO) */
/* --------------------------------------------------------------------------------- */
void sample_reimann_standard(double S, struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out,
                             double n_unit[3], double v_line_L, double v_line_R, double cs_L, double cs_R)
{
    /* we don't actually need to evaluate the fluxes, and we already have P_M and S_M, which define the 
     contact discontinuity where the rho flux = 0; so can simply exit this routine */
    return;
}



/* --------------------------------------------------------------------------------- */
/* the exact (iterative) Riemann solver: this is slower, but exact. 
 however there is a small chance of the iteration diverging,
 so we still cannot completely gaurantee a valid solution */
/*  (written by P. Hopkins; however this is adapted from the iterative solver in 
        ATHENA by J. Stone) */
/* --------------------------------------------------------------------------------- */
int iterative_Riemann_solver(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double v_line_L, double v_line_R, double cs_L, double cs_R)
{
    /* before going on, let's compare this to an exact Riemann solution calculated iteratively */
    double Pg,Pg_prev,W_L,W_R,Z_L,Z_R,tol,pratio; int niter_Riemann=0;
    double a0,a1,a2,dvel,check_vel;
    dvel = v_line_R - v_line_L;
    check_vel = GAMMA_G4 * (cs_R + cs_L) - dvel;
    /* if check_vel<0, this will produce a vacuum: need to use vacuum-specific subroutine */
    if(check_vel < 0) return 0;
    
    tol=100.0;
    Pg = guess_for_pressure(Riemann_vec, Riemann_out, v_line_L, v_line_R, cs_L, cs_R);
    while((tol>TOL_ITER)&&(niter_Riemann<NMAX_ITER))
    {
        Pg_prev=Pg;
        if(Pg>Riemann_vec.L.p)
        {
            /* shock wave */
            a0 = GAMMA_G5 / Riemann_vec.L.rho;
            a1 = GAMMA_G6 * Riemann_vec.L.p;
            a2 = sqrt(a0 / (Pg+a1));
            W_L = (Pg-Riemann_vec.L.p) * a2;
            Z_L = a2 * (1.0 - 0.5*(Pg-Riemann_vec.L.p)/(a1+Pg));
        } else {
            /* rarefaction wave */
            pratio = Pg / Riemann_vec.L.p;
            W_L = GAMMA_G4 * cs_L * (pow(pratio, GAMMA_G1)-1);
            Z_L = 1 / (Riemann_vec.L.rho*cs_L) * pow(Pg/Riemann_vec.L.p, -GAMMA_G2);
        }
        if(Pg>Riemann_vec.R.p)
        {
            /* shock wave */
            a0 = GAMMA_G5 / Riemann_vec.R.rho;
            a1 = GAMMA_G6 * Riemann_vec.R.p;
            a2 = sqrt(a0 / (Pg+a1));
            W_R = (Pg-Riemann_vec.R.p) * a2;
            Z_R = a2 * (1.0 - 0.5*(Pg-Riemann_vec.R.p)/(a1+Pg));
        } else {
            /* rarefaction wave */
            pratio = Pg / Riemann_vec.R.p;
            W_R = GAMMA_G4 * cs_R * (pow(pratio, GAMMA_G1)-1);
            Z_R = 1 / (Riemann_vec.R.rho*cs_R) * pow(pratio, -GAMMA_G2);
        }
        if(niter_Riemann < NMAX_ITER / 2)
            Pg -= (W_L + W_R + dvel) / (Z_L + Z_R);
        else
            Pg -= 0.5 * (W_L + W_R + dvel) / (Z_L + Z_R);
        
        if(Pg < 0.1 * Pg_prev)
            Pg = 0.1 * Pg_prev;
        
        tol = 2.0 * fabs((Pg-Pg_prev)/(Pg+Pg_prev));
        niter_Riemann++;
    }
    if(niter_Riemann<NMAX_ITER)
    {
        Riemann_out->P_M = Pg;
        Riemann_out->S_M = 0.5*(v_line_L+v_line_R) + 0.5*(W_R-W_L);
        return 1;
    } else {
        return 0;
    }
}



/* --------------------------------------------------------------------------------- */
/* get a pressure guess to begin iteration, for the iterative exact Riemann solver(s) */
/*   (written by V. Springel for AREPO, with minor modifications) */
/* --------------------------------------------------------------------------------- */
double guess_for_pressure(struct Input_vec_Riemann Riemann_vec, struct Riemann_outputs *Riemann_out, double v_line_L, double v_line_R, double cs_L, double cs_R)
{
    double pmin, pmax;
    /* start with the usual lowest-order guess for the contact wave pressure */
    double pv = 0.5*(Riemann_vec.L.p+Riemann_vec.R.p) - 0.125*(v_line_R-v_line_L)*(Riemann_vec.L.p+Riemann_vec.R.p)*(cs_L+cs_R);
    pmin = DMIN(Riemann_vec.L.p,Riemann_vec.R.p);
    pmax = DMAX(Riemann_vec.L.p,Riemann_vec.R.p);
    
    /* if one side is vacuum, guess half the mean */
    if(pmin<=0)
        return 0.5*(pmin+pmax);

    /* if the two are sufficiently close, and pv is between both values, return it */
    double qrat = pmax / pmin;
    if(qrat <= 2.0 && (pmin <= pv && pv <= pmax))
        return pv;
    
    if(pv < pmin)
    {
        /* use two-rarefaction solution */
        double pnu = (cs_L+cs_R) - GAMMA_G7 * (v_line_R - v_line_L);
        double pde = cs_L / pow(Riemann_vec.L.p, GAMMA_G1) + cs_R / pow(Riemann_vec.R.p, GAMMA_G1);
        return pow(pnu / pde, GAMMA_G3);
    }
    else
    {
        /* two-shock approximation  */
        double gel = sqrt((GAMMA_G5 / Riemann_vec.L.rho) / (GAMMA_G6 * Riemann_vec.L.p + pv));
        double ger = sqrt((GAMMA_G5 / Riemann_vec.R.rho) / (GAMMA_G6 * Riemann_vec.R.p + pv));
        double x = (gel * Riemann_vec.L.p + ger * Riemann_vec.R.p - (v_line_R - v_line_L)) / (gel + ger);
        if(x < pmin || x > pmax)
            x = pmin;
        return x;
    }
}


/* -------------------------------------------------------------------------------------------------------------- */
/*  Part of exact Riemann solver: */
 /*    take the face state we have calculated from the exact Riemann solution and get the corresponding fluxes */
/*   (written by V. Springel for AREPO, with minor modifications) */
 /* -------------------------------------------------------------------------------------------------------------- */
void convert_face_to_flux(struct Riemann_outputs *Riemann_out, double n_unit[3])
{
    double rho, P, v[3], v_line=0, v_frame=0, h=0; int k;
    rho = Riemann_out->Fluxes.rho;
    P = Riemann_out->Fluxes.p;
    for(k=0;k<3;k++)
    {
        v[k] = Riemann_out->Fluxes.v[k];
        v_line += v[k] * n_unit[k];
        h += v[k] * v[k];
    }
    v_line -= v_frame;
    h *= 0.5 * rho; /* h is the kinetic energy density */
    h += (GAMMA/GAMMA_MINUS1) * P; /* now h is the enthalpy */
    /* now we just compute the standard fluxes for a given face state */
    Riemann_out->Fluxes.p = h * v_line;
    Riemann_out->Fluxes.rho = rho * v_line;
    for(k=0;k<3;k++)
        Riemann_out->Fluxes.v[k] = Riemann_out->Fluxes.rho * v[k] + P * n_unit[k];
    return;
}
