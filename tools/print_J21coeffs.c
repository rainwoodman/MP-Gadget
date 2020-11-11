#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>

#define NSTEP2 100000
#define HP 4.14e-15  //hp = 1 for using E instead of v, not in code currently due to units, Hz to eV

double vH[3] = {13.6,24.59,54.42}; //ionization thresholds
double dens[2] = {0.76,0.06}; //number density fractions of H and He
double kb = 8.617e-5; //eV K^-1 Boltzmann cnst
double Gc[3], Rc[3], Nc[3], Gcq[3], Rcq[3];
double Gct[3], Rct[3], Gct2[3], Rct2[3];

// parameters of sigma from Verner et al (1996)
double crsscn(double v, int sp)
{
  double sigz=0, ezero=0, ya=0, P=0, yw=0, y0=0, y1=0, x, y;
  if(v<vH[sp])return 0.0;
  if(sp==0)
  {
    sigz = 54750; //Mb units
    ezero = 0.4298; //all parameters from Verner et al(1996), eV
    ya = 32.88;
    P = 2.963;
    yw = 0;
    y0 = 0;
    y1 = 0;
  }
  else if(sp==1)
  {
    sigz = 949.2; //Mb units
    ezero = 13.61; //all parameters from Verner et al(1996), eV
    ya = 1.469;
    P = 3.188;
    yw = 2.039;
    y0 = 0.4434;
    y1 = 2.136;
  }
  else if(sp==2)
  {
    sigz = 13690; //Mb units
    ezero = 1.720; //all parameters from Verner et al(1996), eV
    ya = 32.88;
    P = 2.963;
    yw = 0;
    y0 = 0;
    y1 = 0;
  }
  else
  {
    printf("species not defined, crsscn fail, sp = %d\n",sp);
  }
  x = v/ezero - y0;
  y = sqrt(x*x + y1*y1);
  return sigz*1e-18*((x-1)*(x-1) + yw*yw)*pow(y,0.5*P-5.5)*pow(1+sqrt(y/ya),-P); //converted to cm^2
}

double Jtest(double v, double slope)
{
  double Jf;
  Jf = pow((v/vH[0]),(slope*-1));
  if((v>=vH[2]))
  {
    return 0;
  }
  return 6.242e11/4.14e-15*Jf; //strange units, (eV / h_pl Hz ..), makes integrating easier
}

//Heating rate integrand SR
double heatG(double v, int sp, double slope)
{
  return 4*M_PI*Jtest(v,slope)/v*(v - vH[sp])*crsscn(v,sp);
}
//Ionisation rate integrand
double ionR(double v, int sp, double slope)
{
  return 4*M_PI*Jtest(v,slope)/v*crsscn(v,sp);
}

//rates w/o cross section (optically thick for Treion) ONLY FOR INITIAL TEMPS
double heatGtre(double v, int sp, double slope)
{
  return Jtest(v,slope)/v*(v - vH[sp]);
}

double ionRtre(double v, int sp, double slope)
{
  return Jtest(v,slope)/v;
}
double factor(double v, int sp)
{
  return dens[sp]*crsscn(v,0)/(dens[0]*crsscn(v,0) + dens[1]*crsscn(v,1));
}

//thick rates with comparative cross section term
double heatGtre2(double v, int sp, double slope)
{
  return Jtest(v,slope)/v*(v - vH[sp])*factor(v,sp);
}

double ionRtre2(double v, int sp, double slope)
{
  return Jtest(v,slope)/v*factor(v,sp);
}

double sIntegrate(double min, double max, int n, double (*func)(double,int,double), int sp, double slope)
{
  int i;
  double F[NSTEP2];
  double x;
  double sum2=0;
  double h=(max-min)/(n-1);
  for(i=0; i<n; i++)
    {
      x = min+i*h;
      F[i]=func(x,sp,slope);
      if(i==0||i==n-1)
        {
          sum2 = sum2 + F[i];
        }
      else
        {
          if(i%2==1)
            {
              sum2 = sum2 + 4*F[i];
            }
          else
            {
              sum2 = sum2 + 2*F[i];
            }
        }
    }
  sum2 = h*sum2/3;
  return sum2;
}

int main(int argc, char *argv[])
{
  int j,s;
  double J21 = 1;
  double slope;
  for(s=0;s<26;s++)
  {
    for(j=0;j<3;j++)
    {
      slope = 0.2*s;
      Rc[j] = sIntegrate(vH[j],100,NSTEP2,ionR,j,slope)*J21*1e-21;
      Gc[j] = sIntegrate(vH[j],100,NSTEP2,heatG,j,slope)*J21*1e-21;
      Rct[j] = sIntegrate(vH[j],100,NSTEP2,ionRtre,j,slope)*J21*1e-21;
      Gct[j] = sIntegrate(vH[j],100,NSTEP2,heatGtre,j,slope)*J21*1e-21;
      Rct2[j] = sIntegrate(vH[j],100,NSTEP2,ionRtre2,j,slope)*J21*1e-21;
      Gct2[j] = sIntegrate(vH[j],100,NSTEP2,heatGtre2,j,slope)*J21*1e-21;
    }
    double uthin, uthick, Tthin, Tthick;
    uthin = 0.76*Gc[0]/Rc[0] + 0.06*Gc[1]/Rc[1];
    uthick = 0.76*Gct2[0]/Rct2[0] + 0.06*Gct2[1]/Rct2[1];
    Tthin = 1/3./kb*(0.76*Gc[0]/Rc[0] + 0.06*Gc[1]/Rc[1])/0.82;
    Tthick = 1/3./kb*(0.76*Gct2[0]/Rct2[0] + 0.06*Gct2[1]/Rct2[1])/0.82;
    //printf("slope = %.1f | uthin = %.3e | uthick = %.3e | Treionthin = %.3e | Treionthick =  %.3e\n",
    //       slope,uthin,uthick,Tthin,Tthick);
    printf("slope = %.1f | R(HI) = %.3e | G(HI) = %.3e | R(HeI) = %.3e | G(HeI) =  %.3e\n",
           slope,Rc[0],Gc[0],Rc[1],Gc[1]);
    printf("u_reion = %.3e | T_reion(fast) = %.3e | u_excess = %.3e\n",
           uthick,Tthick,uthin);
  }
  printf("-----------------------------------------\n");
  double zr[5] = {3.02,4.11,5.18,6.14,7.25}; 
  double RP[5] = {1.02e-12,8.02e-13,7.32e-13,3.11e-15,2.15e-16};
  double GP[5] = {3.68e-12,2.94e-12,2.76e-12,1.86e-14,1.32e-15};
  double RHM[5] = {8.22e-13,5.46e-13,4.12e-13,2.30e-13,9.28e-14};
  double GHM[5] = {3.25e-12,2.18e-12,1.67e-12,9.89e-13,4.30e-13};

  for(s=0;s<5;s++)
  {
    printf("Puchwein 18 z=%.2f | RHI = %.2e | GHI = %.2e | Ratio = %.3e | Treion = %.3e\n",zr[s],RP[s],GP[s],GP[s]/RP[s],0.76/3.0/kb*GP[s]/RP[s]/0.82);
  }
  printf("-----------------------------------------\n");
  for(s=0;s<5;s++)
  {
    printf("Hrdt Mdu 12 z=%.2f | RHI = %.2e | GHI = %.2e | Ratio = %.3e | Treion = %.3e\n",zr[s],RHM[s],GHM[s],GHM[s]/RHM[s],0.76/3.0/kb*GHM[s]/RHM[s]/0.82);
  }
  return 0;
}
