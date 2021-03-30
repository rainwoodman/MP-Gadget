'''
This code calculates photoionisation and photohating rates at an assumed spectral index
and J_21 == 1: J(nu) = nu(-alpha)
These rates can then be converted to inhomogeneous rates by multiplying by J_21 output
from the excursion set code
'''
from scipy import integrate
import numpy as np

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--min", default=0, type=float, help="spectral slope minimum")
parser.add_argument("--max", default=5, type=float, help="spectral slope maximum")
parser.add_argument("-n","--n_slopes", default=26, type=int, help="spectral slope maximum")
parser.add_argument("-o", "--outfile", help="output file")
parser.add_argument("--noheiii", help="Assume no HeII ionising photons (stop integrating at 54.42eV)",action="store_true")

args = parser.parse_args()

vH = np.array([13.6,24.59,54.42]) #ionization thresholds in eV
slope_min = args.min #minimum spectral slope
slope_max = args.max #maximum spectral slope
n_slope = args.n_slopes #number of spectral slopes in table
noheiii = args.noheiii

# Cross section formulae for H,He from Verner et al (1996)
def crsscn(v, sp):
    if(v<vH[sp]):
        return 0.0
    if(sp==0):
        sigz = 54750 #Mb units
        ezero = 0.4298 #eV
        ya = 32.88
        P = 2.963
        yw = 0
        y0 = 0
        y1 = 0
    elif(sp==1):
        sigz = 949.2 #Mb units
        ezero = 13.61 #eV
        ya = 1.469
        P = 3.188
        yw = 2.039
        y0 = 0.4434
        y1 = 2.136
    elif(sp==2):
        sigz = 13690 #Mb units
        ezero = 1.720 #eV
        ya = 32.88
        P = 2.963
        yw = 0
        y0 = 0
        y1 = 0
    else:
        print(f"species {sp} does not exist")
        return
  
    x = v/ezero - y0
    y = np.sqrt(x*x + y1*y1)
    return sigz*1e-18*((x-1)*(x-1) + yw*yw)*y**(0.5*P-5.5)*(1+np.sqrt(y/ya))**(-P) #converted to cm^2

#specific intensity at a certain frequency with J21 == 1
def Jtest(v,slope):
    #if we assume no helium ionising photons exist
    if noheiii and v >= vH[2]:
        return 0

    Jf = (v/vH[0])**(-slope)
    return 6.242e11/4.14e-15*Jf #strange units, (eV / h_pl Hz ..), makes integrating easier

#Heating rate integrand SR
def heatG(v,sp,slope):
    return 4*np.pi*Jtest(v,slope)/v*(v - vH[sp])*crsscn(v,sp)
#Ionisation rate integrand
def ionR(v,sp,slope):
    return 4*np.pi*Jtest(v,slope)/v*crsscn(v,sp)

out_table = np.zeros((n_slope,7))
s_arr = np.linspace(slope_min,5,num=n_slope)
#for each spectral slope
for i,s in enumerate(s_arr):
    out_table[i,0] = s
    #for each species (H+, He+, He++)
    for j in range(3):
        #calculate heating and ionisation rates at J21 == 1
        integrand_ion = lambda x: ionR(x,j,s)
        integrand_heat = lambda x: heatG(x,j,s)
        #entries 1,2,3 for ion rates
        out_table[i,j+1] = integrate.quad(integrand_ion,vH[j],np.inf)[0]*1e-21
        #entries 4,5,6 for heating rates
        out_table[i,j+4] = integrate.quad(integrand_heat,vH[j],np.inf)[0]*1e-21

np.savetxt(args.outfile,out_table,fmt='%.6e',delimiter=' ')
