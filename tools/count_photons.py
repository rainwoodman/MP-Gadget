'''
This code compares the global ratio of stellar mass to gas mass
with the global neutral fraction over time, as a test of the
global statistics of reionisation.
'''

import argparse
import bigfile as bf
import numpy as np
from matplotlib import pyplot as plt
from os.path import exists
from astropy import cosmology, constants as C, units as U
from scipy import integrate

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("bigfile", help='path to the MP-Gadget output directory')
ap.add_argument("--output", help='path to save the plots')
ap.add_argument("--blocksize", type=int, default=16777216,
                help='number of particles to read at a time')
ap.add_argument("--nion", type=int, default=4000, help='photons per stellar baryon')
ap.add_argument("--fesc-n", type=float, help='ionising photon escape fraction norm')
ap.add_argument("--fesc-s", type=float, help='ionising photon escape fraction halo mass scaling')
ap.add_argument("--snapstart", type=int, default=0, help='starting snapshot')
ap.add_argument("--snapend", type=int, default=-1, help='ending snapshot')
ap.add_argument("--show-plot", help="show plot with matplotlib", action="store_true")

ns = ap.parse_args()

h = 0.7186
Om0 = 0.2814
Ob0 = 0.0464
m_nu = [0., 0., 0.]*U.Unit('eV')
Tcmb0 = 2.7255
Ode0 = 0.7186

cosmo = cosmology.FlatLambdaCDM(H0=h*100,Om0=Om0,Ob0=Ob0,Tcmb0=Tcmb0,m_nu=m_nu)

#set up the list of snapshots, and redshifts
time_list = np.loadtxt(f'{ns.bigfile}/Snapshots.txt', dtype=float, ndmin=2)
if ns.snapend == -1:
    ns.snapend = int(time_list[-1,0])
snapshot_list = np.arange(ns.snapstart, ns.snapend + 1)
time_list = time_list[snapshot_list,1]
redshift_list = 1/time_list - 1

#define mass / neutral fraction arrays
gas_mass = np.zeros(len(snapshot_list))
star_mass = np.zeros(len(snapshot_list))
gas_xhi = np.zeros(len(snapshot_list))
phot_fof = np.zeros(len(snapshot_list))

#snapshot mask to account for missing snapshots
snap_mask = np.ones(len(snapshot_list),dtype=bool)

for i, snap in enumerate(snapshot_list):
    #read in the particle file
    filename = f'{ns.bigfile}/PART_{snap:03d}/'
    fofname = f'{ns.bigfile}/PIG_{snap:03d}/'

    if not exists(filename) or not exists(fofname):
        snap_mask[i] = False
        continue

    print('')

    fin = bf.File(filename)
    
    boxsize = fin['Header'].attrs['BoxSize']

    #we need gas mass, stellar mass, and neutral fraction
    data = bf.Dataset(fin["0/"], ['Mass', 'NeutralHydrogenFraction'])
    stta = bf.Dataset(fin["4/"], ['Mass'])

    mas = data['Mass']
    xhi = data['NeutralHydrogenFraction']

    stm = stta['Mass']

    #read the particles in chunks and add to the total mass
    #weighting the neutral fraction by mass
    pread = 0
    while pread < mas.size:
        xbuf = xhi.read(pread, ns.blocksize)
        mbuf = mas.read(pread, ns.blocksize)

        gas_mass[i] += mbuf.sum()
        gas_xhi[i] += (mbuf*xbuf).sum()

        pread = pread + mbuf.shape[0]

        progress = 100. * pread / mas.size
        print(f'Snapshot {snap} gas, {progress:6.3f} % complete', end='\r')


    #repeat for stellar mass
    pread = 0
    while pread < stm.size:
        mbuf = stm.read(pread, ns.blocksize)
        star_mass[i] += mbuf.sum()

        pread = pread + mbuf.shape[0]

        progress = 100. * pread / stm.size
        print(f'Snapshot {snap:3d} stars, {progress:6.3f} % complete', end='\r')


    fin.close()
    #read in the fof file
    if ns.fesc_n is not None and ns.fesc_s is not None:
        fin = bf.File(fofname)
        
        data = fin['FOFGroups/MassByType']
        star_buf = np.transpose(data[:])[4]
        
        data = fin['FOFGroups/Mass']
        mass_buf = data[:] / h #already in 1e10 solar so the normalisation is at 1
    
        phot_fof[i] += (star_buf * np.minimum(ns.fesc_n * ((mass_buf)**ns.fesc_s),1)).sum()

        fin.close()

#divide my total mass for mass-weighted neutral fraction
gas_xhi /= gas_mass

#h adjustments
boxsize = boxsize*U.Unit('kpc') / h

critdens = cosmo.critical_density0

dm_mass = (critdens * cosmo.Odm0 * boxsize**3).to('M_sun')
b_mass = (critdens * cosmo.Ob0 * boxsize**3).to('M_sun')
b_plot = b_mass.value

#multiply stellar mass by photons per stellar baryon
#and take ratio with gas mass, giving an estimate of
#total number of photons released per hydrogen atom
Y_He = 1 - 0.76
if ns.fesc_n is not None:
    star_photons = star_mass * ns.nion * ns.fesc_n / gas_mass / (1 - 0.75*Y_He)
    star_photons = star_photons[snap_mask]
    if ns.fesc_s is not None:
        fof_photons = phot_fof * ns.nion / gas_mass / (1-0.75*Y_He)
        fof_photons = fof_photons[snap_mask]

gas_xhi = gas_xhi[snap_mask]
redshift_list = redshift_list[snap_mask]

#THOMSON CROSS SECTION TAKEN FROM MERAXES PACKAGE
thomson_cross_section = 6.652e-25 * U.cm * U.cm
density_H = 1.88e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3
# density_He = 0.19e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3
# Not what is in Whythe et al.!
density_He = 0.148e-7 * cosmo.Ob0 * cosmo.h**2 / 0.022 * U.cm**-3

cosmo_factor = lambda z: C.c * (1+z)**2 / cosmo.H(z) * thomson_cross_section

def d_te_postsim(z):
    """This is d/dz scattering depth for redshifts greater than the final
    redshift of the run.
    N.B. THIS ASSUMES THAT THE NEUTRAL FRACTION IS ZERO BY THE END OF THE
    INPUT RUN!
    """
    if z <= 4:
        return (cosmo_factor(z) * (density_H + 2.0*density_He)).decompose()
    else:
        return (cosmo_factor(z) * (density_H + density_He)).decompose()

def d_te_sim(z, xHII):
    """This is d/dz scattering depth for redshifts covered by the run.
    """
    prefac = cosmo_factor(z)
    return (prefac * (density_H*xHII + density_He*xHII)).decompose()

xHII = 1 - np.copy(gas_xhi)[::-1]
xHII = np.minimum(xHII,1)
z_integ = redshift_list[::-1]
scattering_depth = np.zeros_like(z_integ)

for i,z in enumerate(z_integ):
    post_sim_contrib = integrate.quad(d_te_postsim, 0, np.minimum(z,z_integ[0]))[0]

    sim_contrib = integrate.trapz(d_te_sim(z_integ,xHII)[:i+1],z_integ[:i+1],axis=0)

    scattering_depth[i] = sim_contrib + post_sim_contrib

print(f'sigma shape {scattering_depth.shape}')

#plot the neutral fraction and photon ratio vs snapshot
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(121)
if ns.fesc_n is not None:
    ax.plot(redshift_list, star_photons, label=f'stellar photons (flat fesc={ns.fesc_n})')
    ax.plot(redshift_list, star_photons/ns.fesc_n, label='stellar photons (flat fesc=1)')
    if ns.fesc_s is not None:
        ax.plot(redshift_list, fof_photons, label=f'stellar photons (fn = {ns.fesc_n}, fs = {ns.fesc_s})')
ax.plot(redshift_list, 1 - gas_xhi, label='ionised fraction')
ax.set_ylim(0, 1)
ax.set_xlim(5,12)
ax.legend()
ax.set_xlabel('snapshot')
ax.set_ylabel('ratio')

P18_mean = 0.0544
P18_bounds = P18_mean + np.array([-0.0081,0.0070])

ax = fig.add_subplot(122)
ax.plot(z_integ,scattering_depth)
ax.axhline(P18_mean,linestyle=':',color='k')
ax.fill_between(np.array([0,np.amax(z_integ)]),P18_bounds[0],P18_bounds[1],facecolor='black',alpha=0.2,label='Planck18')
ax.set_xlim(5,12)
ax.legend()
ax.set_xlabel('redshift')
ax.set_ylabel('thomson scattering depth')


fig.suptitle('Parametric Model',fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.92)

if ns.output is not None:
    fig.savefig(ns.output)

if ns.show_plot:
    plt.show()

