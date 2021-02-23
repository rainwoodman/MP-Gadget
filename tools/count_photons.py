'''
This code compares the global ratio of stellar mass to gas mass
with the global neutral fraction over time, as a test of the
global statistics of reionisation.
'''

import argparse
import bigfile as bf
import numpy as np
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("bigfile", help='path to the MP-Gadget output directory')
ap.add_argument("--output", help='path to save the plots')
ap.add_argument("--blocksize", type=int, default=16777216,
                help='number of particles to read at a time')
ap.add_argument("--nion", type=int, default=4000, help='photons per stellar baryon')
ap.add_argument("--fesc", type=float, default=1., help='ionising photon escape fraction')
ap.add_argument("--snapstart", type=int, default=0, help='starting snapshot')
ap.add_argument("--snapend", type=int, default=24, help='ending snapshot')
ap.add_argument("--show-plot", help="show plot with matplotlib", action="store_true")

ns = ap.parse_args()

#set up the list of snapshots, and redshifts
snapshot_list = np.arange(ns.snapstart, ns.snapend+1)
time_list = np.loadtxt(f'{ns.bigfile}/Snapshots.txt', dtype=float, ndmin=2)
time_list = time_list[snapshot_list,1]
redshift_list = 1/time_list - 1

#define mass / neutral fraction arrays
gas_mass = np.zeros(len(snapshot_list))
star_mass = np.zeros(len(snapshot_list))
gas_xhi = np.zeros(len(snapshot_list))

for i, snap in enumerate(snapshot_list):
    #read in the particle file
    filename = f'{ns.bigfile}/PART_{snap:03d}/'

    print('')

    fin = bf.File(filename)

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

#divide my total mass for mass-weighted neutral fraction
gas_xhi /= gas_mass

#multiply stellar mass by photons per stellar baryon
#and take ratio with gas mass, giving an estimate of
#total number of photons released per hydrogen atom
Y_He = 1 - 0.76
star_photons = star_mass * ns.nion * ns.fesc / gas_mass / (1 - 0.75*Y_He)

#plot the neutral fraction and photon ratio vs snapshot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(redshift_list, star_photons, label='stellar mass * Nion / gas mass')
ax.plot(redshift_list, 1 - gas_xhi, label='ionised fraction')
ax.set_ylim(0, 1)
ax.legend()
ax.set_xlabel('snapshot')
ax.set_ylabel('ratio')

if ns.show_plot:
    plt.show()

if ns.output is not None:
    fig.savefig(ns.output)
