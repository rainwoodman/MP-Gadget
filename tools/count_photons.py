import bigfile as bf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import sys


import argparse

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("bigfile", help='path to the MP-Gadget output')
ap.add_argument("--output", help='path to save the plots')
ap.add_argument("--blocksize", type=int, default=16777216, help='number of particles to read at a time')
ap.add_argument("--nion", type=int, default=4000, help='photons per stellar baryon')
ap.add_argument("--snapstart",type=int, default=0, help='starting snapshot')
ap.add_argument("--snapend",type=int, default=24, help='ending snapshot')
ap.add_argument("--show-plot", help="show plot with matplotlib",action="store_true")

ns = ap.parse_args()

snapshot_list = np.arange(ns.snapstart,ns.snapend+1)

gas_mass = np.zeros(len(snapshot_list))
star_mass = np.zeros(len(snapshot_list))
gas_xhi = np.zeros(len(snapshot_list))

for i,snap in enumerate(snapshot_list):
    filename = f'{ns.bigfile}/PART_{snap:03d}/'
    
    print('')
    print(f'SNAP {snap}')

    fin = bf.File(filename)

    data = bf.Dataset(fin["0/"],['Mass','NeutralHydrogenFraction'])
    stta = bf.Dataset(fin["4/"],['Mass'])

    mas = data['Mass']
    xhi = data['NeutralHydrogenFraction']

    stm = stta['Mass']

    pread=0
    while pread < mas.size:
        xbuf = xhi.read(pread,ns.blocksize)
        mbuf = mas.read(pread,ns.blocksize)

        gas_mass[i] += mbuf.sum()
        gas_xhi[i] += (mbuf*xbuf).sum()
    
        pread = pread + mbuf.shape[0]

        progress = str(100.*pread/mas.size)+"  % complete            "
        print(progress,end='\r')


    pread=0
    while pread < stm.size:
        mbuf = stm.read(pread,ns.blocksize)
        star_mass[i] += mbuf.sum()
    
        pread = pread + mbuf.shape[0]

        progress = str(100.*pread/stm.size)+"  % complete            "
        print(progress,end='\r')


    fin.close()

gas_xhi /= gas_mass

star_photons = star_mass * ns.nion / gas_mass


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(star_photons,label='stellar mass * Nion / gas mass')
ax.plot(1 - gas_xhi,label='ionised fraction')
ax.set_ylim(0,1)
ax.legend

if ns.show_plot:
    plt.show()

fig.savefig(ns.output)
