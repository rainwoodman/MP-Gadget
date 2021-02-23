'''
simple grid plotting code
takes bigfiles with datasets named {prefix}_grid_z{redshift}
like those generated from get_xgrids.py and plots slices of them
in a number of grid types x number of redshifts grid
'''

import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import gridspec as gs
import bigfile as bf

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("gridfile", help='path to the grid bigfile')
ap.add_argument("--output", help='path to save the plots')
ap.add_argument("--redshifts", default="8,7", help='comma separated list of redshifts to plot')
ap.add_argument("--gridnames", default="d,xhi"
                , help='comma separated list of prefixes of the desired grids')
ap.add_argument("--slice-idx", type=int, default=5, help='index of the plotted slice')
ap.add_argument("--slice-depth", type=int, default=5, help='depth of the slice')
ap.add_argument("--show-plot", help="show plot with matplotlib", action="store_true")

ns = ap.parse_args()
redshifts = ns.redshifts.split(",")
grid_prefix = ns.gridnames.split(",")

n_snap = len(redshifts)
n_types = len(grid_prefix)

#mapping of grids to plot position and prefix/redshift number in the array
grid_map = np.zeros((n_snap*n_types, 2), dtype=int)

#build the list of grid names by redshift and variable
#and save a map of where it exists in redshift/type space
grid_names = []
count = 0
for i, pref in enumerate(grid_prefix):
    for j, z in enumerate(redshifts):
        grid_names.append(f'{pref}_grid_z{z}')
        grid_map[count,0] = i
        grid_map[count,1] = j
        count += 1

#open the file
fin = bf.File(ns.gridfile)
dset = bf.Dataset(fin, grid_names)

#build list of grid slices to plot
slab_idx = lambda x, i, d: x[i-d//2:i+d//2+1,:,:].mean(axis=0)
slab_list = []

#colour limits for plots
c_lims = np.zeros((n_types, 2))

for i, grid in enumerate(grid_names):
    #read in the grid
    buf = dset[grid].read(0, dset[grid].size)

    #assuming a cube here
    length = int(np.cbrt(buf.size))
    buf = buf.reshape((length, length, length))
    #grab the required slice
    buf = slab_idx(buf, ns.slice_idx, ns.slice_depth)
    slab_list.append(buf)

    #save colour limits
    c_lims[grid_map[i,0],0] = min(np.amin(buf), c_lims[grid_map[i,0],0])
    c_lims[grid_map[i,0],1] = max(np.amax(buf), c_lims[grid_map[i,0],1])

fin.close()

#define length and layout of the plot grid
ext = [0, length, 0, length]
gs = gs.GridSpec(n_types, n_snap)

fig = plt.figure(figsize=(8, 6))

#put each slice in the correct place based on redshift / type
for i, slab in enumerate(slab_list):
    pos = grid_map[i, :]
    ax = fig.add_subplot(gs[pos[0],pos[1]])

    im = ax.imshow(slab, cmap=cm.Purples, norm=matplotlib.colors.LogNorm()
                   , origin='lower', extent=ext)
    plt.colorbar(im)
    ax.set_title(grid_names[i])

fig.tight_layout()
fig.savefig(ns.output)

if ns.show_plot:
    plt.show()
