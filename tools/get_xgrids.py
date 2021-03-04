"""
	James' hacked up grid maker
	created from parts of the reionisation model

    Example usage: python get_xgrids.py  output/PART_005 grids/example_grid_z10 --pos=0/Position --weight=0/Mass

    For a 1Mpc resolution grid of gas particle overdensity at the sixth snapshot and save it to grids/example_grid_z10
"""

import argparse
import bigfile
from pmesh.pm import ParticleMesh, RealField, ComplexField
import logging
from mpi4py import MPI
import numpy

ap = argparse.ArgumentParser("get_xgrids.py")
ap.add_argument("fastpm", help='Particle data')
ap.add_argument("output", help='Name of bigfile to store the mesh')
ap.add_argument("--pos", help='Name of the position dataset in the bigfile')
ap.add_argument("--weight", help='Name of the weighting dataset in the bigfile')
ap.add_argument("--dataset", default='grid', help='name of the dataset to write to')
ap.add_argument("--resolution", type=float, default=1.0, help='resolution in Mpc/h')
ap.add_argument("--chunksize", type=int, default=1024*1024*32, help='number of particle to read at once')
logger = logging
logging.basicConfig(level=logging.INFO)

def main():
    ns = ap.parse_args()
    comm = MPI.COMM_WORLD

    ff = bigfile.BigFileMPI(comm, ns.fastpm)
    BoxSize = ff['Header'].attrs['BoxSize'] / 1000
    Redshift = 1/ff['Header'].attrs['Time'] - 1

    Nmesh = int(BoxSize / ns.resolution)
    # round it to 8.
    Nmesh -= Nmesh % 8

    if comm.rank == 0:
        logger.info("source = %s", ns.fastpm)
        logger.info("output = %s", ns.output)
        logger.info("BoxSize = %g", BoxSize)
        logger.info("Redshift = %g", Redshift)
        logger.info("Nmesh = %g", Nmesh)

    pm = ParticleMesh([Nmesh, Nmesh, Nmesh], BoxSize, comm=comm)

    real = RealField(pm)
    real[...] = 0

    #particle field
    pcle = RealField(pm)
    pcle[...] = 0

    with ff[ns.pos] as ds, ff[ns.weight] as dx:
        logger.info(ds.size)
        for i in range(0, ds.size, ns.chunksize):
            sl = slice(i, i + ns.chunksize)
            pos = ds[sl] / 1000. #convert to Mpc
            nh = dx[sl]
            layout = pm.decompose(pos)
            lpos = layout.exchange(pos)
            real.paint(lpos, mass=nh, hold=True)
            pcle.paint(lpos,hold=True)
            print(pos.min(),pos.max())

    mean = real.cmean()

    if comm.rank == 0:
        logger.info("mean particle per cell = %s", mean)

    #if mass, create density contrast grid (1 + delta)
    if 'Mass' in ns.weight:
        real[...] /= mean
    #else, divide by particle grid for average in cell
    #TODO: change to mass weighting, from particle weighting
    else:
        real[...] /= pcle[...]

    buffer = numpy.empty(real.size, real.dtype)
    real.sort(out=buffer)
    if comm.rank == 0:
        logger.info("sorted for output")

    with bigfile.BigFileMPI(comm, ns.output, create=True) as ff:
        with ff.create_from_array(ns.dataset, buffer) as bb:
            bb.attrs['BoxSize'] = BoxSize
            bb.attrs['Redshift'] = Redshift
            bb.attrs['Nmesh'] = Nmesh

    if comm.rank == 0:
        logger.info("done. written at %s", ns.output)

if __name__ == '__main__':
    main()
