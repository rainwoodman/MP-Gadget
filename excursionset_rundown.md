
#21cmFAST in MP-Gadget rundown

*Syncpoints are set up every 10 Myr or so (free parameter) in order to trigger the excursion set.
*Whenever a star particle is created, mass is added to a spatial grid where it formed.
*When we reach one of the UV syncpoints, stellar mass and total mass grids are created by nearest gridpoint, and decomposed into a slab for each task.
*We then step through filters of different scales using FFTW and count photons in each cell (from stellar mass * photons per baryon) to determine which cells are ionised, and calculate J21 if so.
*This J21 is then converted into ionisation and photoheating rates at a certain spectral slope (free parameter) and passed into the existing cooling functions.


##Notes:

*Conversion factors from J21 to rates are passed in a table file via the same parameter as the TREECOOL files, if the excursion set flag is on or off it will treat this file as a conversion table or TREECOOL file repsectively.
*These additions add a couple of global grids for J21 and stellar mass. While these can be relatively low-resolution, it might result in a lot of memory usage with many cores.
*With the small examples I have run so far (256^3 particles at 40Mpc) the excursion set does not take much time compared to the rest of the run (~3 seconds per call with 16 cores)

