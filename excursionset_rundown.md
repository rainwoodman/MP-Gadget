
#21cmFAST in MP-Gadget rundown

*Syncpoints are set up every 10 Myr or so (free parameter) in order to trigger the excursion set.
*When we reach one of the UV syncpoints, three particle meshes are created for mass, stellar mass, and star formation rate, with the same communication and layout as the long-range force code
*We then step through filters of different scales PFFT and count photons in each cell (from stellar mass * photons per baryon) to determine which cells are ionised, and calculate J21 if so.
*This J21 is then converted into ionisation and photoheating rates at a certain spectral slope (free parameter) and passed into the existing cooling functions.


##Notes:

*new parameters:
..*ExcursionSetFlag : 0 or 1 to disable/enable the excursion set
..*ExcursionSetZStop : redshift at which the cooling code reverts to global ionising background
..*ExcursionSetZStart : redshift at which the excursion set starts
..*UVBGdim : dimension of the excursion set grids (can be different from the force grids)
..*ReionRBubbleMax : largest filter radius for the excursion set
..*ReionRBubbleMin : smallest filter radius for the excursion set
..*ReionDeltaRFactor : relative size of subsequent filter radii
..*ReionGammaHaloBias : **UNUSED** bias for calculating reionisation feedback in halos
..*ReionNionPhotPerBary : photons per stellar baryon, usually set to 4000 (TODO: add citation)
..*AlphaUV : spectral slope for ionising radiation, currently a global constant
..*EscapeFraction : escape fraction of photons from ionising sources
..*UVBGTimestep : time in Myr between UVBG timesteps
..*ReionFilterType : Filter to use with 21CMFAST, 0 = real-space top-hat, 1 = k-space top-hat, 2 = gaussian
..*RtoMFilterType : Filter to use with 21CMFAST radius to mass, 0 = top-hat, 1 = gaussian
..*J21CoeffFile : path to the table containing photo-ionisaiton and heating rates versus spectral slope at J21 == 1

*With the small examples I have run so far (256^3 particles at 40Mpc) the excursion set does not take much time compared to the rest of the run

