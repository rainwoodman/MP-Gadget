#ifndef UVBG_H
#define UVBG_H

#include <pfft.h>
#include "petapm.h"
#include "utils/paramset.h"

struct UVBGgrids_type {
    float *J21;
    float *xHI;

    double volume_weighted_global_xHI;
    double mass_weighted_global_xHI;
};

//extern struct UVBGgrids_type UVBGgrids; 

void calculate_uvbg(PetaPM * pm_mass, PetaPM * pm_star, PetaPM * pm_sfr, int WriteSnapshot, int SnapshotFileCount);
void set_uvbg_params(ParameterSet * ps);

#endif
