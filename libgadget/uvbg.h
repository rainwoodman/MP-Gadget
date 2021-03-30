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

extern struct UVBGgrids_type UVBGgrids; 

int grid_index(int i, int j, int k, ptrdiff_t strides[3]);
void calculate_uvbg();
void save_uvbg_grids(int SnapshotFileCount, PetaPM * pm);
void set_uvbg_params(ParameterSet * ps);

#endif
