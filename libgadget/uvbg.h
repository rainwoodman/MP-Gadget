#ifndef UVBG_H
#define UVBG_H

#include <pfft.h>
#include "petapm.h"
#include "utils/paramset.h"

struct UVBGgrids_type {
    float *J21;

    float *xHI;
    //float *z_at_ionization;
    //float *J21_at_ionization;

    //double last_a;  //< Last called expansion factor

    double volume_weighted_global_xHI;
    double mass_weighted_global_xHI;

    //TODO(jdavies): remove this
    //this is a check for debug messages so i don't print them a million times
    int debug_printed;
};

extern struct UVBGgrids_type UVBGgrids; 

int grid_index(int i, int j, int k, ptrdiff_t strides[3]);
double time_to_present(double a);
void calculate_uvbg();
void save_uvbg_grids(int SnapshotFileCount);
void set_uvbg_params(ParameterSet * ps);

#endif
