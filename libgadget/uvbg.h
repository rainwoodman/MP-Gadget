#ifndef UVBG_H
#define UVBG_H

#include <pfft.h>
#include "petapm.h"
#include "utils/paramset.h"

struct UVBGgrids_type {
    //These are the sizes and offsets of ALL ranks used in populate_grids
    //TODO: gather this from regions in populate_grids instead of assign_slabs
    //and then these can be removed
    ptrdiff_t *slab_ni;
    ptrdiff_t *slab_no;
    ptrdiff_t *slab_i_start;
    ptrdiff_t *slab_o_start;
    ptrdiff_t *slab_n_complex;

    //communicator for PFFT
    MPI_Comm comm_cart_2d;

    //Using PetaPM regions here to store local sizes, offsets and strides
    PetaPMRegion local_r_region;
    PetaPMRegion local_c_region;
    
    float *J21;
    float *prev_stars;

    float *stars;

    double *deltax;
    pfft_complex *deltax_filtered;
    double *stars_slab;
    pfft_complex *stars_slab_filtered;
    double *sfr;
    pfft_complex *sfr_filtered;
    
    float *xHI;
    float *z_at_ionization;
    float *J21_at_ionization;

    pfft_plan plan_dft_r2c;
    pfft_plan plan_dft_c2r;

    double last_a;  //< Last called expansion factor

    double volume_weighted_global_xHI;
    double mass_weighted_global_xHI;

    //TODO(jdavies): remove this
    //this is a check for debug messages so i don't print them a million times
    int debug_printed;
};

extern struct UVBGgrids_type UVBGgrids; 

typedef enum index_type {
    INDEX_PADDED = 5674,
    INDEX_REAL,
    INDEX_COMPLEX_HERM,
} index_type;

int pos_to_ngp(double x, double Offset, double side, int nx);
int grid_index(int i, int j, int k, ptrdiff_t strides[3]);
double time_to_present(double a);
void calculate_uvbg();
void calculate_uvbg_new();
void malloc_permanent_uvbg_grids();
void free_permanent_uvbg_grids();
void save_uvbg_grids(int SnapshotFileCount);
void read_star_grids(int snapnum);
void set_uvbg_params(ParameterSet * ps);

#endif
