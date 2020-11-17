#ifndef UVBG_H
#define UVBG_H

#include <pfft.h>
#include "utils/paramset.h"

struct UVBGgrids_type {
    ptrdiff_t *slab_ni;
    ptrdiff_t *slab_no;
    ptrdiff_t *slab_i_start;
    ptrdiff_t *slab_o_start;
    ptrdiff_t *slab_n_complex;

    //communicator for PFFT
    MPI_Comm comm_cart_2d;

    float *J21;
    float *prev_stars;

    float *stars;

    float *deltax;
    pfftf_complex *deltax_filtered;
    float *stars_slab;
    pfftf_complex *stars_slab_filtered;
    float *sfr;
    pfftf_complex *sfr_filtered;
    float *xHI;
    float *z_at_ionization;
    float *J21_at_ionization;

    pfftf_plan plan_dft_r2c;
    pfftf_plan plan_dft_c2r;

    double last_a;  //< Last called expansion factor

    float volume_weighted_global_xHI;
    float mass_weighted_global_xHI;

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
int grid_index(int i, int j, int k, int dim, int longdim, index_type type);
double time_to_present(double a);
void calculate_uvbg();
void malloc_permanent_uvbg_grids();
void free_permanent_uvbg_grids();
void save_uvbg_grids(int SnapshotFileCount);
void read_star_grids(int snapnum);
void set_uvbg_params(ParameterSet * ps);

#endif
