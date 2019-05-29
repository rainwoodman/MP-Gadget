#ifndef UVBG_H
#define UVBG_H

#include <fftw3.h>

// TODO(smutch): This should be a parameter
const int uvbg_dim = 64;

struct UVBGgrids_type {
    ptrdiff_t *slab_nix;
    ptrdiff_t *slab_ix_start;
    ptrdiff_t *slab_n_complex;
    
    float *J21;
    float *stars;

    float *deltax;
    fftwf_complex *deltax_filtered;
    float *stars_slab;
    fftwf_complex *stars_slab_filtered;
    float *xHI;
    float *z_at_ionization;
    float *J21_at_ionization;

    fftwf_plan plan_dft_r2c;
    fftwf_plan plan_dft_c2r;

    float volume_weighted_global_xHI;
    float mass_weighted_global_xHI;
};

extern struct UVBGgrids_type UVBGgrids; 

typedef enum index_type {
    INDEX_PADDED = 5674,
    INDEX_REAL,
    INDEX_COMPLEX_HERM,
} index_type;

int pos_to_ngp(double x, double side, int nx);
int grid_index(int i, int j, int k, int dim, index_type type);
void calculate_uvbg();
void malloc_permanent_uvbg_grids();
void free_permanent_uvbg_grids();

#endif
