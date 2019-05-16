#ifndef UVBG_H
#define UVBG_H

#include <fftw3.h>

typedef struct UVBGgrids {
    ptrdiff_t *slab_nix;
    ptrdiff_t *slab_ix_start;
    ptrdiff_t *slab_n_complex;

    float *deltax;
    fftwf_complex *deltax_filtered;
    float *uvphot;
    fftwf_complex *uvphot_filtered;
    float *xHI;
    float *J21;
    float *z_at_ionization;
    float *J21_at_ionization;

    fftwf_plan plan_dft_r2c;
    fftwf_plan plan_dft_c2r;

    float volume_weighted_global_xHI;
    float mass_weighted_global_xHI;
} UVBGgrids;

void calculate_uvbg();

#endif
