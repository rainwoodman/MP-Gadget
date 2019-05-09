#include <fftw3.h>

typedef struct UVBGgrids {
    ptrdiff_t *slab_nix;
    ptrdiff_t *slab_ix_start;
    ptrdiff_t *slab_n_complex;

    float *deltax;
    fftwf_complex *deltax_filtered;
    // float *uvphot;
    // fftwf_complex *uvphot_filtered;
    // float *xHI;
    // float *J21;
} UVBGgrids;

void calculate_uvbg();
