#pragma once
namespace argtools {

int _roundup(const int value, const int radix);

void init_params ( bool *mode_debug, bool *mode_cuda, bool *mode_pagelock, bool *mode_cuda_um,
                   int *numloops, int *width, int *height, int *channels,
                   int (*dim_block)[3], int (*dim_grid)[2] );

void adjust_cudaparams ( int (*dim_block)[3], int (*dim_grid)[2], bool *mode_pagelock, bool *mode_cuda_um,
                         const bool mode_cuda, const int width, const int height, const int channels );

void get_args ( bool *mode_debug, bool *mode_cuda, bool *mode_pagelock, bool *mode_cuda_um,
                int *numloops, int *width, int *height, int *channels,
                int (*dim_block)[3], int (*dim_grid)[2],
                const int argc, const char **argv );

void disp_args ( const bool mode_debug, const bool mode_cuda, const bool mode_pagelock, const bool mode_cuda_um,
                 const int numloops, const int width, const int height, const int channels,
                 const int dim_block[3], const int dim_grid[2] );

}