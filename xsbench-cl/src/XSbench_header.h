/*******************************************************************************
Copyright (c) 2015 Advanced Micro Devices, Inc. 

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/
#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#ifdef WIN32
#include <io.h>
#include <windows.h>
#include "wintime.h"
#ifdef USE_RESTRICT
#else
#define restrict
#endif

#define snprintf _snprintf 
#define vsnprintf _vsnprintf 
#define strcasecmp _stricmp 
#define strncasecmp _strnicmp
typedef unsigned int uint;


#else
#include<strings.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif


#include<math.h>
#include<omp.h>

// Papi Header
#ifdef __PAPI
#include "/usr/local/include/papi.h"
#endif

// OpenCL header files
#include <CL/cl.h>

// OpenCL configuration and getter routines
int SetupOpenCL(char *kernel_filename);
int CleanupOpenCL();
cl_context GetXSBenchContext();
cl_command_queue GetXSBenchCommandQueue();
cl_kernel GetXSBenchKernel();

// number of materials
#define N_MAT     12

// Variable to add extra flops at each lookup from unionized grid.
//#define ADD_EXTRAS
#define EXTRA_FLOPS 0
#define EXTRA_LOADS 0

// I/O Specifiers
#define INFO 1
#define DEBUG 1
#define SAVE 1
#define PRINT_PAPI_INFO 1

// Structures
typedef struct{
	double energy;
	
	double total_xs;
	double elastic_xs;
	double absorbtion_xs;
	double fission_xs;
	double nu_fission_xs;
} NuclideGridPoint;

typedef struct{
	double energy;
	int * xs_ptrs;
} GridPoint;

typedef struct{
	int nthreads;
	int n_isotopes;
	int n_gridpoints;
	int lookups;
	char * HM;
#ifdef __USE_AMD_OCL__
	int tloops;
	bool run_cpu;
#endif
	bool savegrids;
	bool restoregrids;
        char file_name[256];
} Inputs;

// Function Prototypes
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(unsigned long a);

NuclideGridPoint ** gpmatrix(size_t m, size_t n);

void gpmatrix_free( NuclideGridPoint ** M );

int NGP_compare( const void * a, const void * b );

void generate_grids( NuclideGridPoint ** nuclide_grids,
                     int n_isotopes, int n_gridpoints );
void generate_grids_v( NuclideGridPoint ** nuclide_grids,
                     int n_isotopes, int n_gridpoints );

void sort_nuclide_grids( NuclideGridPoint ** nuclide_grids, int n_isotopes,
                         int n_gridpoints );

GridPoint * generate_energy_grid( int n_isotopes, int n_gridpoints,
                                  NuclideGridPoint ** nuclide_grids);
void free_energy_grid(GridPoint *pe_grid);

void set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints );

int binary_search( NuclideGridPoint * A, double quarry, int n );

void save_grids( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints );

GridPoint * restore_grids( NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints, char *file_name);


#if defined (__USE_AMD_OCL__)


#define VERIFICATION_BUFFER
#ifndef VERIFICATION
#undef VERIFICATION_BUFFER
#endif

typedef struct _OCLKernelParamsMacroXSStruct
{
  unsigned lookups;
  int n_isotopes;
  int n_gridpoints;
  int n_mat;
  int e_sort_tile;
  unsigned enlarged_lookups;
} OCLKernelParamsMacroXSS;

void calculate_xs_ocl(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops);

int AllocArrangeCopyConst(OCLKernelParamsMacroXSS *kern_params, const int *num_nucs, const int **mats, const double **concs);
int AllocOutput(OCLKernelParamsMacroXSS *kern_params);
int ReadOutput(OCLKernelParamsMacroXSS *kern_params);
int AllocCopyInputs(OCLKernelParamsMacroXSS *kern_params, double *pp_energy, int *pmat);
int AllocArrangeCopyEnergyGridPtrs(OCLKernelParamsMacroXSS *kern_params, const GridPoint * energy_grid);
int AllocArrangeCopyNucGrid(OCLKernelParamsMacroXSS *kern_params, const NuclideGridPoint ** nuclide_grids);

void* getOCLOutput(void);

void calculate_macro_xs(   double p_energy, int mat, int n_isotopes,
                           int n_gridpoints, int * restrict num_nucs,
                           double ** restrict concs,
			   GridPoint * restrict energy_grid,
                           NuclideGridPoint ** restrict nuclide_grids,
			   int ** restrict mats,
                           double * restrict macro_xs_vector );

void calculate_micro_xs(   double p_energy, int nuc, int n_isotopes,
                           int n_gridpoints,
                           GridPoint * restrict energy_grid,
                           NuclideGridPoint ** restrict nuclide_grids, int idx,
                           double * restrict xs_vector );

void calculate_macro_xs_v( double p_energy, int mat, int n_isotopes,
                           int n_gridpoints, int * restrict num_nucs,
                           double ** restrict concs,
                           GridPoint * restrict energy_grid,
                           NuclideGridPoint ** restrict nuclide_grids,
                           int ** restrict mats,
                           double * restrict macro_xs_vector,
			   int mype, int thread, int i);


#endif

int grid_search( int n, double quarry, GridPoint * A);

int * load_num_nucs(int n_isotopes);
int ** load_mats( int * num_nucs, int n_isotopes );
double ** load_concs( int * num_nucs );
double ** load_concs_v( int * num_nucs );
int pick_mat(unsigned long * seed);
double rn(unsigned long * seed);
int rn_int(unsigned long * seed);
void counter_stop( int * eventset, int num_papi_events );
void counter_init( int * eventset, int * num_papi_events );
void do_flops(void);
void do_loads( int nuc,
               NuclideGridPoint ** restrict nuclide_grids,
	       int n_gridpoints );	
Inputs read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
double rn_v(void);
double round_double( double input );
unsigned int hash(unsigned char *str, int nbins);
unsigned int hash_l(unsigned char *str, int len, int nbins);

#endif
