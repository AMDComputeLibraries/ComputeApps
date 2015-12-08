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
#include<strings.h>

#ifndef HAVE_AMP
#include<omp.h>
#endif

#include<math.h>
#include<unistd.h>
#include<sys/time.h>

// Papi Header
#ifdef PAPI
#include "papi.h"
#endif

// I/O Specifiers
#define INFO 1
#define DEBUG 1
#define SAVE 1

#ifdef __cplusplus
#define _CPPSTRING_ extern "C"
#else
#define _CPPSTRING_
#endif //__cplusplus


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
	long n_isotopes;
	long n_gridpoints;
	int lookups;
	char * HM;
} Inputs;

// Function Prototypes
void logo(int version);
_CPPSTRING_ void center_print(const char *s, int width);
_CPPSTRING_ void border_print(void);
void fancy_int(long a);

_CPPSTRING_ NuclideGridPoint ** gpmatrix(size_t m, size_t n);

void gpmatrix_free( NuclideGridPoint ** M );

int NGP_compare( const void * a, const void * b );

_CPPSTRING_ void generate_grids( NuclideGridPoint ** nuclide_grids,
                     long n_isotopes, long n_gridpoints );
_CPPSTRING_ void generate_grids_v( NuclideGridPoint ** nuclide_grids,
                     long n_isotopes, long n_gridpoints );

_CPPSTRING_ void sort_nuclide_grids( NuclideGridPoint ** nuclide_grids, long n_isotopes,
                         long n_gridpoints );

_CPPSTRING_ GridPoint * generate_energy_grid( long n_isotopes, long n_gridpoints,
                                  NuclideGridPoint ** nuclide_grids);

_CPPSTRING_ void set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    long n_isotopes, long n_gridpoints );

int binary_search( NuclideGridPoint * A, double quarry, int n );
#ifndef HAVE_AMP
void calculate_macro_xs(   double p_energy, int mat, long n_isotopes,
                           long n_gridpoints, int * __restrict__ num_nucs,
                           double ** __restrict__ concs,
			   GridPoint * __restrict__ energy_grid,
                           NuclideGridPoint ** __restrict__ nuclide_grids,
						   int ** __restrict__ mats,
						   double * __restrict__ macro_xs_vector);

void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           GridPoint * __restrict__ energy_grid,
                           NuclideGridPoint ** __restrict__ nuclide_grids, int idx,
                           double * __restrict__ xs_vector );
long grid_search( long n, double quarry, GridPoint * A);
#else

#ifdef __cplusplus
void calculate_macro_xs(   double p_energy, int mat, long n_isotopes,
                           long n_gridpoints, int * __restrict__ num_nucs,
                           double ** __restrict__ concs,
			   GridPoint * __restrict__ energy_grid,
                           NuclideGridPoint ** __restrict__ nuclide_grids,
						   int ** __restrict__ mats,
						   double * __restrict__ macro_xs_vector) restrict(amp);

void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           GridPoint * __restrict__ energy_grid,
                           NuclideGridPoint ** __restrict__ nuclide_grids, int idx,
                           double * __restrict__ xs_vector ) restrict(amp);
long grid_search( long n, double quarry, GridPoint * A) restrict(amp);
#endif //__cplusplus
#endif //HAVE_AMP



_CPPSTRING_ int * load_num_nucs(long n_isotopes);
_CPPSTRING_ int ** load_mats( int * num_nucs, long n_isotopes );
_CPPSTRING_ double ** load_concs( int * num_nucs );
_CPPSTRING_ double ** load_concs_v( int * num_nucs );
_CPPSTRING_ int pick_mat(unsigned long * seed);
_CPPSTRING_ double rn(unsigned long * seed);
int rn_int(unsigned long * seed);
void counter_stop( int * eventset, int num_papi_events );
void counter_init( int * eventset, int * num_papi_events );
void do_flops(void);
void do_loads( int nuc,
               NuclideGridPoint ** __restrict__ nuclide_grids,
		       long n_gridpoints );
_CPPSTRING_ Inputs read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
_CPPSTRING_ double rn_v(void);
double round_double( double input );
_CPPSTRING_ unsigned int hash(unsigned char *str, int nbins);
size_t estimate_mem_usage( Inputs in );
_CPPSTRING_ void print_inputs(Inputs in, int nprocs, int version);
_CPPSTRING_ void print_results( Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash );
void binary_dump(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid);
void binary_read(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid);
_CPPSTRING_ double timer();

typedef struct _verifyStruct{
  double p_energy;  
  double macro_xs_vector[5];
  int mat;
} verifyStruct;

#endif
