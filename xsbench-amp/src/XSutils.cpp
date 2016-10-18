/*******************************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. 

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

#include "XSbench_header.h"

// Allocates nuclide matrix
NuclideGridPoint * gpmatrix(size_t m, size_t n)
{
  int i,j;
  NuclideGridPoint * full = new NuclideGridPoint[m*n];

  return full;
	
}

// Frees nuclide matrix
void gpmatrix_free( NuclideGridPoint * M )
{
  delete M;
}

// Compare function for two grid points. Used for sorting during init
int NGP_compare( const void * a, const void * b )
{
  NuclideGridPoint *i, *j;

  i = (NuclideGridPoint *) a;
  j = (NuclideGridPoint *) b;

  if( i->energy > j->energy )
    return 1;
  else if ( i->energy < j->energy)
    return -1;
  else
    return 0;
}



// Binary Search function for nuclide grid
// Returns ptr to energy less than the quarry that is closest to the quarry
int binary_search( NuclideGridPoint * A, double quarry, int n )
{
  int min = 0;
  int max = n-1;
  int mid;
	
  // checks to ensure we're not reading off the end of the grid
  if( A[0].energy > quarry )
    return 0;
  else if( A[n-1].energy < quarry )
    return n-2;
	
  // Begins binary search	
  while( max >= min )
    {
      mid = min + floor( (max-min) / 2.0);
      if( A[mid].energy < quarry )
	min = mid+1;
      else if( A[mid].energy > quarry )
	max = mid-1;
      else
	return mid;
    }
  return max;
}

// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn(unsigned long * seed)
{
  double ret;
  unsigned long n1;
  unsigned long a = 16807;
  unsigned long m = 2147483647;
  n1 = ( a * (*seed) ) % m;
  *seed = n1;
  ret = (double) n1 / m;
  return ret;
}



// RNG Used for Verification Option.
// This one has a static seed (must be set manually in source).
// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn_v(void)
{
  static unsigned long seed = 1337;
  double ret;
  unsigned long n1;
  unsigned long a = 16807;
  unsigned long m = 2147483647;
  n1 = ( a * (seed) ) % m;
  seed = n1;
  ret = (double) n1 / m;
  return ret;
}

unsigned int hash(unsigned char *str, int nbins)
{
  unsigned int hash = 5381;
  int c;

  while (c = *str++)
    hash = ((hash << 5) + hash) + c;

  return hash % nbins;
}

size_t estimate_mem_usage( Inputs in )
{
  size_t single_nuclide_grid = in.n_gridpoints * sizeof( NuclideGridPoint );
  size_t all_nuclide_grids   = in.n_isotopes * single_nuclide_grid;
  size_t size_GridPoint      = sizeof(GridPoint) + in.n_isotopes*sizeof(int);
  size_t size_UEG            = in.n_isotopes*in.n_gridpoints * size_GridPoint;
  size_t memtotal;

  memtotal          = all_nuclide_grids + size_UEG;
  all_nuclide_grids = all_nuclide_grids / 1048576;
  size_UEG          = size_UEG / 1048576;
  memtotal          = memtotal / 1048576;
  return memtotal;
}

void binary_dump(long n_isotopes, long n_gridpoints, NuclideGridPoint * nuclide_grids, GridPoint * energy_grid)
{
  FILE * fp = fopen("XS_data.dat", "wb");
  // Dump Nuclide Grid Data
  //for( long i = 0; i < n_isotopes; i++ )
  fwrite(nuclide_grids, sizeof(NuclideGridPoint), n_isotopes * n_gridpoints, fp);
  // Dump UEG Data
  for( long i = 0; i < n_isotopes * n_gridpoints; i++ )
    {
      // Write energy level
      fwrite(&energy_grid[i].energy, sizeof(double), 1, fp);

      // Write index data array (xs_ptrs array)
      fwrite(energy_grid[i].xs_ptrs, sizeof(int), n_isotopes, fp);
    }

  fclose(fp);
}

void binary_read(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid)
{
  int stat;
  FILE * fp = fopen("XS_data.dat", "rb");
  // Read Nuclide Grid Data
  //for( long i = 0; i < n_isotopes; i++ )
  stat = fread(nuclide_grids, sizeof(NuclideGridPoint), n_isotopes * n_gridpoints, fp);
  // Dump UEG Data
  for( long i = 0; i < n_isotopes * n_gridpoints; i++ )
    {
      // Write energy level
      stat = fread(&energy_grid[i].energy, sizeof(double), 1, fp);

      // Write index data array (xs_ptrs array)
      stat = fread(energy_grid[i].xs_ptrs, sizeof(int), n_isotopes, fp);
    }

  fclose(fp);

}

double timer()
{
  struct timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec + time.tv_usec / 1000000.0;
}
