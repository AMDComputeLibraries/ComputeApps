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

#ifdef DOMPI
#include<mpi.h>
#endif

// Generates randomized energy grid for each nuclide
// Note that this is done as part of initialization (serial), so
// rand() is used.
void generate_grids( NuclideGridPoint * nuclide_grids,
                     long n_isotopes, long n_gridpoints ) {
  for( long i = 0; i < n_isotopes; i++ )
    for( long j = 0; j < n_gridpoints; j++ )
      {
	nuclide_grids[i*n_gridpoints + j].energy       =((double)rand()/(double)RAND_MAX);
	nuclide_grids[i*n_gridpoints + j].total_xs     =((double)rand()/(double)RAND_MAX);
	nuclide_grids[i*n_gridpoints + j].elastic_xs   =((double)rand()/(double)RAND_MAX);
	nuclide_grids[i*n_gridpoints + j].absorbtion_xs=((double)rand()/(double)RAND_MAX);
	nuclide_grids[i*n_gridpoints + j].fission_xs   =((double)rand()/(double)RAND_MAX);
	nuclide_grids[i*n_gridpoints + j].nu_fission_xs=((double)rand()/(double)RAND_MAX);
      }
}

// Verification version of this function (tighter control over RNG)
void generate_grids_v( NuclideGridPoint * nuclide_grids,
		       long n_isotopes, long n_gridpoints ) {
  for( long i = 0; i < n_isotopes; i++ )
    for( long j = 0; j < n_gridpoints; j++ )
      {
	nuclide_grids[i*n_gridpoints + j].energy       = rn_v();
	nuclide_grids[i*n_gridpoints + j].total_xs     = rn_v();
	nuclide_grids[i*n_gridpoints + j].elastic_xs   = rn_v();
	nuclide_grids[i*n_gridpoints + j].absorbtion_xs= rn_v();
	nuclide_grids[i*n_gridpoints + j].fission_xs   = rn_v();
	nuclide_grids[i*n_gridpoints + j].nu_fission_xs= rn_v();
      }
}

// Sorts the nuclide grids by energy (lowest -> highest)
void sort_nuclide_grids( NuclideGridPoint * nuclide_grids, long n_isotopes,
                         long n_gridpoints )
{
  int (*cmp) (const void *, const void *);
  cmp = NGP_compare;
	
  for( long i = 0; i < n_isotopes; i++ )
    qsort( &nuclide_grids[i*n_gridpoints], n_gridpoints, sizeof(NuclideGridPoint), cmp );
	
  // error debug check
  /*
    for( int i = 0; i < n_isotopes; i++ )
    {
    printf("NUCLIDE %d ==============================\n", i);
    for( int j = 0; j < n_gridpoints; j++ )
    printf("E%d = %lf\n", j, nuclide_grids[i][j].energy);
    }
  */
}

// Allocates unionized energy grid, and assigns union of energy levels
// from nuclide grids to it.
GridPoint * generate_energy_grid( long n_isotopes, long n_gridpoints,
                                  NuclideGridPoint * nuclide_grids) {
  int mype = 0;

#ifdef DOMPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
#endif
	
  if( mype == 0 ) printf("Generating Unionized Energy Grid...\n");
	
  long n_unionized_grid_points = n_isotopes*n_gridpoints;
  int (*cmp) (const void *, const void *);
  cmp = NGP_compare;
	
  GridPoint * energy_grid = (GridPoint *)malloc( n_unionized_grid_points
						 * sizeof( GridPoint ) );
  if( mype == 0 ) printf("Copying and Sorting all nuclide grids...\n");
	
  NuclideGridPoint * n_grid_sorted = gpmatrix( n_isotopes, n_gridpoints );
	  	
  memcpy( n_grid_sorted, nuclide_grids, n_isotopes*n_gridpoints*
	  sizeof( NuclideGridPoint ) );
	
  qsort( &n_grid_sorted[0], n_unionized_grid_points, sizeof(NuclideGridPoint), cmp);
	
  if( mype == 0 ) printf("Assigning energies to unionized grid...\n");
	
  for( long i = 0; i < n_unionized_grid_points; i++ )
    energy_grid[i].energy = n_grid_sorted[i].energy;
	
  delete n_grid_sorted;
	
  int * full = new int[n_isotopes * n_unionized_grid_points];
	
  for( long i = 0; i < n_unionized_grid_points; i++ )
    energy_grid[i].xs_ptrs = &full[n_isotopes * i];
	
  // debug error checking
  /*
    for( int i = 0; i < n_unionized_grid_points; i++ )
    printf("E%d = %lf\n", i, energy_grid[i].energy);
  */

  return energy_grid;
}

// Searches each nuclide grid for the closest energy level and assigns
// pointer from unionized grid to the correct spot in the nuclide grid.
// This process is time consuming, as the number of binary searches
// required is:  binary searches = n_gridpoints * n_isotopes^2
void set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint * nuclide_grids,
                    long n_isotopes, long n_gridpoints )
{
  int mype = 0;

#ifdef DOMPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
#endif
	
  if( mype == 0 ) printf("Assigning pointers to Unionized Energy Grid...\n");
  for( long i = 0; i < n_isotopes * n_gridpoints ; i++ )
    {
      double quarry = energy_grid[i].energy;
      if( INFO && mype == 0 && i % 200 == 0 )
	printf("\rAligning Unionized Grid...(%.0lf%% complete)",
	       100.0 * (double) i / (n_isotopes*n_gridpoints));
      for( long j = 0; j < n_isotopes; j++ )
	{
	  // j is the nuclide i.d.
	  // log n binary search
	  energy_grid[i].xs_ptrs[j] = 
	    binary_search( &nuclide_grids[j*n_gridpoints], quarry, n_gridpoints);
	}
    }
  if( mype == 0 ) printf("\n");

  //test
  /*
    for( int i=0; i < n_isotopes * n_gridpoints; i++ )
    for( int j = 0; j < n_isotopes; j++ )
    printf("E = %.4lf\tNuclide %d->%p->%.4lf\n",
    energy_grid[i].energy,
    j,
    energy_grid[i].xs_ptrs[j],
    (energy_grid[i].xs_ptrs[j])->energy
    );
  */
}
