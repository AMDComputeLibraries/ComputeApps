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
#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

// Generates randomized energy grid for each nuclide
// Note that this is done as part of initialization (serial), so
// rand() is used.
void generate_grids( NuclideGridPoint ** nuclide_grids,
                     int n_isotopes, int n_gridpoints ) {
	for( int i = 0; i < n_isotopes; i++ )
		for( int j = 0; j < n_gridpoints; j++ )
		{
			nuclide_grids[i][j].energy       =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].total_xs     =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].elastic_xs   =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].absorbtion_xs=((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].fission_xs   =((double)rand()/(double)RAND_MAX);
			nuclide_grids[i][j].nu_fission_xs=((double)rand()/(double)RAND_MAX);
		}
}

// Verification version of this function (tighter control over RNG)
void generate_grids_v( NuclideGridPoint ** nuclide_grids,
                     int n_isotopes, int n_gridpoints ) {
	for( int i = 0; i < n_isotopes; i++ )
		for( int j = 0; j < n_gridpoints; j++ )
		{
			nuclide_grids[i][j].energy       = rn_v();
			nuclide_grids[i][j].total_xs     = rn_v();
			nuclide_grids[i][j].elastic_xs   = rn_v();
			nuclide_grids[i][j].absorbtion_xs= rn_v();
			nuclide_grids[i][j].fission_xs   = rn_v();
			nuclide_grids[i][j].nu_fission_xs= rn_v();
		}
}

// Sorts the nuclide grids by energy (lowest -> highest)
void sort_nuclide_grids( NuclideGridPoint ** nuclide_grids, int n_isotopes,
                         int n_gridpoints )
{
	int (*cmp) (const void *, const void *);
	cmp = NGP_compare;
	
	for( int i = 0; i < n_isotopes; i++ )
		qsort( nuclide_grids[i], n_gridpoints, sizeof(NuclideGridPoint),
		       cmp );
	
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
GridPoint * generate_energy_grid( int n_isotopes, int n_gridpoints,
                                  NuclideGridPoint ** nuclide_grids) {
	int mype = 0;

	#ifdef MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif
	
	if( mype == 0 ) printf("Generating Unionized Energy Grid...\n");
	
	int n_unionized_grid_points = n_isotopes*n_gridpoints;
	int (*cmp) (const void *, const void *);
	cmp = NGP_compare;
	
	GridPoint * energy_grid = (GridPoint *)malloc( n_unionized_grid_points
	                                               * sizeof( GridPoint ) );
	if( mype == 0 ) printf("Copying and Sorting all nuclide grids...\n");
	
	NuclideGridPoint ** n_grid_sorted = gpmatrix( n_isotopes, n_gridpoints );
	
	  	
	memcpy( n_grid_sorted[0], nuclide_grids[0], n_isotopes*n_gridpoints*
	                                      sizeof( NuclideGridPoint ) );
	
	qsort( &n_grid_sorted[0][0], n_unionized_grid_points,
	       sizeof(NuclideGridPoint), cmp);
	
	if( mype == 0 ) printf("Assigning energies to unionized grid...\n");
	
	for( int i = 0; i < n_unionized_grid_points; i++ )
		energy_grid[i].energy = n_grid_sorted[0][i].energy;
	

	gpmatrix_free(n_grid_sorted);
	
	int * full = (int *) malloc( n_isotopes * n_unionized_grid_points
	                             * sizeof(int) );
	
	for( int i = 0; i < n_unionized_grid_points; i++ )
		energy_grid[i].xs_ptrs = &full[n_isotopes * i];
	
	// debug error checking
	/*
	for( int i = 0; i < n_unionized_grid_points; i++ )
		printf("E%d = %lf\n", i, energy_grid[i].energy);
	*/

	return energy_grid;
}
void free_energy_grid(GridPoint *pe_grid)
{
	if (pe_grid && pe_grid[0].xs_ptrs)
	{
		free(pe_grid[0].xs_ptrs);
	}
	if ( pe_grid )
	{
		free(pe_grid);
	}
}


// Searches each nuclide grid for the closest energy level and assigns
// pointer from unionized grid to the correct spot in the nuclide grid.
// This process is time consuming, as the number of binary searches
// required is:  binary searches = n_gridpoints * n_isotopes^2
void set_grid_ptrs( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints )
{
	int mype = 0;

	#ifdef MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif
	
	if( mype == 0 ) printf("Assigning pointers to Unionized Energy Grid...\n");
	
	//#pragma omp parallel for default(none)			\
	//shared( energy_grid, nuclide_grids, n_isotopes, n_gridpoints, mype )
	for( int i = 0; i < n_isotopes * n_gridpoints ; i++ )
	{
		double quarry = energy_grid[i].energy;
		if( INFO && mype == 0 /*&& omp_get_thread_num() == 0*/ && i % 200 == 0 )
			printf("\rAligning Unionized Grid...(%.0lf%% complete)",
			       100.0 * (double) i / (n_isotopes*n_gridpoints ));
		// omp_get_num_threads())     );
		for( int j = 0; j < n_isotopes; j++ )
		{
			// j is the nuclide i.d.
			// log n binary search
			energy_grid[i].xs_ptrs[j] = 
				binary_search( nuclide_grids[j], quarry, n_gridpoints);
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

//
// A routine to save the sorted energy and nuclide grid info to disk
// The data is saved so that it can be restored, allowing the app to 
// bypassthe expensive steps of sorting.
//
void save_grids( GridPoint * energy_grid, NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints )
{
  FILE *fp;
  long size;
  char grids_filename[] = {"grids"};
  int n_gp = n_isotopes*n_gridpoints;
  GridPoint temp_egp;
  int * base;

  fp = fopen("grids", "w");
  if(fp == NULL)
    {
      printf("cannot open grids file: %s\n", grids_filename);
      exit (1);
    }
//first write out the nuclide grid data
  for(int i = 0;i<n_isotopes;i++)
  {
    fwrite(nuclide_grids[i],sizeof(NuclideGridPoint),n_gridpoints,fp);
  }
// Now write the energy grid info.
//
  for( int i = 0; i < n_gp; i++ )
  {
//    for(int j = 0;j<n_isotopes;j++)
//    {
//     tempegp.energy = energy_grid[i].energy;
//      tempegp.xs_ptrs[j] = energy_grid[i].xs_ptrs[j];
//    }
    fwrite(&energy_grid[i].energy,sizeof(double),1,fp);
    fwrite(energy_grid[i].xs_ptrs,sizeof(int),n_isotopes,fp);
  }
  fclose(fp);
}

// A routine to restore grid data from the saved file.  The file must have
// been saved with the same number of idotopes and gridpoints.
//
// nuclide grid should be allocated, and should be filled in with pointers to 
// allocated NGP's
// energy_grid should be allocated, but does not have to be initialized.
//
GridPoint * restore_grids( NuclideGridPoint ** nuclide_grids,
                    int n_isotopes, int n_gridpoints, char *filename)
{
  FILE *fp;
  long size;
  int n_gp = n_isotopes*n_gridpoints;
  float tempf;

  fp = fopen(filename, "r");
  if(fp == NULL)
  {
    printf("cannot open grids file: %s\n", filename);
    exit (1);
  }

//first read the nuclide grid data
  for(int i = 0;i<n_isotopes;i++)
  {
    fread(nuclide_grids[i],sizeof(NuclideGridPoint),n_gridpoints,fp);
  }

//we have to duplicate calling rand() the same number of times as for
//live code.  This turns out to be n_isotopes*n_gridpoints
//
  for( int i = 0; i < n_isotopes; i++ )
    for( int j = 0; j < n_gridpoints; j++ )
      for( int k = 0; k < 6; k++ )
        #ifdef VERIFICATION
           tempf = rn_v();
        #else
           tempf = rand();
        #endif
//
//next read the sorted energy grid data, which includes the xs_ptrs
// we also need to allocate space for the energy grid.
  GridPoint * energy_grid = (GridPoint *)malloc( n_gp*sizeof(GridPoint));
  int * full = (int *) malloc( n_isotopes * n_gp * sizeof(int) );
  for( int i = 0; i < n_gp ; i++ )
  {
    energy_grid[i].xs_ptrs = &full[n_isotopes * i];
    fread(&energy_grid[i].energy,sizeof(double),1,fp);
    fread(energy_grid[i].xs_ptrs,sizeof(int),n_isotopes,fp);
  }
  fclose(fp);
  return energy_grid;
}
