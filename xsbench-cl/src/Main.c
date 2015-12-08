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

#include <time.h>
int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	
	int version = 10;
	int mype = 0;

	int max_procs = omp_get_num_procs();
	int n_isotopes, n_gridpoints, lookups, i, thread, nthreads;
	unsigned long seed;
	double omp_start, omp_end;
	int mat;
	double p_energy;
	GridPoint * energy_grid;

//	clock_t cstart, cend;
	double cstart, cend;
	//double p_energy;
	char * HM;
	unsigned long long vhash = 0;

	int err = SetupOpenCL("xsbench_kernels.cl");
	if(err != CL_SUCCESS)
	  {
	    printf("Failed to start OpenCL with error %d\n", err);
	    CleanupOpenCL();
	    exit(1);
	  }

	#ifdef MPI
	int nprocs;
	MPI_Status stat;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif
	
	// rand() is only used in the serial initialization stages.
	// A custom RNG is used in parallel portions.
	#ifdef VERIFICATION
	srand(26);
	#else
	srand(time(NULL));
	#endif

	// Process CLI Fields
	Inputs input = read_CLI( argc, argv );
	
	// Set CLI variables
	nthreads =     input.nthreads; // force to 1 for gem5
	n_isotopes =   input.n_isotopes;
	n_gridpoints = input.n_gridpoints;
	lookups =      input.lookups;
	HM =           input.HM;

	// Set number of OpenMP Threads
	//omp_set_num_threads(nthreads); 
		
	// =====================================================================
	// Calculate Estimate of Memory Usage
	// =====================================================================

	size_t single_nuclide_grid = n_gridpoints * sizeof( NuclideGridPoint );
	size_t all_nuclide_grids   = n_isotopes * single_nuclide_grid;
	size_t size_GridPoint      = sizeof(GridPoint) + n_isotopes*sizeof(int);
	size_t size_UEG            = n_isotopes * n_gridpoints * size_GridPoint;
	size_t memtotal;
	int mem_tot;

	memtotal          = all_nuclide_grids + size_UEG;
	all_nuclide_grids = all_nuclide_grids / 1048576;
	size_UEG          = size_UEG / 1048576;
	memtotal          = memtotal / 1048576;
	mem_tot           = memtotal;

	// =====================================================================
	// Print-out of Input Summary
	// =====================================================================
	
	if( mype == 0 )
	{
		logo(version);
		center_print("INPUT SUMMARY", 79);
		border_print();
		#ifdef VERIFICATION
		printf("Verification Mode:            on\n");
		#endif
		printf("Materials:                    %d\n", N_MAT);
		printf("H-M Benchmark Size:           %s\n", HM);
		printf("Total Nuclides:               %d\n", n_isotopes);
		printf("Gridpoints (per Nuclide):     ");
		fancy_int(n_gridpoints);
		printf("Unionized Energy Gridpoints:  ");
		fancy_int(n_isotopes*n_gridpoints);
		printf("XS Lookups:                   "); fancy_int(lookups);
		#ifdef MPI
		printf("MPI Ranks:                    %d\n", nprocs);
		//printf("OMP Threads per MPI Rank:     %d\n", nthreads);
		printf("Mem Usage per MPI Rank (MB):  "); fancy_int(mem_tot);
		#else
		printf("Threads:                      %d\n", nthreads);
		printf("Est. Memory Usage (MB):       "); fancy_int(mem_tot);
		#endif
		if( EXTRA_FLOPS > 0 )
			printf("Extra Flops:                  %d\n", EXTRA_FLOPS);
		if( EXTRA_LOADS > 0 )
			printf("Extra Loads:                  %d\n", EXTRA_LOADS);
		border_print();
		center_print("INITIALIZATION", 79);
		border_print();
	}

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// =====================================================================

	// Allocate & fill energy grids
	if( mype == 0) printf("Generating Nuclide Energy Grids...\n");
	
	NuclideGridPoint ** nuclide_grids = gpmatrix( n_isotopes, n_gridpoints );
if(input.restoregrids)
{
	if( mype == 0 ) printf("restoring grids from file %s...\n",input.file_name);
        energy_grid = restore_grids(nuclide_grids, n_isotopes, n_gridpoints, input.file_name);
}
else
{
	#ifdef VERIFICATION
	generate_grids_v( nuclide_grids, n_isotopes, n_gridpoints );	
	#else
	generate_grids( nuclide_grids, n_isotopes, n_gridpoints );	
	#endif
	
	// Sort grids by energy
	if( mype == 0) printf("Sorting Nuclide Energy Grids...\n");
	sort_nuclide_grids( nuclide_grids, n_isotopes, n_gridpoints );
	// Prepare Unionized Energy Grid Framework

	energy_grid = generate_energy_grid( n_isotopes, n_gridpoints,
	                                                nuclide_grids ); 	

	// Double Indexing. Filling in energy_grid with pointers to the
	// nuclide_energy_grids.
	set_grid_ptrs( energy_grid, nuclide_grids, n_isotopes, n_gridpoints );
}
if(input.savegrids)
{
	if( mype == 0 ) printf("saving grids...\n");
        save_grids(energy_grid, nuclide_grids, n_isotopes, n_gridpoints);
}
	
	// Get material data
	if( mype == 0 ) printf("Loading Mats...\n");
	int *num_nucs  = load_num_nucs(n_isotopes);
	int **mats     = load_mats(num_nucs, n_isotopes);
	#ifdef VERIFICATION
	double **concs = load_concs_v(num_nucs);
	double * xs_vectors = (double*)malloc(lookups * 5 * sizeof(double));
	#else
	double **concs = load_concs(num_nucs);
	#endif



	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================
	
	if( mype == 0 )
	{
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}



	int * pmat = (int*)malloc(lookups*sizeof(int));
	double * pp_energy = (double*)malloc(lookups*sizeof(double));
	#if defined(VERIFICATION)
	    for( int i = 0; i < lookups; i++)
	    {
		     pp_energy[i] = rn_v();
		     pmat[i]      = pick_mat(&seed); 
	    }
        #else
		thread = omp_get_thread_num();
		seed   = (thread+1)*19+17;
	    for( int i = 0; i < lookups; i++)
	    {
			 pp_energy[i] = rn(&seed);
			 pmat[i]      = pick_mat(&seed); 	    
	    }
        #endif


OCLKernelParamsMacroXSS krnl_param;
uint t_loops = input.tloops;

const char * krnl_param_nm =
"bitonicSortTiled3";

	memset(&krnl_param, 0, sizeof(OCLKernelParamsMacroXSS));
	krnl_param.n_mat = N_MAT;
	krnl_param.lookups = input.lookups;
	krnl_param.n_isotopes =   input.n_isotopes;
	krnl_param.n_gridpoints = input.n_gridpoints;
	AllocOutput(&krnl_param);
	AllocArrangeCopyConst(&krnl_param, (const int *)num_nucs, (const int **)mats, (const double **)concs);
	AllocCopyInputs(&krnl_param, pp_energy, pmat);
	AllocArrangeCopyEnergyGridPtrs(&krnl_param, (const GridPoint * )energy_grid);
	AllocArrangeCopyNucGrid(&krnl_param, (const NuclideGridPoint ** )nuclide_grids);
    // warm up and send data

	calculate_xs_ocl(&krnl_param, krnl_param_nm, 1);
	printf("Calculating OCL XS's..., kernel: %s (%d loops)\n",  krnl_param_nm, t_loops);
	cstart = omp_get_wtime();
#ifdef VERIFICATION
	t_loops = 1;
#endif
	calculate_xs_ocl(&krnl_param, krnl_param_nm, t_loops);

	cend = omp_get_wtime();
	printf("OCL XS's calculation completed, kernel: %s (%d loops)\n",  krnl_param_nm, t_loops);
	ReadOutput(&krnl_param);



	omp_start = omp_get_wtime();

	
	#ifdef __PAPI
	int eventset = PAPI_NULL; 
	int num_papi_events;
	counter_init(&eventset, &num_papi_events);
	#endif


   if (input.run_cpu)
   {


	// OpenMP compiler directives - declaring variables as shared or private
#ifdef VERIFICATION
	#pragma omp parallel default(none) \
	private(i, thread, p_energy, mat, seed) \
	shared( max_procs, n_isotopes, n_gridpoints, \
	energy_grid, nuclide_grids, lookups, nthreads, \
	mats, concs, num_nucs, mype, vhash, \
	pp_energy, pmat, xs_vectors) 
#else
	#pragma omp parallel default(none) \
	private(i, thread, p_energy, mat, seed) \
	shared( max_procs, n_isotopes, n_gridpoints, \
	energy_grid, nuclide_grids, lookups, nthreads, \
	mats, concs, num_nucs, mype, vhash) 
#endif
	{	
		double macro_xs_vector[5];
		thread = omp_get_thread_num();
		seed   = (thread+1)*19+17;
		#pragma omp for
		for( i = 0; i < lookups; i++ )
		{
			// Status text
			if( INFO && mype == 0 && thread == 0 && i % 1000 == 0 )
				printf("\rCalculating XS's... (%.0lf%% completed)",
						i / ( lookups / (double) nthreads ) * 100.0);
			
			// Randomly pick an energy and material for the particle
			#ifdef VERIFICATION
			p_energy = pp_energy[i];
			mat      = pmat[i]; 
			#else
			p_energy = rn(&seed);
			mat      = pick_mat(&seed); 
			#endif
			
			// debugging
			//printf("E = %lf mat = %d\n", p_energy, mat);
				
			// This returns the macro_xs_vector, but we're not going
			// to do anything with it in this program, so return value
			// is written over.
#ifdef VERIFICATION
			calculate_macro_xs_v( p_energy, mat, n_isotopes,
			                    n_gridpoints, num_nucs, concs,
			                    energy_grid, nuclide_grids, mats,
                                            macro_xs_vector,
					    mype, thread, i);
#else
			calculate_macro_xs( p_energy, mat, n_isotopes,
			                    n_gridpoints, num_nucs, concs,
			                    energy_grid, nuclide_grids, mats,
                                            macro_xs_vector );
#endif

			// Verification hash calculation
			// This method provides a consistent hash accross
			// architectures and compilers.
#ifdef VERIFICATION

			char line[256];


#if 0
			sprintf(line, "%.5lf %d %.5lf %.5lf %.5lf %.5lf %.5lf",
			       p_energy, mat,
				   macro_xs_vector[0],
				   macro_xs_vector[1],
				   macro_xs_vector[2],
				   macro_xs_vector[3],
				   macro_xs_vector[4]);
			unsigned long long vhash_local = hash((unsigned char*)line, 10000);
#else

#ifdef VERIFICATION_BUFFER
			for(int v = 0; v < 5; v++)
			{
				   xs_vectors[i*5 + v] = macro_xs_vector[v];
			}
#endif // #ifdef VERIFICATION_BUFFER 

			memcpy(line, macro_xs_vector, 5*sizeof(double));
			unsigned long long vhash_local = hash_l((unsigned char*)line, 5*sizeof(double), 10000);


			#pragma omp atomic
			vhash += vhash_local;
#endif  //#ifdef VERIFICATION
#endif  //#ifdef VERIFICATION

		}
		if( INFO && mype == 0 && thread == 0)
		{
			printf("\n");
		}
	}

#ifdef VERIFICATION_BUFFER
double * OCLxs_vectors = (double *)getOCLOutput();
bool match = true;
    for( int l = 0; l < lookups && match; l++)
	{
		for ( int v = 0; v < 5 && match; v++)
		{
			if ( abs(xs_vectors[l*5 +v] - OCLxs_vectors[l*5 +v]) > 0.00001 )
			{
				printf("No match : %d %d %lf %lf\n", l, v, xs_vectors[l*5 +v], OCLxs_vectors[l*5 +v]);
				match = false;
				
			}
		}
	}
#endif

   }





	if( mype == 0)	
	{	
		printf("\n" );
		printf("Simulation complete.\n" );
	}

	omp_end = omp_get_wtime();

	
	// =====================================================================
	// Print / Save Results and Exit
	// =====================================================================
	
	// Calculate Lookups per sec
	int lookups_per_sec = (int) ((double) lookups / (omp_end-omp_start));
//	double sec = (((double)cend - (double)cstart)*1.0e-3) / (double)t_loops;
	double sec = (((double)cend - (double)cstart)) / (double)t_loops;
	int ocl_lookups_per_sec = (int) ( (double)lookups/sec);
	
	// If running in MPI, reduce timing statistics and calculate average
	#ifdef MPI
	int total_lookups = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&lookups_per_sec, &total_lookups, 1, MPI_INT,
	           MPI_SUM, 0, MPI_COMM_WORLD);
	//total_lookups = total_lookups / nprocs;
	#endif
	
	// Print output
	if( mype == 0 )
	{
		border_print();
		center_print("RESULTS", 79);
		border_print();

		// Print the results
		printf("Threads:     %d\n", nthreads);
		#ifdef MPI
		printf("MPI ranks:   %d\n", nprocs);
		#endif
		if( EXTRA_FLOPS > 0 )
		printf("Extra Flops: %d\n", EXTRA_FLOPS);
		if( EXTRA_LOADS > 0 )
		printf("Extra Loads: %d\n", EXTRA_LOADS);
		#ifdef MPI
		printf("Total Lookups/s:            ");
		fancy_int(total_lookups);
		printf("Avg Lookups/s per MPI rank: ");
		fancy_int(total_lookups / nprocs);
		#else
		printf("Lookups:     "); fancy_int(lookups);
                if (input.run_cpu)
                {
		  printf("CPU Runtime:     %.3lf seconds\n", omp_end-omp_start);

		  printf("CPU Lookups/s:   ");
		  fancy_int(lookups_per_sec);
                }
		printf("OCL Runtime:     %.6lf seconds\n", sec);
		printf("OCL Lookups/s:   ");
		fancy_int(ocl_lookups_per_sec);

		#endif
		#ifdef VERIFICATION
		printf("Verification checksum: %llu\n", vhash);
		#endif
		border_print();

		// For bechmarking, output lookup/s data to file
		if( SAVE )
		{
			FILE * out = fopen( "results.txt", "a" );
			fprintf(out, "%d\t%d\n", nthreads, lookups_per_sec);
			fclose(out);
		}
	}	

	#ifdef __PAPI
	counter_stop(&eventset, num_papi_events);
	#endif

	#ifdef MPI
	MPI_Finalize();
	#endif

	gpmatrix_free ( nuclide_grids );

	free_energy_grid(energy_grid);

	#ifdef VERIFICATION
	if ( xs_vectors )
	{
		free(xs_vectors);
	}
	if (pmat)
	{
		free(pmat);
	}

    if (pp_energy)
	{
		free(pp_energy);
	}
	#endif

	return 0;
}
