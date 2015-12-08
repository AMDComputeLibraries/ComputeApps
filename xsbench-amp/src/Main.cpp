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
#ifdef DOMPI
#include<mpi.h>
#endif

#ifdef HAVE_AMP
#include <amp.h>
using namespace concurrency;
#define BLOCKSIZE 2048
#define WGSIZE 256
#endif

static void sort(double *energy, int *mat, unsigned low, unsigned high)
{
	if (low >= high)
		return;
	const unsigned old_lo = low, old_hi = high;
	const double pivot = energy[(high + low + 1) / 2];
	while(low < high) {
		while (energy[low] < pivot)
			++low;
		while (energy[high] > pivot)
			--high;
		if (low < high) {
			std::swap(energy[low], energy[high]);
//			std::swap(mat[low], mat[high]);
			++low; --high;
		}
	}
	sort(energy, mat, old_lo, low - 1);
	sort(energy, mat, low, old_hi);
}



extern "C" int main( int argc, char* argv[] )
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 13;
	int mype = 0;
	#ifndef HAVE_AMP
	int max_procs = omp_get_num_procs();
	int thread, mat;
	#endif	
	int i;
	unsigned long seed;
	#ifndef HAVE_AMP
	double omp_start, omp_end, p_energy;
	#endif
	double timer_start, timer_end;
	unsigned long long vhash = 0;
	int nprocs;
	
	#ifdef DOMPI
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

	// Process CLI Fields -- store in "Inputs" structure
	Inputs in = read_CLI( argc, argv );
	
	// Set number of OpenMP Threads
	#ifndef HAVE_AMP
	omp_set_num_threads(in.nthreads);
	#endif

	// Print-out of Input Summary
	if( mype == 0 )
		print_inputs( in, nprocs, version );

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// =====================================================================
	NuclideGridPoint ** nuclide_grids = NULL;

	// Allocate & fill energy grids
	#ifndef BINARY_READ
	if( mype == 0) printf("Generating Nuclide Energy Grids...\n");
	
	nuclide_grids = gpmatrix(in.n_isotopes,in.n_gridpoints);
	
	#ifdef VERIFICATION
	generate_grids_v( nuclide_grids, in.n_isotopes, in.n_gridpoints );	
	#else
	generate_grids( nuclide_grids, in.n_isotopes, in.n_gridpoints );	
	#endif
	
	// Sort grids by energy
	if( mype == 0) printf("Sorting Nuclide Energy Grids...\n");
	sort_nuclide_grids( nuclide_grids, in.n_isotopes, in.n_gridpoints );
	#endif

	// Prepare Unionized Energy Grid Framework
	#ifndef BINARY_READ
	GridPoint * energy_grid = generate_energy_grid( in.n_isotopes,
	                          in.n_gridpoints, nuclide_grids ); 	
	#else
	nuclide_grids = gpmatrix(in.n_isotopes,in.n_gridpoints);
	GridPoint * energy_grid = (GridPoint *)malloc( in.n_isotopes *
	                           in.n_gridpoints * sizeof( GridPoint ) );
	int * index_data = (int *) malloc( in.n_isotopes * in.n_gridpoints
	                   * in.n_isotopes * sizeof(int));
	for( i = 0; i < in.n_isotopes*in.n_gridpoints; i++ )
		energy_grid[i].xs_ptrs = &index_data[i*in.n_isotopes];
	#endif

	// Double Indexing. Filling in energy_grid with pointers to the
	// nuclide_energy_grids.
	#ifndef BINARY_READ
	set_grid_ptrs( energy_grid, nuclide_grids, in.n_isotopes, in.n_gridpoints );
	#endif

	#ifdef BINARY_READ
	if( mype == 0 ) printf("Reading data from \"%s\" file...\n",
	                       in.filename);
	binary_read(in.n_isotopes, in.n_gridpoints, nuclide_grids, energy_grid,
	            in.filename);
	#endif
	
	// Get material data
	if( mype == 0 )
		printf("Loading Mats...\n");
	int *num_nucs  = load_num_nucs(in.n_isotopes);
	int **mats     = load_mats(num_nucs, in.n_isotopes);

	#ifdef VERIFICATION
	double **concs = load_concs_v(num_nucs);
	#else
	double **concs = load_concs(num_nucs);
	#endif

	#ifdef BINARY_DUMP
	if( mype == 0 ) printf("Dumping data to binary file...\n");
	binary_dump(in.n_isotopes, in.n_gridpoints, nuclide_grids, energy_grid);
	if( mype == 0 ) printf("Binary file \"XS_data.dat\" written! Exiting...\n");
	return 0;
	#endif

	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================

	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	//initialize papi with one thread (master) here
	#ifdef PAPI
	if ( PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT){
		fprintf(stderr, "PAPI library init error!\n");
		exit(1);
	}
	#endif

        #ifdef VERIFICATION
	verifyStruct * verifyArray =
	  (verifyStruct *) malloc(in.lookups * sizeof(verifyStruct));
	#endif
	
	#ifndef HAVE_AMP
	#ifdef VERIFICATION
        #pragma omp parallel default(none)		\
	private(i, thread, p_energy, mat, seed) \
	shared( max_procs, in, energy_grid, nuclide_grids, \
	        mats, concs, num_nucs, mype, vhash, verifyArray) 
	{
	#else
	#pragma omp parallel default(none) \
	private(i, thread, p_energy, mat, seed) \
	shared( max_procs, in, energy_grid, nuclide_grids, \
	        mats, concs, num_nucs, mype, vhash) 
	{
        #endif //VERIFICATION
        #endif //HAVE_AMP
		
	// Initialize parallel PAPI counters
        #ifdef PAPI
	int eventset = PAPI_NULL; 
	int num_papi_events;
	{
	  counter_init(&eventset, &num_papi_events);
	}
        #endif
	
        #ifndef HAVE_AMP
	thread = omp_get_thread_num();
	seed   = (thread+1)*19+17;
        #else
	seed = 13; 
        #endif
	
	int * pickedMats = (int *) malloc(in.lookups * sizeof(int));
	double * pickedP_energy = (double *) malloc(in.lookups * sizeof(double));
        #ifdef VERIFICATION	
	for(int j = 0; j < in.lookups; ++j){	  
	  pickedP_energy[j] = rn_v();
	  pickedMats[j] = pick_mat(&seed); 
	}
        #else
	for(int j = 0; j < in.lookups; ++j){
	  pickedP_energy[j] = rn(&seed);
	  pickedMats[j] = pick_mat(&seed); 
	}
        #endif

	timer_start = timer();

	#ifndef HAVE_AMP
        #pragma omp for schedule(dynamic)
	for( i = 0; i < in.lookups; i++ )
	{
	#else
	double check = 0.0;
	parallel_for_each(extent<1>(in.lookups),[=, &check] (index<1> idx) restrict(amp){
	    int i = idx[0];
        #endif
	    // Randomly pick an energy and material for the particle
	    int mat;
	    double p_energy;
	    double macro_xs_vector[5];	    
	    
	    mat = pickedMats[i];
	    p_energy = pickedP_energy[i]; 
	    
	    // debugging
	    //printf("E = %lf mat = %d\n", p_energy, mat);
	    
	    // This returns the macro_xs_vector, but we're not going
	    // to do anything with it in this program, so return value
	    // is written over.
	    calculate_macro_xs( p_energy, mat, in.n_isotopes,
				in.n_gridpoints, num_nucs, concs,
				energy_grid,
				nuclide_grids,
				mats,
				macro_xs_vector );
	    // Make sure the code is not DCE
	    if (i == 0)
		    check += macro_xs_vector[0] + macro_xs_vector[1] +
		             macro_xs_vector[2] + macro_xs_vector[3] +
		             macro_xs_vector[4];
	    
	    // Verification hash calculation
	    // This method provides a consistent hash across
	    // architectures and compilers.
	    
        #ifdef VERIFICATION
	    verifyArray[i].p_energy = p_energy;
	    verifyArray[i].mat = mat;
	    verifyArray[i].macro_xs_vector[0] = macro_xs_vector[0];
	    verifyArray[i].macro_xs_vector[1] = macro_xs_vector[1];
	    verifyArray[i].macro_xs_vector[2] = macro_xs_vector[2];
	    verifyArray[i].macro_xs_vector[3] = macro_xs_vector[3];
	    verifyArray[i].macro_xs_vector[4] = macro_xs_vector[4];	
        #endif
        #ifndef HAVE_AMP
	  }
	#else
	  });	       
	#endif
	
	timer_end = timer();
	// Prints out thread local PAPI counters
        #ifdef PAPI
	if( mype == 0 && thread == 0 )
	  {
	    printf("\n");
	    border_print();
	    center_print("PAPI COUNTER RESULTS", 79);
	    border_print();
	    printf("Count          \tSmybol      \tDescription\n");
	  }
	counter_stop(&eventset, num_papi_events);
        #endif

	#ifndef HAVE_AMP
	}
	#endif
	
        #ifdef VERIFICATION
	for(int i = 0; i < in.lookups; ++i){
	  char line[256];
	  sprintf(line, "%.5lf %d %.5lf %.5lf %.5lf %.5lf %.5lf",
		  verifyArray[i].p_energy, verifyArray[i].mat,
		  verifyArray[i].macro_xs_vector[0],
		  verifyArray[i].macro_xs_vector[1],
		  verifyArray[i].macro_xs_vector[2],
		  verifyArray[i].macro_xs_vector[3],
		  verifyArray[i].macro_xs_vector[4]);
	          vhash += hash((unsigned char*)line, 10000);
	}
        #endif
	
	#ifndef PAPI
	if( mype == 0)	
	{	
	  printf("\n" );
	  printf("Simulation complete.\n" );
	}
	#endif

	
	// Print / Save Results and Exit

        print_results( in, mype, timer_end-timer_start, nprocs, vhash );

	#ifdef DOMPI
	MPI_Finalize();
	#endif

	return 0;
}
