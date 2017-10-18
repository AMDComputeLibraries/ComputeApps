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

#if ARRAY_TYPE == 0
#define HCC_ARRAY_STRUC(type, name, size, ptr) array_view<type> name(size, ptr)
#define HCC_ID(name)
#define HCC_SYNC(type, name, size, ptr) name.synchronize()

#elif ARRAY_TYPE == 1
#define HCC_ARRAY_STRUC(type, name, size, ptr) array<type> name(size); copy(ptr, name)
#define HCC_ID(name) ,&name
#define HCC_SYNC(type, name, size, ptr) copy(name, ptr)

#elif ARRAY_TYPE == 2
#define HCC_ARRAY_STRUC(type, name, size, ptr) type * name = (type *)am_alloc(size*sizeof(type), acc, 0); acc_view.copy(ptr, name, size*sizeof(type));
#define HCC_ID(name) ,name
#define HCC_SYNC(type, name, size, ptr) acc_view.copy(name, ptr, size*sizeof(type));
#endif

#include <hc.hpp>
#include <hc_am.hpp>
using namespace hc;

int main( int argc, char* argv[] )
{
  // =====================================================================
  // Initialization & Command Line Read-In
  // =====================================================================
  int version = 13;
  int mype = 0;
  int i;
  unsigned long seed;
  double timer_start, timer_end;
  unsigned long long vhash = 0;
  int nprocs;
  int nuc_size = 0;

  accelerator acc;
  std::vector<accelerator> node_acc = acc.get_all();
  int num_acc = node_acc.size();  
  for(int i = 0; i < num_acc; ++i){
    if(!node_acc[i].is_hsa_accelerator()){
      node_acc.erase(node_acc.begin()+i);
    }
  }
  acc = node_acc[WHICH_ACC];
  accelerator_view acc_view = acc.get_default_view();
	
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
	
  // Print-out of Input Summary
  if( mype == 0 )
    print_inputs( in, nprocs, version );

  // =====================================================================
  // Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
  // =====================================================================
  NuclideGridPoint * nuclide_grids = NULL;

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

  GridPoint * energy_grid = new GridPoint[in.n_isotopes * in.n_gridpoints];
  int * index_data = new int[in.n_isotopes * in.n_gridpoints * in.n_isotopes];
	
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
	
  int *num_nucs_idx = new int[12];
  for(int i = 0; i < 12; i++){
    num_nucs_idx[i] = nuc_size;
    nuc_size += num_nucs[i];
  }

  int *mats     = load_mats(num_nucs, in.n_isotopes, num_nucs_idx);

#ifdef VERIFICATION
  double *concs = load_concs_v(num_nucs);
#else
  double *concs = load_concs(num_nucs);
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

  if( mype == 0 ){
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

			
  // Initialize parallel PAPI counters
#ifdef PAPI
  int eventset = PAPI_NULL; 
  int num_papi_events;
  {
    counter_init(&eventset, &num_papi_events);
  }
#endif
	
  seed = 13; 
  int n_isotopes_t = in.n_isotopes;
  int n_gridpoints_t = in.n_gridpoints;
	
  int * pickedMats = new int[in.lookups];
  double * pickedP_energy = new double[in.lookups];
  double * energy_grid_energy = new double[in.n_isotopes*in.n_gridpoints];
  int * energy_grid_xs = new int[in.n_isotopes*in.n_gridpoints*in.n_isotopes];

  timer_start = timer();
	
  for(int i = 0; i< in.n_isotopes*in.n_gridpoints; ++i){
    energy_grid_energy[i] = energy_grid[i].energy;
    for(int j = 0; j < in.n_isotopes; ++j){
      energy_grid_xs[i*in.n_isotopes + j] = energy_grid[i].xs_ptrs[j];
    }
  }
		
#ifdef VERIFICATION
  double * verify_p_energy = new double[in.lookups];
  int * verify_mat = new int[in.lookups];
  double * verify_macro_xs_vector = new double[in.lookups * 5];

  HCC_ARRAY_STRUC(double, verify_p_energy_t, in.lookups, verify_p_energy);
  HCC_ARRAY_STRUC(int, verify_mat_t, in.lookups, verify_mat);
  HCC_ARRAY_STRUC(double, verify_macro_xs_vector_t, in.lookups*5, verify_macro_xs_vector);
  
  for(int j = 0; j < in.lookups; ++j){	  
    pickedP_energy[j] = rn_v();
    pickedMats[j] = pick_mat(&seed); 
  }
	
#else
  double * check = new double[1];  
  HCC_ARRAY_STRUC(double, check_t, 1, check);
  
  for(int j = 0; j < in.lookups; ++j){
    pickedP_energy[j] = rn(&seed);
    pickedMats[j] = pick_mat(&seed); 
  }
	
#endif //VERIFICATION

  HCC_ARRAY_STRUC(int, pickedMats_t, in.lookups, pickedMats);
  HCC_ARRAY_STRUC(double, pickedP_energy_t, in.lookups, pickedP_energy);
  HCC_ARRAY_STRUC(double, energy_grid_energy_t, in.n_isotopes*in.n_gridpoints, energy_grid_energy);
  HCC_ARRAY_STRUC(int, energy_grid_xs_t, in.n_isotopes*in.n_gridpoints*in.n_isotopes, energy_grid_xs);
  HCC_ARRAY_STRUC(NuclideGridPoint, nuclide_grids_t, in.n_isotopes*in.n_gridpoints, nuclide_grids);
  HCC_ARRAY_STRUC(int, num_nucs_t, 12, num_nucs);
  HCC_ARRAY_STRUC(int, num_nucs_idx_t, 12, num_nucs_idx);
  HCC_ARRAY_STRUC(double, concs_t, nuc_size, concs);
  HCC_ARRAY_STRUC(int, mats_t, nuc_size, mats);  
	
  completion_future fut = parallel_for_each(acc_view, extent<1>(in.lookups),[
                                                                   #if ARRAY_TYPE == 2
                                                                   n_isotopes_t, n_gridpoints_t
                                                                   #else
                                                                   =
                                                                   #endif
                                                                   HCC_ID(pickedMats_t)
								   HCC_ID(pickedP_energy_t)
								   HCC_ID(energy_grid_energy_t)
								   HCC_ID(energy_grid_xs_t)
								   HCC_ID(nuclide_grids_t)
								   HCC_ID(num_nucs_t)
								   HCC_ID(num_nucs_idx_t)
								   HCC_ID(concs_t)
								   HCC_ID(mats_t)
                                                                   #ifdef VERIFICATION
                                                                   HCC_ID(verify_p_energy_t)
								   HCC_ID(verify_mat_t)
								   HCC_ID(verify_macro_xs_vector_t)
								   #else
								   HCC_ID(check_t)
								   #endif
								   ] (index<1> idx) restrict(amp){
      int i = idx[0];
      int mat;
      double p_energy;
      double macro_xs_vector[5];	    

      long index = 0;	
      double xs_vector[5];
      int p_nuc; 
      double conc; 

      long lowerLimit = 0;
      long upperLimit = n_isotopes_t*n_gridpoints_t - 1;
      long examinationPoint;
      long length = upperLimit - lowerLimit;
	    
      mat = pickedMats_t[i];
      p_energy = pickedP_energy_t[i];

      for( int k = 0; k < 5; k++ )
	macro_xs_vector[k] = 0;

      //idx = grid_search( n_isotopes * n_gridpoints, p_energy,
      //		 energy_grid_energy);

      while( length > 1 ){
	examinationPoint = lowerLimit + ( length / 2 );
		    
	if( energy_grid_energy_t[examinationPoint] > p_energy )
	  upperLimit = examinationPoint;
	else
	  lowerLimit = examinationPoint;
		    
	length = upperLimit - lowerLimit;
      }
      index = lowerLimit;

      for( int k = lowerLimit; k < upperLimit; k++ ){
	if( energy_grid_energy_t[k] <= p_energy )
	  index = k;
	else
	  break;
      }

      for( int j = 0; j < num_nucs_t[mat]; j++ ){
	p_nuc = mats_t[num_nucs_idx_t[mat] + j];
	conc = concs_t[num_nucs_idx_t[mat] + j];
	//calculate_micro_xs( p_energy, p_nuc, n_isotopes,
	//		      n_gridpoints, energy_grid_energy,
	//		      energy_grid_xs,
	//		      nuclide_grids, index, xs_vector );

	double f;
	NuclideGridPoint low, high;

	if( energy_grid_xs_t[index*n_isotopes_t + p_nuc] == n_gridpoints_t - 1 ){
	  low = nuclide_grids_t[p_nuc*n_gridpoints_t + energy_grid_xs_t[index*n_isotopes_t + p_nuc] - 1];
	  high = nuclide_grids_t[p_nuc*n_gridpoints_t + energy_grid_xs_t[index*n_isotopes_t + p_nuc]];
	}
	else{
	  low = nuclide_grids_t[p_nuc*n_gridpoints_t + energy_grid_xs_t[index*n_isotopes_t + p_nuc]];
	  high = nuclide_grids_t[p_nuc*n_gridpoints_t + energy_grid_xs_t[index*n_isotopes_t + p_nuc] + 1];
        }

	f = (high.energy - p_energy) /
	    (high.energy - low.energy);
	xs_vector[0] = high.total_xs - f *
	    (high.total_xs - low.total_xs);	
	xs_vector[1] = high.elastic_xs - f *
	    (high.elastic_xs - low.elastic_xs);
	xs_vector[2] = high.absorbtion_xs - f *
	    (high.absorbtion_xs - low.absorbtion_xs);
	xs_vector[3] = high.fission_xs - f *
	    (high.fission_xs - low.fission_xs);
	xs_vector[4] = high.nu_fission_xs - f *
        (high.nu_fission_xs - low.nu_fission_xs);
        
	for( int k = 0; k < 5; k++ )
	  macro_xs_vector[k] += xs_vector[k] * conc;
      }

#ifndef VERIFICATION
      if (i == 0)
	check_t[0] = macro_xs_vector[0] + macro_xs_vector[1] +
	  macro_xs_vector[2] + macro_xs_vector[3] +
	  macro_xs_vector[4];
#else
      // Verification hash calculation
      // This method provides a consistent hash across
      // architectures and compilers.
      verify_p_energy_t[i] = p_energy;
      verify_mat_t[i] = mat;
      verify_macro_xs_vector_t[i*5 + 0] = macro_xs_vector[0];
      verify_macro_xs_vector_t[i*5 + 1] = macro_xs_vector[1];
      verify_macro_xs_vector_t[i*5 + 2] = macro_xs_vector[2];
      verify_macro_xs_vector_t[i*5 + 3] = macro_xs_vector[3];
      verify_macro_xs_vector_t[i*5 + 4] = macro_xs_vector[4];
#endif

    });
  fut.wait();  
  timer_end = timer();
  
  // Prints out thread local PAPI counters
#ifdef PAPI
  if( mype == 0 && thread == 0 ){
    printf("\n");
    border_print();
    center_print("PAPI COUNTER RESULTS", 79);
    border_print();
    printf("Count          \tSmybol      \tDescription\n");
  }
  counter_stop(&eventset, num_papi_events);
#endif
	
#ifndef VERIFICATION
  HCC_SYNC(double, check_t, 1, check);
  vhash = check[0];
#else
  HCC_SYNC(double, verify_p_energy_t, in.lookups, verify_p_energy);
  HCC_SYNC(int, verify_mat_t, in.lookups, verify_mat);
  HCC_SYNC(double, verify_macro_xs_vector_t, in.lookups*5, verify_macro_xs_vector);
  
  for(int i = 0; i < in.lookups; ++i){
    char line[256];
    sprintf(line, "%.5lf %d %.5lf %.5lf %.5lf %.5lf %.5lf",
	    verify_p_energy[i], verify_mat[i],
	    verify_macro_xs_vector[i*5 + 0],
	    verify_macro_xs_vector[i*5 + 1],
	    verify_macro_xs_vector[i*5 + 2],
	    verify_macro_xs_vector[i*5 + 3],
	    verify_macro_xs_vector[i*5 + 4]);
    vhash += hash((unsigned char*)line, 10000);
  }
#endif
	
#ifndef PAPI
  if( mype == 0)	{	
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
