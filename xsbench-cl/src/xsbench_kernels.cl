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
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

//#define VERIFICATION

typedef struct _OCLKernelParamsMacroXSStruct
{
  unsigned lookups;
  int n_isotopes;
  int n_gridpoints;
  int n_mat;
  int e_sort_tile;
  unsigned enlarged_lookups;
} OCLKernelParamsMacroXSS;

typedef struct NuclideGridPointS{
	double energy;
	
	double total_xs;
	double elastic_xs;
	double absorbtion_xs;
	double fission_xs;
	double nu_fission_xs;
} NuclideGridPoint;

// (fixed) binary search for energy on unionized energy grid
// returns lower index
int grid_search( int n, double quarry, const __global double * energy)
{
  int lowerLimit = 0;
  int upperLimit = n-1;
  int examinationPoint;
  int length = upperLimit - lowerLimit;
  
  while( length > 1 )
    {
      examinationPoint = lowerLimit + ( length / 2 );
      
      if( energy[examinationPoint] > quarry )
	      upperLimit = examinationPoint;
      else
	      lowerLimit = examinationPoint;
      
      length = upperLimit - lowerLimit;
    }
  
  return lowerLimit;
}


__global uint* get_mat_data(__global const char *params_in, uint * n_nuc, uint * mat_off, uint mat_id)
{
  __global const OCLKernelParamsMacroXSS *params = (__global const OCLKernelParamsMacroXSS *)params_in; 
  __global uint* mats =( __global uint*)(params_in + sizeof(OCLKernelParamsMacroXSS));
  __global uint *mat_data = mats + params->n_mat *2;

  *n_nuc = mats[mat_id*2];
  *mat_off = mats[mat_id*2 + 1];
  return mat_data;
}

void get_nuc_data(__global uint *mat_data, uint* nuc, double* nuc_conc, uint n_nucs, uint mat_off, uint nuc_id)
{
  __global uint * pnuc_ids = mat_data + (mat_off >> 2);
  *nuc = pnuc_ids[nuc_id];
  __global double * pnuc_concs = (__global double*)(pnuc_ids + n_nucs);
  *nuc_conc = pnuc_concs[nuc_id];
}	

__kernel 
void calculate_xs_null(  __global const OCLKernelParamsMacroXSS *params, 
			 __global double *macro_xs_vector
			     )
{
  int i = get_global_id(0);

  if(i > params->lookups) return;
  macro_xs_vector[i*5] = (double)i;
}

__kernel
void calculate_xs_param(__global const char *params_in, 
			__global double *macro_xs_vector
			     )
{
  int i = get_global_id(0);

}

__kernel
void calculate_xs_input(__global const char *params_in,
                 __global const double *penergy,
		 __global const int *pmat,
		 __global double *macro_xs_vector
			     )
{
  int i = get_global_id(0);

}
__kernel
void calculate_xs_energy_grid(__global const char *params_in,
                 __global const double *penergy,
		 __global const int *pmat,
                 __global const double *energyGrid,
		 __global const uint *xs_nucGridPtrs,
		 __global double *macro_xs_vector
			     )
{
  int i = get_global_id(0);

}

// Calculates the microscopic cross section for a given nuclide & energy
void calculate_micro_xs(   double p_energy, int nuc, uint nucGrid_off,
                           int n_gridpoints,
                           const __global NuclideGridPoint *xs_nucGrid,
                           double *xs_vector )
{     
  // Variables
  double f;
  uint low, high;
  // of the end
  nucGrid_off -= (nucGrid_off == n_gridpoints -1 )? 1 : 0;
  low = nuc * n_gridpoints + nucGrid_off;
  
  high = low + 1;
  
  // calculate the re-useable interpolation factor
  f = (xs_nucGrid[high].energy - p_energy) / (xs_nucGrid[high].energy - xs_nucGrid[low].energy);
  
  // Total XS
  xs_vector[0] = xs_nucGrid[high].total_xs - f * (xs_nucGrid[high].total_xs - xs_nucGrid[low].total_xs);
  
  
  // Elastic XS
  xs_vector[1] = xs_nucGrid[high].elastic_xs - f * (xs_nucGrid[high].elastic_xs - xs_nucGrid[low].elastic_xs);
  
  
  // Absorbtion XS
  xs_vector[2] = xs_nucGrid[high].absorbtion_xs - f * (xs_nucGrid[high].absorbtion_xs - xs_nucGrid[low].absorbtion_xs);
  
  
  // Fission XS
  xs_vector[3] = xs_nucGrid[high].fission_xs - f * (xs_nucGrid[high].fission_xs - xs_nucGrid[low].fission_xs);
  
  // Nu Fission XS
  xs_vector[4] = xs_nucGrid[high].nu_fission_xs - f * (xs_nucGrid[high].nu_fission_xs - xs_nucGrid[low].nu_fission_xs);
  
  
  //test
  /*	
	if( omp_get_thread_num() == 0 )
	{
	printf("Lookup: Energy = %lf, nuc = %d\n", p_energy, nuc);
	printf("e_h = %lf e_l = %lf\n", high->energy , low->energy);
	printf("xs_h = %lf xs_l = %lf\n", high->elastic_xs, low->elastic_xs);
	printf("total_xs = %lf\n\n", xs_vector[1]);
	}
  */
	
}


//#define VERIFICATION
__kernel
void unionized_grid_search(__global const char *params_in,
                __global const double *sorted_energy,
		__global const uint* unsorted_eng_index,
                __global const double *energyGrid,
		__global uint *energyGripIndxes
			     )
{
  int i = get_global_id(0);
  __global const OCLKernelParamsMacroXSS *params = (__global const OCLKernelParamsMacroXSS *)params_in; 
  double energy;
  uint energy_id;
  uint unsorted_id;
  uint n_unionized_grid_points = params->n_isotopes*params->n_gridpoints;

  if(i >= params->lookups) return; 

  energy = sorted_energy[i];


  energy_id = grid_search(n_unionized_grid_points, energy, energyGrid);

  uint unsorted_i = unsorted_eng_index[i];

// unsort index and energy
  energyGripIndxes[i] = energy_id;

}




__kernel
void calculate_xs_loop(__global const char *params_in,
                 __global const double *penergy,
		 __global const int *pmat,
                 __global const double *energyGrid,
		 __global const uint *xs_nucGridPtrs,
		 __global const NuclideGridPoint *xs_nucGrid,
		 __global double *macro_xs_vector
			     )
{
  int i = get_global_id(0);
  __global const OCLKernelParamsMacroXSS *params = (__global const OCLKernelParamsMacroXSS *)params_in; 
  double xs_vector[5];
  double xs_vector_out[5];
  double energy;
  uint mat_id;
  __global uint *mat_data;
  uint n_nucs;
  uint mat_off;
  uint energy_id;
  const __global uint * xs_nuc_ptrs;
  uint n_unionized_grid_points = params->n_isotopes*params->n_gridpoints;

  for( int i = 0; i < 5; i++)
  { 
    xs_vector_out[i] = 0.;
  }
  if(i >= params->lookups) return; 

  energy = penergy[i];

  energy_id = grid_search(n_unionized_grid_points, energy, energyGrid);
  xs_nuc_ptrs = xs_nucGridPtrs +  energy_id * params->n_isotopes;

  mat_id = pmat[i];
  mat_data = get_mat_data(params_in, &n_nucs, &mat_off, mat_id);


#if 0 //def VERIFICATION
        if( i == 0 )
		{
			printf("\n%d %lf %d %d %d\n", i, energy, energy_id, mat_id, n_nucs); 
		}

#endif

  for(int n = 0; n < n_nucs; n++)
  { 
     uint nuc;
     double nuc_conc;
     get_nuc_data(mat_data, &nuc, &nuc_conc, n_nucs, mat_off, n);
     uint nucGrid_off = xs_nuc_ptrs[nuc];
     calculate_micro_xs( energy, nuc, nucGrid_off,
                         params->n_gridpoints,
                         xs_nucGrid,
                         xs_vector );
		   
#if 0 // def VERIFICATION
           if( i == 0 )
		   {
		      printf("%d %d %lf %lf %lf %lf %lf %lf\n", n, nuc, nuc_conc, xs_vector[0],xs_vector[1],xs_vector[2],xs_vector[3],xs_vector[4]);
		   }
#endif
     for (int c = 0; c < 5; c++){
       xs_vector_out[c] += xs_vector[c] * nuc_conc;
     }
  }

  for (int c = 0; c < 5; c++) {
#ifdef VERIFICATION_BUFFER
    macro_xs_vector[i*5 + c] = xs_vector_out[c];
#else
    macro_xs_vector[0 + c] = xs_vector_out[c];
#endif
  }
}


__kernel
void calculate_xs_sorted(__global const char *params_in,
                         __global const double *penergy,
                         __global const uint *energyId,
			 __global const int *mat,
			 __global const int *unsorted_eng_indx,
			 __global const uint *xs_nucGridPtrs,
			 __global const NuclideGridPoint *xs_nucGrid,
			 __global double *macro_xs_vector
			     )
{
  int i = get_global_id(0);

  __global const OCLKernelParamsMacroXSS *params = (__global const OCLKernelParamsMacroXSS *)params_in; 
  double xs_vector[5];
  double xs_vector_out[5];
  uint mat_id;
  __global uint *mat_data;
  uint n_nucs;
  uint mat_off;
  double energy;
  uint energy_id;
  const __global uint * xs_nuc_ptrs;

  for( int i = 0; i < 5; i++) { 
    xs_vector_out[i] = 0.;
  }
  if(i >= params->lookups) return;

// find original usorted position of the energy/index pair
  uint unsorted_mat_i = unsorted_eng_indx[i];
// get original material related to the pair

  mat_id = mat[unsorted_mat_i];
  mat_data = get_mat_data(params_in, &n_nucs, &mat_off, mat_id);

// take energy and index from the sorted array
  energy = penergy[i];
  energy_id = energyId[i];

// get an array of pointers into nuc grid
  xs_nuc_ptrs = xs_nucGridPtrs +  energy_id * params->n_isotopes;

  for(int n = 0; n < n_nucs; n++) { 
     uint nuc;
     double nuc_conc;
     uint nucGrid_off;
     get_nuc_data(mat_data, &nuc, &nuc_conc, n_nucs, mat_off, n);
     nucGrid_off = xs_nuc_ptrs[nuc];
     calculate_micro_xs(  energy, nuc, nucGrid_off, params->n_gridpoints, xs_nucGrid, xs_vector );

		   
     for (int c = 0; c < 5; c++) {
        xs_vector_out[c] += xs_vector[c] * nuc_conc;
     }

  }

  for (int c = 0; c < 5; c++) {
#ifdef VERIFICATION_BUFFER
// return result in unsorted original order 
     macro_xs_vector[unsorted_mat_i*5 + c] = xs_vector_out[c];
#else
     macro_xs_vector[0 + c] = xs_vector_out[c];
#endif
  }
}



void bitonicSort(__global double * tiledArray,
                  uint stage, 
                  uint passOfStage,
                  uint direction,
		  uint threadId)
{
    uint sortIncreasing = direction;
    
    uint pairDistance = 1 << (stage - passOfStage);
    uint blockWidth   = 2 * pairDistance;

    uint leftId = (threadId % pairDistance) + (threadId / pairDistance) * blockWidth;

    uint rightId = leftId + pairDistance;
    
    double leftElement = tiledArray[leftId];
    double rightElement = tiledArray[rightId];

    uint sameDirectionBlockWidth = (1 << stage);
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1) {
        sortIncreasing = 1 - sortIncreasing;
    }

    double greater;
    double lesser;
    if(leftElement > rightElement) {
        greater = leftElement;
        lesser  = rightElement;
    }
    else {
        greater = rightElement;
        lesser  = leftElement;
    }
    
    if(sortIncreasing) {
        tiledArray[leftId]  = lesser;
        tiledArray[rightId] = greater;
    }
    else {
        tiledArray[leftId]  = greater;
        tiledArray[rightId] = lesser;
    }
}

void bitonicSortLcl(__local double * tiledArray,
                  __local uint *indexArray,
                  uint stage, 
                  uint passOfStage,
                  uint direction,
		  uint threadId,
		  uint leftId,
		  uint rightId)
{
    uint sortIncreasing = direction;   
    double leftElement = tiledArray[leftId];
    double rightElement = tiledArray[rightId];

    uint sameDirectionBlockWidth = (1 << stage);
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1) {
        sortIncreasing = 1 - sortIncreasing;
    }

    double greater;
    double lesser;
    if(leftElement > rightElement) {
        greater = leftElement;
        lesser  = rightElement;
    }
    else {
        greater = rightElement;
        lesser  = leftElement;
    }
    
    double newLeftElement;
    if(sortIncreasing) {
        tiledArray[leftId]  = lesser;
        tiledArray[rightId] = greater;
	newLeftElement = lesser;
    }
    else {
        tiledArray[leftId]  = greater;
        tiledArray[rightId] = lesser;
	newLeftElement = greater;
    }
    if ( newLeftElement != leftElement ) {
	uint tindex = indexArray[leftId];
	indexArray[leftId] = indexArray[rightId];
	indexArray[rightId] = tindex;
    }
}

void bitonicSortGlbl(__global double * tiledArray,
                  __global uint *indexArray,
                  uint stage, 
                  uint passOfStage,
                  uint direction,
		  uint threadId,
		  uint leftId,
		  uint rightId)
{
    uint sortIncreasing = direction; 
    double leftElement = tiledArray[leftId];
    double rightElement = tiledArray[rightId];

    uint sameDirectionBlockWidth = (1 << stage);
    
    if((threadId/sameDirectionBlockWidth) % 2 == 1) {
        sortIncreasing = 1 - sortIncreasing;
    }

    double greater;
    double lesser;
    if(leftElement > rightElement) {
        greater = leftElement;
        lesser  = rightElement;
    }
    else {
        greater = rightElement;
        lesser  = leftElement;
    }
    
    double newLeftElement;
    if(sortIncreasing) {
        tiledArray[leftId]  = lesser;
        tiledArray[rightId] = greater;
	newLeftElement = lesser;
    }
    else {
        tiledArray[leftId]  = greater;
        tiledArray[rightId] = lesser;
	newLeftElement = greater;
    }
    if ( newLeftElement != leftElement ) {
	uint tindex = indexArray[leftId];
	indexArray[leftId] = indexArray[rightId];
	indexArray[rightId] = tindex;
    }
}





__kernel 
void bitonicSortTiled(__global double * theArray,
                      uint total_length, 
                      uint tile_length,
                      uint n_stages,
                      uint direction)
{
    uint groupID = get_group_id(0);
    __global double* tiledArray = &theArray[groupID * tile_length];
    for(uint stage = 0; stage < n_stages; ++stage) {
        
        // Every stage has stage + 1 passes
    for(uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
		// run over block
	uint threadId = get_local_id(0);
	for(uint sub_block = 0; sub_block < (1 << (n_stages - 8)); sub_block++, threadId += 256) {

                 bitonicSort(tiledArray,
                             stage, 
                             passOfStage,
                             direction,
							 threadId);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);

	}
    }

}

#define LOG2_BLOCK_SIZE 17
#define LG_MAX_LOCAL_DISTANCE 9
#define MAX_LOCAL_DISTANCE (1<<LG_MAX_LOCAL_DISTANCE)

void bitonicSort256(__global double *tiledArray, __local double * left_right, __global uint* indexArray, __local uint *lr_indexes,
			uint n_stages,
			uint direction,
			uint start_stage, uint end_stage,
	                uint start_pass, uint end_pass
			)
{
   uint lcl_id = get_local_id(0);
   uint pairDistance = MAX_LOCAL_DISTANCE;
   uint blockWidth = pairDistance*2;
// loop over local memory-size buffers
   for( uint threadId = lcl_id; threadId < (1 << (n_stages-1)); threadId += MAX_LOCAL_DISTANCE) {
// read into local memory once
	uint leftId0 = (threadId % pairDistance) + (threadId / pairDistance) * blockWidth;

	for(uint l = lcl_id; l < blockWidth; l += 256) {

		left_right[l] = tiledArray[leftId0 + l - lcl_id];
		lr_indexes[l] = indexArray[leftId0 + l - lcl_id];

	}
	barrier(CLK_LOCAL_MEM_FENCE);
	uint loop_counter = 0;
	for(uint stage = start_stage; stage < end_stage; ++stage) {
// all passes needed
           for(uint passOfStage = start_pass; passOfStage < stage+1; ++passOfStage, loop_counter++) {
	       uint pairDistance = 1 << (stage - passOfStage);
               uint blockWidth   = 2 * pairDistance;
	
               for( uint l = lcl_id, lcl_threadId = threadId; l < MAX_LOCAL_DISTANCE; l+=256, lcl_threadId+=256) {

	            uint leftId = (l % pairDistance) + (l / pairDistance) * blockWidth;
	            uint rightId = leftId + pairDistance;

                    bitonicSortLcl(left_right, lr_indexes,
                                    stage, 
                                    passOfStage,
                                    direction,
			            lcl_threadId,
				    leftId,
				    rightId);
	 	    }
		    barrier(CLK_LOCAL_MEM_FENCE);
	    	}
	   }
           for(uint l = lcl_id; l < blockWidth; l += 256) {

		    tiledArray[leftId0 + l - lcl_id] = left_right[l];
		    indexArray[leftId0 + l - lcl_id] = lr_indexes[l];

	   }
           barrier(CLK_GLOBAL_MEM_FENCE);
       }

}


__kernel 
void bitonicSortTiled3(__global const double * theArray,
                      __global double * sortedArray,
                      __global uint *theIndexArray,
                      uint total_length, 
                      uint tile_length,
                      uint n_stages,
                      uint direction)
{
	uint groupID = get_group_id(0);
	uint lcl_id = get_local_id(0);
	__global double* origArray = &theArray[groupID * tile_length];
	__global double* tiledArray = &sortedArray[groupID * tile_length];
	__global uint *indexArray = &theIndexArray[groupID * tile_length];
	__local double left_right[MAX_LOCAL_DISTANCE*2];
	__local uint lr_indexes[MAX_LOCAL_DISTANCE*2];
	uint passOfStage;
	uint index = lcl_id;	
	for(uint index = lcl_id; index < (1 << n_stages); index += 256)
	{
	   indexArray[index] = index + groupID * tile_length;
	   tiledArray[index] = origArray[index];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	bitonicSort256(tiledArray, left_right, indexArray, lr_indexes, n_stages,direction, 0, LG_MAX_LOCAL_DISTANCE + 1, 0, LG_MAX_LOCAL_DISTANCE + 1 );

	for(uint stage = LG_MAX_LOCAL_DISTANCE + 1; stage < n_stages; ++stage) {
        
        // Every stage has stage + 1 passes
           for(passOfStage = 0; (passOfStage < stage + 1) && ((stage - passOfStage ) > LG_MAX_LOCAL_DISTANCE); ++passOfStage) {
		uint pairDistance = 1 << (stage - passOfStage);
        	uint blockWidth   = 2 * pairDistance;
		// group
		for(uint threadId = lcl_id; threadId < (1 << (n_stages - 1)); threadId += 256) {

                   // left ID
                   uint leftId = (threadId % pairDistance) + (threadId / pairDistance) * blockWidth;	
                   // + global right distance

                   bitonicSortGlbl(tiledArray, indexArray, stage, passOfStage, direction, threadId, leftId, (leftId + pairDistance));

		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	   }
	   bitonicSort256(tiledArray, left_right, indexArray, lr_indexes, n_stages, direction, stage, stage+1, passOfStage, stage+1);

	}
}

