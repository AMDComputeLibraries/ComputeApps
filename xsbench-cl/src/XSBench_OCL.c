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
#include "XSBench_OCL.h"


static struct OCL_ConfigS *OCLConfigs  = NULL;
static int instances = 0;

int SetupOpenCL(char *kernel_filename)
{
  if(instances == 0)
  {
     instances = 1;
     OCLConfigs = (struct OCL_ConfigS*)malloc(sizeof(struct OCL_ConfigS));
     memset(OCLConfigs, 0, sizeof(struct OCL_ConfigS));
  }

  OCLConfigs->context = 0;
  OCLConfigs->command_queue = 0;
  OCLConfigs->program = 0;
  OCLConfigs->dev_id = 0;
  OCLConfigs->kernel = 0;
  OCLConfigs->kernelfilename = kernel_filename;


  // initialize OpenCL here
  int err;
  err = CreateContext(OCLConfigs);
  if(err == 0) err = CreateCommandQueue(OCLConfigs);
  if(err == 0) err = CreateProgram(OCLConfigs);
  if(err == 0) err = CreateOCLKernels(OCLConfigs);

  if(err != 0)
  {
     printf("Could not initialize OpenCL: error = %d\n", err);
     return(err);
  }


  // get max device memory alloc size and kernel workgroup size
  GetInstanceInfo(OCLConfigs); 

  return(CL_SUCCESS);
}

int CleanupOpenCL()
{
  if(instances == 0) return 0;
  ReleaseOCLBuffer(OCLConfigs->consts.mat_concs);
  ReleaseOCLBuffer(OCLConfigs->outputs.xs_output);
  for (int i = 0; i < __MAX_N_KERNELS__; i++)
  {
     if ( OCLConfigs->OCL_kernels[i] )
     {
        clReleaseKernel(OCLConfigs->OCL_kernels[i]);
        OCLConfigs->OCL_kernels[i] = 0;
     }
  }

  //  clean up OpenCL here
  if(OCLConfigs->command_queue != 0) clReleaseCommandQueue(OCLConfigs->command_queue);
  if(OCLConfigs->kernel != 0) clReleaseKernel(OCLConfigs->kernel);
  if(OCLConfigs->program != 0) clReleaseProgram(OCLConfigs->program);
  if(OCLConfigs->context != 0) clReleaseContext(OCLConfigs->context);



  free(OCLConfigs);
  instances = 0;

  return 0;
}

struct OCL_ConfigS* GetOCLConfig(void)
{
  return(OCLConfigs);
}


cl_context GetXSBenchContext()
{
  if(instances == 0) return 0;
  
  return(OCLConfigs->context);
}

cl_command_queue GetXSBenchCommandQueue()
{
  if(instances == 0) return 0;

  return(OCLConfigs->command_queue);
}

cl_kernel GetXSBenchKernel()
{
  if(instances == 0) return 0;

  return(OCLConfigs->kernel);
}




void* getOCLOutput(void)
{
  struct OCL_ConfigS * config =  GetOCLConfig();
  OCLBuffer *buf = &config->outputs.xs_output[0]; 
  return(buf->sys);
}

cl_kernel GetXSBenchOCLKernel(const char * name)
{
cl_kernel ret = 0;
  for (int i = 0; sKernel_nms[i] != NULL; i++)
  {
      if ( !strcmp(sKernel_nms[i], name) )
      {
         ret = OCLConfigs->OCL_kernels[i];
      }
  }
  return (ret);
}

int AllocArrangeCopyConst(OCLKernelParamsMacroXSS *kern_params, const int *num_nucs, const int **mats, const double **concs)
{
  int err = CL_SUCCESS;
  struct OCL_ConfigS * config =  GetOCLConfig();
  int n_mat = kern_params->n_mat;
// totla structure size + material off + num_of_nuc
  int total_const_mem = sizeof (OCLKernelParamsMacroXSS) + n_mat * (sizeof(uint)*2);
  for (int i = 0; i < n_mat; i++)
  {
     total_const_mem += num_nucs[i] * (sizeof(uint) + sizeof(double)); 
  }
  OCLBuffer *buf = &config->consts.mat_concs[0];
  buf->len = total_const_mem;
  if ( (err = CreateOCLBuffer(config->context, buf, CL_MEM_READ_ONLY)) != CL_SUCCESS)
  {
     printf("Error allocating OCLBuffer in AllocArrangeCopyConst, size = %d\n",buf->len);
     exit(err);
  }
  memcpy(buf->sys, kern_params, sizeof(OCLKernelParamsMacroXSS));
  uint * consts_offs = (uint*)((char*)buf->sys + sizeof(OCLKernelParamsMacroXSS));
	// number of nucl per mat and offsets of nucs ids and their concs
  uint tot_off = 0;
  for ( int i = 0; i < n_mat; i++)
  {
	consts_offs[2*i] = num_nucs[i];
	consts_offs[2*i + 1] =  tot_off;
	tot_off += num_nucs[i] * (sizeof(uint) + sizeof(double));
  }

  uint * consts = consts_offs + n_mat * 2;
  for ( int i = 0; i < n_mat; i++)
  {
	uint n_nuc = consts_offs[2*i];
	uint mat_offset = consts_offs[2*i + 1];
	char * mat = ((char*)consts + mat_offset);
	uint * mat_nucs = (uint*)mat;
	double *nucs_concs = (double*)(mat_nucs + n_nuc);
	for( uint j = 0; j < n_nuc; j++)
	{
               	mat_nucs[j] = mats[i][j];
		nucs_concs[j] = concs[i][j];
	}
  }
// send it to device
  CopyToDevice(config->command_queue, buf);


  return(err);
}

int AllocOutput(OCLKernelParamsMacroXSS *kern_params)
{
 int err = CL_SUCCESS;
 struct OCL_ConfigS * config =  GetOCLConfig();
 OCLBuffer *buf = &config->outputs.xs_output[0];
#ifdef VERIFICATION_BUFFER
     buf->len = kern_params->lookups * 5 * sizeof(double);
#else
     buf->len = 5 * sizeof(double);
#endif
     if((err = CreateOCLBuffer(config->context, buf, CL_MEM_WRITE_ONLY))!= CL_SUCCESS)
     {
	printf("Error allocating OCLBuffer in AllocOutput, size = %d,%d\n",buf->len,err);
	exit(0);
     }
     return(err);
}

int AllocArrangeCopyEnergyGridPtrs(OCLKernelParamsMacroXSS *kern_params, const GridPoint * energy_grid)
{
 int err = CL_SUCCESS;
 struct OCL_ConfigS * config =  GetOCLConfig();
 	int n_mat = kern_params->n_mat;
	int n_unionized_grid_points = kern_params->n_isotopes*kern_params->n_gridpoints;
	// energy
	OCLBuffer *buf = &config->enGripPoints.enGP[0];
	buf->len = n_unionized_grid_points* sizeof(double);
	if ( (err = CreateOCLBuffer(config->context, buf, CL_MEM_READ_ONLY)) != CL_SUCCESS)
	{
		printf("Error allocating OCLBuffer in AllocArrangeCopyEnergyGridPtrs1, size = %d\n",buf->len);
		exit(0);
	}
	for(uint n = 0; n < n_unionized_grid_points; n++)
	{
          ((double*)buf->sys)[n] = energy_grid[n].energy;
	}
// send it to device
	CopyToDevice(config->command_queue, buf);


	// xs enery ptrs(offsets)
	buf = &config->enGripPoints.enGP[1];
	buf->len = n_unionized_grid_points* kern_params->n_isotopes*sizeof(uint);
	if ( (err = CreateOCLBuffer(config->context, buf, CL_MEM_READ_ONLY)) != CL_SUCCESS)
	{
		printf("Error allocating OCLBuffer in AllocArrangeCopyEnergyGridPtrs2, size = %d\n",buf->len);
		exit(0);
	}
	uint * ocl_sys = (uint*)buf->sys;
	for(uint m = 0; m < n_unionized_grid_points; m++)
	{

		memcpy(ocl_sys, energy_grid[m].xs_ptrs,kern_params->n_isotopes *sizeof(uint));
		ocl_sys += kern_params->n_isotopes;

	}
// send it to device
	CopyToDevice(config->command_queue, buf);


    return(err);
}

int AllocArrangeCopyNucGrid(OCLKernelParamsMacroXSS *kern_params, const NuclideGridPoint ** nuclide_grids)
{
 int err = CL_SUCCESS;
 struct OCL_ConfigS * config =  GetOCLConfig();
 	int n_mat = kern_params->n_mat;
	int n_grid_points = kern_params->n_isotopes*kern_params->n_gridpoints;
	// nuc Grid
	OCLBuffer *buf = &config->nuGripPoints.nucGP[0];
	buf->len = n_grid_points* sizeof(NuclideGridPoint);
	if ( (err = CreateOCLBuffer(config->context, buf, CL_MEM_READ_ONLY)) != CL_SUCCESS)
	{
		printf("Error allocating OCLBuffer in AllocArrangeCopyNucGrid, size = %d\n",buf->len);
		exit(0);
	}
        NuclideGridPoint *ocl_sys = (NuclideGridPoint *)buf->sys;
	for(int i = 0; i < kern_params->n_isotopes; i++, ocl_sys += kern_params->n_gridpoints)
	{
		memcpy(ocl_sys, nuclide_grids[i], kern_params->n_gridpoints * sizeof(NuclideGridPoint));
	}
// send it to device
	CopyToDevice(config->command_queue, buf);


    return(err);
}

typedef int (*cmp_func)(const void * a, const void * b );

typedef struct _double_index{
   double d;
   uint   i;
} DOUBLE_INDEX;

typedef struct _uint_index{
   uint   ui;
   uint   i;
} UINT_INDEX;

static
int double_index_compare( const void * a, const void * b )
{

	if( ((DOUBLE_INDEX*)a)->d > ((DOUBLE_INDEX*)b)->d )
		return 1;
	else if ( ((DOUBLE_INDEX*)a)->d < ((DOUBLE_INDEX*)b)->d)
		return -1;
	else
		return 0;
}

static
int uint_index_compare( const void * a, const void * b )
{

	if( ((UINT_INDEX*)a)->ui > ((UINT_INDEX*)b)->ui )
		return 1;
	else if (((UINT_INDEX*)a)->ui < ((UINT_INDEX*)b)->ui)
		return -1;
	else
		return 0;
}


// Compare function for two grid points. Used for sorting during init
static
int double_compare( const void * a, const void * b )
{

	if( *(double*)a > *(double*)b )
		return 1;
	else if ( *(double*)a < *(double*)b)
		return -1;
	else
		return 0;
}

static
int int_compare( const void * a, const void * b )
{

	if( *(int*)a > *(int*)b )
		return 1;
	else if ( *(int*)a < *(int*)b)
		return -1;
	else
		return 0;
}

static
int ulonglong_compare( const void * a, const void * b )
{

	if( *(unsigned long long *)a > *(unsigned long long *)b )
		return 1;
	else if ( *(unsigned long long *)a < *(unsigned long long *)b)
		return -1;
	else
		return 0;
}


static
void block_double_qsort(double *ptr, uint size, uint block_sz)
{

   for(uint off = 0, left = size; off < size; off+=block_sz, left-=block_sz)
   {
	uint len = (left>block_sz)? block_sz:left;
	qsort(&ptr[off], len, sizeof(double), double_compare);
   }
}

static
void block_qsort(void *ptr, uint type_sz, uint size, uint block_sz, cmp_func func )
{

   for(uint off = 0, left = size; off < size; off+=block_sz, left-=block_sz)
   {
	uint len = (left>block_sz)? block_sz:left;
	qsort(&((char*)ptr)[off*type_sz], len, type_sz, func);
   }
}

static
void block_int_qsort(int *ptr, uint size, uint block_sz)
{

   for(uint off = 0, left = size; off < size; off+=block_sz, left-=block_sz)
   {
	uint len = (left>block_sz)? block_sz:left;
	qsort(&ptr[off], len, sizeof(int), int_compare);
   }
}

int AllocCopyInputs(OCLKernelParamsMacroXSS *kern_params, double *pp_energy, int *pmat)
{
 int err = CL_SUCCESS;
#if 0 //def VERIFICATION
 uint e_sort_block = 16384;
#else
 uint e_sort_block = (1 << 17);
#endif
 struct OCL_ConfigS * config =  GetOCLConfig();
 OCLBuffer *eng_buf = &config->inputs.xs_input[0]  ;
 OCLBuffer *mat_buf = &config->inputs.xs_input[1]  ;
 OCLBuffer *index_buf2 = &config->inputs.xs_input[2]  ;
 OCLBuffer *index_buf3 = &config->inputs.xs_input[3]  ;
 OCLBuffer *index_buf4 = &config->inputs.xs_input[4]  ;
 uint actual_len = (kern_params->lookups > e_sort_block) ? kern_params->lookups : e_sort_block;
 double log_len = ceil(log((double)actual_len) / log(2.));
 uint ilog_len = (uint)log_len;
 uint len = (1 << ilog_len) * sizeof(double);

     eng_buf->len = len; //kern_params->lookups * sizeof(double);

     kern_params->e_sort_tile = e_sort_block;
     if (CL_SUCCESS != (err = CreateOCLBuffer(config->context, eng_buf, CL_MEM_READ_WRITE)))
     {
	printf("Error allocating OCLBuffer in AllocCopyInputs1, size = %d\n", eng_buf->len);
        exit(0);
     }

//	int (*cmp) (const void *, const void *);
     for(int i = 0; i < kern_params->lookups; i++)
     {
	 ((double*)eng_buf->sys)[i] = pp_energy[i];
	 if ( i < 16 )
	 {
		 printf("%.3lf ", ((double*)eng_buf->sys)[i]);
	 }
     }

     uint rem;
     uint blocked_lookups = ( (actual_len + kern_params->e_sort_tile - 1)/ kern_params->e_sort_tile) * kern_params->e_sort_tile;
// padding last sorted block with big values
// to push non-used entry into the end
     if ((rem = (blocked_lookups - kern_params->lookups)) > 0 )
     {
	 for (int i = blocked_lookups - 1, k = rem - 1; k >= 0; k--, i--)
	 {
		 ((double*)eng_buf->sys)[i] = 1.0;
	 }
     }



     printf("\n");

     kern_params->enlarged_lookups = blocked_lookups;

// send it to device
     CopyToDevice(config->command_queue, eng_buf);
     mat_buf->len = sizeof(int) * (1 << ilog_len); // kern_params->lookups;
     if (CL_SUCCESS != (err = CreateOCLBuffer(config->context, mat_buf, CL_MEM_READ_WRITE)))
     {
	printf("Error allocating OCLBuffer in AllocCopyInputs2, size = %d\n",mat_buf->len);
        exit(0);
     }
     for(int i = 0; i < kern_params->lookups; i++)
     {
	 ((int*)mat_buf->sys)[i] = pmat[i];
	 if ( i < 16 )
	 {
		 printf("%u ", ((int*)mat_buf->sys)[i]);
	 }
     }
     printf("\n");

     uint m_sort_block = e_sort_block;
     blocked_lookups = ( (actual_len + kern_params->e_sort_tile - 1)/ kern_params->e_sort_tile) * kern_params->e_sort_tile;
// padding last sorted block with big values
     if ((rem = (blocked_lookups - kern_params->lookups)) > 0 )
     {
	 for (int i = blocked_lookups - 1, k = rem - 1; k >= 0; k--, i--)
	 {
		 ((uint*)mat_buf->sys)[i] = 0x7fffffff;
	 }
     }
// send it to device
     CopyToDevice(config->command_queue, mat_buf);

// temp buffer keeping unsorted indexes
     index_buf2->len = sizeof(int) * (1 << ilog_len); // kern_params->lookups;
     if (CL_SUCCESS != (err = CreateOCLBuffer(config->context, index_buf2, CL_MEM_READ_WRITE)))
     {
	printf("Error allocating OCLBuffer in AllocCopyInputs3, size = %d\n",index_buf2->len);
        exit(0);
     }
// temp buffer keeping indexes into unionized energy array
     index_buf3->len = sizeof(int) * (1 << ilog_len); // kern_params->lookups;
     if (CL_SUCCESS != (err = CreateOCLBuffer(config->context, index_buf3, CL_MEM_READ_WRITE)))
     {
	printf("Error allocating OCLBuffer in AllocCopyInputs4, size = %d\n",index_buf3->len);
        exit(0);
     }
// temp buffer keep usorted energy
     index_buf4->len = len; // kern_params->lookups;
     if (CL_SUCCESS != (err = CreateOCLBuffer(config->context, index_buf4, CL_MEM_READ_WRITE)))
     {
	printf("Error allocating OCLBuffer in AllocCopyInputs5, size = %d\n",index_buf4->len);
        exit(0);
     }
	return(err);
}

int ReadOutput(OCLKernelParamsMacroXSS *kern_params)
{
  int err = CL_SUCCESS;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  OCLBuffer *buf = &config->outputs.xs_output[0];

  CopyFromDevice(commandQueue,buf);
  int n_mat = kern_params->n_mat;
  double * out_data = (double *)buf->sys;

  return(err);
}

static
void calculate_xs_null(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{
  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel kernel = GetXSBenchOCLKernel("calculate_xs_null");


  
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_null: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

 
  err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
                                0, NULL, NULL);
  

  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_null: failed to launch kernel with error %d\n", err);
      exit(1);
  }

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static
void calculate_xs_param(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{
  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel kernel = GetXSBenchOCLKernel( kernel_nm);


  
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_param: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }
  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_param: failed to launch kernel with error %d\n", err);
      exit(1);
  }

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static
void calculate_xs_input(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{


  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel kernel = GetXSBenchOCLKernel( kernel_nm);


  
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_input: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }
  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_input: failed to launch kernel with error %d\n", err);
      exit(1);
  }

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static
void calculate_xs_energy_grid(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{


  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel kernel = GetXSBenchOCLKernel( kernel_nm);


  
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_energy_grid: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }
  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_energy_grid: failed to launch kernel with error %d\n", err);
      exit(1);
  }

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static
void calculate_xs_loop(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{


  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel kernel = GetXSBenchOCLKernel( kernel_nm);

  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->nuGripPoints.nucGP[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_loop: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {256, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }
  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_loop: failed to launch kernel with error %d\n", err);
      exit(1);
  }

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static
void calculate_xs_shared(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{


  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel kernel = GetXSBenchOCLKernel( kernel_nm);


  
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[1].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->nuGripPoints.nucGP[0].mem);
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_shared: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }
  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_shared: failed to launch kernel with error %d\n", err);
      exit(1);
  }

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static
void sort_input_energy(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{

  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel sort_kernel = GetXSBenchOCLKernel( kernel_nm);
  uint acs = 1;
  uint n_stages = 0;
  uint n_groups = kern_params->enlarged_lookups / kern_params->e_sort_tile;
  size_t sort_local_work_size[2] = {256, 0};
  size_t sort_global_work_size[2] = {n_groups*sort_local_work_size[0], 0};

  uint temp;
  for(temp = kern_params->e_sort_tile; temp > 1; temp >>= 1)
        ++n_stages;

#if VERIFICATION
  DOUBLE_INDEX *d_i = (DOUBLE_INDEX*)malloc(sizeof(DOUBLE_INDEX) * kern_params->enlarged_lookups);
  double *d2 = (double*)malloc(sizeof(double) * kern_params->enlarged_lookups);
  double * in_energy = (double*)config->inputs.xs_input[0].sys;
  for ( uint i = 0; i < kern_params->enlarged_lookups; i++ )
  {
	  d2[i] = d_i[i].d = in_energy[i];
	  d_i[i].i = i;

  }
 
  block_qsort(d_i, sizeof(DOUBLE_INDEX), kern_params->enlarged_lookups, kern_params->e_sort_tile, double_index_compare );
#endif

  n_arg = 0;
// energy
  err = clSetKernelArg(sort_kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[0].mem);
// sorted energy
  err = clSetKernelArg(sort_kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[4].mem);
// unsorted indexes
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[2].mem);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &kern_params->lookups);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &kern_params->e_sort_tile);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &n_stages);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &acs);

    err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, sort_kernel, 1, NULL, sort_global_work_size, sort_local_work_size, 
			       0, NULL, NULL);
  }

  if(err != CL_SUCCESS)
    {
      printf("sort_input_energy: failed to launch kernel with error %d\n", err);
      exit(1);
    }

 // clFinish(commandQueue);
#if VERIFICATION
  OCLBuffer *buf_d = &config->inputs.xs_input[4];
    CopyFromDevice(commandQueue,buf_d);
  OCLBuffer *buf_i = &config->inputs.xs_input[2];
    CopyFromDevice(commandQueue,buf_i);
  double * out_d = (double *)buf_d->sys;
  uint * out_i = (uint *)buf_i->sys;
  for(uint i = 0; i < kern_params->lookups; i++ )
  {
	 if ( abs(out_d[i] -  d_i[i].d) > 0.00000 )
	 {
		 printf("e error: %d %lf %lf\n", i, out_d[i] ,  d_i[i].d);
		 break;
	 }
#if 1
//	 if ( d_i[i].i == 0 )
	 {
//		 printf("e: %d %d %lf\n", i, out_i[i], out_d[i] );
	 }
	 if ( out_i[i] !=  d_i[i].i  && d2[out_i[i]] != d2[d_i[i].i])
	 {
		 printf("ei error: %d %d %d\n", i, out_i[i] ,  d_i[i].i);
		 break;
	 }
#endif
  }
  free(d2);
  free(d_i);
#endif
  return;
}




static
void sort_input_material(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{
  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();
  cl_kernel sort_kernel = GetXSBenchOCLKernel( kernel_nm);
  uint acs = 1;
  uint n_stages = 0;
  uint n_groups = kern_params->enlarged_lookups / kern_params->e_sort_tile;
  size_t sort_local_work_size[2] = {256, 0};
  size_t sort_global_work_size[2] = {n_groups*sort_local_work_size[0], 0};

  uint temp;
  for(temp = kern_params->e_sort_tile; temp > 1; temp >>= 1)
        ++n_stages;

#if VERIFICATION
  unsigned long long *ui_i = (unsigned long long *)malloc(sizeof(unsigned long long) * kern_params->enlarged_lookups);
  uint *i2 = (uint*)malloc(sizeof(uint) * kern_params->enlarged_lookups);
  uint * in_mat = (uint*)config->inputs.xs_input[1].sys;
  for ( uint i = 0; i < kern_params->enlarged_lookups; i++ )
  {
	  i2[i] = in_mat[i];
	  ui_i[i] = ((unsigned long long)in_mat[i] << 32 ) | (unsigned long long)i;

  }
 
  block_qsort(ui_i, sizeof(unsigned long long), kern_params->enlarged_lookups, kern_params->e_sort_tile, ulonglong_compare );
#endif

  n_arg = 0;
// unsorted -> sorted material
  err = clSetKernelArg(sort_kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
// unsorted material indexes
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[2].mem);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &kern_params->lookups);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &kern_params->e_sort_tile);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &n_stages);
  err |= clSetKernelArg(sort_kernel, n_arg++, sizeof(uint), &acs);

  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, sort_kernel, 1, NULL, sort_global_work_size, sort_local_work_size, 
			       0, NULL, NULL);
  }

  if(err != CL_SUCCESS)
  {
      printf("sort_input_material: failed to launch kernel with error %d\n", err);
      exit(1);
  }
#if VERIFICATION
  OCLBuffer *buf_mat = &config->inputs.xs_input[1];
     CopyFromDevice(commandQueue,buf_mat);
  OCLBuffer *buf_i = &config->inputs.xs_input[2];
     CopyFromDevice(commandQueue,buf_i);
  uint * out_mat = (uint *)buf_mat->sys;
  uint * out_i = (uint *)buf_i->sys;
  for(uint i = 0; i < kern_params->lookups; i++ )
  {

//	 printf("M: %d %d\n", i,out_mat[i]);

	 if ( out_mat[i] -  (uint)(ui_i[i] >> 32) != 0 )
	 {
		 printf("m error: %d %d %d\n", i, out_mat[i] ,  (uint)(ui_i[i] >> 32));
		 break;
	 }
#if 1
	 if ( out_i[i] !=  (uint)ui_i[i]  && i2[out_i[i]] != i2[(uint)ui_i[i]])
	 {
		 printf("mi error: %d %d %d\n", i, out_i[i] ,  (uint)ui_i[i]);
		 break;
	 }
#endif
  }
  free(i2);
  free(ui_i);
#endif
  return;
}


static
void unionized_grid_search(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{


  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();



  cl_kernel kernel = GetXSBenchOCLKernel(kernel_nm);

  n_arg = 0;
// constrol/const
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
// sorted energy
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[4].mem);
// unsorted index
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[2].mem);
// unionized grid
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[0].mem);
// unsorted energy indexes
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[3].mem);


  if(err != CL_SUCCESS)
  {
      printf("unionized_grid_search: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {256, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }
  if(err != CL_SUCCESS)
  {
      printf("unionized_grid_search: failed to launch kernel with error %d\n", err);
      exit(1);
  }


  return;
}

static
void calculate_xs_sorted_mat(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{
  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();

  cl_kernel kernel = GetXSBenchOCLKernel(kernel_nm);

  n_arg = 0;
// control info
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
// unsorted energy
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[0].mem);
// usorted enegy indexes
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[3].mem);
// sorted mat 
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
// unsorted mat indx (original indexes)
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[2].mem);
// xs nuc grid pointers
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[1].mem);
// xs nuc grid
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->nuGripPoints.nucGP[0].mem);
// xs vector(s)
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_sorted_mat: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }

  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_sorted_mat: failed to launch kernel with error %d\n", err);
      exit(1);
  }
}

static
void calculate_xs_sorted(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{
  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();

  cl_kernel kernel = GetXSBenchOCLKernel(kernel_nm);

  n_arg = 0;
// control info
  err = clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->consts.mat_concs[0].mem);
// sorted energy
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[4].mem);
// sorted enegy indexes
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[3].mem);
// unsorted sorted mat 
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[1].mem);
// unsorted indx (original indexes)
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->inputs.xs_input[2].mem);
// xs nuc grid pointers
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->enGripPoints.enGP[1].mem);
// xs nuc grid
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->nuGripPoints.nucGP[0].mem);
// xs vector(s)
  err |= clSetKernelArg(kernel, n_arg++, sizeof(cl_mem), &config->outputs.xs_output[0].mem);


  if(err != CL_SUCCESS)
  {
      printf("calculate_xs_sorted: failed to set kernel arguments with error %d\n", err);
      exit(1);
  }

  size_t global_work_size[2] = {kern_params->lookups, 0};
  size_t local_work_size[2] = {64, 0};

 
  err = CL_SUCCESS;

  for (int i =0; i < (int)t_loops; i++)
  {
       err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 
			       0, NULL, NULL);
  }

  if(err != CL_SUCCESS)
    {
      printf("calculate_xs_sorted: failed to launch kernel with error %d\n", err);
      exit(1);
    }
}


static
void calculate_xs_withsort(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{


  int err;
  int n_arg = 0;
  struct OCL_ConfigS * config =  GetOCLConfig();
  cl_context context = GetXSBenchContext();
  cl_command_queue commandQueue = GetXSBenchCommandQueue();


// run the three kernels
  sort_input_energy(kern_params, "bitonicSortTiled3", t_loops);
  unionized_grid_search(kern_params, "unionized_grid_search", t_loops);
  calculate_xs_sorted(kern_params, "calculate_xs_sorted", t_loops);

  int ID = 0;
  //err = clSetEventCallback(event, CL_COMPLETE, &eventCallback, (void*)ID);
  //err = clSetUserEventStatus(usr_event, CL_SUCCESS);
  clFinish(commandQueue);

  return;
}

static OCL_PATH sOclPathes[] = 
{

calculate_xs_loop,
calculate_xs_withsort,
};

void calculate_xs_ocl(OCLKernelParamsMacroXSS *kern_params, const char * kernel_nm, uint t_loops)
{
	for(int p = 0; sKernel_nms[p] != NULL; p++ )
	{
	    if (!strcmp(sKernel_nms[p],kernel_nm))
	    {
		    sOclPathes[p](kern_params, kernel_nm, t_loops);
			break;
	    }
	}
}
