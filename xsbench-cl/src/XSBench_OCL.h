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

#ifndef __XSBENCH_OCL__
#define __XSBENCH_OCL__


#if defined (__USE_AMD_OCL__)
typedef struct {
	uint len;
	void * sys;
	cl_mem mem;
} OCLBuffer;

typedef struct{
	OCLBuffer nucGP[1];
} OCLNuclideGridPoint;

typedef struct{
	 OCLBuffer enGP[2];
} OCLEneryGridPoint;

typedef struct{
	OCLBuffer mat_concs[1];
} OCLXSConsts;

typedef struct{
	OCLBuffer xs_output[1];
} OCLXSOutput;

typedef struct{
// energy
// mat
// original unsorted index
// indexed in unionized grid
	OCLBuffer xs_input[5];
} OCLXSInput;

#endif
#define __MAX_N_KERNELS__ 64
struct OCL_ConfigS
{
  cl_context context;
  cl_command_queue command_queue;
  cl_kernel kernel;
  cl_program program;
  cl_device_id dev_id;
  cl_ulong max_mem_alloc_size;
  size_t kernel_workgroup_size;
  char *kernelfilename;
#if defined (__USE_AMD_OCL__)
  cl_kernel OCL_kernels[__MAX_N_KERNELS__];
  OCLNuclideGridPoint nuGripPoints;
  OCLEneryGridPoint enGripPoints;
  OCLXSConsts consts;
  OCLXSOutput outputs;
  OCLXSInput inputs;
#endif
};

int CreateContext(struct OCL_ConfigS *config);
int CreateCommandQueue(struct OCL_ConfigS *config);
int CreateProgram(struct OCL_ConfigS *config);
void GetInstanceInfo(struct OCL_ConfigS *config);
int CreateKernel(struct OCL_ConfigS *config, char *kernelname);
struct OCL_ConfigS* GetOCLConfig(void);

#if defined (__USE_AMD_OCL__)
typedef void (*OCL_PATH)(OCLKernelParamsMacroXSS *kern_params, const char * path_nm, uint tloops);
static const char *sKernel_nms[] = 
{

	"calculate_xs_loop",
        "bitonicSortTiled3",
	"unionized_grid_search",
	"calculate_xs_sorted",
	NULL
};

cl_kernel GetXSBenchOCLKernel(const char * name);
int CreateOCLKernels(struct OCL_ConfigS *config);
int CreateOCLBuffer(cl_context context,OCLBuffer * buf, uint buffer_flags);
int CopyToDevice(cl_command_queue commandQueue, OCLBuffer * buf);
int CopyFromDevice(cl_command_queue commandQueue, OCLBuffer * buf);
int ReleaseOCLBuffer(OCLBuffer * buf);

#endif


#endif
