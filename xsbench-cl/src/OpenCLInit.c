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

int CreateContext(struct OCL_ConfigS *config)
{
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstID;
  
  config->context = NULL;

  errNum = clGetPlatformIDs(1, &firstID, &numPlatforms);
  if(errNum != CL_SUCCESS) return errNum;

  cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM,
						(cl_context_properties)firstID,
						0};
  config->context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

  return errNum;
}

int CreateCommandQueue(struct OCL_ConfigS *config)
{
  cl_int errNum;
  cl_device_id *devices;
  size_t deviceBufferSize = -1;
  config->command_queue = NULL;

  errNum = clGetContextInfo(config->context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  if(errNum != CL_SUCCESS) return errNum;

  if(deviceBufferSize <=0) return -99999999;

  devices = (cl_device_id*)malloc(deviceBufferSize);

  errNum = clGetContextInfo(config->context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
  if(errNum != CL_SUCCESS) return errNum;
  
  config->command_queue = clCreateCommandQueue(config->context, devices[0], 0, NULL);
  if(config->command_queue == NULL)
    {
      free(devices);
      return -99999999;
    }

  config->dev_id = devices[0];

  free(devices);

  return CL_SUCCESS;
}

int CreateProgram(struct OCL_ConfigS *config)
{
  cl_int errNum;
  FILE *fp;
  long size;
  
  fp = fopen(config->kernelfilename, "r");
  if(fp == NULL)
    {
      printf("cannot open kernel source file: %s\n", config->kernelfilename);
      return -99999999;
    }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  rewind(fp);

  char *srcStr = (char*)malloc(size+1);
  memset(srcStr, '\0', size+1);

  fread(srcStr, size, 1, fp);

  config->program = clCreateProgramWithSource(config->context, 1, (const char**)&srcStr, NULL, NULL);

  if(config->program == NULL)
    {
      printf("Failed to create CL program from source\n");
      return -99999999;
    }

char *p_build_options = NULL;
#ifdef VERIFICATION_BUFFER
    p_build_options = "-DVERIFICATION_BUFFER";
#endif

  errNum = clBuildProgram(config->program, 0, NULL, p_build_options, NULL, NULL);
  if(errNum != CL_SUCCESS)
    {
      char buildLog[16384];
      clGetProgramBuildInfo(config->program, config->dev_id, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), 
			    buildLog, NULL);
      printf("Build Log\n %s\n\n", buildLog);
      clReleaseProgram(config->program);
      return errNum;
    }

  return CL_SUCCESS;
}

void GetInstanceInfo(struct OCL_ConfigS *config)
{
  int err;
  config->max_mem_alloc_size = 0;

  err = clGetDeviceInfo(config->dev_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
			sizeof(config->max_mem_alloc_size), &config->max_mem_alloc_size, NULL);
 
 if(err == CL_SUCCESS)
   printf("Maximum device memory alloc size (MBytes): %ul\n", config->max_mem_alloc_size/(1024*1024));
 

 err = clGetKernelWorkGroupInfo(config->kernel, config->dev_id, 
				CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), 
				&config->kernel_workgroup_size, 0);

 if(err == CL_SUCCESS)
   printf("Maximum kernel work group size = %d\n", (int)config->kernel_workgroup_size);

 return;
}

int CreateKernel(struct OCL_ConfigS *config, char *kernelname)
{
  cl_int err;
  config->kernel = clCreateKernel(config->program, kernelname, &err);

  return err;
}



int CreateOCLKernels(struct OCL_ConfigS *config)
{
  cl_int err;
  for (int i = 0; sKernel_nms[i] != NULL; i++)
  {
	  config->OCL_kernels[i] = clCreateKernel(config->program, sKernel_nms[i], &err);
	  if (err != CL_SUCCESS)
	  {
		  printf ("error creating kernel: %s\n",sKernel_nms[i]); 
	  }

  }

  return err;
}


int CreateOCLBuffer(cl_context context,OCLBuffer * buf, uint buffer_flags)
{
  int BUFFLAGS = buffer_flags;
  int err;

   if (!buf || buf->len == 0 )
   {
	err = -1;
        printf("error buf params\n");
        return err;
   }

   buf->sys = malloc(buf->len);

   if ( !buf->sys )
   {
	err = -1;
        printf("error buf allocation\n");
        return err;
   }

   buf->mem = clCreateBuffer(context, BUFFLAGS, buf->len, NULL, &err);
   if(err != CL_SUCCESS)
   {
      printf("error creating OCLBuffer: %d\n", err);
      return err;
   }

   return err;
}

int ReleaseOCLBuffer(OCLBuffer * buf)
{
  int err = CL_SUCCESS;


   if ( buf->sys )
   {
	   free(buf->sys);
   }

   if (buf->mem)
   {
	  err = clReleaseMemObject(buf->mem);
   }

   memset(buf, 0, sizeof(OCLBuffer));

   return err;
}


int CopyToDevice(cl_command_queue commandQueue, OCLBuffer * buf)
{
  int err;

  if(buf->len == 0 || !buf->sys || !buf->mem)
    {
      printf("wrong data\n");
      return(-1);
    }
  
  err = clEnqueueWriteBuffer(commandQueue, buf->mem, CL_TRUE,0, buf->len, buf->sys, 0, NULL, NULL);
  if(err != CL_SUCCESS)
    {
      printf("error writing data to device: %d\n", err);
      return err;
    }
  return(err);
}

int CopyFromDevice(cl_command_queue commandQueue, OCLBuffer * buf)
{
  int err;

  if(buf->len == 0 || !buf->sys || !buf->mem)
    {
      printf("wrong data\n");
      return(-1);
    }
  
  err = clEnqueueReadBuffer(commandQueue, buf->mem, CL_TRUE,0, buf->len, buf->sys, 0, NULL, NULL);
  if(err != CL_SUCCESS)
    {
      printf("error reading data from device: %d\n", err);
      return err;
    }
  return(err);
}

