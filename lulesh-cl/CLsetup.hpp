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

#ifndef CLSETUP_HPP_
#define CLSETUP_HPP_ 

#include <CL/cl.h>
#include <string>
#include <map>

class CLsetup {
public:
	static cl_platform_id platform;
	static cl_device_id device;
	static cl_context context;
	static cl_command_queue queue;
	static cl_command_queue queueForReads;
	static cl_program program;
	static cl_int err;
    static std::map<std::string, cl_kernel> kernels;

	
	static inline void
	checkErr(cl_int err, const char * name)
	{
        std::string errMessage;
		if (err != CL_SUCCESS) {
            switch (err) {
                case -1:  errMessage = "CL_DEVICE_NOT_FOUND"; break;
				case -2:  errMessage = "CL_DEVICE_NOT_AVAILABLE"; break;
				case -3:  errMessage = "CL_COMPILER_NOT_AVAILABLE"; break;
				case -4:  errMessage = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
				case -5:  errMessage = "CL_OUT_OF_RESOURCES"; break;
				case -6:  errMessage = "CL_OUT_OF_HOST_MEMORY"; break;
				case -7:  errMessage = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
				case -8:  errMessage = "CL_MEM_COPY_OVERLAP"; break;
				case -9:  errMessage = "CL_IMAGE_FORMAT_MISMATCH"; break;
				case -10: errMessage = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
				case -11: errMessage = "CL_BUILD_PROGRAM_FAILURE"; break;
				case -12: errMessage = "CL_MAP_FAILURE"; break;
				case -13: errMessage = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
				case -14: errMessage = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
				case -15: errMessage = "CL_COMPILE_PROGRAM_FAILURE"; break;
				case -16: errMessage = "CL_LINKER_NOT_AVAILABLE"; break;
				case -17: errMessage = "CL_LINK_PROGRAM_FAILURE"; break;
				case -18: errMessage = "CL_DEVICE_PARTITION_FAILED"; break;
				case -19: errMessage = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;
				case -30: errMessage = "CL_INVALID_VALUE"; break;
				case -31: errMessage = "CL_INVALID_DEVICE_TYPE"; break;
				case -32: errMessage = "CL_INVALID_PLATFORM"; break;
				case -33: errMessage = "CL_INVALID_DEVICE"; break;
				case -34: errMessage = "CL_INVALID_CONTEXT"; break;
				case -35: errMessage = "CL_INVALID_QUEUE_PROPERTIES"; break;
				case -36: errMessage = "CL_INVALID_COMMAND_QUEUE"; break;
				case -37: errMessage = "CL_INVALID_HOST_PTR"; break;
				case -38: errMessage = "CL_INVALID_MEM_OBJECT"; break;
				case -39: errMessage = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
				case -40: errMessage = "CL_INVALID_IMAGE_SIZE"; break;
				case -41: errMessage = "CL_INVALID_SAMPLER"; break;
				case -42: errMessage = "CL_INVALID_BINARY"; break;
				case -43: errMessage = "CL_INVALID_BUILD_OPTIONS"; break;
				case -44: errMessage = "CL_INVALID_PROGRAM"; break;
				case -45: errMessage = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
				case -46: errMessage = "CL_INVALID_KERNEL_NAME"; break;
				case -47: errMessage = "CL_INVALID_KERNEL_DEFINITION"; break;
				case -48: errMessage = "CL_INVALID_KERNEL"; break;
				case -49: errMessage = "CL_INVALID_ARG_INDEX"; break;
				case -50: errMessage = "CL_INVALID_ARG_VALUE"; break;
				case -51: errMessage = "CL_INVALID_ARG_SIZE"; break;
				case -52: errMessage = "CL_INVALID_KERNEL_ARGS"; break;
				case -53: errMessage = "CL_INVALID_WORK_DIMENSION"; break;
				case -54: errMessage = "CL_INVALID_WORK_GROUP_SIZE"; break;
				case -55: errMessage = "CL_INVALID_WORK_ITEM_SIZE"; break;
				case -56: errMessage = "CL_INVALID_GLOBAL_OFFSET"; break;
				case -57: errMessage = "CL_INVALID_EVENT_WAIT_LIST"; break;
				case -58: errMessage = "CL_INVALID_EVENT"; break;
				case -59: errMessage = "CL_INVALID_OPERATION"; break;
				case -60: errMessage = "CL_INVALID_GL_OBJECT"; break;
				case -61: errMessage = "CL_INVALID_BUFFER_SIZE"; break;
				case -62: errMessage = "CL_INVALID_MIP_LEVEL"; break;
				case -63: errMessage = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
				case -64: errMessage = "CL_INVALID_PROPERTY"; break;
				case -65: errMessage = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
				case -66: errMessage = "CL_INVALID_COMPILER_OPTIONS"; break;
				case -67: errMessage = "CL_INVALID_LINKER_OPTIONS"; break;
				case -68: errMessage = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
                default: errMessage = "Unknown error!";
            }
			std::cerr << "ERROR: " << name << " (" << errMessage << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	CLsetup()
	{
	}

	~CLsetup()
	{
	}

	static void init(std::string kernelFile, unsigned int pl, unsigned int dev, int blockSize);
};

#endif
