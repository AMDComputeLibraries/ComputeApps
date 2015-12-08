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

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <CL/cl.h>
#include "CLsetup.hpp"

using namespace std;

cl_platform_id CLsetup::platform;
cl_device_id CLsetup::device;
cl_context CLsetup::context;
cl_command_queue CLsetup::queue;
cl_command_queue CLsetup::queueForReads;
cl_program CLsetup::program;
std::map<std::string, cl_kernel> CLsetup::kernels;
cl_int CLsetup::err;

void
CLsetup::init(string kernelFile, unsigned int pl, unsigned int dev, int blockSize)
{
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(err, "clGetPlatformIDs");

    platforms = (cl_platform_id *)malloc(numPlatforms*sizeof(cl_platform_id));
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkErr(err, "clGetPlatformIDs");

    cerr << numPlatforms << " platform(s) detected" << endl;

    if ( pl > numPlatforms ) {
        cerr << "Desired platform is unavailable!" << endl;
        exit(EXIT_FAILURE);
    }

    platform = platforms[pl];

    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    checkErr(err, "clGetDeviceIDs()");

    devices = (cl_device_id *)malloc(numDevices*sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    checkErr(err, "clGetDeviceIDs()");

    if (dev > numDevices) {
        cerr << "Selected device is unavailable!" << endl;
        exit(EXIT_FAILURE);
    }

    device = devices[dev];

    char devName[1024];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
    checkErr(err, "clGetDeviceInfo()");
    cout << "Using device " << devName << endl;

    int computeUnits;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    checkErr(err, "clGetDeviceInfo()");
    size_t workGroupSize;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, NULL);

    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &err);
    checkErr(err, "clCreateContext()");

    queue = clCreateCommandQueue(context, device,  CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "clCreateCommandQueue()");
    queueForReads = clCreateCommandQueue(context, device,  CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "clCreateCommandQueue()");

    std::ifstream kernelFileHandler( kernelFile.c_str() );
    checkErr(kernelFileHandler.is_open() ? CL_SUCCESS:-1, "Opening kernels.cl failed");

    string kernels(
            std::istreambuf_iterator<char>(kernelFileHandler),
            (std::istreambuf_iterator<char>())
            );

    const char *kernelsString = kernels.c_str();
    program = clCreateProgramWithSource(context, 1, &kernelsString, NULL, &err);
    checkErr(err, "clCreateProgramWithSource()");

    string args = "-D BLOCKSIZE=";
    stringstream s;
    s << blockSize;
    args += s.str();
    args += " ";
#if defined DEBUG
    cout << "debug" << endl;
    args += "-g";
#elif defined SINGLE
    cout << "single precision" << endl;
    args += "-D SINGLE";
#else
    cout << "double precision" << endl;
#endif
    err = clBuildProgram(program, numDevices, devices, args.c_str(), NULL, NULL);
    checkErr(err, "clBuildProgram()");

    size_t paramValueSize = 0;
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &paramValueSize);
    checkErr(err, "clGetProgramBuildInfo()");
    char options[paramValueSize];
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_OPTIONS, sizeof(options), options, NULL);
    checkErr(err, "clGetProgramBuildInfo()");
    std::cout << "Build Options:\t" << options << std::endl;
    if(err != CL_SUCCESS){
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 0, NULL, &paramValueSize);
        checkErr(err, "clGetProgramBuildInfo()");
        char status[paramValueSize];
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), status, NULL);
        checkErr(err, "clGetProgramBuildInfo()");

        std::cout << "Build Status: " << status<< std::endl;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 0, NULL, &paramValueSize);
        checkErr(err, "clGetProgramBuildInfo()");
        char log[paramValueSize];
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(log), log, NULL);
        checkErr(err, "clGetProgramBuildInfo()");
        std::cout << "Build Log:\t " << log << std::endl;
    }
    cout << "OpenCL environment has been setup!" << endl;
}
