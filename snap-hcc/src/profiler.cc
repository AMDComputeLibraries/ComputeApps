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

// profiling for HCC is not implemented.  We leave the OpenCL code for
// reference.

#include "profiler.h"

double wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1.0E-6;
}

void outer_profiler(struct timers * timers)
{
    if (!profiling)
        return;
#if 0
    cl_int err;

    // Times are in nanoseconds
    cl_ulong tick, tock;

    // Get outer souce update times
    err = clGetEventProfilingInfo(outer_source_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
    check_ocl(err, "Getting outer source start time");
    err = clGetEventProfilingInfo(outer_source_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
    check_ocl(err, "Getting outer source end time");
    timers->outer_source_time += (double)(tock - tick) * 1.0E-9;

    // Get outer parameter times
    // Start is velocity delta start, end is denominator end
    err = clGetEventProfilingInfo(velocity_delta_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
    check_ocl(err, "Getting velocity delta start time");
    err = clGetEventProfilingInfo(denominator_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
    check_ocl(err, "Getting denominator end time");
    timers->outer_params_time += (double)(tock - tick) * 1.0E-9;
#endif
}

void inner_profiler(struct timers * timers, struct problem * problem)
{
    if (!profiling)
        return;
#if 0
    cl_int err;

    // Times are in nanoseconds
    cl_ulong tick, tock;

    // Get inner source update times
    err = clGetEventProfilingInfo(inner_source_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
    check_ocl(err, "Getting inner source start time");
    err = clGetEventProfilingInfo(inner_source_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
    check_ocl(err, "Getting inner source end time");
    timers->inner_source_time += (double)(tock - tick) * 1.0E-9;

    // Get scalar flux reduction times
    err = clGetEventProfilingInfo(scalar_flux_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
    check_ocl(err, "Getting scalar flux start time");
    err = clGetEventProfilingInfo(scalar_flux_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
    check_ocl(err, "Getting scalar flux end time");
    timers->reduction_time += (double)(tock - tick) * 1.0E-9;
    if (problem->cmom-1 > 0)
    {
        err = clGetEventProfilingInfo(scalar_flux_moments_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting scalar flux moments start time");
        err = clGetEventProfilingInfo(scalar_flux_moments_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting scalar flux moments end time");
        timers->reduction_time += (double)(tock - tick) * 1.0E-9;
    }
#endif
}


void chunk_profiler(struct timers * timers)
{
    if (!profiling)
        return;
#if 0
    cl_int err;

    // Times are in nanoseconds
    cl_ulong tick, tock;

    // Get recv writes
    if (flux_i_write_event)
    {
        err = clGetEventProfilingInfo(flux_i_write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux i write start time");
        err = clGetEventProfilingInfo(flux_i_write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux i write stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }

    if (flux_j_write_event)
    {
        err = clGetEventProfilingInfo(flux_j_write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux j write start time");
        err = clGetEventProfilingInfo(flux_j_write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux j write stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }

    // Get send reads
    if (flux_i_read_event)
    {
        err = clGetEventProfilingInfo(flux_i_read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux i read start time");
        err = clGetEventProfilingInfo(flux_i_read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux i read stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }

    if (flux_j_read_event)
    {
        err = clGetEventProfilingInfo(flux_j_read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux j read start time");
        err = clGetEventProfilingInfo(flux_j_read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux j read stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }
#endif
}

