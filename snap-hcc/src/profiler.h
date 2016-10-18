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

#pragma once

/** \file
* \brief Routines and data structures to time sections of the code
*/

#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "global.h"
//#include "ocl_global.h"

static const bool profiling = true;

/** \brief Timers */
struct timers
{
    /** \brief Time to setup MPI, OpenCL and initialise memory */
    double setup_time;

    /** \brief Total time running the outer source kernel */
    double outer_source_time;

    /** \brief Total time running the inner source kernel */
    double inner_source_time;

    /** \brief Total time sweeping (the sweep kernel and MPI calls) */
    double sweep_time;

    /** \brief Total time transfering data over PCIe during sweep */
    double sweep_transfer_time;

    /** \brief Total time calculating scalar flux and scalar flux moments */
    double reduction_time;

    /** \brief Time from start of first timestep to end of last timestep */
    double simulation_time;

    /** \brief Total time calculating convergence of scalar flux */
    double convergence_time;

    /** \brief Total time calculating the parameters each outer */
    double outer_params_time;
};

extern double sweep_mpi_time;
extern double sweep_mpi_recv_time;

#if 0
/** @{ \brief OpenCL Events used to later read compute timings if profiling is on */
cl_event outer_source_event;
cl_event inner_source_event;

cl_event scalar_flux_event;
cl_event scalar_flux_moments_event;

cl_event velocity_delta_event;
cl_event denominator_event;

cl_event flux_i_read_event;
cl_event flux_j_read_event;
cl_event flux_i_write_event;
cl_event flux_j_write_event;
#endif

/** @} */

/** \brief Get the current wallclock time */
double wtime(void);

/** \brief Update the timers every outer */
void outer_profiler(struct timers * timers);

/** \brief Update the timers every inner */
void inner_profiler(struct timers * timers, struct problem * problem);

/** \brief Update the timers every chunk with transfer times */
void chunk_profiler(struct timers * timers);

