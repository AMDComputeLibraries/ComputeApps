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

/** \file
* \brief Instantiate hcc array objects
*/


/** \brief 
This file included only once in main.
It instantiates all of the hc::array objects for the life of the program.
*/
    // Angular flux
    nsize = problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
    for (int i=0;i<8;i++)
    {
       buffers.angular_flux_in[i] = new GPU_VEC(double)(nsize);
       buffers.angular_flux_out[i] = new GPU_VEC(double)(nsize);
    }

    // Edge fluxes
    nsize = problem.nang*problem.ng*rankinfo.ny*rankinfo.nz;
    buffers.flux_i = new GPU_VEC(double)(nsize);
    nsize = problem.nang*problem.ng*rankinfo.nx*rankinfo.nz;
    buffers.flux_j = new GPU_VEC(double)(nsize);
    nsize = problem.nang*problem.ng*rankinfo.nx*rankinfo.ny;
    buffers.flux_k = new GPU_VEC(double)(nsize);

    // Scalar flux
    // grid * ng
    nsize = rankinfo.nx*rankinfo.ny*rankinfo.nz*problem.ng;
    buffers.scalar_flux = new GPU_VEC(double)(nsize);

    //Scalar flux moments
    if (problem.cmom-1 == 0)
    {
    // can't have a null reference, so we'll just allocate room for a double
        buffers.scalar_flux_moments = new GPU_VEC(double)(1);
    }
    else
    {
        nsize = (problem.cmom-1)*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
        buffers.scalar_flux_moments = new GPU_VEC(double)(nsize);
    }

    // Weights and cosines
    nsize = problem.nang;
    buffers.quad_weights = new GPU_VEC(double)(nsize);
    buffers.mu = new GPU_VEC(double)(nsize);
    buffers.eta = new GPU_VEC(double)(nsize);
    buffers.xi = new GPU_VEC(double)(nsize);

    // Scattering coefficient
    nsize = problem.nang*problem.cmom*8;
    buffers.scat_coeff = new GPU_VEC(double)(nsize);

    // Material cross section
    nsize = problem.ng;
    buffers.mat_cross_section= new GPU_VEC(double)(nsize);

    // Source terms
    nsize = problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
    buffers.fixed_source = new GPU_VEC(double)(nsize);
    nsize = problem.cmom*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
    buffers.outer_source = new GPU_VEC(double)(nsize);
    buffers.inner_source = new GPU_VEC(double)(nsize);

    // Scattering terms
    nsize = problem.nmom*problem.ng*problem.ng;
    buffers.scattering_matrix = new GPU_VEC(double)(nsize);

    // Diamond diference co-efficients
    nsize = 1;
    buffers.dd_i = new GPU_VEC(double)(nsize);
    nsize = problem.nang;
    buffers.dd_j = new GPU_VEC(double)(nsize);
    buffers.dd_k = new GPU_VEC(double)(nsize);

    // Velocities
    nsize = problem.ng;
    buffers.velocities = new GPU_VEC(double)(nsize);
    buffers.velocity_delta = new GPU_VEC(double)(nsize);

    // Denominator array
    nsize = problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
    buffers.denominator = new GPU_VEC(double)(nsize);
