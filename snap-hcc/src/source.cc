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

#include "source.h"


void compute_outer_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{
    unsigned int nang = problem->nang;
    unsigned int nx = rankinfo->nx;
    unsigned int ny = rankinfo->ny;
    unsigned int nz = rankinfo->nz;
    unsigned int ng = problem->ng;
    unsigned int cmom = problem->cmom;
    unsigned int nmom = problem->nmom;

    GPU_VEC(double) & outer_source = * buffers->outer_source;
    GPU_VEC(double) & fixed_source = * buffers->fixed_source;
    GPU_VEC(double) & scalar_flux = * buffers->scalar_flux;
    GPU_VEC(double) & scattering_matrix = * buffers->scattering_matrix;
    GPU_VEC(double) & scalar_flux_moments = * buffers->scalar_flux_moments;

    parallel_for_each(hc::extent<3>(nx, ny, nz),
                    [=, &outer_source,
                        &fixed_source,
                        &scalar_flux,
                        &scattering_matrix, 
                        &scalar_flux_moments]
                    (hc::index<3> idx) __HC__
    {
// 3D kernel, in local nx,ny,nz dimensions
// Probably not going to vectorise very well..
    const size_t i = idx[0];
    const size_t j = idx[1];
    const size_t k = idx[2];

    for (unsigned int g = 0; g < ng; g++)
    {
        // Set first moment to the fixed source
        outer_source(0,g,i,j,k) = fixed_source(g,i,j,k);

        // Loop over groups and moments to compute out-of-group scattering
        for (unsigned int g2 = 0; g2 < ng; g2++)
        {
            if (g == g2)
                continue;
            // Compute scattering source
            outer_source(0,g,i,j,k) += scattering_matrix(0,g2,g) * scalar_flux(g2,i,j,k);
            // Other moments
            unsigned int mom = 1;
            for (unsigned int l = 1; l < nmom; l++)
            {
                for (unsigned int m = 0; m < 2*l+1; m++)
                {
                    outer_source(mom,g,i,j,k) += scattering_matrix(l,g2,g) * scalar_flux_moments(mom-1,g2,i,j,k);
                    mom += 1;
                }
            }
        }
    }
    });

}


void compute_inner_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{
    unsigned int nang = problem->nang;
    unsigned int nx = rankinfo->nx;
    unsigned int ny = rankinfo->ny;
    unsigned int nz = rankinfo->nz;
    unsigned int ng = problem->ng;
    unsigned int cmom = problem->cmom;
    unsigned int nmom = problem->nmom;

    GPU_VEC(double) & outer_source = * buffers->outer_source;
    GPU_VEC(double) & inner_source = * buffers->inner_source;
    GPU_VEC(double) & scalar_flux = * buffers->scalar_flux;
    GPU_VEC(double) & scattering_matrix = * buffers->scattering_matrix;
    GPU_VEC(double) & scalar_flux_moments = * buffers->scalar_flux_moments;

    parallel_for_each(hc::extent<3>(nx, ny, nz),
                    [=, &outer_source,
                        &inner_source,
                        &scalar_flux,
                        &scattering_matrix, 
                        &scalar_flux_moments]
                    (hc::index<3> idx) __HC__
    {
// 3D kernel, in local nx,ny,nz dimensions
    const size_t i = idx[0];
    const size_t j = idx[1];
    const size_t k = idx[2];

    for (unsigned int g = 0; g < ng; g++)
    {
        // Set first moment to outer source plus scattering contribution of scalar flux
        inner_source(0,g,i,j,k) = outer_source(0,g,i,j,k) + scattering_matrix(0,g,g) * scalar_flux(g,i,j,k);

        // Set other moments similarly based on scalar flux moments
        unsigned int mom = 1;
        for (unsigned int l = 1; l < nmom; l++)
        {
            for (unsigned int m = 0; m < 2*l+1; m++)
            {
                inner_source(mom,g,i,j,k) = outer_source(mom,g,i,j,k) + scattering_matrix(l,g,g) * scalar_flux_moments(mom-1,g,i,j,k);
                mom += 1;
            }
        }
    }

    });
}
