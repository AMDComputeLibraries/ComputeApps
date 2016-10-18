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

#include <stdlib.h>

#include "problem.h"
#include "allocate.h"

void allocate_memory(struct problem * problem, struct rankinfo * rankinfo, struct memory * memory)
{

    // Allocate edge arrays
    std::vector<double> flux_i(problem->nang*problem->ng*rankinfo->ny*rankinfo->nz);
    std::vector<double> flux_j(problem->nang*problem->ng*rankinfo->nx*rankinfo->nz);
    std::vector<double> flux_k(problem->nang*problem->ng*rankinfo->nx*rankinfo->ny);
    memory->flux_i = flux_i;
    memory->flux_j = flux_j;
    memory->flux_k = flux_k;

    // Scalar flux
    // grid * ng
    size_t nsize = rankinfo->nx*rankinfo->ny*rankinfo->nz*problem->ng;
    std::vector<double> scalar_flux(nsize);
    std::vector<double> old_inner_scalar_flux(nsize);
    std::vector<double> old_outer_scalar_flux(nsize);
    memory->scalar_flux = scalar_flux;
    memory->old_inner_scalar_flux = old_inner_scalar_flux;
    memory->old_outer_scalar_flux = old_outer_scalar_flux;

    //Scalar flux moments
    if (problem->cmom-1 == 0)
        memory->scalar_flux_moments = (std::vector<double>) 0;
    else
        memory->scalar_flux_moments = std::vector<double>((problem->cmom-1)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,0.0);

    // Cosine coefficients
    std::vector<double> mu(problem->nang);
    std::vector<double> eta(problem->nang);
    std::vector<double> xi(problem->nang);
    memory->mu = mu;
    memory->eta = eta;
    memory->xi = xi;

    // Material cross section
    std::vector<double> mat_cross_section(problem->ng);
    memory->mat_cross_section = mat_cross_section;
}

// this routine no longer has anything to do, but we leave it in
void free_memory(struct memory * memory)
{
}
