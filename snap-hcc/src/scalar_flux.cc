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

#include "scalar_flux.h"
#include <iostream>

void compute_scalar_flux(
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{

    // get closest power of 2 to nang
    size_t power = 1 << (unsigned int)ceil(log2((double)problem->nang));

    unsigned int nang = problem->nang;
    unsigned int nx = rankinfo->nx;
    unsigned int ny = rankinfo->ny;
    unsigned int nz = rankinfo->nz;
    unsigned int ng = problem->ng;

    GPU_VEC(double) & angular_flux_out_0 = * buffers->angular_flux_out[0];
    GPU_VEC(double) & angular_flux_out_1 = * buffers->angular_flux_out[1];
    GPU_VEC(double) & angular_flux_out_2 = * buffers->angular_flux_out[2];
    GPU_VEC(double) & angular_flux_out_3 = * buffers->angular_flux_out[3];
    GPU_VEC(double) & angular_flux_out_4 = * buffers->angular_flux_out[4];
    GPU_VEC(double) & angular_flux_out_5 = * buffers->angular_flux_out[5];
    GPU_VEC(double) & angular_flux_out_6 = * buffers->angular_flux_out[6];
    GPU_VEC(double) & angular_flux_out_7 = * buffers->angular_flux_out[7];
    GPU_VEC(double) & angular_flux_in_0  = * buffers->angular_flux_in[0];
    GPU_VEC(double) & angular_flux_in_1  = * buffers->angular_flux_in[1];
    GPU_VEC(double) & angular_flux_in_2  = * buffers->angular_flux_in[2];
    GPU_VEC(double) & angular_flux_in_3  = * buffers->angular_flux_in[3];
    GPU_VEC(double) & angular_flux_in_4  = * buffers->angular_flux_in[4];
    GPU_VEC(double) & angular_flux_in_5  = * buffers->angular_flux_in[5];
    GPU_VEC(double) & angular_flux_in_6  = * buffers->angular_flux_in[6];
    GPU_VEC(double) & angular_flux_in_7  = * buffers->angular_flux_in[7];
    GPU_VEC(double) & quad_weights = * buffers->quad_weights;
    GPU_VEC(double) & scalar_flux = * buffers->scalar_flux;
    GPU_VEC(double) & velocity_delta = * buffers->velocity_delta;

    hc::extent<2> reduce_extent(power * ng, nx*ny*nz);
    hc::tiled_extent<2> reduce_grid = reduce_extent.tile(power,1);

    reduce_grid.set_dynamic_group_segment_size(sizeof(double)*power);

    parallel_for_each(reduce_grid,
                    [=, &velocity_delta,
                        &quad_weights,
                        &angular_flux_in_0,
                        &angular_flux_in_1,
                        &angular_flux_in_2,
                        &angular_flux_in_3,
                        &angular_flux_in_4,
                        &angular_flux_in_5,
                        &angular_flux_in_6,
                        &angular_flux_in_7,
                        &angular_flux_out_0,
                        &angular_flux_out_1,
                        &angular_flux_out_2,
                        &angular_flux_out_3,
                        &angular_flux_out_4,
                        &angular_flux_out_5,
                        &angular_flux_out_6,
                        &angular_flux_out_7,
                        &scalar_flux]
                    (hc::tiled_index<2> tidx) __HC__
    {
#define ANGULAR_FLUX_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define SCALAR_FLUX_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))


#define angular_flux_in_0(a,g,i,j,k) angular_flux_in_0[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_1(a,g,i,j,k) angular_flux_in_1[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_2(a,g,i,j,k) angular_flux_in_2[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_3(a,g,i,j,k) angular_flux_in_3[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_4(a,g,i,j,k) angular_flux_in_4[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_5(a,g,i,j,k) angular_flux_in_5[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_6(a,g,i,j,k) angular_flux_in_6[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_7(a,g,i,j,k) angular_flux_in_7[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_0(a,g,i,j,k) angular_flux_out_0[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_1(a,g,i,j,k) angular_flux_out_1[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_2(a,g,i,j,k) angular_flux_out_2[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_3(a,g,i,j,k) angular_flux_out_3[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_4(a,g,i,j,k) angular_flux_out_4[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_5(a,g,i,j,k) angular_flux_out_5[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_6(a,g,i,j,k) angular_flux_out_6[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_7(a,g,i,j,k) angular_flux_out_7[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define scalar_flux(g,i,j,k) scalar_flux[SCALAR_FLUX_INDEX((g),(i),(j),(k),ng,nx,ny)]
// We want to perform a weighted sum of angles in each cell in each energy group
// One work-group per cell per energy group, and reduce within a work-group
// Work-groups must be power of two sized
    const size_t a = tidx.local[0];
    const size_t g = tidx.tile[0];

    const size_t gid1 = tidx.global[1];
    const size_t local_size0 = tidx.tile_dim[0];
    const size_t i = gid1 % nx;
    const size_t j = (gid1 / nx) % ny;
    const size_t k = gid1 / (nx * ny);

#if !defined(HCC_BACKEND_AMDGPU)
// this is how to get a variable sized chunk of tile static.
// In this case, it is the only one, so we use the entire space.
    typedef __attribute__((address_space(3))) double group_t;
    group_t * local_scalar = (group_t *) hc::get_group_segment_addr(0);
#else
// but the hcc-lc for discrete GPUs doesn't implement get_group_segment_addr
    tile_static double local_scalar[1024];
#endif
    // Load into local memory
    local_scalar[a] = 0.0;
    tidx.barrier.wait();
    for (unsigned int aa = a; aa < nang; aa += local_size0)
    {
        const double w = quad_weights[aa];
        if (velocity_delta[g] != 0.0)
        {
            local_scalar[a] +=
                w * (0.5 * (angular_flux_out_0(aa,g,i,j,k) + angular_flux_in_0(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_1(aa,g,i,j,k) + angular_flux_in_1(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_2(aa,g,i,j,k) + angular_flux_in_2(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_3(aa,g,i,j,k) + angular_flux_in_3(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_4(aa,g,i,j,k) + angular_flux_in_4(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_5(aa,g,i,j,k) + angular_flux_in_5(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_6(aa,g,i,j,k) + angular_flux_in_6(aa,g,i,j,k))) +
                w * (0.5 * (angular_flux_out_7(aa,g,i,j,k) + angular_flux_in_7(aa,g,i,j,k)));
        }
        else
        {
            local_scalar[a] +=
                w * angular_flux_out_0(aa,g,i,j,k) +
                w * angular_flux_out_1(aa,g,i,j,k) +
                w * angular_flux_out_2(aa,g,i,j,k) +
                w * angular_flux_out_3(aa,g,i,j,k) +
                w * angular_flux_out_4(aa,g,i,j,k) +
                w * angular_flux_out_5(aa,g,i,j,k) +
                w * angular_flux_out_6(aa,g,i,j,k) +
                w * angular_flux_out_7(aa,g,i,j,k);
        }
    }
    tidx.barrier.wait();

    // Reduce in local memory
    for (unsigned int offset = local_size0 / 2; offset > 0; offset /= 2)
    {
        if (a < offset)
        {
            local_scalar[a] += local_scalar[a + offset];
        }
        tidx.barrier.wait();
    }

    // Save result
    if (a == 0)
    {
        scalar_flux(g,i,j,k) = local_scalar[0];
    }
    });

}

void compute_scalar_flux_moments(
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{

    // get closest power of 2 to nang
    size_t power = 1 << (unsigned int)ceil(log2((double)problem->nang));

    unsigned int nang = problem->nang;
    unsigned int nx = rankinfo->nx;
    unsigned int ny = rankinfo->ny;
    unsigned int nz = rankinfo->nz;
    unsigned int ng = problem->ng;
    unsigned int cmom = problem->cmom;

    GPU_VEC(double) & angular_flux_out_0 = * buffers->angular_flux_out[0];
    GPU_VEC(double) & angular_flux_out_1 = * buffers->angular_flux_out[1];
    GPU_VEC(double) & angular_flux_out_2 = * buffers->angular_flux_out[2];
    GPU_VEC(double) & angular_flux_out_3 = * buffers->angular_flux_out[3];
    GPU_VEC(double) & angular_flux_out_4 = * buffers->angular_flux_out[4];
    GPU_VEC(double) & angular_flux_out_5 = * buffers->angular_flux_out[5];
    GPU_VEC(double) & angular_flux_out_6 = * buffers->angular_flux_out[6];
    GPU_VEC(double) & angular_flux_out_7 = * buffers->angular_flux_out[7];
    GPU_VEC(double) & angular_flux_in_0  = * buffers->angular_flux_in[0];
    GPU_VEC(double) & angular_flux_in_1  = * buffers->angular_flux_in[1];
    GPU_VEC(double) & angular_flux_in_2  = * buffers->angular_flux_in[2];
    GPU_VEC(double) & angular_flux_in_3  = * buffers->angular_flux_in[3];
    GPU_VEC(double) & angular_flux_in_4  = * buffers->angular_flux_in[4];
    GPU_VEC(double) & angular_flux_in_5  = * buffers->angular_flux_in[5];
    GPU_VEC(double) & angular_flux_in_6  = * buffers->angular_flux_in[6];
    GPU_VEC(double) & angular_flux_in_7  = * buffers->angular_flux_in[7];
    GPU_VEC(double) & quad_weights = * buffers->quad_weights;
    GPU_VEC(double) & scat_coeff = * buffers->scat_coeff;
    GPU_VEC(double) & scalar_flux_moments = * buffers->scalar_flux_moments;
    GPU_VEC(double) & velocity_delta = * buffers->velocity_delta;

    hc::extent<2> reduce_extent(power * ng, nx*ny*nz);
    hc::tiled_extent<2> reduce_grid = reduce_extent.tile(power,1);

    reduce_grid.set_dynamic_group_segment_size(sizeof(double)*power);

    parallel_for_each(reduce_grid,
                    [=, &velocity_delta,
                        &quad_weights,
                        &angular_flux_in_0,
                        &angular_flux_in_1,
                        &angular_flux_in_2,
                        &angular_flux_in_3,
                        &angular_flux_in_4,
                        &angular_flux_in_5,
                        &angular_flux_in_6,
                        &angular_flux_in_7,
                        &angular_flux_out_0,
                        &angular_flux_out_1,
                        &angular_flux_out_2,
                        &angular_flux_out_3,
                        &angular_flux_out_4,
                        &angular_flux_out_5,
                        &angular_flux_out_6,
                        &angular_flux_out_7,
                        &scat_coeff,
                        &scalar_flux_moments]
                    (hc::tiled_index<2> tidx) __HC__
    {
#define SCALAR_FLUX_MOMENTS_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+((cmom-1)*(g))+((cmom-1)*(ng)*(i))+((cmom-1)*(ng)*(nx)*(j))+((cmom-1)*(ng)*(nx)*(ny)*(k)))
#define SCAT_COEFF_INDEX(a,l,o,nang,cmom) ((a)+((nang)*(l))+((nang)*(cmom)*o))

#define scalar_flux_moments(l,g,i,j,k) scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX((l),(g),(i),(j),(k),cmom,ng,nx,ny)]
#define scat_coeff(a,l,o) scat_coeff[SCAT_COEFF_INDEX((a),(l),(o),nang,cmom)]

// We want to perform a weighted sum of angles in each cell in each energy group for each moment
// One work-group per cell per energy group, and reduce within a work-group
// Work-groups must be power of two sized
    const size_t a = tidx.local[0];
    const size_t g = tidx.tile[0];

    const size_t gid1 = tidx.global[1];
    const size_t local_size0 = tidx.tile_dim[0];
    const size_t i = gid1 % nx;
    const size_t j = (gid1 / nx) % ny;
    const size_t k = gid1 / (nx * ny);

#if !defined(HCC_BACKEND_AMDGPU)
    typedef __attribute__((address_space(3))) double group_t;
    group_t * local_scalar = (group_t *) hc::get_group_segment_addr(0);
#else
// but the hcc-lc for discrete GPUs doesn't implement get_group_segment_addr
    tile_static double local_scalar[256];
#endif

    for (unsigned int l = 0; l < cmom-1; l++)
    {
        // Load into local memory
        local_scalar[a] = 0.0;
        for (unsigned int aa = a; aa < nang; aa += local_size0)
        {
            const double w = quad_weights[aa];
            if (velocity_delta[g] != 0.0)
            {
                local_scalar[a] +=
                    scat_coeff(aa,l+1,0) * w * (0.5 * (angular_flux_out_0(aa,g,i,j,k) + angular_flux_in_0(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,1) * w * (0.5 * (angular_flux_out_1(aa,g,i,j,k) + angular_flux_in_1(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,2) * w * (0.5 * (angular_flux_out_2(aa,g,i,j,k) + angular_flux_in_2(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,3) * w * (0.5 * (angular_flux_out_3(aa,g,i,j,k) + angular_flux_in_3(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,4) * w * (0.5 * (angular_flux_out_4(aa,g,i,j,k) + angular_flux_in_4(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,5) * w * (0.5 * (angular_flux_out_5(aa,g,i,j,k) + angular_flux_in_5(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,6) * w * (0.5 * (angular_flux_out_6(aa,g,i,j,k) + angular_flux_in_6(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,7) * w * (0.5 * (angular_flux_out_7(aa,g,i,j,k) + angular_flux_in_7(aa,g,i,j,k)));
            }
            else
            {
                local_scalar[a] +=
                    scat_coeff(aa,l+1,0) * w * angular_flux_out_0(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,1) * w * angular_flux_out_1(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,2) * w * angular_flux_out_2(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,3) * w * angular_flux_out_3(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,4) * w * angular_flux_out_4(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,5) * w * angular_flux_out_5(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,6) * w * angular_flux_out_6(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,7) * w * angular_flux_out_7(aa,g,i,j,k);
            }
        }

        tidx.barrier.wait();

        // Reduce in local memory
        for (unsigned int offset = local_size0 / 2; offset > 0; offset /= 2)
        {
            if (a < offset)
            {
                local_scalar[a] += local_scalar[a + offset];
            }
            tidx.barrier.wait();
        }
        // Save result
        if (a == 0)
        {
            scalar_flux_moments(l,g,i,j,k) = local_scalar[0];
        }
    }

    });

}

void copy_back_scalar_flux(
    struct problem *problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers,
    vdouble& scalar_flux,
    bool blocking
    )
{
#if ARRAY_VIEW
     buffers->scalar_flux.synchronize();
#else
     copy(*buffers->scalar_flux,scalar_flux.data());
#endif
}
