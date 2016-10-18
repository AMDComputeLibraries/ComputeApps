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

#include "problem.h"
#include "sweep.h"
#include <math.h>

#include <iostream>

#include <stdio.h>
#include <stdlib.h>


void init_quadrature_weights(
    const struct problem * problem,
    struct buffers * buffers
    )
{
    // Create tempoary on host for quadrature weights
    std::vector<double> m_quad_weights(problem->nang);
    // Uniform weights
    for (unsigned int a = 0; a < problem->nang; a++)
    {
        m_quad_weights[a] = 0.125 / (double)(problem->nang);
    }

    // Copy to device
    copy(m_quad_weights.data(),*buffers->quad_weights);
}
#define min(a,b) ((a>b)?b:a)
void print_stuff(
    GPU_VEC(double) &flux,
    size_t nsize,
    std::string name
    )
{
    // Create temporary on host for data
    std::vector<double> m_flux(nsize);
    // Copy from device
    copy(flux,m_flux.data());
    //print it out
    std::cout << name << std::endl;
    for (int i=0;i<min(32,nsize);i++)
       std::cout << m_flux[i] << " ";
    std::cout << std::endl;
}

void print_planes(
    const struct problem * problem,
    const struct buffers * buffers,
    const struct plane * planes,
    const unsigned int num_planes
    )
{
    // Create temporary on host for data
    //print it out
    std::cout << "planes" << std::endl;
    for (int p=0;p<num_planes;p++)
    {
       std::vector<struct cell_id> vstuff(planes[p].num_cells);

       std::cout << "  plane " << p << std::endl;
       // Copy from device
       copy(*buffers->planes[p],vstuff.data());
        for (int j=0;j<planes[p].num_cells;j++)
        {
          struct cell_id ptr = vstuff[j];
          std::cout << j << " " << ptr.i << " "
          << ptr.j << " " << ptr.k << std::endl;
        }
    }
    std::cout << std::endl;
}

void calculate_cosine_coefficients(const struct problem * problem,
    const struct buffers * buffers,
    vdouble& mu,
    vdouble& eta,
    vdouble& xi
    )
{

    double dm = 1.0 / problem->nang;

    mu[0] = 0.5 * dm;
    eta[0] = 1.0 - 0.5 * dm;
    double t = mu[0] * mu[0] + eta[0] * eta[0];
    xi[0] = sqrt(1.0 - t);

    for (unsigned int a = 1; a < problem->nang; a++)
    {
        mu[a] = mu[a-1] + dm;
        eta[a] = eta[a-1] - dm;
        t = mu[a] * mu[a] + eta[a] * eta[a];
        xi[a] = sqrt(1.0 - t);
    }
    // Copy to device
    copy(mu.data(),*buffers->mu);
    copy(eta.data(),*buffers->eta);
    copy(xi.data(),*buffers->xi);
}

void calculate_scattering_coefficients(
    const struct problem * problem,
    const struct buffers * buffers,
    const vdouble& mu,
    const vdouble& eta,
    const vdouble& xi
    )
{
    // Allocate temporary on host for scattering coefficients
    vdouble scat_coeff(problem->nang*problem->cmom*8);
    // (mu*eta*xi)^l starting at 0
    for (int id = 0; id < 2; id++)
    {
        double is = (id == 1) ? 1.0 : -1.0;
        for (int jd = 0; jd < 2; jd++)
        {
            double js = (jd == 1) ? 1.0 : -1.0;
            for (int kd = 0; kd < 2; kd++)
            {
                double ks = (kd == 1) ? 1.0 : -1.0;
                int oct = 4*id + 2*jd + kd;
                // Init first moment
                for (unsigned int a = 0; a < problem->nang; a++)
                    scat_coeff[SCAT_COEFF_INDEX(a,0,oct,problem->nang,problem->cmom)] = 1.0;
                // Init other moments
                int mom = 1;
                for (int l = 1; l < problem->nmom; l++)
                {
                    for (int m = 0; m < 2*l+1; m++)
                    {
                        for (unsigned int a = 0; a < problem->nang; a++)
                        {
                            scat_coeff[SCAT_COEFF_INDEX(a,mom,oct,problem->nang,problem->cmom)] = pow(is*mu[a], 2.0*l-1.0) * pow(ks*xi[a]*js*eta[a], m);
                        }
                        mom += 1;
                    }
                }
            }
        }
    }

    // Copy to device
    copy(scat_coeff.data(),*buffers->scat_coeff);
}

void init_material_data(
    const struct problem * problem,
    struct buffers * buffers,
    vdouble& mat_cross_section
    )
{
    mat_cross_section[0] = 1.0;
    for (unsigned int g = 1; g < problem->ng; g++)
    {
        mat_cross_section[g] = mat_cross_section[g-1] + 0.01;
    }
    // Copy to device
    copy(mat_cross_section.data(),*buffers->mat_cross_section);
}

void init_fixed_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct buffers * buffers
    )
{
    // Allocate temporary array for fixed source
    std::vector<double> fixed_source(problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);

    // Source everywhere, set at strength 1.0
    // This is src_opt == 0 in original SNAP
    for(unsigned int k = 0; k < rankinfo->nz; k++)
        for(unsigned int j = 0; j < rankinfo->ny; j++)
            for(unsigned int i = 0; i < rankinfo->nx; i++)
                for(unsigned int g = 0; g < problem->ng; g++)
                    fixed_source[FIXED_SOURCE_INDEX(g,i,j,k,problem->ng,rankinfo->nx,rankinfo->ny)] = 1.0;

    // Copy to device
    copy(fixed_source.data(),*buffers->fixed_source);
}

void init_scattering_matrix(
    const struct problem * problem,
    const struct buffers * buffers,
    const vdouble& mat_cross_section
    )
{
    // Allocate temporary array for scattering matrix
    std::vector<double> scattering_matrix(problem->nmom*problem->ng*problem->ng);

    // 10% up scattering
    // 20% in group scattering
    // 70% down scattering
    // First and last group, no up/down scattering
    for (unsigned int g = 0; g < problem->ng; g++)
    {
        if (problem->ng == 1)
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,problem->nmom,problem->ng)] = mat_cross_section[g] * 0.5;
            break;
        }

        scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g,problem->nmom,problem->ng)] = 0.2 * 0.5 * mat_cross_section[g];

        if (g > 0)
        {
            double t = 1.0 / (double)(g);
            for (unsigned int g2 = 0; g2 < g; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,problem->nmom,problem->ng)] = 0.1 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,problem->nmom,problem->ng)] = 0.3 * 0.5 * mat_cross_section[0];
        }

        if (g < (problem->ng) - 1)
        {
            double t = 1.0 / (double)(problem->ng - g - 1);
            for (unsigned int g2 = g + 1; g2 < problem->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,problem->nmom,problem->ng)] = 0.7 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,problem->ng-1,problem->ng-1,problem->nmom,problem->ng)] = 0.9 * 0.5 * mat_cross_section[problem->ng-1];
        }
    }

    // Set scattering moments (up to 4)
    // Second moment 10% of first, subsequent half of previous
    if (problem->nmom > 1)
    {
        for (unsigned int g1 = 0; g1 < problem->ng; g1++)
        {
            for (unsigned int g2 = 0; g2 < problem->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(1,g1,g2,problem->nmom,problem->ng)] = 0.1 * scattering_matrix[SCATTERING_MATRIX_INDEX(0,g1,g2,problem->nmom,problem->ng)];
                for (unsigned int m = 2; m < problem->nmom; m++)
                {
                    scattering_matrix[SCATTERING_MATRIX_INDEX(m,g1,g2,problem->nmom,problem->ng)] = 0.5 * scattering_matrix[SCATTERING_MATRIX_INDEX(m-1,g1,g2,problem->nmom,problem->ng)];
                }
            }
        }
    }

    // Copy to device
    copy(scattering_matrix.data(),*buffers->scattering_matrix);
}

void init_velocities(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // Allocate tempoary array for velocities
    std::vector<double> velocities(problem->ng);

    for (unsigned int g = 0; g < problem->ng; g++)
        velocities[g] = (double)(problem->ng - g);

    // Copy to device
    copy(velocities.data(),*buffers->velocities);
}

void init_velocity_delta(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    double dt = problem->dt;
    GPU_VEC(double) & velocities = *buffers->velocities;
    GPU_VEC(double) & velocity_delta = *buffers->velocity_delta;
    parallel_for_each(hc::extent<1>(problem->ng),
                    [=, &velocities , &velocity_delta]
                    (hc::index<1> idx) __HC__
    {
// Calculate the time absorbtion coefficient
       size_t g = idx[0];
       velocity_delta[g] = 2.0 / (dt * velocities[g]);
    });
}

void calculate_dd_coefficients(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    double dx = problem->dx;
    double dy = problem->dy;
    double dz = problem->dz;
    GPU_VEC(double) & xi = * buffers->xi;
    GPU_VEC(double) & eta = * buffers->eta;
    GPU_VEC(double) & dd_i = * buffers->dd_i;
    GPU_VEC(double) & dd_j = * buffers->dd_j;
    GPU_VEC(double) & dd_k = * buffers->dd_k;
    parallel_for_each(hc::extent<1>(problem->nang),
                    [=, &xi, &eta, &dd_i, &dd_j, &dd_k]
                    (hc::index<1> idx) __HC__
    {
       size_t a = idx[0];

    // There is only one dd_i so just get the first work-item to do this
       if (a == 0)
          dd_i[0] = 2.0 / dx;

       dd_j[a] = (2.0 / dy) * eta[a];
       dd_k[a] = (2.0 / dz) * xi[a];

    });
}

void calculate_denominator(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
#define DENOMINATOR_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define denominator(a,g,i,j,k) denominator[DENOMINATOR_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]

    unsigned int nang = problem->nang;
    unsigned int nx = rankinfo->nx;
    unsigned int ny = rankinfo->ny;
    unsigned int nz = rankinfo->nz;
    unsigned int ng = problem->ng;
    GPU_VEC(double) & mu = * buffers->mu;
    GPU_VEC(double) & dd_i = * buffers->dd_i;
    GPU_VEC(double) & dd_j = * buffers->dd_j;
    GPU_VEC(double) & dd_k = * buffers->dd_k;
    GPU_VEC(double) & mat_cross_section = * buffers->mat_cross_section;
    GPU_VEC(double) & velocity_delta = * buffers->velocity_delta;
    GPU_VEC(double) & denominator = * buffers->denominator;

    parallel_for_each(hc::extent<2>(problem->nang,problem->ng),
                    [=, &mu, &dd_i, &dd_j, &dd_k,
                     &mat_cross_section,
                     &velocity_delta,
                     &denominator]
                    (hc::index<2> idx) __HC__
    {
       size_t a = idx[0];
       size_t g = idx[1];

       for (unsigned int k = 0; k < nz; k++)
           for (unsigned int j = 0; j < ny; j++)
               for (unsigned int i = 0; i < nx; i++)
                   denominator(a,g,i,j,k) = 1.0 / (mat_cross_section[g] + velocity_delta[g] + mu[a]*dd_i[0] + dd_j[a] + dd_k[a]);

    });
}
