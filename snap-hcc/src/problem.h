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
* \brief Calculate "static" problem data
*
* Routines to initilise the data arrays based on the problem inputs.
* Also contains data calculated every outer, which could only be done once
* per timestep.
*/

#pragma once

#include "global.h"
#include "buffers.h"

#include "profiler.h"

/** @defgroup MEM Memory access patterns
* \brief Macros for indexing multi-dimensional arrays
* @{*/

/** \brief Index for scattering coefficient array */
#define SCAT_COEFF_INDEX(a,m,o,nang,cmom) ((a)+((nang)*(m))+((nang)*(cmom)*o))

/** \brief Index for fixed source array */
#define FIXED_SOURCE_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))

/** \brief Index for scattering matrix array */
#define SCATTERING_MATRIX_INDEX(m,g1,g2,nmom,ng) ((m)+((nmom)*(g1))+((nmom)*(ng)*(g2)))

/** \brief Index for transport denominator array */
#define DENOMINATOR_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))

/** \brief Index for scalar flux array */
#define SCALAR_FLUX_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))

/** \brief Index for scalar flux moments array */
#define SCALAR_FLUX_MOMENTS_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+(((cmom)-1)*(g))+(((cmom)-1)*(ng)*(i))+(((cmom)-1)*(ng)*(nx)*(j))+(((cmom)-1)*(ng)*(nx)*(ny)*(k)))

/**@}*/


/** \brief Initilise quadrature weights
*
* Set to uniform weights: number of angles divided by eight.
*/
void init_quadrature_weights(const struct problem * problem, struct buffers * buffers);
void print_stuff(GPU_VEC(double) &flux, size_t nsize, std::string name);

void print_planes(const struct problem * problem, const struct buffers * buffers,
    const struct plane * planes,const unsigned int num_planes);

/** \brief Calculate cosine coefficients
*
* Populates the \a mu, \a eta and \a xi arrays.
*/
void calculate_cosine_coefficients(const struct problem * problem, const struct buffers * buffers, vdouble&  mu, vdouble&  eta, vdouble& xi);

/** \brief Calculate the scattering coefficients
*
* Populates the \a scat_coef array based on the cosine coefficients.
* Set as \f$(\mu*\eta*\xi)^l\f$ starting at 0, for the lth moment.
*/
void calculate_scattering_coefficients(const struct problem * problem, const struct buffers * buffers, const vdouble& mu, const vdouble& eta, const vdouble& xi);

/** \brief Set material cross sections
*
* We one have one material across the whole grid. Set to 1.0 for the first group, and + 0.01 for each subsequent group.
*/
void init_material_data(const struct problem * problem, struct buffers * buffers, vdouble&  mat_cross_section);

/** /brief Set fixed source data
*
* Source is applied everywhere, set at strenght 1.0.
* This is fixed src_opt == 0 in original SNAP
*/
void init_fixed_source(const struct problem * problem, const struct rankinfo * rankinfo, const struct buffers * buffers);

/** \brief Setup group to group scattering information
*
* Scattering is 10% upscattering, 20% in group and 70% down scattering in every group,
* except first and last which have no up/down scattering.
* Data is initilised for all moments.
*/
void init_scattering_matrix(const struct problem * problem, const struct buffers * buffers, const vdouble&  mat_cross_section);


/** \brief Set velocities array
*
* Fake data on group velocity.
*/
void init_velocities(const struct problem * problem, const struct buffers * buffers);

/** \brief Set velocity time delta array on device (non-blocking) */
void init_velocity_delta(const struct problem * problem, const struct buffers * buffers);

/** \brief Calculate the spatial diamond difference coefficients on device (non-blocking)
*
* Called every outer. Includes the cosine coefficient terms.
*/
void calculate_dd_coefficients(const struct problem * problem, const struct buffers * buffers);

/** \brief Calculate the denominator to the transport equation update on device (non-blocking)
*
* Called every outer.
*/
void calculate_denominator(const struct problem * problem, const struct rankinfo * rankinfo, const struct buffers * buffers);
