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
* \brief Manage the allocation of arrays
*
* All problem scope arrays are allocated in the host DRAM using these functions calls.
*/

#pragma once

#include "global.h"

/** \brief Struct to hold the buffers
*
* All the memory arrays are stored here
*/
struct memory
{
	/**@{*/
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, ny, nz)
	*
	* Note, rankinfo spatial dimension
	*/
	std::vector<double> flux_i;
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, nx, nz)
	*
	* Note, rankinfo spatial dimension
	*/
	std::vector<double> flux_j;
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, nx, ny)
	*
	* Note, rankinfo spatial dimension
	*/
	std::vector<double> flux_k;
	/**@}*/

	/**@{ \brief Scalar flux */
	/**
	* Size: (ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimensions
	*/
	std::vector<double> scalar_flux;
	std::vector<double> old_inner_scalar_flux;
	std::vector<double> old_outer_scalar_flux;
	/**@}*/
	/**@{*/
	/** \brief Scalar flux moments */
	/** Size: (cmom-1, ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimensions
	*/
	std::vector<double> scalar_flux_moments;
//	double *scalar_flux_moments;
	/**@}*/

	/**@{ \brief Cosine coefficients */
	/** Size: (nang) */
	std::vector<double> mu;
	std::vector<double> eta;
	std::vector<double> xi;
	/**@}*/

	/** \brief Material cross sections
	*
	* ASSUME ONE MATERIAL
	*
	* Size: (ng)
	*/
	std::vector<double> mat_cross_section;
};

/** \brief Allocate the problem arrays */
void allocate_memory(struct problem * problem, struct rankinfo * rankinfo, struct memory * memory);

/** \brief Free the arrays sroted in the \a mem struct */
void free_memory(struct memory * memory);

