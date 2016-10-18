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

#pragma once

/** \file
* \brief Sweep calculation routines
*/

#include <stdlib.h>

#include "global.h"
#include "buffers.h"

/** \brief Structure to hold a 3D cell index for use in storing planes */
struct cell_id
{
    /** @{ \brief Cell index */
    unsigned int i, j, k;
    /** @} */
};

/** \brief Structure to hold list of cells in each plane */
struct plane
{
    /** \brief Number of cells in this plane */
    unsigned int num_cells;
    /** \brief Array of cell indexes in this plane */
    struct cell_id * cell_ids;
};

/** \brief Create a list of cell indexes in the planes in the XY plane determined by chunk */
void init_planes(struct plane** planes, unsigned int *num_planes, struct problem * problem, struct rankinfo * rankinfo);

/** \brief Enqueue the kernels to sweep a plane */
void sweep_plane(const unsigned int z_pos, const int octant, const int istep, const int jstep, const int kstep, const unsigned int plane_num, const struct plane * planes, struct problem * problem, struct rankinfo * rankinfo, struct buffers * buffers);

