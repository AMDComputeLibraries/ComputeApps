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
* \brief Structures containing problem sizes and MPI information
*/

#include <stdbool.h>

#include <hc.hpp>

#define ARRAY_VIEW 0

#if ARRAY_VIEW
  #define GPU_VEC(type) hc::array_view<type>
#else
  #define GPU_VEC(type) hc::array<type>
#endif

typedef std::vector<double> vdouble;

/** \brief Problem dimensions
*
* Read in from input file or calculated from those inputs
*/
struct problem
{
    /**@{ \brief Global grid size */
    unsigned int nx, ny, nz;
    /**@}*/

    /**@{ \brief Physical grid size */
    double lx, ly, lz;
    /**@}*/

    /**@{ \brief Width of spatial cells */
    double dx, dy, dz;
    /**@}*/

    /** \brief Energy groups */
    unsigned int ng;

    /** \brief Angles per octant
        (3D assumed) */
    unsigned int nang;

    /** \brief Number of expansion moments */
    unsigned int nmom;

    /**  \brief Number of computational moments
    *
    * = nmom*nmom */
    unsigned int cmom;

    /**@{*/
    /** \brief Number of inner iterations */
    unsigned int iitm;
    /** \brief Number of outer iterations */
    unsigned int oitm;
    /**@}*/

    /**@{*/
    /** \brief Number of timesteps */
    unsigned int nsteps;
    /** \brief Total time to simulate */
    double tf;
    /** \brief Time domain stride */
    double dt;
    /**@}*/

    /** \brief Convergence criteria */
    double epsi;

    /**@{ \brief Number of MPI tasks in each direction */
    unsigned int npex, npey, npez;
    /**@}*/

    /** \brief KBA chunk size */
    unsigned int chunk;

    /** \brief Global variable to determine if there are multiple GPUs per node (if so we have to get GPUs VERY carefully) */
    bool multigpu;
};


/** \brief MPI Information
*
* Holds rankinfo information about tile size and MPI rank
*/
struct rankinfo
{
    /**  \brief My WORLD rank */
    int rank;

    /** \brief My MPI Cartesian co-ordinate ranks */
    int ranks[3];

    /**@{ \brief Local grid size */
    unsigned int nx, ny, nz;
    /**@}*/

    /**@{ \brief Global grid corners of MPI partition */
    unsigned int ilb, iub;
    unsigned int jlb, jub;
    unsigned int klb, kub;
    /**@}*/

    /**@{ \brief Global ranks of my neighbours */
    int xup, xdown;
    int yup, ydown;
    int zup, zdown;
    /**@}*/
};
