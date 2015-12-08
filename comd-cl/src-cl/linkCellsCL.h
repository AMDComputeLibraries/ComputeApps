/*******************************************************************************
Copyright (c) 2015 Advanced Micro Devices, Inc. 

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

#ifndef LINKCELLSCL_H
#define LINKCELLSCL_H

#include "cl_utils.h"
#include "CoMDTypes.h"

typedef struct HostBoxesSt 
{
   cl_int nLocalBoxes;
   cl_int nHaloBoxes;
   cl_int nTotalBoxes;

   cl_int* neighborList;
   cl_int* nNeighbors;
   cl_int* nAtoms;

   cl_real* boxSize;
   cl_real* invBoxSize;

   int nTotalBoxesIntSize;

   int nLocalBoxesIntSize;
   int nLocalBoxesNeighborSize;
   int nLocalBoxesRealSize;
} HostBoxes;

typedef struct DevBoxesSoaSt 
{
	cl_int nLocalBoxes;

   DevVec rBox;
   cl_mem neighborList;
   cl_mem nNeighbors;
   cl_mem nAtoms;
   cl_mem boxSize;
   cl_mem invBoxSize;
   cl_mem nAtomsPfx;

} DevBoxesSoa;

typedef struct DevBoxesAosSt 
{
   cl_mem rBox;
   cl_mem neighborList;
   cl_mem nNeighbors;
   cl_mem nAtoms;
   cl_mem boxSize;
   cl_mem invBoxSize;
} DevBoxesAos;

HostBoxes* initHostBoxesSoa(SimFlat* sim);

HostBoxes* initHostBoxesAos(SimFlat* sim);

void createDevBoxesSoa(DevBoxesSoa *boxesDev, HostBoxes* hostBoxes);

void createDevBoxesAos(DevBoxesAos *boxesDev, HostBoxes* hostBoxes);

void putBoxesSoa(HostBoxes* boxesHost, DevBoxesSoa* boxesDev);

void putBoxesAos(HostBoxes* boxesHost, DevBoxesAos* boxesDev);

#endif
