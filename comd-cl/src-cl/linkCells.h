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

/// \file
/// Functions to maintain link cell structures for fast pair finding.

#ifndef __LINK_CELLS_H_
#define __LINK_CELLS_H_

#include "mytype.h"

/// The maximum number of atoms that can be stored in a link cell.
#define MAXATOMS 64 

struct DomainSt;
struct AtomsSt;

/// Link cell data.  For convenience, we keep a copy of the localMin and
/// localMax coordinates that are also found in the DomainsSt.
typedef struct LinkCellSt
{
   int gridSize[3];     //!< number of boxes in each dimension on processor
   int nInnerBoxes;     //!< boxes which do not contribute to halos
   int nLocalBoxes;     //!< total number of local boxes on processor
   int nHaloBoxes;      //!< total number of remote halo/ghost boxes on processor

   int nTotalBoxes;     //!< total number of boxes on processor
                        //!< nLocalBoxes + nHaloBoxes
   real3 localMin;      //!< minimum local bounds on processor
   real3 localMax;      //!< maximum local bounds on processor
   real3 boxSize;       //!< size of box in each dimension
   real3 invBoxSize;    //!< inverse size of box in each dimension

   int* nAtoms;         //!< total number of atoms in each box
} LinkCell;

LinkCell* initLinkCells(const struct DomainSt* domain, real_t cutoff);
void destroyLinkCells(LinkCell** boxes);

int getNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes);
void putAtomInBox(LinkCell* boxes, struct AtomsSt* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz);
int getBoxFromTuple(LinkCell* boxes, int x, int y, int z);

void moveAtom(LinkCell* boxes, struct AtomsSt* atoms, int iId, int iBox, int jBox);

/// Update link cell data structures when the atoms have moved.
void updateLinkCells(LinkCell* boxes, struct AtomsSt* atoms);

int maxOccupancy(LinkCell* boxes);


#endif
