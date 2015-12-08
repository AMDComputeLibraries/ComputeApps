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

#include "helpers.h"

HostBoxes* initHostBoxesSoa(SimFlat* sim)
{
   HostBoxes* boxes;

   boxes = malloc(sizeof(HostBoxes));

   boxes->boxSize      = malloc(3*sizeof(cl_real));
   boxes->invBoxSize   = malloc(3*sizeof(cl_real));

   boxes->nTotalBoxes = sim->boxes->nTotalBoxes;
   boxes->nHaloBoxes = sim->boxes->nHaloBoxes;
   boxes->nLocalBoxes = sim->boxes->nLocalBoxes;

   printf("Allocating space on host for %d local and %d halo boxes\n", boxes->nLocalBoxes, boxes->nHaloBoxes);

   boxes->nLocalBoxesNeighborSize = boxes->nLocalBoxes*NUMNEIGHBORS*sizeof(cl_int);
   boxes->nLocalBoxesRealSize     = boxes->nLocalBoxes*sizeof(cl_real);
   boxes->nLocalBoxesIntSize      = boxes->nLocalBoxes*sizeof(cl_int);

   boxes->nTotalBoxesIntSize      = boxes->nTotalBoxes*sizeof(cl_int);

   for (int j=0;j<3;j++)
   {
      boxes->boxSize[j]    = sim->boxes->boxSize[j];
      boxes->invBoxSize[j] = sim->boxes->invBoxSize[j];
   }

   // only need neighbor lists for local boxes
   boxes->neighborList = malloc(boxes->nLocalBoxesNeighborSize);
   boxes->nNeighbors   = malloc(boxes->nLocalBoxesIntSize);
   for (int iBox=0;iBox<sim->boxes->nLocalBoxes;iBox++)
   {

      int nbrBoxes[27];
      boxes->nNeighbors[iBox] = getNeighborBoxes(sim->boxes,iBox, nbrBoxes);
      // check if any boxes are missing neighbors
      if (boxes->nNeighbors[iBox] < 27)
         printf("Box %d has only %d neighbors\n", iBox, boxes->nNeighbors[iBox]);

      for (int j=0;j<boxes->nNeighbors[iBox];j++)
      {
         boxes->neighborList[NUMNEIGHBORS*iBox + j] = nbrBoxes[j];
      }

   }

   // need nAtoms for all boxes
   boxes->nAtoms       = malloc(boxes->nTotalBoxesIntSize);
   for (int iBox=0;iBox<sim->boxes->nTotalBoxes;iBox++)
   {
      boxes->nAtoms[iBox] = sim->boxes->nAtoms[iBox];
   }

   // cehck box occupancy for all local boxes
   int nMaxInBox = 0;
   int boxWithMax = 0;
   for (int iBox=0;iBox<boxes->nLocalBoxes;iBox++)
   {
      if (sim->boxes->nAtoms[iBox] > nMaxInBox)
      {
         nMaxInBox = sim->boxes->nAtoms[iBox];
         boxWithMax = iBox;
      }
   }
   printf("Max atom count %d in box %d\n", nMaxInBox, boxWithMax);

   return boxes;
}

HostBoxes* initHostBoxesAos(SimFlat* sim)
{
   HostBoxes* boxes;

   boxes = malloc(sizeof(HostBoxes));

   boxes->nTotalBoxes = sim->boxes->nTotalBoxes;
   boxes->nLocalBoxes = sim->boxes->nLocalBoxes;

   int nMaxInBox = 0;
   int boxWithMax = 0;
   for (int iBox=0;iBox<boxes->nTotalBoxes;iBox++)
   {
      if (sim->boxes->nAtoms[iBox] > nMaxInBox)
      {
         nMaxInBox = sim->boxes->nAtoms[iBox];
         boxWithMax = iBox;
      }
   }
   printf("Max atom count %d in box %d\n", nMaxInBox, boxWithMax);

   boxes->nLocalBoxesNeighborSize = sim->boxes->nLocalBoxes*NUMNEIGHBORS*sizeof(cl_int);
   boxes->nLocalBoxesRealSize = sim->boxes->nLocalBoxes*sizeof(cl_real);
   boxes->nTotalBoxesIntSize = sim->boxes->nTotalBoxes*sizeof(cl_int);

   boxes->neighborList = malloc(boxes->nLocalBoxesNeighborSize);
   boxes->nNeighbors = malloc(boxes->nLocalBoxesIntSize);
   boxes->nAtoms     = malloc(boxes->nTotalBoxesIntSize);

   boxes->boxSize    = malloc(sizeof(cl_real4));
   boxes->invBoxSize    = malloc(sizeof(cl_real4));


   for (int j=0;j<3;j++)
   {
      boxes->boxSize[j] = sim->boxes->boxSize[j];
      boxes->invBoxSize[j] = sim->boxes->invBoxSize[j];
   }

   for (int iBox=0;iBox<sim->boxes->nLocalBoxes;iBox++)
   {

      int* nbrBoxes;
      getNeighborBoxes(sim->boxes,iBox, nbrBoxes);

      boxes->nAtoms[iBox] = sim->boxes->nAtoms[iBox];

      int j;
      boxes->nNeighbors[iBox] = nbrBoxes[-1];
      for (int j=0;j<boxes->nNeighbors[iBox];j++)
      {
         boxes->neighborList[NUMNEIGHBORS*iBox + j] = nbrBoxes[j];
      }
   }

   return boxes;
}

void createDevBoxesSoa(DevBoxesSoa* boxesDev, HostBoxes* hostBoxes) 
{
   /// Create the device buffers to hold the boxes data:

   oclCreateReadWriteBuffer(&boxesDev->neighborList, hostBoxes->nLocalBoxesNeighborSize);
   oclCreateReadWriteBuffer(&boxesDev->nNeighbors, hostBoxes->nLocalBoxesIntSize);
   oclCreateReadWriteBuffer(&boxesDev->nAtoms, hostBoxes->nTotalBoxesIntSize);

   oclCreateReadWriteBuffer(&boxesDev->boxSize, sizeof(cl_real)*3);
   oclCreateReadWriteBuffer(&boxesDev->invBoxSize, sizeof(cl_real)*3);
	oclCreateReadWriteBuffer(&boxesDev->nAtomsPfx, sizeof(int)*hostBoxes->nLocalBoxes);

	boxesDev->nLocalBoxes = hostBoxes->nLocalBoxes;
}

void createDevBoxesAos(DevBoxesAos* boxesDev, HostBoxes* hostBoxes) 
{
   /// Create the device buffers to hold the boxes data:

   oclCreateReadWriteBuffer(&boxesDev->neighborList, hostBoxes->nLocalBoxesNeighborSize);
   oclCreateReadWriteBuffer(&boxesDev->nNeighbors, hostBoxes->nLocalBoxesIntSize);
   oclCreateReadWriteBuffer(&boxesDev->nAtoms, hostBoxes->nTotalBoxesIntSize);

   oclCreateReadWriteBuffer(&boxesDev->boxSize, sizeof(cl_real4));
   oclCreateReadWriteBuffer(&boxesDev->invBoxSize, sizeof(cl_real4));
}

void putBoxesSoa(HostBoxes* boxesHost, DevBoxesSoa* boxesDev)
{
   oclCopyToDevice(boxesHost->nNeighbors, boxesDev->nNeighbors, boxesHost->nLocalBoxesIntSize, 0);
   oclCopyToDevice(boxesHost->neighborList, boxesDev->neighborList, boxesHost->nLocalBoxesNeighborSize, 0);

   oclCopyToDevice(boxesHost->nAtoms, boxesDev->nAtoms, boxesHost->nTotalBoxesIntSize, 0);

   oclCopyToDevice(boxesHost->boxSize, boxesDev->boxSize, sizeof(cl_real)*3, 0);
   oclCopyToDevice(boxesHost->invBoxSize, boxesDev->invBoxSize, sizeof(cl_real)*3, 0);
}

void putBoxesAos(HostBoxes* boxesHost, DevBoxesAos* boxesDev)
{
   oclCopyToDevice(boxesHost->nNeighbors, boxesDev->nNeighbors, boxesHost->nLocalBoxesIntSize, 0);
   oclCopyToDevice(boxesHost->neighborList, boxesDev->neighborList, boxesHost->nLocalBoxesNeighborSize, 0);

   oclCopyToDevice(boxesHost->nAtoms, boxesDev->nAtoms, boxesHost->nTotalBoxesIntSize, 0);

   oclCopyToDevice(boxesHost->boxSize, boxesDev->boxSize, sizeof(cl_real4), 0);
   oclCopyToDevice(boxesHost->invBoxSize, boxesDev->invBoxSize, sizeof(cl_real4), 0);
}

