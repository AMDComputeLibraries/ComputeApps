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

//Initial implementation of the MD code

/** 
  Since OpenCL doesn't pick up #include properly, we need to manually switch real_t from 
  float to double in each kernel file individually.
 **/

#define UNROLL 1
#define N_MAX_NEIGHBORS 27
#define PERIODIC 1

// diagnostic flag to allow multiple levels of debug output (on CPU only)
#define KERN_DIAG 0

#define POT_SHIFT 1.0

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* CL_REAL_T is set to single or double depending on compile time flags */
typedef CL_REAL_T real_t;

int binSearch(global int *boxes, int n, int nboxes, int tot_atoms)
{
	int l = 0, r = nboxes - 1, mid = 0, box = 0;

	while(l <= r)
	{
		mid = (l + r)/2;
                if (mid <= 0)
                {
                        break;
                }

		if(n >= boxes[mid-1] && n < boxes[mid])
		{
			box = mid;
			break;
		}
		else if(mid == (nboxes - 1) && n >= boxes[mid] && n < tot_atoms)
		{
			box = mid + 1;
			break;
		}
		else if(boxes[mid] > n)
			r = mid - 1;
		else
			l = mid + 1;
	}

	return (box == 0) ? 0 : box - 1;
}

// Simple version without local blocking to check for correctness
__kernel void ljForce(
      __global const real_t* xPos,
      __global const real_t* yPos,
      __global const real_t* zPos,
      __global real_t* fx,
      __global real_t* fy,
      __global real_t* fz,
      __global real_t* U,
      __global const int* neighborList,
      __global const int* nNeighbors,
      __global const int* nAtoms,
      const real_t sigma,
      const real_t epsilon,
      const real_t cutoff,
	  const int boxSize,
	  global int *boxes) 
{

   // no loop unrolling
   /*
   int iAtom = get_global_id(0);
   int iBox = get_global_id(1);
   int maxAtoms = get_global_size(0);
   */

   // loop unrolling
	/*
   int iLocal = get_global_id(0);
   int iLine = get_global_id(1);
   int lineLength = get_global_size(0);

   int maxAtoms = lineLength/UNROLL;
   int iBox = iLine*UNROLL + iLocal/maxAtoms;
   int iAtom = iLocal  % maxAtoms;
*/

	 // prefix sum array to compute iBox
	int tid = get_global_id(0);
	int iBox = binSearch(boxes, tid, boxSize, get_global_size(0));

	int iAtom = tid - boxes[iBox];

   real_t rCut = cutoff;
   real_t rCut2 = rCut*rCut;
   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);

   // zero out local force and energy accumulators 
   real_t fxItem = 0.0;
   real_t fyItem = 0.0;
   real_t fzItem = 0.0;

   real_t uItem = 0.0;

   int iOffset = iBox*64;//maxAtoms; //N_MAX_ATOMS;
   int iParticle = iOffset + iAtom;

   if (iAtom < nAtoms[iBox])
   {// each thread executes on a single atom in the box

#if(KERN_DIAG > 1) 
      printf("i = %d, %f, %f, %f\n", iParticle, xPos[iParticle], yPos[iParticle], zPos[iParticle]);
#endif

#if(KERN_DIAG > 0) 
      printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
#endif

      for (int j = 0; j<nNeighbors[iBox]; j++)
      {// loop over neighbor cells
         int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
         int jOffset = jBox*64;//maxAtoms; //N_MAX_ATOMS;

         for (int jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
         {// loop over all groups in neighbor cell 

            int jParticle = jOffset + jAtom; // global offset of particle

            real_t dx = xPos[iParticle] - xPos[jParticle];
            real_t dy = yPos[iParticle] - yPos[jParticle];
            real_t dz = zPos[iParticle] - zPos[jParticle];

#if(KERN_DIAG > 1) 
            //printf("dx, dy, dz = %f, %f, %f\n", dx, dy, dz);
            //printf("i = %d, j = %d, %f, %f, %f\n", iParticle, jParticle, xPos[jParticle], yPos[jParticle], zPos[jParticle]);
#endif

            real_t r2 = dx*dx + dy*dy + dz*dz;

            if ( r2 <= rCut2 && r2 > 0.0)
            {// no divide by zero

#if(KERN_DIAG > 1) 
               printf("%d, %d, %f\n", iParticle, jParticle, r2);
               //printf("r2, rCut = %f, %f\n", r2, rCut);
#endif

               // reciprocal of r2 now
               r2 = (real_t)1.0/r2;

               real_t r6 = s6*r2*r2*r2;

               uItem += r6*(r6 - 1.0) - eShift;

#if(KERN_DIAG > 1) 
               //printf("%d, %d, %f\n", iParticle, jParticle, r2);
               //printf("iParticle = %d, jParticle = %d, i_b = %d, r6 = %f\n", iParticle, jParticle, i_b, r6);
#endif

               real_t fr = - 4.0*epsilon*r2*r6*(12.0*r6 - 6.0);

               fxItem += dx*fr;
               fyItem += dy*fr;
               fzItem += dz*fr;

            } 
            else 
            {
            }


         } // loop over all atoms
      } // loop over neighbor cells

      fx[iParticle] = fxItem;
      fy[iParticle] = fyItem;
      fz[iParticle] = fzItem;

      // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
      U[iParticle] = uItem*2.0*epsilon;

   }
}

__kernel void ljForceLocal(
      __global real_t* xPos,
      __global real_t* yPos,
      __global real_t* zPos,
      __global real_t* fx,
      __global real_t* fy,
      __global real_t* fz,
      __global real_t* U,
      __global int* neighborList,
      __global int* nNeighbors,
      __global int* nAtoms,
      __local real_t* x_ii,
      __local real_t* y_ii,
      __local real_t* z_ii,
      __local real_t* x_ij,
      __local real_t* y_ij,
      __local real_t* z_ij,
      const real_t sigma,
      const real_t epsilon,
      const real_t cutoff) 
{




   real_t dx, dy, dz;
   real_t r2, r6;
   real_t fr;
   real_t dxbox, dybox, dzbox;

   real_t rCut = 5.0*sigma;
   real_t rCut2 = rCut*rCut;
   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   int i_p;

   int iOffset, jOffset;
   int jBox;
   int iParticle, jParticle;

   // zero out forces, energy on particles
   real_t fx_ii = 0.0;
   real_t fy_ii = 0.0;
   real_t fz_ii = 0.0;

   real_t u_ii = 0.0;

   int iBox = get_global_id(1);
   int iLocal = get_local_id(0);
   int maxAtoms = get_global_size(0);
   int nGroups = get_num_groups(0);
   int nItems = get_local_size(0);
   int groupId = get_group_id(0);

   int group_offset = nItems*groupId;
   int iAtom = group_offset + iLocal;
   iOffset = iBox*maxAtoms; //N_MAX_ATOMS;
   iParticle = iOffset + iAtom;

#if(KERN_DIAG) 
   printf("Number of work groups: %d\n", nGroups);
   printf("Number of work items: %d\n", nItems);
#endif

   if (iAtom < nAtoms[iBox])
   {// each thread executes on a single atom in the box

#if(KERN_DIAG) 
      //printf("iParticle = %d\n", iParticle);
      //printf("i_global = %d, iLocal = %d, i_b = %d, nItems = %d\n", i_global, iLocal, i_b, nItems);
#endif

      // load particle data into local arrays
      x_ii[iLocal] = xPos[iParticle];
      y_ii[iLocal] = yPos[iParticle];
      z_ii[iLocal] = zPos[iParticle];

      barrier(CLK_LOCAL_MEM_FENCE);

#if(KERN_DIAG) 
      //printf("x_ii, y_ii, z_ii = %f, %f, %f\n", x_ii[iLocal], y_ii[iLocal], z_ii[iLocal]);
      printf("%d, %f, %f, %f\n", iParticle, x_ii[iLocal], y_ii[iLocal], z_ii[iLocal]);
#endif

      for (int j = 0; j<nNeighbors[iBox]; j++)
      {// loop over neighbor cells
         jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
         jOffset = jBox*maxAtoms; //N_MAX_ATOMS;

         for (int j_b = 0; j_b<nGroups; j_b++)
         {// loop over all groups in neighbor cell 

            // use iLocal to load data in blocks of size nItems
            x_ij[iLocal] = xPos[iLocal + j_b*nItems + jOffset];
            y_ij[iLocal] = yPos[iLocal + j_b*nItems + jOffset];
            z_ij[iLocal] = zPos[iLocal + j_b*nItems + jOffset];

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int jLocal=0;jLocal < nItems; jLocal ++)
            {// loop over all atoms in group

               jParticle = jLocal+ j_b*nItems; // global offset of particle

               dx = x_ii[iLocal] - x_ij[jLocal];
               dy = y_ii[iLocal] - y_ij[jLocal];
               dz = z_ii[iLocal] - z_ij[jLocal];

#if(KERN_DIAG) 
               printf("%d, %f, %f, %f\n", jParticle, x_ij[jLocal], y_ij[jLocal], z_ij[jLocal]);
               printf("%d, %d, %f, %f, %f\n", iParticle, jParticle, dx, dy, dz);
#endif

               r2 = dx*dx + dy*dy + dz*dz;

#if(KERN_DIAG) 
               printf("%d, %d, %f\n", iParticle, jParticle, r2);
               //printf("r2, rCut = %f, %f\n", r2, rCut);
#endif

               if ( r2 <= rCut2 && r2 > 0.0)
               {// no divide by zero

                  // reciprocal of r2 now
                  r2 = (real_t)1.0/r2;

                  r6 = s6*r2*r2*r2;

                  u_ii += r6*(r6 - 1.0);

#if(KERN_DIAG) 
                  //printf("%d, %d, %f\n", iParticle, jParticle, r2);
                  //printf("iParticle = %d, jParticle = %d, i_b = %d, r6 = %f\n", iParticle, jParticle, i_b, r6);
#endif

                  fr = 4.0*epsilon*s6*r2*r6*(12.0*r6*s6 - 6.0);

                  fx_ii += dx*fr;
                  fy_ii += dy*fr;
                  fz_ii += dz*fr;

               } 
               else 
               {
               }

            } // loop over all atoms in group

         } // loop over all groups in neighbor cell
      } // loop over neighbor cells

      fx[iParticle] = fx_ii;
      fy[iParticle] = fy_ii;
      fz[iParticle] = fz_ii;


      // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
      U[iParticle] = u_ii*2.0*epsilon*s6;

      barrier(CLK_LOCAL_MEM_FENCE);
   }

}

