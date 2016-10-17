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
/*
  This is a Version 2.0 MPI + OpenMP Beta implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
* Single source distribution supports pure serial, pure OpenMP, MPI-only, 
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt 
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal
*    gas material, and the same sedov blast wave problem is still the only
*    problem its hardcoded to solve.
* Regions allow two things important to making this proxy app more representative:
*   Four of the LULESH routines are now performed on a region-by-region basis,
*     making the memory access patterns non-unit stride
*   Artificial load imbalances can be easily introduced that could impact
*     parallelization strategies.  
* The load balance flag changes region assignment.  Region number is raised to
*   the power entered for assignment probability.  Most likely regions changes
*   with MPI process id.
* The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
*   entered multiple. The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source
*   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/
#include <climits>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>

#include <iostream>
#include <iomanip>
#include <string.h>
#include <vector>
#include <map>
#include <fstream>

#include "lulesh.h"

#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>

#include <hc.hpp>
#include <hc_math.hpp>
using namespace hc;

#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000

#define X 0
#define Y 1
#define Z 2

#define MINEQ(a,b) (a)=(((a)<(b))?(a):(b))  
//#define BLOCKSIZE  256

inline Real_t  SQRT(Real_t  arg) restrict(amp) { return hc::precise_math::sqrt(arg) ; }
inline Real_t  CBRT(Real_t  arg) restrict(amp) { return hc::precise_math::cbrt(arg) ; }
inline Real_t  FABS(Real_t  arg) restrict(amp) { return hc::precise_math::fabs(arg) ; }
inline Real_t  FMAX(Real_t  arg1,Real_t  arg2) restrict(amp) { return hc::precise_math::fmax(arg1,arg2) ; }

  
double getTime() {
    struct timeval tp;
    static long start=0, startu;
    if (!start) {
        gettimeofday(&tp, NULL);
        start = tp.tv_sec;
        startu = tp.tv_usec;
        return(0.0);
    }
    gettimeofday(&tp, NULL);
    return( ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 );
}

/*********************************/
/* Data structure implementation */
/*********************************/

/* might want to add access methods so that memory can be */
/* better managed, as in luleshFT */

template <typename T>
T *Allocate(size_t size)
{
   return static_cast<T *>(malloc(sizeof(T)*size)) ;
}

template <typename T>
void Release(T **ptr)
{
   if (*ptr != NULL) {
      free(*ptr) ;
      *ptr = NULL ;
   }
}

/******************************************/

/* Work Routines */

static inline
void TimeIncrement(Domain& domain)
{
   Real_t targetdt = domain.stoptime() - domain.time() ;

   if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain.deltatime() ;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt ;
      if (domain.dtcourant() < gnewdt) {
         gnewdt = domain.dtcourant() / Real_t(2.0) ;
      }
      if (domain.dthydro() < gnewdt) {
         gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

      newdt = gnewdt;
      
      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain.deltatimemultub()) {
            newdt = olddt*domain.deltatimemultub() ;
         }
      }

      if (newdt > domain.dtmax()) {
         newdt = domain.dtmax() ;
      }
      domain.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain.deltatime()) &&
       (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
   }

   if (targetdt < domain.deltatime()) {
      domain.deltatime() = targetdt ;
   }

   domain.time() += domain.deltatime() ;

   ++domain.cycle() ;
}

/******************************************/

static inline
void InitStressTermsForElems(struct MeshGPU *meshGPU,
			     Index_t numElem)  
{

  if (numElem <= 0) return;
  HCC_ARRAY_OBJECT(Real_t, q) = meshGPU->q;
  HCC_ARRAY_OBJECT(Real_t, p) = meshGPU->p;
  HCC_ARRAY_OBJECT(Real_t, sigxx) = meshGPU->sigxx;
  HCC_ARRAY_OBJECT(Real_t, sigyy) = meshGPU->sigyy;
  HCC_ARRAY_OBJECT(Real_t, sigzz) = meshGPU->sigzz;

  extent<1> numElemExt(numElem);
  completion_future fut = parallel_for_each(numElemExt, [=
							 HCC_ID(sigxx)
							 HCC_ID(sigyy)
							 HCC_ID(sigzz)
							 HCC_ID(p)
							 HCC_ID(q)](index<1> idx) restrict(amp){
    int i = idx[0];
    sigxx[i] = sigyy[i] = sigzz[i] =  - p[i] - q[i] ;
  });
  fut.wait();
  
}

/******************************************/


static inline
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3) restrict(amp)
{
   Real_t bisectX0 = (Real_t)(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = (Real_t)(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = (Real_t)(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = (Real_t)(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = (Real_t)(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = (Real_t)(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = (Real_t)(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = (Real_t)(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = (Real_t)(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

static inline
void IntegrateStressForElems(struct MeshGPU *meshGPU,Index_t numNode,
    int& badvol, Index_t numElem)
{
  if (numElem <= 0) return;
  HCC_ARRAY_OBJECT(Real_t, sigxx) = meshGPU->sigxx;
  HCC_ARRAY_OBJECT(Real_t, sigyy) = meshGPU->sigyy;
  HCC_ARRAY_OBJECT(Real_t, sigzz) = meshGPU->sigzz;
  HCC_ARRAY_OBJECT(Real_t, x) = meshGPU->x;
  HCC_ARRAY_OBJECT(Real_t, y) = meshGPU->y;
  HCC_ARRAY_OBJECT(Real_t, z) = meshGPU->z;
  HCC_ARRAY_OBJECT(Real_t, determ) = meshGPU->determ;
  HCC_ARRAY_OBJECT(Index_t, nodelist) = meshGPU->nodelist;
  HCC_ARRAY_OBJECT(Real_t, fx) = meshGPU->fx;
  HCC_ARRAY_OBJECT(Real_t, fy) = meshGPU->fy;
  HCC_ARRAY_OBJECT(Real_t, fz) = meshGPU->fz;
  HCC_ARRAY_OBJECT(Real_t, fx_elem) = meshGPU->fx_elem;
  HCC_ARRAY_OBJECT(Real_t, fy_elem) = meshGPU->fy_elem;
  HCC_ARRAY_OBJECT(Real_t, fz_elem) = meshGPU->fz_elem;
  HCC_ARRAY_OBJECT(Int_t, nodeElemCount) = meshGPU->nodeElemCount;
  HCC_ARRAY_OBJECT(Index_t, nodeElemCornerList) = meshGPU->nodeElemCornerList;

  extent<1> elemExt(PAD(numElem,BLOCKSIZE));
  tiled_extent<1> tElemExt(elemExt,BLOCKSIZE);
  
  completion_future fut = parallel_for_each(tElemExt,[=
						      HCC_ID(sigxx)
						      HCC_ID(sigyy)
						      HCC_ID(sigzz)
						      HCC_ID(determ)
						      HCC_ID(fx_elem)
						      HCC_ID(fy_elem)
						      HCC_ID(fz_elem)
						      HCC_ID(nodelist)
						      HCC_ID(x)
						      HCC_ID(y)
						      HCC_ID(z)]
					    (tiled_index<1> idx) restrict(amp){
      Index_t k=idx.global[0] ;				      
      if(k < numElem){					      
        Real_t B[3][8] ;// shape function derivatives
	Real_t x_local[8] ;
	Real_t y_local[8] ;
	Real_t z_local[8] ;

	B[0][0] = (Real_t)(0.0); B[0][1] = (Real_t)(0.0);
	B[0][2] = (Real_t)(0.0); B[0][3] = (Real_t)(0.0);
	B[0][4] = (Real_t)(0.0); B[0][5] = (Real_t)(0.0);
	B[0][6] = (Real_t)(0.0); B[0][7] = (Real_t)(0.0);
	B[1][0] = (Real_t)(0.0); B[1][1] = (Real_t)(0.0);
	B[1][2] = (Real_t)(0.0); B[1][3] = (Real_t)(0.0);
	B[1][4] = (Real_t)(0.0); B[1][5] = (Real_t)(0.0);
	B[1][6] = (Real_t)(0.0); B[1][7] = (Real_t)(0.0);
	B[2][0] = (Real_t)(0.0); B[2][1] = (Real_t)(0.0);
	B[2][2] = (Real_t)(0.0); B[2][3] = (Real_t)(0.0);
	B[2][4] = (Real_t)(0.0); B[2][5] = (Real_t)(0.0);
	B[2][6] = (Real_t)(0.0); B[2][7] = (Real_t)(0.0);
	
	for( Index_t lnode=0 ; lnode<8 ; ++lnode ){
	  Index_t gnode = nodelist[k+lnode*numElem];
	  x_local[lnode] = x[gnode];
	  y_local[lnode] = y[gnode];
	  z_local[lnode] = z[gnode];
	}
	    
	Real_t fjxxi, fjxet, fjxze;
	Real_t fjyxi, fjyet, fjyze;
	Real_t fjzxi, fjzet, fjzze;
	Real_t cjxxi, cjxet, cjxze;
	Real_t cjyxi, cjyet, cjyze;
	Real_t cjzxi, cjzet, cjzze;
	    
	fjxxi = (Real_t)(.125) * ( (x_local[6]-x_local[0]) +
				   (x_local[5]-x_local[3]) -
				   (x_local[7]-x_local[1]) -
				   (x_local[4]-x_local[2]) );
	fjxet = (Real_t)(.125) * ( (x_local[6]-x_local[0]) -
				   (x_local[5]-x_local[3]) +
				   (x_local[7]-x_local[1]) -
				   (x_local[4]-x_local[2]) );
	fjxze = (Real_t)(.125) * ( (x_local[6]-x_local[0]) +
				   (x_local[5]-x_local[3]) +
				   (x_local[7]-x_local[1]) +
				   (x_local[4]-x_local[2]) );
	fjyxi = (Real_t)(.125) * ( (y_local[6]-y_local[0]) +
				   (y_local[5]-y_local[3]) -
				   (y_local[7]-y_local[1]) -
				   (y_local[4]-y_local[2]) );
	fjyet = (Real_t)(.125) * ( (y_local[6]-y_local[0]) -
				   (y_local[5]-y_local[3]) +
				   (y_local[7]-y_local[1]) -
				   (y_local[4]-y_local[2]) );
	fjyze = (Real_t)(.125) * ( (y_local[6]-y_local[0]) +
				   (y_local[5]-y_local[3]) +
				   (y_local[7]-y_local[1]) +
				   (y_local[4]-y_local[2]) );
	fjzxi = (Real_t)(.125) * ( (z_local[6]-z_local[0]) +
				   (z_local[5]-z_local[3]) -
				   (z_local[7]-z_local[1]) -
				   (z_local[4]-z_local[2]) );
	fjzet = (Real_t)(.125) * ( (z_local[6]-z_local[0]) -
				   (z_local[5]-z_local[3]) +
				   (z_local[7]-z_local[1]) -
				   (z_local[4]-z_local[2]) );
	fjzze = (Real_t)(.125) * ( (z_local[6]-z_local[0]) +
				   (z_local[5]-z_local[3]) +
				   (z_local[7]-z_local[1]) +
				   (z_local[4]-z_local[2]) );
	    
	// compute cofactors 
	cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
	cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
	cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);
	    
	cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
	cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
	cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);
	    
	cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
	cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
	cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);
	    
	/* calculate jacobian determinant (volume) */
	determ[k] = (Real_t)(8.) * ( fjxet * cjxet + fjyet * cjyet +
				     fjzet * cjzet);
	    
	/* computation is done per element faces (each element has 6 faces)
	 * each element has 8 nodes and 6 faces 
	 * if one thread is spawned for each node and these
	 * threads are used for the element's faces
	 * 2 of the threads will be remaining idle
	 */
	/*CalcElemNodeNormals( B[0] , B[1], B[2],
	  x_local, y_local, z_local );*/
       	
	// evaluate face one: nodes 0, 1, 2, 3
	SumElemFaceNormal(&(B[0][0]), &(B[1][0]), &(B[2][0]),
			  &(B[0][1]), &(B[1][1]), &(B[2][1]),
			  &(B[0][2]), &(B[1][2]), &(B[2][2]),
			  &(B[0][3]), &(B[1][3]), &(B[2][3]),
			  x_local[0], y_local[0], z_local[0],
			  x_local[1], y_local[1], z_local[1],
			  x_local[2], y_local[2], z_local[2],
			  x_local[3], y_local[3], z_local[3]);

	// evaluate face two: nodes 0, 4, 5, 1
	SumElemFaceNormal(&(B[0][0]), &(B[1][0]), &(B[2][0]),
			  &(B[0][4]), &(B[1][4]), &(B[2][4]),
			  &(B[0][5]), &(B[1][5]), &(B[2][5]),
			  &(B[0][1]), &(B[1][1]), &(B[2][1]),
			  x_local[0], y_local[0], z_local[0],
			  x_local[4], y_local[4], z_local[4],
			  x_local[5], y_local[5], z_local[5],
			  x_local[1], y_local[1], z_local[1]);
	
	// evaluate face three: nodes 1, 5, 6, 2
	SumElemFaceNormal(&(B[0][1]), &(B[1][1]), &(B[2][1]),
			  &(B[0][5]), &(B[1][5]), &(B[2][5]),
			  &(B[0][6]), &(B[1][6]), &(B[2][6]),
			  &(B[0][2]), &(B[1][2]), &(B[2][2]),
			  x_local[1], y_local[1], z_local[1],
			  x_local[5], y_local[5], z_local[5],
			  x_local[6], y_local[6], z_local[6],
			  x_local[2], y_local[2], z_local[2]);
	
	// evaluate face four: nodes 2, 6, 7, 3 
	SumElemFaceNormal(&(B[0][2]), &(B[1][2]), &(B[2][2]),
			  &(B[0][6]), &(B[1][6]), &(B[2][6]),
			  &(B[0][7]), &(B[1][7]), &(B[2][7]),
			  &(B[0][3]), &(B[1][3]), &(B[2][3]),
			  x_local[2], y_local[2], z_local[2],
			  x_local[6], y_local[6], z_local[6],
			  x_local[7], y_local[7], z_local[7],
			  x_local[3], y_local[3], z_local[3]);

	// evaluate face five: nodes 3, 7, 4, 0
	SumElemFaceNormal(&(B[0][3]), &(B[1][3]), &(B[2][3]),
			  &(B[0][7]), &(B[1][7]), &(B[2][7]),
			  &(B[0][4]), &(B[1][4]), &(B[2][4]),
			  &(B[0][0]), &(B[1][0]), &(B[2][0]),
			  x_local[3], y_local[3], z_local[3],
			  x_local[7], y_local[7], z_local[7],
			  x_local[4], y_local[4], z_local[4],
			  x_local[0], y_local[0], z_local[0]);

	// evaluate face six: nodes 4, 7, 6, 5 
	SumElemFaceNormal(&(B[0][4]), &(B[1][4]), &(B[2][4]),
			  &(B[0][7]), &(B[1][7]), &(B[2][7]),
			  &(B[0][6]), &(B[1][6]), &(B[2][6]),
			  &(B[0][5]), &(B[1][5]), &(B[2][5]),
			  x_local[4], y_local[4], z_local[4],
			  x_local[7], y_local[7], z_local[7],
			  x_local[6], y_local[6], z_local[6],
			  x_local[5], y_local[5], z_local[5]);

	/* computation is done per nodes */
	/*SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
	  &fx_elem[k], &fy_elem[k], &fz_elem[k], numElem ) ;*/

	fx_elem[k + 0*numElem] = -( sigxx[k] * B[0][0] );
	fx_elem[k + 1*numElem] = -( sigxx[k] * B[0][1] );
	fx_elem[k + 2*numElem] = -( sigxx[k] * B[0][2] );
	fx_elem[k + 3*numElem] = -( sigxx[k] * B[0][3] );
	fx_elem[k + 4*numElem] = -( sigxx[k] * B[0][4] );
	fx_elem[k + 5*numElem] = -( sigxx[k] * B[0][5] );
	fx_elem[k + 6*numElem] = -( sigxx[k] * B[0][6] );
	fx_elem[k + 7*numElem] = -( sigxx[k] * B[0][7] );
	    
	fy_elem[k + 0*numElem] = -( sigyy[k] * B[1][0] );
	fy_elem[k + 1*numElem] = -( sigyy[k] * B[1][1] );
	fy_elem[k + 2*numElem] = -( sigyy[k] * B[1][2] );
	fy_elem[k + 3*numElem] = -( sigyy[k] * B[1][3] );
	fy_elem[k + 4*numElem] = -( sigyy[k] * B[1][4] );
	fy_elem[k + 5*numElem] = -( sigyy[k] * B[1][5] );
	fy_elem[k + 6*numElem] = -( sigyy[k] * B[1][6] );
	fy_elem[k + 7*numElem] = -( sigyy[k] * B[1][7] );
	    
	fz_elem[k + 0*numElem] = -( sigzz[k] * B[2][0] );
	fz_elem[k + 1*numElem] = -( sigzz[k] * B[2][1] );
	fz_elem[k + 2*numElem] = -( sigzz[k] * B[2][2] );
	fz_elem[k + 3*numElem] = -( sigzz[k] * B[2][3] );
	fz_elem[k + 4*numElem] = -( sigzz[k] * B[2][4] );
	fz_elem[k + 5*numElem] = -( sigzz[k] * B[2][5] );
	fz_elem[k + 6*numElem] = -( sigzz[k] * B[2][6] );
	fz_elem[k + 7*numElem] = -( sigzz[k] * B[2][7] );
      }
	});
  fut.wait();

  fut = parallel_for_each(extent<1>(numElem),[=
                                              HCC_ID(nodeElemCount)
					      HCC_ID(nodeElemCornerList)
					      HCC_ID(fx)
					      HCC_ID(fy)
					      HCC_ID(fz)
					      HCC_ID(fx_elem)
					      HCC_ID(fy_elem)
					      HCC_ID(fz_elem)](index<1> idx) restrict(amp){
    Int_t i=idx[0];
    Int_t count=nodeElemCount[i];
    Real_t fx_local,fy_local,fz_local;
    fx_local=fy_local=fz_local=(Real_t)(0.0);
    for (int j=0;j<count;j++) {
      Index_t elem=nodeElemCornerList[i+numNode*j];
      fx_local+=fx_elem[elem];
      fy_local+=fy_elem[elem];
      fz_local+=fz_elem[elem];
    }
    fx[i]=fx_local;
    fy[i]=fy_local;
    fz[i]=fz_local;
  });
  fut.wait();

  badvol=0; 
}

/******************************************/

static inline
void CalcFBHourglassForceForElems(struct MeshGPU *meshGPU,
				  Real_t hourg,
				  Index_t numNode,
				  Index_t numElem)
{
  completion_future fut;
  HCC_ARRAY_OBJECT(Real_t, determ) = meshGPU->determ;
  HCC_ARRAY_OBJECT(Real_t, xd) = meshGPU->xd;
  HCC_ARRAY_OBJECT(Real_t, yd) = meshGPU->yd;
  HCC_ARRAY_OBJECT(Real_t, zd) = meshGPU->zd;
  HCC_ARRAY_OBJECT(Index_t, nodelist) = meshGPU->nodelist;
  HCC_ARRAY_OBJECT(Real_t, fx) = meshGPU->fx;
  HCC_ARRAY_OBJECT(Real_t, fy) = meshGPU->fy;
  HCC_ARRAY_OBJECT(Real_t, fz) = meshGPU->fz;
  HCC_ARRAY_OBJECT(Real_t, dvdx) = meshGPU->dvdx;
  HCC_ARRAY_OBJECT(Real_t, dvdy) = meshGPU->dvdy;
  HCC_ARRAY_OBJECT(Real_t, dvdz) = meshGPU->dvdz;
  HCC_ARRAY_OBJECT(Real_t, x8n) = meshGPU->x8n;
  HCC_ARRAY_OBJECT(Real_t, y8n) = meshGPU->y8n;
  HCC_ARRAY_OBJECT(Real_t, z8n) = meshGPU->z8n;
  HCC_ARRAY_OBJECT(Real_t, fx_elem) = meshGPU->fx_elem;
  HCC_ARRAY_OBJECT(Real_t, fy_elem) = meshGPU->fy_elem;
  HCC_ARRAY_OBJECT(Real_t, fz_elem) = meshGPU->fz_elem;
  HCC_ARRAY_OBJECT(Real_t, ss) = meshGPU->ss;
  HCC_ARRAY_OBJECT(Real_t, elemMass) = meshGPU->elemMass;
  HCC_ARRAY_OBJECT(Int_t, nodeElemCount) = meshGPU->nodeElemCount;
  HCC_ARRAY_OBJECT(Index_t, nodeElemCornerList) = meshGPU->nodeElemCornerList;
  if (numElem > 0){
  extent<1> elemExt(PAD(numElem,BLOCKSIZE));
  tiled_extent<1> tElemExt(elemExt,BLOCKSIZE);
    
  fut = parallel_for_each(tElemExt,[=
				    HCC_ID(determ)
				    HCC_ID(xd)
				    HCC_ID(yd)
				    HCC_ID(zd)
				    HCC_ID(fx_elem)
				    HCC_ID(fy_elem)
				    HCC_ID(fz_elem)
				    HCC_ID(nodelist)
				    HCC_ID(dvdx)
				    HCC_ID(dvdy)
				    HCC_ID(dvdz)
				    HCC_ID(x8n)
				    HCC_ID(y8n)
				    HCC_ID(z8n)
				    HCC_ID(ss)
				    HCC_ID(elemMass)]
			  (tiled_index<1> idx) restrict(amp){
    uint elem = idx.global[0]; 
    static const Real_t gamma[4][8] =
    {
      { +1, +1, -1, -1, -1, -1, +1, +1 }, 
      { +1, -1, -1, +1, -1, +1, +1, -1 }, 
      { +1, -1, +1, -1, +1, -1, +1, -1 }, 
      { -1, +1, -1, +1, +1, -1, +1, -1 }
    };
    
    Real_t coefficient;
    Real_t hgfx, hgfy, hgfz;
    Real_t hourgam[4][8];
	  
    /*************************************************/
    /*    compute the hourglass modes */
	  
    if (elem>=numElem)
      elem=numElem-1; // don't return -- need thread to participate in sync operations
	  
    Real_t volinv = (Real_t)(1.0)/determ[elem];

    Real_t hourmodx[4];
    Real_t hourmody[4];
    Real_t hourmodz[4];
	  
    for (int i = 0; i < 4; i++)
      {
	hourmodx[i]=0;
	hourmodx[i] += x8n[elem+numElem*0] * gamma[i][0];
	hourmodx[i] += x8n[elem+numElem*1] * gamma[i][1];
	hourmodx[i] += x8n[elem+numElem*2] * gamma[i][2];
	hourmodx[i] += x8n[elem+numElem*3] * gamma[i][3];
	hourmodx[i] += x8n[elem+numElem*4] * gamma[i][4];
	hourmodx[i] += x8n[elem+numElem*5] * gamma[i][5];
	hourmodx[i] += x8n[elem+numElem*6] * gamma[i][6];
	hourmodx[i] += x8n[elem+numElem*7] * gamma[i][7];
      }

    for (int i = 0; i < 4; i++)
      {
	hourmody[i]=0;
	hourmody[i] += y8n[elem+numElem*0] * gamma[i][0];
	hourmody[i] += y8n[elem+numElem*1] * gamma[i][1];
	hourmody[i] += y8n[elem+numElem*2] * gamma[i][2];
	hourmody[i] += y8n[elem+numElem*3] * gamma[i][3];
	hourmody[i] += y8n[elem+numElem*4] * gamma[i][4];
	hourmody[i] += y8n[elem+numElem*5] * gamma[i][5];
	hourmody[i] += y8n[elem+numElem*6] * gamma[i][6];
	hourmody[i] += y8n[elem+numElem*7] * gamma[i][7];
      }
	  
    for (int i = 0; i < 4; i++)
      {
	hourmodz[i]=0;
	hourmodz[i] += z8n[elem+numElem*0] * gamma[i][0];
	hourmodz[i] += z8n[elem+numElem*1] * gamma[i][1];
	hourmodz[i] += z8n[elem+numElem*2] * gamma[i][2];
	hourmodz[i] += z8n[elem+numElem*3] * gamma[i][3];
	hourmodz[i] += z8n[elem+numElem*4] * gamma[i][4];
	hourmodz[i] += z8n[elem+numElem*5] * gamma[i][5];
	hourmodz[i] += z8n[elem+numElem*6] * gamma[i][6];
	hourmodz[i] += z8n[elem+numElem*7] * gamma[i][7];
      }
	  
    for (int i = 0; i < 4; i++)
      {
	hourgam[i][0] = gamma[i][0] -
	volinv*(dvdx[elem+numElem*0]*hourmodx[i] +
		dvdy[elem+numElem*0]*hourmody[i] +
		dvdz[elem+numElem*0]*hourmodz[i]);
	hourgam[i][1] = gamma[i][1] -
	volinv*(dvdx[elem+numElem*1]*hourmodx[i] +
		dvdy[elem+numElem*1]*hourmody[i] +
		dvdz[elem+numElem*1]*hourmodz[i]);
	hourgam[i][2] = gamma[i][2] -
	volinv*(dvdx[elem+numElem*2]*hourmodx[i] +
		dvdy[elem+numElem*2]*hourmody[i] +
		dvdz[elem+numElem*2]*hourmodz[i]);
	hourgam[i][3] = gamma[i][3] -
	volinv*(dvdx[elem+numElem*3]*hourmodx[i] +
		dvdy[elem+numElem*3]*hourmody[i] +
		dvdz[elem+numElem*3]*hourmodz[i]);
	hourgam[i][4] = gamma[i][4] -
	volinv*(dvdx[elem+numElem*4]*hourmodx[i] +
		dvdy[elem+numElem*4]*hourmody[i] +
		dvdz[elem+numElem*4]*hourmodz[i]);
	hourgam[i][5] = gamma[i][5] -
	volinv*(dvdx[elem+numElem*5]*hourmodx[i] +
		dvdy[elem+numElem*5]*hourmody[i] +
		dvdz[elem+numElem*5]*hourmodz[i]);
	hourgam[i][6] = gamma[i][6] -
	volinv*(dvdx[elem+numElem*6]*hourmodx[i] +
		dvdy[elem+numElem*6]*hourmody[i] +
		dvdz[elem+numElem*6]*hourmodz[i]);
	hourgam[i][7] = gamma[i][7] -
	volinv*(dvdx[elem+numElem*7]*hourmodx[i] +
		dvdy[elem+numElem*7]*hourmody[i] +
		dvdz[elem+numElem*7]*hourmodz[i]);
      }
	  
    coefficient = - hourg * (Real_t)(0.01) *
                    ss[elem] * elemMass[elem] / CBRT(determ[elem]);  
	  
    Index_t ni[8];
    ni[0] = nodelist[elem+numElem*0];
    ni[1] = nodelist[elem+numElem*1];
    ni[2] = nodelist[elem+numElem*2];
    ni[3] = nodelist[elem+numElem*3];
    ni[4] = nodelist[elem+numElem*4];
    ni[5] = nodelist[elem+numElem*5];
    ni[6] = nodelist[elem+numElem*6];
    ni[7] = nodelist[elem+numElem*7];
	  
    Real_t h[4];   
    for (int i=0;i<4;i++)
      {      
	h[i] = 0;
	h[i]+=hourgam[i][0]*xd[ni[0]];
	h[i]+=hourgam[i][1]*xd[ni[1]];
	h[i]+=hourgam[i][2]*xd[ni[2]];
	h[i]+=hourgam[i][3]*xd[ni[3]];
	h[i]+=hourgam[i][4]*xd[ni[4]];
	h[i]+=hourgam[i][5]*xd[ni[5]];
	h[i]+=hourgam[i][6]*xd[ni[6]];
	h[i]+=hourgam[i][7]*xd[ni[7]];
      }
	  
    for( int node = 0; node < 8; node++)
      {
	hgfx = 0;
	hgfx += hourgam[0][node] * h[0];
	hgfx += hourgam[1][node] * h[1];
	hgfx += hourgam[2][node] * h[2];
	hgfx += hourgam[3][node] * h[3];
	fx_elem[elem+numElem*node]=hgfx * coefficient;
      }       
	  
    for (int i = 0; i < 4; i++)
      {      
	h[i] = 0;
	h[i]+=hourgam[i][0]*yd[ni[0]];
	h[i]+=hourgam[i][1]*yd[ni[1]];
	h[i]+=hourgam[i][2]*yd[ni[2]];
	h[i]+=hourgam[i][3]*yd[ni[3]];
	h[i]+=hourgam[i][4]*yd[ni[4]];
	h[i]+=hourgam[i][5]*yd[ni[5]];
	h[i]+=hourgam[i][6]*yd[ni[6]];
	h[i]+=hourgam[i][7]*yd[ni[7]];
      }   
	  
    for( int node = 0; node < 8; node++)
      {
	      
	hgfy = 0;
	hgfy += hourgam[0][node] * h[0];
	hgfy += hourgam[1][node] * h[1];
	hgfy += hourgam[2][node] * h[2];
	hgfy += hourgam[3][node] * h[3];
	fy_elem[elem+numElem*node]=hgfy * coefficient;
      }       
	  
    for (int i=0;i<4;i++)
      {      
	h[i] = 0;
	h[i]+=hourgam[i][0]*zd[ni[0]];
	h[i]+=hourgam[i][1]*zd[ni[1]];
	h[i]+=hourgam[i][2]*zd[ni[2]];
	h[i]+=hourgam[i][3]*zd[ni[3]];
	h[i]+=hourgam[i][4]*zd[ni[4]];
	h[i]+=hourgam[i][5]*zd[ni[5]];
	h[i]+=hourgam[i][6]*zd[ni[6]];
	h[i]+=hourgam[i][7]*zd[ni[7]];
      }   
	  
    for( int node = 0; node < 8; node++)
      {
	hgfz = 0;
	hgfz += hourgam[0][node] * h[0];
	hgfz += hourgam[1][node] * h[1];
	hgfz += hourgam[2][node] * h[2];
	hgfz += hourgam[3][node] * h[3];
	fz_elem[elem+numElem*node]=hgfz * coefficient;
      }
    });
  fut.wait();
  
  }
  if (numNode > 0){
  fut = parallel_for_each(extent<1>(numNode),
		      [=
                       HCC_ID(nodeElemCount)
		       HCC_ID(nodeElemCornerList)
		       HCC_ID(fx)
                       HCC_ID(fy)
                       HCC_ID(fz)
		       HCC_ID(fx_elem)
                       HCC_ID(fy_elem)
                       HCC_ID(fz_elem)]
		       (index<1> idx) restrict(amp){
    Int_t i=idx[0];
    Int_t count=nodeElemCount[i];
    Real_t fx_local = (Real_t)(0.0);
    Real_t fy_local = (Real_t)(0.0);
    Real_t fz_local = (Real_t)(0.0);
    for (int j=0;j<count;j++) {
      Index_t elem=nodeElemCornerList[i+numNode*j];
      fx_local+=fx_elem[elem];
      fy_local+=fy_elem[elem];
      fz_local+=fz_elem[elem];
    }
    fx[i]+=fx_local;
    fy[i]+=fy_local;
    fz[i]+=fz_local;
  });
  fut.wait();

  }
		      
}

/******************************************/

static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz) restrict(amp)
{
  const Real_t twelfth = (Real_t)1.0 / (Real_t)12.0 ;
	
  *dvdx =
  (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
  (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
  (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
  *dvdy =
  - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
  (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
  (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);
	
  *dvdz =
  - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
  (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
  (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);
	
  *dvdx *= twelfth;
  *dvdy *= twelfth;
  *dvdz *= twelfth;
}

static inline
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8]) restrict(amp)
{
  VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
	  y[1], y[2], y[3], y[4], y[5], y[7],
	  z[1], z[2], z[3], z[4], z[5], z[7],
	  &dvdx[0], &dvdy[0], &dvdz[0]);
  VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
	  y[0], y[1], y[2], y[7], y[4], y[6],
	  z[0], z[1], z[2], z[7], z[4], z[6],
	  &dvdx[3], &dvdy[3], &dvdz[3]);
  VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
	  y[3], y[0], y[1], y[6], y[7], y[5],
	  z[3], z[0], z[1], z[6], z[7], z[5],
	  &dvdx[2], &dvdy[2], &dvdz[2]);
  VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
	  y[2], y[3], y[0], y[5], y[6], y[4],
	  z[2], z[3], z[0], z[5], z[6], z[4],
	  &dvdx[1], &dvdy[1], &dvdz[1]);
  VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
	  y[7], y[6], y[5], y[0], y[3], y[1],
	  z[7], z[6], z[5], z[0], z[3], z[1],
	  &dvdx[4], &dvdy[4], &dvdz[4]);
  VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
	  y[4], y[7], y[6], y[1], y[0], y[2],
	  z[4], z[7], z[6], z[1], z[0], z[2],
	  &dvdx[5], &dvdy[5], &dvdz[5]);
  VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
	  y[5], y[4], y[7], y[2], y[1], y[3],
	  z[5], z[4], z[7], z[2], z[1], z[3],
	  &dvdx[6], &dvdy[6], &dvdz[6]);
  VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
	  y[6], y[5], y[4], y[3], y[2], y[0],
	  z[6], z[5], z[4], z[3], z[2], z[0],
	  &dvdx[7], &dvdy[7], &dvdz[7]);
}

static inline
void CalcHourglassControlForElems(struct MeshGPU *meshGPU,Index_t numElem,
                                  Index_t numNode,
                                  Real_t hgcoef)
{
  
  if (numElem > 0){
  HCC_ARRAY_OBJECT(Real_t, v) = meshGPU->v;
  HCC_ARRAY_OBJECT(Real_t, volo) = meshGPU->volo;
  HCC_ARRAY_OBJECT(Real_t, x) = meshGPU->x;
  HCC_ARRAY_OBJECT(Real_t, y) = meshGPU->y;
  HCC_ARRAY_OBJECT(Real_t, z) = meshGPU->z;
  HCC_ARRAY_OBJECT(Real_t, determ) = meshGPU->determ;
  HCC_ARRAY_OBJECT(Index_t, nodelist) = meshGPU->nodelist;
  HCC_ARRAY_OBJECT(Real_t, fx) = meshGPU->fx;
  HCC_ARRAY_OBJECT(Real_t, fy) = meshGPU->fy;
  HCC_ARRAY_OBJECT(Real_t, fz) = meshGPU->fz;
  HCC_ARRAY_OBJECT(Real_t, dvdx) = meshGPU->dvdx;
  HCC_ARRAY_OBJECT(Real_t, dvdy) = meshGPU->dvdy;
  HCC_ARRAY_OBJECT(Real_t, dvdz) = meshGPU->dvdz;
  HCC_ARRAY_OBJECT(Real_t, x8n) = meshGPU->x8n;
  HCC_ARRAY_OBJECT(Real_t, y8n) = meshGPU->y8n;
  HCC_ARRAY_OBJECT(Real_t, z8n) = meshGPU->z8n;
    
    extent<1> elemExt(PAD(numElem,BLOCKSIZE));
    tiled_extent<1> tElemExt(elemExt,BLOCKSIZE);
    
    completion_future fut = parallel_for_each(tElemExt,
		    [=
		     HCC_ID(x)
		     HCC_ID(y)
		     HCC_ID(z)
		     HCC_ID(determ)
		     HCC_ID(nodelist)
		     HCC_ID(volo)
		     HCC_ID(v)
		     HCC_ID(dvdx)
		     HCC_ID(dvdy)
		     HCC_ID(dvdz)
		     HCC_ID(x8n)
		     HCC_ID(y8n)
		     HCC_ID(z8n)]
		    (tiled_index<1> idx) restrict(amp){
	  
    uint elem = idx.global[0]; //get_global_id(X);
    Real_t x1[8],y1[8],z1[8];
    Real_t pfx[8],pfy[8],pfz[8];
    
    if (elem>=numElem)
      elem = numElem - 1; 
	  
    for (int node = 0; node < 8; node++) {
      Index_t idx = elem + numElem * node;
      Index_t ni = nodelist[idx];
	  
      x1[node] = x[ni];
      y1[node] = y[ni];
      z1[node] = z[ni];
    }
    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1); 
	
    for (int node = 0; node < 8; node++) {
      Index_t idx = elem + numElem * node;
      dvdx[idx] = pfx[node];
      dvdy[idx] = pfy[node];
      dvdz[idx] = pfz[node];
      x8n[idx]  = x1[node];
      y8n[idx]  = y1[node];
      z8n[idx]  = z1[node];
    }
    determ[elem] = volo[elem] * v[elem];	  
  });
  fut.wait();

  }

  if ( hgcoef > Real_t(0.) ) {

    CalcFBHourglassForceForElems(meshGPU,hgcoef,
				 numNode,
				 numElem);
  }
}

/******************************************/

static inline
void CalcVolumeForceForElems(Domain& mesh, struct MeshGPU *meshGPU)
{
  Index_t numNode = mesh.numNode() ;
  Index_t numElem = mesh.numElem();
  Real_t  hgcoef = mesh.hgcoef() ;
  int badvol;
  
  /* Sum contributions to total stress tensor */
  InitStressTermsForElems(meshGPU, numElem);      

  // call elemlib stress integration loop to produce nodal forces from
  // material stresses.

  IntegrateStressForElems(meshGPU,numNode, badvol, numElem);


  // check for negative element volume
  if (badvol) exit(VolumeError) ;

      
  CalcHourglassControlForElems(meshGPU,numElem,
                          numNode,
                          hgcoef);
}

/******************************************/

static inline
void CalcForceForNodes(Domain& mesh, struct MeshGPU *meshGPU)
{
  Index_t numNode = mesh.numNode() ;
  if (numNode > 0){
  HCC_ARRAY_OBJECT(Real_t, fx) = meshGPU->fx;
  HCC_ARRAY_OBJECT(Real_t, fy) = meshGPU->fy;
  HCC_ARRAY_OBJECT(Real_t, fz) = meshGPU->fz;
    completion_future fut = parallel_for_each(extent<1>(numNode),
		    [=
		     HCC_ID(fx)
		     HCC_ID(fy)
		     HCC_ID(fz)]
		    (index<1> idx) restrict(amp){
    Int_t i = idx[0];
    fx[i] = (Real_t)(0.0) ;
    fy[i] = (Real_t)(0.0) ;
    fz[i] = (Real_t)(0.0) ;
  });
    fut.wait();
  }
    
  CalcVolumeForceForElems(mesh, meshGPU);
}

/******************************************/


static inline
void CalcAccelerationForNodes(Domain &mesh, struct MeshGPU *meshGPU,
			      Index_t numNode)
{
  if (numNode <= 0) return;
  HCC_ARRAY_OBJECT(Real_t, nodalMass) = meshGPU->nodalMass;
  HCC_ARRAY_OBJECT(Real_t, xdd) = meshGPU->xdd;
  HCC_ARRAY_OBJECT(Real_t, ydd) = meshGPU->ydd;
  HCC_ARRAY_OBJECT(Real_t, zdd) = meshGPU->zdd;
  HCC_ARRAY_OBJECT(Real_t, fx) = meshGPU->fx;
  HCC_ARRAY_OBJECT(Real_t, fy) = meshGPU->fy;
  HCC_ARRAY_OBJECT(Real_t, fz) = meshGPU->fz;
  completion_future fut = parallel_for_each(extent<1>(numNode),[=
                                                             HCC_ID(xdd)
                                                             HCC_ID(ydd)
                                                             HCC_ID(zdd)
                                                             HCC_ID(fx)
                                                             HCC_ID(fy)
                                                             HCC_ID(fz)
                                                             HCC_ID(nodalMass)]
                                            (index<1> idx) restrict(amp){

    Int_t i=idx[0];
    xdd[i]=fx[i]/nodalMass[i];
    ydd[i]=fy[i]/nodalMass[i];
    zdd[i]=fz[i]/nodalMass[i];
  });
  fut.wait();

}

/******************************************/

static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain& mesh,
						 struct MeshGPU *meshGPU,
						 Index_t numNodeBC)
{
  if (numNodeBC <= 0) return;
  HCC_ARRAY_OBJECT(Real_t, xdd) = meshGPU->xdd;
  HCC_ARRAY_OBJECT(Real_t, ydd) = meshGPU->ydd;
  HCC_ARRAY_OBJECT(Real_t, zdd) = meshGPU->zdd;
  HCC_ARRAY_OBJECT(Index_t, symmX) = meshGPU->symmX;
  HCC_ARRAY_OBJECT(Index_t, symmY) = meshGPU->symmY;
  HCC_ARRAY_OBJECT(Index_t, symmZ) = meshGPU->symmZ;
  completion_future fut = parallel_for_each(extent<1>(numNodeBC),[=
                                                                  HCC_ID(xdd)
                                                                  HCC_ID(ydd)
                                                                  HCC_ID(zdd)
                                                                  HCC_ID(symmX)
                                                                  HCC_ID(symmY)
                                                                  HCC_ID(symmZ)]
                                            (index<1> idx) restrict(amp)
    {
      Int_t i=idx[0];
      xdd[symmX[i]] = (Real_t)(0.0) ;
      ydd[symmY[i]] = (Real_t)(0.0) ;
      zdd[symmZ[i]] = (Real_t)(0.0) ;
    });
  fut.wait();

}

/******************************************/

static inline
void CalcVelocityForNodes(Domain &mesh, struct MeshGPU *meshGPU,
			  const Real_t dt,
			  const Real_t u_cut,
			  Index_t numNode)
{
  if (numNode <= 0) return;
  HCC_ARRAY_OBJECT(Real_t, xd) = meshGPU->xd;
  HCC_ARRAY_OBJECT(Real_t, yd) = meshGPU->yd;
  HCC_ARRAY_OBJECT(Real_t, zd) = meshGPU->zd;
  HCC_ARRAY_OBJECT(Real_t, xdd) = meshGPU->xdd;
  HCC_ARRAY_OBJECT(Real_t, ydd) = meshGPU->ydd;
  HCC_ARRAY_OBJECT(Real_t, zdd) = meshGPU->zdd;
  completion_future fut = parallel_for_each(extent<1>(numNode),[=
                                                                HCC_ID(xd)
                                                                HCC_ID(yd)
                                                                HCC_ID(zd)
								HCC_ID(xdd)
								HCC_ID(ydd)
								HCC_ID(zdd)]
					    (index<1> idx) restrict(amp)
  {
    Int_t i = idx[0]; 
    Real_t xdtmp, ydtmp, zdtmp ;
        
    xdtmp = xd[i] + xdd[i] * dt ;
    if( FABS(xdtmp) < u_cut ) xdtmp = (Real_t)(0.0);
    xd[i] = xdtmp ;
    
    ydtmp = yd[i] + ydd[i] * dt ;
    if( FABS(ydtmp) < u_cut ) ydtmp = (Real_t)(0.0);
    yd[i] = ydtmp ;
    
    zdtmp = zd[i] + zdd[i] * dt ;
    if( FABS(zdtmp) < u_cut ) zdtmp = (Real_t)(0.0);
    zd[i] = zdtmp ;
  });
  fut.wait();
}

/******************************************/

static inline
void CalcPositionForNodes(Domain &mesh, struct MeshGPU *meshGPU,
			  const Real_t dt,
			  Index_t numNode)
{
  if (numNode <= 0) return;
  HCC_ARRAY_OBJECT(Real_t, x) = meshGPU->x;
  HCC_ARRAY_OBJECT(Real_t, y) = meshGPU->y;
  HCC_ARRAY_OBJECT(Real_t, z) = meshGPU->z;
  HCC_ARRAY_OBJECT(Real_t, xd) = meshGPU->xd;
  HCC_ARRAY_OBJECT(Real_t, yd) = meshGPU->yd;
  HCC_ARRAY_OBJECT(Real_t, zd) = meshGPU->zd;
  completion_future fut = parallel_for_each(extent<1>(numNode),[=
                                                                HCC_ID(x)
                                                                HCC_ID(y)
                                                                HCC_ID(z)
                                                                HCC_ID(xd)
                                                                HCC_ID(yd)
                                                                HCC_ID(zd)]
                                             (index<1> idx) restrict(amp)
    {
      Int_t i = idx[0]; 
      x[i] += xd[i] * dt;
      y[i] += yd[i] * dt;
      z[i] += zd[i] * dt;
    });
  fut.wait();
}

/******************************************/

static inline
void LagrangeNodal(Domain& mesh, struct MeshGPU *meshGPU)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   Domain_member fieldData[6] ;
#endif

   const Real_t delt = mesh.deltatime() ;
   Real_t u_cut = mesh.u_cut() ;

   Index_t numNode = mesh.numNode() ;
   Index_t numElem = mesh.numElem();
   Index_t numNodeBC = (mesh.sizeX()+1)*(mesh.sizeX()+1) ;
    
   if (numElem != 0) {

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
     CalcForceForNodes(mesh, meshGPU);

     CalcAccelerationForNodes(mesh, meshGPU, numNode);

     ApplyAccelerationBoundaryConditionsForNodes(mesh, meshGPU, numNodeBC);
      
     CalcVelocityForNodes(mesh, meshGPU, delt, u_cut, numNode);
   
     CalcPositionForNodes( mesh, meshGPU, delt, numNode);
   }
   
  return;
}

/******************************************/

static inline
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
	       const Real_t z6, const Real_t z7 ) restrict(amp)
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

/******************************************/

//inline
static inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] ) restrict(amp)
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

static inline
Real_t CalcElemVolume_nonamp( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
	       const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

/******************************************/

//inline

Real_t CalcElemVolume_nonamp( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume_nonamp( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

/******************************************/

static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3) restrict(amp)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

/******************************************/

static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume) restrict(amp)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = FMAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

/******************************************/

static inline
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
			       Real_t* const d ) restrict(amp)
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = (Real_t)( .5) * ( dxddy + dyddx );
  d[4]  = (Real_t)( .5) * ( dxddz + dzddx );
  d[3]  = (Real_t)( .5) * ( dzddy + dyddz );
}

/******************************************/


static inline
void CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume ) restrict(amp)
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = (Real_t)(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = (Real_t)(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = (Real_t)(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = (Real_t)(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = (Real_t)(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = (Real_t)(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = (Real_t)(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = (Real_t)(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = (Real_t)(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */

  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = (Real_t)(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}


static inline
void CalcKinematicsForElems(struct MeshGPU *meshGPU,Index_t numElem, Real_t dt)
{
  if (numElem <= 0) return;
  HCC_ARRAY_OBJECT(Index_t, nodelist) = meshGPU->nodelist;
  HCC_ARRAY_OBJECT(Real_t, volo) = meshGPU->volo;
  HCC_ARRAY_OBJECT(Real_t, v) = meshGPU->v;
  HCC_ARRAY_OBJECT(Real_t, x) = meshGPU->x;
  HCC_ARRAY_OBJECT(Real_t, y) = meshGPU->y;
  HCC_ARRAY_OBJECT(Real_t, z) = meshGPU->z;
  HCC_ARRAY_OBJECT(Real_t, xd) = meshGPU->xd;
  HCC_ARRAY_OBJECT(Real_t, yd) = meshGPU->yd;
  HCC_ARRAY_OBJECT(Real_t, zd) = meshGPU->zd;
  HCC_ARRAY_OBJECT(Real_t, vnew) = meshGPU->vnew;
  HCC_ARRAY_OBJECT(Real_t, delv) = meshGPU->delv;
  HCC_ARRAY_OBJECT(Real_t, arealg) = meshGPU->arealg;
  HCC_ARRAY_OBJECT(Real_t, dxx) = meshGPU->dxx;
  HCC_ARRAY_OBJECT(Real_t, dyy) = meshGPU->dyy;
  HCC_ARRAY_OBJECT(Real_t, dzz) = meshGPU->dzz;

  extent<1> elemExt(PAD(numElem,BLOCKSIZE));
  tiled_extent<1> tElemExt(elemExt,BLOCKSIZE);  
  completion_future fut = parallel_for_each(tElemExt,
		    [=
		     HCC_ID(nodelist)
		     HCC_ID(volo)
		     HCC_ID(v)
		     HCC_ID(x)
		     HCC_ID(y)
		     HCC_ID(z)
		     HCC_ID(xd)
		     HCC_ID(yd)
		     HCC_ID(zd)
		     HCC_ID(vnew)
		     HCC_ID(delv)
		     HCC_ID(arealg)
		     HCC_ID(dxx)
		     HCC_ID(dyy)
		     HCC_ID(dzz)](tiled_index<1> idx) restrict(amp)
	{
	  int k=idx.global[0];
	  if(k < numElem){
	  Real_t B[3][8] ; /** shape function derivatives */
	  Real_t D[6] ;
	  Real_t x_local[8] ;
	  Real_t y_local[8] ;
	  Real_t z_local[8] ;
	  Real_t xd_local[8] ;
	  Real_t yd_local[8] ;
	  Real_t zd_local[8] ;
	  Real_t detJ = (Real_t)(0.0) ;
	  
	  
	  Real_t volume ;
	  Real_t relativeVolume ;
	    
	  // get nodal coordinates from global arrays and copy into local arrays.
	  for( Index_t lnode=0 ; lnode<8 ; ++lnode )
	    {
	      Index_t gnode = nodelist[k+lnode*numElem];
	      x_local[lnode] = x[gnode];
	      y_local[lnode] = y[gnode];
	      z_local[lnode] = z[gnode];
	    }
	    
	  // volume calculations
	  volume = CalcElemVolume(x_local, y_local, z_local );
	  relativeVolume = volume / volo[k] ;
	  vnew[k] = relativeVolume ;
	  delv[k] = relativeVolume - v[k] ;
	    
	  // set characteristic length
	  arealg[k] = CalcElemCharacteristicLength(x_local,y_local,z_local,volume);
	    
	  // get nodal velocities from global array and copy into local arrays.
	  for( Index_t lnode=0 ; lnode<8 ; ++lnode )
	    {
	      Index_t gnode = nodelist[k+lnode*numElem];
	      xd_local[lnode] = xd[gnode];
	      yd_local[lnode] = yd[gnode];
	      zd_local[lnode] = zd[gnode];
	    }
	    
	  Real_t dt2 = (Real_t)(0.5) * dt;
	  for ( Index_t j=0 ; j<8 ; ++j )
	    {
	      x_local[j] -= dt2 * xd_local[j];
	      y_local[j] -= dt2 * yd_local[j];
	      z_local[j] -= dt2 * zd_local[j];
	    }
	    
	  CalcElemShapeFunctionDerivatives(x_local,y_local,z_local,B,&detJ);
	    
	  CalcElemVelocityGradient(xd_local,yd_local,zd_local,B,detJ,D);
	  // put velocity gradient quantities into their global arrays.
	  dxx[k] = D[0];
	  dyy[k] = D[1];
	  dzz[k] = D[2];
	  }
	});
  fut.wait();
}

/******************************************/

static inline
void CalcLagrangeElements(Domain& mesh, struct MeshGPU *meshGPU)
{
  Index_t numElem = mesh.numElem() ;
  
  const Real_t deltatime = mesh.deltatime();
  if (numElem > 0) {
  HCC_ARRAY_OBJECT(Real_t, dxx) = meshGPU->dxx;
  HCC_ARRAY_OBJECT(Real_t, dyy) = meshGPU->dyy;
  HCC_ARRAY_OBJECT(Real_t, dzz) = meshGPU->dzz;
  HCC_ARRAY_OBJECT(Real_t, vdov) = meshGPU->vdov;

  CalcKinematicsForElems( meshGPU, numElem, deltatime);

  completion_future fut = parallel_for_each(extent<1>(numElem),
		      [=
		       HCC_ID(vdov)
		       HCC_ID(dxx)
		       HCC_ID(dyy)
                       HCC_ID(dzz)](index<1> idx) restrict(amp)
        {
	  int k=idx[0];
	  Real_t vdovNew = dxx[k] + dyy[k] + dzz[k] ;
	  Real_t vdovthird = vdovNew/(Real_t)(3.0) ;
	    
	  // make the rate of deformation tensor deviatoric
	  vdov[k] = vdovNew ;
	  dxx[k] -= vdovthird ;
	  dyy[k] -= vdovthird ;
	  dzz[k] -= vdovthird ;
	});
  fut.wait();
  }
}

/******************************************/


static inline
void CalcMonotonicQGradientsForElems(struct MeshGPU *meshGPU,Index_t numElem)
{
  if (numElem <= 0) return;
  HCC_ARRAY_OBJECT(Index_t, nodelist) = meshGPU->nodelist;
  HCC_ARRAY_OBJECT(Real_t, volo) = meshGPU->volo;
  HCC_ARRAY_OBJECT(Real_t, v) = meshGPU->v;
  HCC_ARRAY_OBJECT(Real_t, x) = meshGPU->x;
  HCC_ARRAY_OBJECT(Real_t, y) = meshGPU->y;
  HCC_ARRAY_OBJECT(Real_t, z) = meshGPU->z;
  HCC_ARRAY_OBJECT(Real_t, xd) = meshGPU->xd;
  HCC_ARRAY_OBJECT(Real_t, yd) = meshGPU->yd;
  HCC_ARRAY_OBJECT(Real_t, zd) = meshGPU->zd;
  HCC_ARRAY_OBJECT(Real_t, vnew) = meshGPU->vnew;
  HCC_ARRAY_OBJECT(Real_t, delx_zeta) = meshGPU->delx_zeta;
  HCC_ARRAY_OBJECT(Real_t, delv_zeta) = meshGPU->delv_zeta;
  HCC_ARRAY_OBJECT(Real_t, delx_eta) = meshGPU->delx_eta;
  HCC_ARRAY_OBJECT(Real_t, delv_eta) = meshGPU->delv_eta;
  HCC_ARRAY_OBJECT(Real_t, delx_xi) = meshGPU->delx_xi;
  HCC_ARRAY_OBJECT(Real_t, delv_xi) = meshGPU->delv_xi;
  extent<1> elemExt(PAD(numElem,BLOCKSIZE));
  tiled_extent<1> tElemExt(elemExt,BLOCKSIZE);
  completion_future fut = parallel_for_each(tElemExt,
		    [=
		     HCC_ID(nodelist)
		     HCC_ID(volo)
		     HCC_ID(v)
		     HCC_ID(x)
		     HCC_ID(y)
		     HCC_ID(z)
		     HCC_ID(xd)
		     HCC_ID(yd)
		     HCC_ID(zd)
		     HCC_ID(vnew)
		     HCC_ID(delx_zeta)
		     HCC_ID(delv_zeta)
		     HCC_ID(delx_xi)
		     HCC_ID(delv_xi)
		     HCC_ID(delx_eta)
                     HCC_ID(delv_eta)](tiled_index<1> idx) restrict(amp)
    {
   int i=idx.global[0];
   if(i < numElem){
#define SUM4(a,b,c,d) (a + b + c + d)
   const Real_t ptiny = (Real_t)(1.e-36) ;

   Real_t ax,ay,az ;
   Real_t dxv,dyv,dzv ;

   Index_t n0 = nodelist[i+0*numElem] ;
   Index_t n1 = nodelist[i+1*numElem] ;
   Index_t n2 = nodelist[i+2*numElem] ;
   Index_t n3 = nodelist[i+3*numElem] ;
   Index_t n4 = nodelist[i+4*numElem] ;
   Index_t n5 = nodelist[i+5*numElem] ;
   Index_t n6 = nodelist[i+6*numElem] ;
   Index_t n7 = nodelist[i+7*numElem] ;

   Real_t x0 = x[n0] ; Real_t y0 = y[n0] ; Real_t z0 = z[n0] ;
   Real_t x1 = x[n1] ; Real_t y1 = y[n1] ; Real_t z1 = z[n1] ;
   Real_t x2 = x[n2] ; Real_t y2 = y[n2] ; Real_t z2 = z[n2] ;
   Real_t x3 = x[n3] ; Real_t y3 = y[n3] ; Real_t z3 = z[n3] ;
   Real_t x4 = x[n4] ; Real_t y4 = y[n4] ; Real_t z4 = z[n4] ;
   Real_t x5 = x[n5] ; Real_t y5 = y[n5] ; Real_t z5 = z[n5] ;
   Real_t x6 = x[n6] ; Real_t y6 = y[n6] ; Real_t z6 = z[n6] ;
   Real_t x7 = x[n7] ; Real_t y7 = y[n7] ; Real_t z7 = z[n7] ;

   Real_t xv0 = xd[n0] ; Real_t yv0 = yd[n0] ; Real_t zv0 = zd[n0] ;
   Real_t xv1 = xd[n1] ; Real_t yv1 = yd[n1] ; Real_t zv1 = zd[n1] ;
   Real_t xv2 = xd[n2] ; Real_t yv2 = yd[n2] ; Real_t zv2 = zd[n2] ;
   Real_t xv3 = xd[n3] ; Real_t yv3 = yd[n3] ; Real_t zv3 = zd[n3] ;
   Real_t xv4 = xd[n4] ; Real_t yv4 = yd[n4] ; Real_t zv4 = zd[n4] ;
   Real_t xv5 = xd[n5] ; Real_t yv5 = yd[n5] ; Real_t zv5 = zd[n5] ;
   Real_t xv6 = xd[n6] ; Real_t yv6 = yd[n6] ; Real_t zv6 = zd[n6] ;
   Real_t xv7 = xd[n7] ; Real_t yv7 = yd[n7] ; Real_t zv7 = zd[n7] ;

   Real_t vol = volo[i]*vnew[i] ;
   Real_t norm = (Real_t)(1.0) / ( vol + ptiny ) ;

   Real_t dxj = (Real_t)(-0.25)*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
   Real_t dyj = (Real_t)(-0.25)*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
   Real_t dzj = (Real_t)(-0.25)*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

   Real_t dxi = (Real_t)( 0.25)*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
   Real_t dyi = (Real_t)( 0.25)*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
   Real_t dzi = (Real_t)( 0.25)*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

   Real_t dxk = (Real_t)( 0.25)*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
   Real_t dyk = (Real_t)( 0.25)*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
   Real_t dzk = (Real_t)( 0.25)*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

   /* find delvk and delxk ( i cross j ) */

   ax = dyi*dzj - dzi*dyj ;
   ay = dzi*dxj - dxi*dzj ;
   az = dxi*dyj - dyi*dxj ;

   delx_zeta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = (Real_t)(0.25)*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
   dyv = (Real_t)(0.25)*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
   dzv = (Real_t)(0.25)*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

   delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

   /* find delxi and delvi ( j cross k ) */

   ax = dyj*dzk - dzj*dyk ;
   ay = dzj*dxk - dxj*dzk ;
   az = dxj*dyk - dyj*dxk ;

   delx_xi[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = (Real_t)(0.25)*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
   dyv = (Real_t)(0.25)*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
   dzv = (Real_t)(0.25)*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

   delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

   /* find delxj and delvj ( k cross i ) */

   ax = dyk*dzi - dzk*dyi ;
   ay = dzk*dxi - dxk*dzi ;
   az = dxk*dyi - dyk*dxi ;

   delx_eta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = (Real_t)(-0.25)*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
   dyv = (Real_t)(-0.25)*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
   dzv = (Real_t)(-0.25)*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

   delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
#undef SUM4
   }
    });
  fut.wait();
}

/******************************************/


static inline
void CalcMonotonicQRegionForElems(struct MeshGPU *meshGPU,
				  Index_t regionStart,
				  Real_t qlc_monoq,
				  Real_t qqc_monoq,
				  Real_t monoq_limiter_mult,
				  Real_t monoq_max_slope,
				  Real_t ptiny,
				  Index_t elength )
{
  if (elength <= 0) return;
  HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
  HCC_ARRAY_OBJECT(Real_t, volo) = meshGPU->volo;
  HCC_ARRAY_OBJECT(Real_t, vnew) = meshGPU->vnew;
  HCC_ARRAY_OBJECT(Real_t, vdov) = meshGPU->vdov;
  HCC_ARRAY_OBJECT(Real_t, delx_zeta) = meshGPU->delx_zeta;
  HCC_ARRAY_OBJECT(Real_t, delv_zeta) = meshGPU->delv_zeta;
  HCC_ARRAY_OBJECT(Real_t, delx_xi) = meshGPU->delx_xi;
  HCC_ARRAY_OBJECT(Real_t, delv_xi) = meshGPU->delv_xi;
  HCC_ARRAY_OBJECT(Real_t, delx_eta) = meshGPU->delx_eta;
  HCC_ARRAY_OBJECT(Real_t, delv_eta) = meshGPU->delv_eta;
  HCC_ARRAY_OBJECT(Index_t, elemBC) = meshGPU->elemBC;
  HCC_ARRAY_OBJECT(Index_t, lxim) = meshGPU->lxim;
  HCC_ARRAY_OBJECT(Index_t, lxip) = meshGPU->lxip;
  HCC_ARRAY_OBJECT(Index_t, letam) = meshGPU->letam;
  HCC_ARRAY_OBJECT(Index_t, letap) = meshGPU->letap;
  HCC_ARRAY_OBJECT(Index_t, lzetam) = meshGPU->lzetam;
  HCC_ARRAY_OBJECT(Index_t, lzetap) = meshGPU->lzetap;
  HCC_ARRAY_OBJECT(Real_t, elemMass) = meshGPU->elemMass;
  HCC_ARRAY_OBJECT(Real_t, qq) = meshGPU->qq;
  HCC_ARRAY_OBJECT(Real_t, ql) = meshGPU->ql;
  completion_future fut = parallel_for_each(extent<1>(elength),
		    [=
		     HCC_ID(matElemlist)
		     HCC_ID(volo)
		     HCC_ID(vnew)
		     HCC_ID(vdov)
		     HCC_ID(delx_zeta)
		     HCC_ID(delv_zeta)
		     HCC_ID(delx_xi)
		     HCC_ID(delv_xi)
		     HCC_ID(delx_eta)
		     HCC_ID(delv_eta)
		     HCC_ID(elemBC)
		     HCC_ID(lxim)
		     HCC_ID(lxip)
		     HCC_ID(letam)
		     HCC_ID(letap)
		     HCC_ID(lzetam)
		     HCC_ID(lzetap)
		     HCC_ID(elemMass)
		     HCC_ID(qq)
                     HCC_ID(ql)](index<1> idx) restrict(amp)
    {
      int ielem=idx[0];
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = matElemlist[regionStart + ielem];
	
      Int_t bcMask = elemBC[i] ;
      Real_t delvm, delvp ;
	
      //  phixi     
      Real_t norm = (Real_t)(1.) / ( delv_xi[i] + ptiny ) ;
	
      switch (bcMask & XI_M) {
      case 0:         delvm = delv_xi[lxim[i]] ; break ; 
      case XI_M_SYMM: delvm = delv_xi[i]       ; break ; 
      case XI_M_FREE: delvm = (Real_t)(0.0)    ; break ; 
      default:        delvm = (Real_t)(0.0); break ;
      }
      switch (bcMask & XI_P) {
      case 0:         delvp = delv_xi[lxip[i]] ; break ; 
      case XI_P_SYMM: delvp = delv_xi[i]       ; break ; 
      case XI_P_FREE: delvp = (Real_t)(0.0)    ; break ; 
      default:        delvp = (Real_t)(0.0); break ; 
      }
	
      delvm = delvm * norm ;
      delvp = delvp * norm ;
	
      phixi = (Real_t)(.5) * ( delvm + delvp ) ;
	
      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < (Real_t)(0.)) phixi = (Real_t)(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      //  phieta     
      norm = (Real_t)(1.) / ( delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
      case 0:          delvm = delv_eta[letam[i]] ; break ; 
      case ETA_M_SYMM: delvm = delv_eta[i]        ; break ; 
      case ETA_M_FREE: delvm = (Real_t)(0.0)      ; break ; 
      default:         delvm = (Real_t)(0.0)      ; break ; 
      }


      // If you uncomment the case 0 below, then the program will
      //    segfault. I'm not sure why yet.
      switch (bcMask & ETA_P) {
      case 0:          delvp = delv_eta[letap[i]] ; break ; 
      case ETA_P_SYMM: delvp = delv_eta[i]        ; break ; 
      case ETA_P_FREE: delvp = (Real_t)(0.0)      ; break ; 
      default:         delvp = (Real_t)(0.0)      ; break ; 
      }
	
      delvm = delvm * norm ;
      delvp = delvp * norm ;
	
      phieta = (Real_t)(.5) * ( delvm + delvp ) ;
	
      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;
	
      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < (Real_t)(0.)) phieta = (Real_t)(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      //  phizeta
      norm = (Real_t)(1.) / ( delv_zeta[i] + ptiny ) ;
	
      switch (bcMask & ZETA_M) {
      case 0:           delvm = delv_zeta[lzetam[i]] ; break ; 
      case ZETA_M_SYMM: delvm = delv_zeta[i]         ; break ; 
      case ZETA_M_FREE: delvm = (Real_t)(0.0)        ; break ; 
      default:          delvm = (Real_t)(0.0); break ; 
      }
      switch (bcMask & ZETA_P) {
      case 0:           delvp = delv_zeta[lzetap[i]] ; break ; 
      case ZETA_P_SYMM: delvp = delv_zeta[i]         ; break ; 
      case ZETA_P_FREE: delvp = (Real_t)(0.0)        ; break ; 
      default:          delvp = (Real_t)(0.0); break ; 
      }
	
      delvm = delvm * norm ;
      delvp = delvp * norm ;
	
      phizeta = (Real_t)(.5) * ( delvm + delvp ) ;
	
      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;
	
      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < (Real_t)(0.)) phizeta = (Real_t)(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;
	
      // Remove length scale
	
      if ( vdov[i] > (Real_t)(0.) )  {
	qlin  = (Real_t)(0.) ;
	qquad = (Real_t)(0.) ;
      }
      else {
	Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
	Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
	Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

	if ( delvxxi   > (Real_t)(0.) ) delvxxi   = (Real_t)(0.) ;
	if ( delvxeta  > (Real_t)(0.) ) delvxeta  = (Real_t)(0.) ;
	if ( delvxzeta > (Real_t)(0.) ) delvxzeta = (Real_t)(0.) ;
	  
	Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;
	  
	qlin = -qlc_monoq * rho *
	  (  delvxxi   * ((Real_t)(1.) - phixi) +
	     delvxeta  * ((Real_t)(1.) - phieta) +
	     delvxzeta * ((Real_t)(1.) - phizeta)  ) ;
	  
	qquad = qqc_monoq * rho *
	  (  delvxxi*delvxxi     * ((Real_t)(1.) - phixi*phixi) +
	     delvxeta*delvxeta   * ((Real_t)(1.) - phieta*phieta) +
	     delvxzeta*delvxzeta * ((Real_t)(1.) - phizeta*phizeta)  ) ;
      }
	
      qq[i] = qquad ;
      ql[i] = qlin  ;
    });
  fut.wait();
		    
}

/******************************************/

static inline
void CalcMonotonicQForElems(Domain& mesh,
			    struct MeshGPU *meshGPU)
{  
   //
   // initialize parameters
   // 
   const Real_t ptiny        = Real_t(1.e-36) ;
   Real_t monoq_max_slope    = mesh.monoq_max_slope() ;
   Real_t monoq_limiter_mult = mesh.monoq_limiter_mult() ;
   
   //
   // calculate the monotonic q for pure regions
   //
   for (Index_t r=0 ; r<mesh.numReg() ; ++r) {
     
     Index_t elength = mesh.regElemSize(r) ;      
     if (elength > 0) {

       Real_t qlc_monoq = mesh.qlc_monoq();
       Real_t qqc_monoq = mesh.qqc_monoq();
       Index_t regionStart = mesh.regStartPosition(r);
       CalcMonotonicQRegionForElems(meshGPU,
				    regionStart,
				    qlc_monoq,
				    qqc_monoq,
				    monoq_limiter_mult,
				    monoq_max_slope,
				    ptiny,
				    elength );
      }
   }
}

/******************************************/

static inline
void CalcQForElems(Domain& mesh, struct MeshGPU *meshGPU)
{
   //
   // MONOTONIC Q option
   //

   Index_t numElem = mesh.numElem() ;

   if (numElem != 0) {
      /* Calculate velocity gradients */

     CalcMonotonicQGradientsForElems(meshGPU, numElem);

     CalcMonotonicQForElems(mesh, meshGPU);
   }
}

/******************************************/

static inline
void CalcPressureForElems(struct MeshGPU *meshGPU,
			  Index_t regionStart,
                          HCC_ARRAY_OBJECT(Real_t, p_new),
                          HCC_ARRAY_OBJECT(Real_t, compression),
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          Index_t length)
{
  Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
  if (length <= 0) return;
  HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
  HCC_ARRAY_OBJECT(Real_t, bvc) = meshGPU->bvc;
  HCC_ARRAY_OBJECT(Real_t, pbvc) = meshGPU->pbvc;
  HCC_ARRAY_OBJECT(Real_t, e_old) = meshGPU->e_new;
  HCC_ARRAY_OBJECT(Real_t, vnewc) = meshGPU->vnewc;
  completion_future fut = parallel_for_each(extent<1>(length),
		      [=
		       HCC_ID(matElemlist)
		       HCC_ID(p_new)
		       HCC_ID(bvc)
		       HCC_ID(pbvc)
		       HCC_ID(e_old)
		       HCC_ID(compression)
		       HCC_ID(vnewc)](index<1> idx) restrict(amp)
    {
      int i=idx[0];
      bvc[i] = c1s * (compression[i] + (Real_t)(1.));
      pbvc[i] = c1s;
	
      p_new[i] = bvc[i] * e_old[i] ;
	
      if (FABS(p_new[i]) < p_cut)
	p_new[i] = (Real_t)(0.0) ;
      
      int elem = matElemlist[regionStart+i];
      if ( vnewc[elem] >= eosvmax ) /* impossible condition here? */
	p_new[i] = (Real_t)(0.0) ;

      if (p_new[i] < pmin)
	p_new[i]   = pmin ;
    });
  fut.wait();
		      
}

/******************************************/

static inline
void CalcEnergyForElems(struct MeshGPU *meshGPU, Index_t regionStart,
			Real_t pmin, 
                        Real_t p_cut,
			Real_t e_cut,
			Real_t q_cut,
			Real_t emin,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length)
{
  if (length <= 0) return;
  
  const Real_t sixth = Real_t(1.0) / Real_t(6.0);
  HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
  HCC_ARRAY_OBJECT(Real_t, work) = meshGPU->work;
  HCC_ARRAY_OBJECT(Real_t, vnewc) = meshGPU->vnewc;
  HCC_ARRAY_OBJECT(Real_t, delvc) = meshGPU->delvc;
  HCC_ARRAY_OBJECT(Real_t, e_new) = meshGPU->e_new;
  HCC_ARRAY_OBJECT(Real_t, p_new) = meshGPU->p_new;
  HCC_ARRAY_OBJECT(Real_t, q_new) = meshGPU->q_new;
  HCC_ARRAY_OBJECT(Real_t, bvc) = meshGPU->bvc;
  HCC_ARRAY_OBJECT(Real_t, pbvc) = meshGPU->pbvc;
  HCC_ARRAY_OBJECT(Real_t, e_old) = meshGPU->e_old;
  HCC_ARRAY_OBJECT(Real_t, p_old) = meshGPU->p_old;
  HCC_ARRAY_OBJECT(Real_t, q_old) = meshGPU->q_old;
  HCC_ARRAY_OBJECT(Real_t, compression) = meshGPU->compression;
  HCC_ARRAY_OBJECT(Real_t, compHalfStep) = meshGPU->compHalfStep;
  HCC_ARRAY_OBJECT(Real_t, pHalfStep) = meshGPU->pHalfStep;
  HCC_ARRAY_OBJECT(Real_t, qq) = meshGPU->qq_old;
  HCC_ARRAY_OBJECT(Real_t, ql) = meshGPU->ql_old;

  
  completion_future fut = parallel_for_each(extent<1>(length),
		     [=
		      HCC_ID(e_new)
		      HCC_ID(p_old)
		      HCC_ID(e_old)
		      HCC_ID(q_old)
		      HCC_ID(work)
		      HCC_ID(delvc)
                      ](index<1> idx) restrict(amp)
    {
      int i=idx[0];
      e_new[i] = e_old[i] - (Real_t)(0.5) * delvc[i] * (p_old[i] + q_old[i])
	+ (Real_t)(0.5) * work[i];
        
      if (e_new[i] < emin) {
	e_new[i] = emin ;
      }
    });
  fut.wait();
    CalcPressureForElems(meshGPU,
                         regionStart,
                         pHalfStep,compHalfStep,
                         pmin, p_cut, eosvmax, length);
    
   fut = parallel_for_each(extent<1>(length),
		     [=
		      HCC_ID(e_new)
		      HCC_ID(q_new)
		      HCC_ID(bvc)
		      HCC_ID(pbvc)
		      HCC_ID(p_old)
		      HCC_ID(q_old)
		      HCC_ID(compHalfStep)
		      HCC_ID(work)
		      HCC_ID(delvc)
		      HCC_ID(qq)
		      HCC_ID(ql)
                      HCC_ID(pHalfStep)](index<1> idx) restrict(amp)
   {
    int i=idx[0];
    Real_t vhalf = (Real_t)(1.) / ((Real_t)(1.) + compHalfStep[i]) ;

    if ( delvc[i] > (Real_t)(0.) ) {
      q_new[i] = (Real_t)(0.) ;
    }
    else {
      Real_t ssc = ( pbvc[i] * e_new[i] + vhalf *
		     vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;
      if ( ssc <= (Real_t)(0.) ) {
	ssc = (Real_t)(.333333e-36) ;
      } else {
	ssc = SQRT(ssc) ;
      }

      q_new[i] = (ssc*ql[i] + qq[i]) ;
    }

    e_new[i] = e_new[i] + (Real_t)(0.5) * delvc[i] 
      * (  (Real_t)(3.0)*(p_old[i]     + q_old[i])   
	   - (Real_t)(4.0)*( pHalfStep[i] + q_new[i] )) ;

    e_new[i] += (Real_t)(0.5) * work[i];

    if (FABS(e_new[i]) < e_cut) {
      e_new[i] = (Real_t)(0.)  ;
    }
    if (     e_new[i]  < emin ) {
      e_new[i] = emin ;
    }
   });
   fut.wait();
   
   CalcPressureForElems(meshGPU, regionStart,
                        p_new,compression,
                        pmin, p_cut, eosvmax, length);

   fut = parallel_for_each(extent<1>(length),
		     [=
		      HCC_ID(matElemlist)
		      HCC_ID(p_new)
		      HCC_ID(e_new)
		      HCC_ID(q_new)
		      HCC_ID(bvc)
		      HCC_ID(pbvc)
		      HCC_ID(p_old)
		      HCC_ID(q_old)
		      HCC_ID(vnewc)
		      HCC_ID(delvc)
		      HCC_ID(qq)
		      HCC_ID(ql)
                      HCC_ID(pHalfStep)](index<1> idx) restrict(amp)
	{

	  int i=idx[0]; 
	  Real_t q_tilde ;

	  if (delvc[i] > (Real_t)(0.)) {
	    q_tilde = (Real_t)(0.) ;
	  }
	  else {
	    Index_t elem = matElemlist[regionStart+i];
	    Real_t ssc = ( pbvc[i] * e_new[i]
			   + vnewc[elem] * vnewc[elem] *
			   bvc[i] * p_new[i] ) / rho0 ;

	    if ( ssc <= (Real_t)(0.) ) {
	      ssc = (Real_t)(.333333e-36) ;
	    } else {
	      ssc = SQRT(ssc) ;
	    }
	      
	    q_tilde = (ssc*ql[i] + qq[i]) ;
	  }

	  e_new[i] = e_new[i] - (  (Real_t)(7.0)*(p_old[i]     + q_old[i])
				   - (Real_t)(8.0)*(pHalfStep[i] + q_new[i])
				   + (p_new[i] + q_tilde)) * delvc[i]*sixth ;
	    
	  if (FABS(e_new[i]) < e_cut) {
	    e_new[i] = (Real_t)(0.)  ;
	  }
	  if ( e_new[i] < emin ) {
	    e_new[i] = emin ;
	  }
	});
   fut.wait();

  CalcPressureForElems(meshGPU, regionStart,
                        p_new,compression,
                        pmin, p_cut, eosvmax, length);

    fut = parallel_for_each(extent<1>(length),
		      [=
		       HCC_ID(matElemlist)
		       HCC_ID(p_new)
		       HCC_ID(e_new)
		       HCC_ID(q_new)
		       HCC_ID(bvc)
		       HCC_ID(pbvc)
		       HCC_ID(vnewc)
		       HCC_ID(delvc)
		       HCC_ID(qq)
		       HCC_ID(ql)](index<1> idx) restrict(amp)	      
	{
	  int i=idx[0]; 
	  if ( delvc[i] <= (Real_t)(0.) ) {
	    Index_t elem = matElemlist[regionStart+i];
	    Real_t ssc = ( pbvc[i] * e_new[i] +
			   vnewc[elem] * vnewc[elem] *
			   bvc[i] * p_new[i] ) / rho0 ;

	    if ( ssc <= (Real_t)(0.) ) {
	      ssc = (Real_t)(.333333e-36) ;
	    } else {
	      ssc = SQRT(ssc) ;
	    }

	    q_new[i] = (ssc*ql[i] + qq[i]) ;

	    if (FABS(q_new[i]) < q_cut) q_new[i] = (Real_t)(0.) ;
	  }
	});
    fut.wait();
   return ;
}

/******************************************/

static inline
void CalcSoundSpeedForElems(struct MeshGPU *meshGPU,
                            Index_t regionStart,
			    Real_t rho0,
			    Real_t ss4o3,
			    Index_t nz)
{
  if (nz <= 0) return;
  HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
  HCC_ARRAY_OBJECT(Real_t, ss) = meshGPU->ss;
  HCC_ARRAY_OBJECT(Real_t, vnewc) = meshGPU->vnewc;
  HCC_ARRAY_OBJECT(Real_t, enewc) = meshGPU->e_new;
  HCC_ARRAY_OBJECT(Real_t, pnewc) = meshGPU->p_new;
  HCC_ARRAY_OBJECT(Real_t, bvc) = meshGPU->bvc;
  HCC_ARRAY_OBJECT(Real_t, pbvc) = meshGPU->pbvc;
    completion_future fut = parallel_for_each(extent<1>(nz),
		      [=
		       HCC_ID(vnewc)
		       HCC_ID(enewc)
		       HCC_ID(pnewc)
		       HCC_ID(pbvc)
		       HCC_ID(bvc)
		       HCC_ID(matElemlist)
		       HCC_ID(ss)](index<1> idx) restrict(amp)
    {
      int i=idx[0];
      Index_t iz = matElemlist[regionStart+i];
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[iz] *
		      vnewc[iz] * bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= (Real_t)(.1111111e-36)) {
	ssTmp = (Real_t)(.3333333e-18);
      }
      ss[iz] = SQRT(ssTmp);
    });
    fut.wait();
}

/******************************************/

static inline
void EvalEOSForElems(Index_t regionStart, Domain& mesh,
		     struct MeshGPU *meshGPU,
		     Int_t numElemReg, Int_t rep)
{
  Real_t  e_cut = mesh.e_cut();
  Real_t  p_cut = mesh.p_cut();
  Real_t  ss4o3 = mesh.ss4o3();
  Real_t  q_cut = mesh.q_cut();

  Real_t eosvmax = mesh.eosvmax() ;
  Real_t eosvmin = mesh.eosvmin() ;
  Real_t pmin    = mesh.pmin() ;
  Real_t emin    = mesh.emin() ;
  Real_t rho0    = mesh.refdens() ;
  HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
  HCC_ARRAY_OBJECT(Real_t, vnewc) = meshGPU->vnewc;
  HCC_ARRAY_OBJECT(Real_t, e) = meshGPU->e;
  HCC_ARRAY_OBJECT(Real_t, delv) = meshGPU->delv;
  HCC_ARRAY_OBJECT(Real_t, p) = meshGPU->p;
  HCC_ARRAY_OBJECT(Real_t, q) = meshGPU->q;
  HCC_ARRAY_OBJECT(Real_t, qq_old) = meshGPU->qq_old;
  HCC_ARRAY_OBJECT(Real_t, ql_old) = meshGPU->ql_old;
  HCC_ARRAY_OBJECT(Real_t, delvc) = meshGPU->delvc;
  HCC_ARRAY_OBJECT(Real_t, work) = meshGPU->work;
  HCC_ARRAY_OBJECT(Real_t, e_new) = meshGPU->e_new;
  HCC_ARRAY_OBJECT(Real_t, p_new) = meshGPU->p_new;
  HCC_ARRAY_OBJECT(Real_t, q_new) = meshGPU->q_new;
  HCC_ARRAY_OBJECT(Real_t, e_old) = meshGPU->e_old;
  HCC_ARRAY_OBJECT(Real_t, p_old) = meshGPU->p_old;
  HCC_ARRAY_OBJECT(Real_t, q_old) = meshGPU->q_old;
  HCC_ARRAY_OBJECT(Real_t, compression) = meshGPU->compression;
  HCC_ARRAY_OBJECT(Real_t, compHalfStep) = meshGPU->compHalfStep;
  HCC_ARRAY_OBJECT(Real_t, qq) = meshGPU->qq;
  HCC_ARRAY_OBJECT(Real_t, ql) = meshGPU->ql;
  completion_future fut;
    
    //loop to add load imbalance based on region number 
  for(Int_t j = 0; j < rep; j++) {
    if (numElemReg > 0){
    fut = parallel_for_each(extent<1>(numElemReg),
			[=
			 HCC_ID(vnewc)
			 HCC_ID(matElemlist)
			 HCC_ID(e)
			 HCC_ID(delv)
			 HCC_ID(p)
			 HCC_ID(q)
			 HCC_ID(qq)
			 HCC_ID(ql)
			 HCC_ID(qq_old)
			 HCC_ID(ql_old)
			 HCC_ID(e_old)
			 HCC_ID(delvc)
			 HCC_ID(p_old)
			 HCC_ID(q_old)
			 HCC_ID(compression)
			 HCC_ID(compHalfStep)
			 HCC_ID(work)
                         ](index<1> idx) restrict(amp)
	    {

	      int i=idx[0];
	      Index_t zidx = matElemlist[regionStart+i];
	      e_old[i] = e[zidx];
	      delvc[i] = delv[zidx];
	      p_old[i] = p[zidx];
	      q_old[i] = q[zidx];
		
	      Real_t vchalf ;
	      compression[i] = (Real_t)(1.) / vnewc[zidx] - (Real_t)(1.);
	      vchalf = vnewc[zidx] - delvc[i] * (Real_t)(.5);
	      compHalfStep[i] = (Real_t)(1.) / vchalf - (Real_t)(1.);
		
	      if ( eosvmin != (Real_t)(0.) ) {
		if (vnewc[zidx] <= eosvmin) { 
		  compHalfStep[i] = compression[i] ;
		}
	      }
	      if ( eosvmax != (Real_t)(0.) ) {
		if (vnewc[zidx] >= eosvmax) { 
		  p_old[i]        = (Real_t)(0.) ;
		  compression[i]  = (Real_t)(0.) ;
		  compHalfStep[i] = (Real_t)(0.) ;
		}
	      }
		
	      qq_old[i] = qq[zidx] ;
	      ql_old[i] = ql[zidx] ;
	      work[i] = (Real_t)(0.) ; 
	    });
    fut.wait();
  }

      CalcEnergyForElems(meshGPU, regionStart, 
			 pmin,
			 p_cut,
			 e_cut,
			 q_cut,
			 emin,
			 rho0,
			 eosvmax,
			 numElemReg);
    }
    if (numElemReg > 0){
    fut = parallel_for_each(extent<1>(numElemReg),
			    [=
                             HCC_ID(p)
                             HCC_ID(e)
                             HCC_ID(q)
			     HCC_ID(p_new)
                             HCC_ID(e_new)
                             HCC_ID(q_new)
                             HCC_ID(matElemlist)]
		            (index<1> idx) restrict(amp)
	{
	      int i=idx[0];
	      Index_t zidx = matElemlist[regionStart+i] ;
	      p[zidx] = p_new[i];
	      e[zidx] = e_new[i];
	      q[zidx] = q_new[i];
	});
    fut.wait();
  }
    CalcSoundSpeedForElems(meshGPU, regionStart, 
			   rho0,
			   ss4o3, numElemReg) ;

}

/******************************************/

static inline
void ApplyMaterialPropertiesForElems(Domain& mesh, struct MeshGPU *meshGPU)
{
  Index_t length = mesh.numElem() ;
  Real_t eosvmin = mesh.eosvmin() ;
  Real_t eosvmax = mesh.eosvmax() ;
  HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
  HCC_ARRAY_OBJECT(Real_t, vnew) = meshGPU->vnew;
  HCC_ARRAY_OBJECT(Real_t, vnewc) = meshGPU->vnewc;

  completion_future fut;
  if (length > 0){
  fut = parallel_for_each(extent<1>(length),
		    [=
                     HCC_ID(matElemlist)
                     HCC_ID(vnew)
                     HCC_ID(vnewc)](index<1> idx) restrict(amp)
    {
      int i=idx[0];
      Index_t zn = matElemlist[i] ;
      vnewc[i] = vnew[i] ;

      if (eosvmin != (Real_t)(0.)) {
	if (vnewc[i] < eosvmin)
	  vnewc[i] = eosvmin ;
      }

      if (eosvmax != (Real_t)(0.)) {
	if (vnewc[i] > eosvmax)
	  vnewc[i] = eosvmax ;
      }
    });
  fut.wait();
  }
    
    //TODO: add this check to a kernel
    //#warning Fixme: volume error check needs to be added to a kernel
    /*
    for (Index_t i=0; i<length; ++i) {
       Index_t zn = mesh.matElemlist(i) ;
       Real_t vc = mesh.v(zn) ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin)
             vc = eosvmin ;
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax)
             vc = eosvmax ;
       }
       if (vc <= 0.) {
          exit(VolumeError) ;
       }
    }
    */
    
    for (Int_t r=0 ; r<mesh.numReg() ; r++) {
       Index_t numElemReg = mesh.regElemSize(r);
       Index_t regionStart = mesh.regStartPosition(r);
       Int_t rep;
       //Determine load imbalance for this region
       //round down the number with lowest cost
       if (r < mesh.numReg()/2)
           rep = 1;
       //you don't get an expensive region unless you at least have 5 regions
       else if (r < (mesh.numReg() - (mesh.numReg()+15)/20))
           rep = 1 + mesh.cost();
       //very expensive regions
       else
           rep = 10 * (1+ mesh.cost());
       if (numElemReg > 0) {
           EvalEOSForElems(regionStart, mesh, meshGPU, 
			   numElemReg, rep);

       }
    }
}

/******************************************/

static inline
void UpdateVolumesForElems(Domain &mesh, struct MeshGPU *meshGPU,
                           Real_t v_cut, Index_t length)
{
   HCC_ARRAY_OBJECT(Real_t, vnew) = meshGPU->vnew;
   HCC_ARRAY_OBJECT(Real_t, v) = meshGPU->v;
   if (length > 0) 
   {
     completion_future fut = parallel_for_each(extent<1>(length),
		       [=
                        HCC_ID(vnew)
                        HCC_ID(v)](index<1> idx) restrict(amp)
     {
       int i=idx[0];
         Real_t tmpV ;
         tmpV = vnew[i] ;

         if ( FABS(tmpV - (Real_t)(1.0)) < v_cut )
            tmpV = (Real_t)(1.0) ;
         v[i] = tmpV ;
     });
     fut.wait();
    
   }
}

/******************************************/

static inline
void LagrangeElements(Domain& mesh, struct MeshGPU *meshGPU, Index_t numElem)
{
  CalcLagrangeElements(mesh, meshGPU);

  // Not the problem
  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(mesh, meshGPU); 

  ApplyMaterialPropertiesForElems(mesh, meshGPU);

  UpdateVolumesForElems(mesh, meshGPU, mesh.v_cut(), numElem);

}

/******************************************/

static inline
void CalcCourantConstraintForElems(Domain &mesh, struct MeshGPU *meshGPU,
				   Index_t regionStart, Index_t length,
                                   Real_t qqc, Real_t& dtcourant)
{

    Real_t qqc2 = Real_t(64.0) * qqc * qqc ;

    HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
    HCC_ARRAY_OBJECT(Real_t, ss) = meshGPU->ss;
    HCC_ARRAY_OBJECT(Real_t, arealg) = meshGPU->arealg;
    HCC_ARRAY_OBJECT(Real_t, vdov) = meshGPU->vdov;
    HCC_ARRAY_OBJECT(Real_t, mindtcourant) = meshGPU->mindtcourant;

    size_t localThreads = BLOCKSIZE;
    size_t globalThreads = PAD(length, localThreads);
    const unsigned int numBlocks = globalThreads/localThreads;


    extent<1> lengthExt(globalThreads);
    tiled_extent<1> tlengthExt(lengthExt, BLOCKSIZE);
    completion_future fut = parallel_for_each(tlengthExt,
		      [=
		       HCC_ID(ss)
		       HCC_ID(vdov)
		       HCC_ID(arealg)
		       HCC_ID(mindtcourant)
		       HCC_ID(matElemlist)]
		      (tiled_index<1> t_idx) restrict(amp)
	{
	  tile_static Real_t minArray[BLOCKSIZE];
	  int i=t_idx.global[0];
	  Real_t dtcourant_l = (Real_t)(1.0e+20) ;
	  if (i<length) {
	    Index_t indx = matElemlist[regionStart+i] ;
	    Real_t dtf = ss[indx] * ss[indx] ;
	    if ( vdov[indx] < (Real_t)(0.) ) {
	      dtf = dtf
                + qqc2 * arealg[indx] * arealg[indx]
                * vdov[indx] * vdov[indx] ;
	    }
	    dtf = SQRT(dtf) ;
	    dtf = arealg[indx] / dtf ;

	    if (vdov[indx] != (Real_t)(0.)) {
	      if ( dtf < dtcourant_l ) {
                dtcourant_l = dtf ;
	      }
	    }
	  }
	  int tid = t_idx.local[0];
	  minArray[tid] = dtcourant_l;
	  
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 128) {
	    if (minArray[tid] > minArray[tid + 128])
	      minArray[tid] = minArray[tid + 128];
	      }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 64) {
	    if (minArray[tid] > minArray[tid + 64])
	      minArray[tid] = minArray[tid + 64];
	  }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 32) {
	    if (minArray[tid] > minArray[tid + 32])
	      minArray[tid] = minArray[tid + 32];
	  }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 16) {
	    if (minArray[tid] > minArray[tid + 16])
	      minArray[tid] = minArray[tid + 16];
	  }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 8) {
	    if (minArray[tid] > minArray[tid + 8])
	      minArray[tid] = minArray[tid + 8];
	  }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 4) {
	    if (minArray[tid] > minArray[tid + 4])
	      minArray[tid] = minArray[tid + 4];
	  }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid < 2) {
	    if (minArray[tid] > minArray[tid + 2])
	      minArray[tid] = minArray[tid + 2];
	  }
	  t_idx.barrier.wait_with_tile_static_memory_fence();
	  if (tid == 0) {
	    if (minArray[tid] > minArray[tid + 1])
	      minArray[tid] = minArray[tid + 1];
	    mindtcourant[t_idx.tile[0]] = minArray[tid];
	  }
	});
    fut.wait();
    HCC_SYNC(mindtcourant,mesh.dev_mindtcourant.data());

    // finish the MIN computation over the thread blocks
    for (unsigned int i=0; i<numBlocks; i++) {
      MINEQ(dtcourant,mesh.dev_mindtcourant[i]);
    }

    if (dtcourant < Real_t(1.0e+20))
        mesh.dtcourant() = dtcourant ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &mesh, struct MeshGPU *meshGPU, 
				 Index_t regionStart, Index_t length,
                                 Real_t dvovmax,
				 Real_t& dthydro)
{

    size_t localThreads = BLOCKSIZE;
    size_t globalThreads = PAD(length, localThreads);
    const unsigned int numBlocks = globalThreads/localThreads;
    HCC_ARRAY_OBJECT(Index_t, matElemlist) = meshGPU->matElemlist;
    HCC_ARRAY_OBJECT(Real_t, vdov) = meshGPU->vdov;
    HCC_ARRAY_OBJECT(Real_t, mindthydro) = meshGPU->mindthydro;

      
    extent<1> lengthExt(globalThreads);
    tiled_extent<1> tlengthExt(lengthExt, BLOCKSIZE);
    completion_future fut = parallel_for_each(tlengthExt,
                       [=
                        HCC_ID(matElemlist)
                        HCC_ID(vdov)
                        HCC_ID(mindthydro)](tiled_index<1> t_idx) restrict(amp){  
      tile_static Real_t minArray[BLOCKSIZE];
      
      int i = t_idx.global[0];
      Real_t dthydro_l = (Real_t)(1.0e+20) ;
      if (i<length) {
	Index_t indx = matElemlist[regionStart+i] ;
	if (vdov[indx] != (Real_t)(0.)) {
	  Real_t dtdvov = dvovmax / (FABS(vdov[indx])+(Real_t)(1.e-20)) ;
	  if ( dthydro_l > dtdvov ) {
            dthydro_l = dtdvov;
	  }
	}
      }
      
      int tid = t_idx.local[0];
      minArray[tid] = dthydro_l;

      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 128) {
	if (minArray[tid] > minArray[tid + 128])
	  minArray[tid] = minArray[tid + 128];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 64) {
	if (minArray[tid] > minArray[tid + 64])
	  minArray[tid] = minArray[tid + 64];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 32) {
	if (minArray[tid] > minArray[tid + 32])
	  minArray[tid] = minArray[tid + 32];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 16) {
	if (minArray[tid] > minArray[tid + 16])
	  minArray[tid] = minArray[tid + 16];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 8) {
	if (minArray[tid] > minArray[tid + 8])
	  minArray[tid] = minArray[tid + 8];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 4) {
	if (minArray[tid] > minArray[tid + 4])
	  minArray[tid] = minArray[tid + 4];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid < 2) {
	if (minArray[tid] > minArray[tid + 2])
	  minArray[tid] = minArray[tid + 2];
      }
      t_idx.barrier.wait_with_tile_static_memory_fence();
      if (tid == 0) {
	if (minArray[tid] > minArray[tid + 1])
	  minArray[tid] = minArray[tid + 1];
	mindthydro[t_idx.tile[0]]=minArray[0];
      }
      
    });
    fut.wait();
    HCC_SYNC(mindthydro, mesh.dev_mindthydro.data());

    // finish the MIN computation over the thread blocks
    for (unsigned int i=0; i<numBlocks; i++) {
      MINEQ(dthydro,mesh.dev_mindthydro[i]);
    }
    
    if (dthydro < Real_t(1.0e+20))
        mesh.dthydro() = dthydro ;

    return;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems(Domain& domain,
				 struct MeshGPU *meshGPU) {
   domain.dtcourant() = 1.0e+20;
   domain.dthydro() = 1.0e+20;
   
   for (Index_t r=0 ; r < domain.numReg() ; ++r) {
      Index_t regionStart = domain.regStartPosition(r);
      Index_t numElemReg = domain.regElemSize(r);

      if (numElemReg > 0) {

	CalcCourantConstraintForElems(domain, meshGPU,
				      regionStart,
				      numElemReg,
				      domain.qqc(),
				      domain.dtcourant());

	CalcHydroConstraintForElems(domain, meshGPU,
				    regionStart,
				    numElemReg,
				    domain.dvovmax(),
				    domain.dthydro());
      }
   }
}

/******************************************/

static inline
  void LagrangeLeapFrog(Domain& mesh, struct MeshGPU *meshGPU)
{
#ifdef SEDOV_SYNC_POS_VEL_LATE
   Domain_member fieldData[6] ;
#endif


   const Real_t delt = mesh.deltatime() ;
   Real_t u_cut = mesh.u_cut() ;

   Index_t numNode = mesh.numNode() ;
   Index_t numElem = mesh.numElem();
   Index_t numNodeBC = (mesh.sizeX()+1)*(mesh.sizeX()+1) ;
    
   if (numElem != 0) {
      
     LagrangeNodal(mesh, meshGPU);

     LagrangeElements(mesh, meshGPU, numElem);

     CalcTimeConstraintsForElems(mesh, meshGPU);
   }

}

/******************************************/

int main(int argc, char *argv[])
{
  Domain *locDom;
//  struct MeshGPU *meshGPU;
   Int_t numRanks ;
   Int_t myRank ;
   struct cmdLineOpts opts;

   numRanks = 1;
   myRank = 0;

   /* Set defaults that can be overridden by command line opts */
   opts.its = 9999999;
   opts.nx  = 30;
   opts.numReg = 11;
   opts.numFiles = (int)(numRanks+10)/9;
   opts.showProg = 0;
   opts.quiet = 0;
   opts.viz = 0;
   opts.balance = 1;
   opts.cost = 1;

   ParseCommandLineOptions(argc, argv, myRank, &opts);

   if ((myRank == 0) && (opts.quiet == 0)) {
      printf("Running problem size %d^3 per domain until completion\n", opts.nx);
      printf("Num processors: %d\n", numRanks);
      printf("Total number of elements: %lld\n\n", (long long int)numRanks*opts.nx*opts.nx*opts.nx);
      printf("To run other sizes, use -s <integer>.\n");
      printf("To run a fixed number of iterations, use -i <integer>.\n");
      printf("To run a more or less balanced region set, use -b <integer>.\n");
      printf("To change the relative costs of regions, use -c <integer>.\n");
      printf("To print out progress, use -p\n");
      printf("To write an output file for VisIt, use -v\n");
      printf("See help (-h) for more options\n\n");
   }

   // Set up the mesh and decompose. Assumes regular cubes for now
   Int_t col, row, plane, side;
   InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

   // Build the main data structure and initialize it
   locDom = new Domain(numRanks, col, row, plane, opts.nx,
                       side, opts.numReg, opts.balance, opts.cost) ;

   locDom->AllocateNodeElemIndexes();
   /* Create a material IndexSet (entire mesh same material for now) */
   
   // BEGIN timestep to solution //
   Real_t start;
   Real_t elapsedTime;
   start = getTime();


   const Real_t delt = locDom->deltatime() ;
   Real_t u_cut = locDom->u_cut() ;

   Index_t numNode = locDom->numNode() ;
   Index_t numElem = locDom->numElem();
   Index_t numNodeBC = (locDom->sizeX()+1)*(locDom->sizeX()+1) ;

  HCC_ARRAY_STRUC(Index_t, matElemlist_av,
		  locDom->m_matElemlist.size(),
		  locDom->m_matElemlist.data());
  HCC_ARRAY_STRUC(Real_t, ss_av,
		  locDom->m_ss.size(),
		  locDom->m_ss.data());
  HCC_ARRAY_STRUC(Real_t, vdov_av,
		  locDom->m_vdov.size(),
		  locDom->m_vdov.data());
  HCC_ARRAY_STRUC(Real_t, arealg_av,
		  locDom->m_arealg.size(),
		  locDom->m_arealg.data());
  HCC_ARRAY_STRUC(Index_t, nodelist_av,
		  locDom->m_nodelist.size(),
		  locDom->m_nodelist.data());
  HCC_ARRAY_STRUC(Real_t, x_av,
		  locDom->m_x.size(),
		  locDom->m_x.data());
  HCC_ARRAY_STRUC(Real_t, y_av,
		  locDom->m_y.size(),
		  locDom->m_y.data());
  HCC_ARRAY_STRUC(Real_t, z_av,
		  locDom->m_z.size(),
		  locDom->m_z.data());
  HCC_ARRAY_STRUC(Real_t, xd_av,
		  locDom->m_xd.size(),
		  locDom->m_xd.data());
  HCC_ARRAY_STRUC(Real_t, yd_av,
		  locDom->m_yd.size(),
		  locDom->m_yd.data());
  HCC_ARRAY_STRUC(Real_t, zd_av,
		  locDom->m_zd.size(),
		  locDom->m_zd.data());
  HCC_ARRAY_STRUC(Real_t, fx_av,
		  locDom->m_fx.size(),
		  locDom->m_fx.data());
  HCC_ARRAY_STRUC(Real_t, fy_av,
		  locDom->m_fy.size(),
		  locDom->m_fy.data());
  HCC_ARRAY_STRUC(Real_t, fz_av,
		  locDom->m_fz.size(),
		  locDom->m_fz.data());
  HCC_ARRAY_STRUC(Real_t, elemMass_av,
		  locDom->m_elemMass.size(),
		  locDom->m_elemMass.data());
  HCC_ARRAY_STRUC(Int_t, nodeElemCount_av,
		  locDom->m_nodeElemCount.size(),
		  locDom->m_nodeElemCount.data());
  HCC_ARRAY_STRUC(Index_t, nodeElemCornerList_av,
		  locDom->m_nodeElemCornerList.size(),
		  locDom->m_nodeElemCornerList.data());
  HCC_ARRAY_STRUC(Real_t, volo_av,
		  locDom->m_volo.size(),
		  locDom->m_volo.data());
  HCC_ARRAY_STRUC(Real_t, v_av,
		  locDom->m_v.size(),
		  locDom->m_v.data());
  HCC_ARRAY_STRUC(Real_t, xdd_av,
		  locDom->m_xdd.size(),
		  locDom->m_xdd.data());
  HCC_ARRAY_STRUC(Real_t, ydd_av,
		  locDom->m_ydd.size(),
		  locDom->m_ydd.data());
  HCC_ARRAY_STRUC(Real_t, zdd_av,
		  locDom->m_zdd.size(),
		  locDom->m_zdd.data());
  HCC_ARRAY_STRUC(Real_t, nodalMass_av,
		  locDom->m_nodalMass.size(),
		  locDom->m_nodalMass.data());
  HCC_ARRAY_STRUC(Index_t, symmX_av,
		  locDom->m_symmX.size(),
		  locDom->m_symmX.data());
  HCC_ARRAY_STRUC(Index_t, symmY_av,
		  locDom->m_symmY.size(),
		  locDom->m_symmY.data());
  HCC_ARRAY_STRUC(Index_t, symmZ_av,
		  locDom->m_symmZ.size(),
		  locDom->m_symmZ.data());
  HCC_ARRAY_STRUC(Real_t, vnew_av,
		  locDom->m_vnew.size(),
		  locDom->m_vnew.data());
  HCC_ARRAY_STRUC(Real_t, delv_av,
		  locDom->m_delv.size(),
		  locDom->m_delv.data());
  HCC_ARRAY_STRUC(Real_t, dxx_av,
		  locDom->m_dxx.size(),
		  locDom->m_dxx.data());	     
  HCC_ARRAY_STRUC(Real_t, dyy_av,
		  locDom->m_dyy.size(),
		  locDom->m_dyy.data());
  HCC_ARRAY_STRUC(Real_t, dzz_av,
		  locDom->m_dzz.size(),
		  locDom->m_dzz.data());
  HCC_ARRAY_STRUC(Real_t, delx_zeta_av,
		  locDom->m_delx_zeta.size(),
		  locDom->m_delx_zeta.data());
  HCC_ARRAY_STRUC(Real_t, delv_zeta_av,
		  locDom->m_delv_zeta.size(),
		  locDom->m_delv_zeta.data());
  HCC_ARRAY_STRUC(Real_t, delx_xi_av,
		  locDom->m_delx_xi.size(),
		  locDom->m_delx_xi.data());
  HCC_ARRAY_STRUC(Real_t, delv_xi_av,
		  locDom->m_delv_xi.size(),
		  locDom->m_delv_xi.data());
  HCC_ARRAY_STRUC(Real_t, delx_eta_av,
		  locDom->m_delx_eta.size(),
		  locDom->m_delx_eta.data());
  HCC_ARRAY_STRUC(Real_t, delv_eta_av,
		  locDom->m_delv_eta.size(),
		  locDom->m_delv_eta.data());
  HCC_ARRAY_STRUC(Index_t, elemBC_av,
		  locDom->m_elemBC.size(),
		  locDom->m_elemBC.data());
  HCC_ARRAY_STRUC(Index_t, lxim_av,
		  locDom->m_lxim.size(),
		  locDom->m_lxim.data());
  HCC_ARRAY_STRUC(Index_t, lxip_av,
		  locDom->m_lxip.size(),
		  locDom->m_lxip.data());
  HCC_ARRAY_STRUC(Index_t, letam_av,
		  locDom->m_letam.size(),
		  locDom->m_letam.data());
  HCC_ARRAY_STRUC(Index_t, letap_av,
		  locDom->m_letap.size(),
		  locDom->m_letap.data());
  HCC_ARRAY_STRUC(Index_t, lzetam_av,
		  locDom->m_lzetam.size(),
		  locDom->m_lzetam.data());
  HCC_ARRAY_STRUC(Index_t, lzetap_av,
		  locDom->m_lzetap.size(),
		  locDom->m_lzetap.data());
  HCC_ARRAY_STRUC(Real_t, qq_av,
		  locDom->m_qq.size(),
		  locDom->m_qq.data());
  HCC_ARRAY_STRUC(Real_t, ql_av,
		  locDom->m_ql.size(),
		  locDom->m_ql.data());
  HCC_ARRAY_STRUC(Real_t, e_av,
		  locDom->m_e.size(),
		  locDom->m_e.data());
  HCC_ARRAY_STRUC(Real_t, p_av,
		  locDom->m_p.size(),
		  locDom->m_p.data());
  HCC_ARRAY_STRUC(Real_t, q_av,
		  locDom->m_q.size(),
		  locDom->m_q.data());
//routine temps
  HCC_ARRAY_STRUC(Real_t, vnewc_av,
		  locDom->p_vnewc.size(),
		  locDom->p_vnewc.data());
  HCC_ARRAY_STRUC(Real_t, mindthydro, 
                  locDom->dev_mindthydro.size(), 
                  locDom->dev_mindthydro.data());

  HCC_ARRAY_STRUC(Real_t, mindtcourant, 
                  locDom->dev_mindtcourant.size(), 
                  locDom->dev_mindtcourant.data());
  HCC_ARRAY_STRUC(Real_t, e_old, numElem, locDom->p_e_old.data());
  HCC_ARRAY_STRUC(Real_t, delvc, numElem, locDom->p_delvc.data());
  HCC_ARRAY_STRUC(Real_t, p_old, numElem, locDom->p_p_old.data());
  HCC_ARRAY_STRUC(Real_t, q_old, numElem, locDom->p_q_old.data());
  HCC_ARRAY_STRUC(Real_t, compression, numElem, locDom->p_compression.data());
  HCC_ARRAY_STRUC(Real_t, compHalfStep, numElem, locDom->p_compHalfStep.data());
  HCC_ARRAY_STRUC(Real_t, qq_old, numElem, locDom->p_qq_old.data());
  HCC_ARRAY_STRUC(Real_t, ql_old, numElem, locDom->p_ql_old.data());
  HCC_ARRAY_STRUC(Real_t, work, numElem, locDom->p_work.data());
  HCC_ARRAY_STRUC(Real_t, p_new, numElem, locDom->p_p_new.data());
  HCC_ARRAY_STRUC(Real_t, e_new, numElem, locDom->p_e_new.data());
  HCC_ARRAY_STRUC(Real_t, q_new, numElem, locDom->p_q_new.data());
  HCC_ARRAY_STRUC(Real_t, bvc, numElem, locDom->p_bvc.data());
  HCC_ARRAY_STRUC(Real_t, pbvc, numElem, locDom->p_pbvc.data());
  HCC_ARRAY_STRUC(Real_t, pHalfStep, numElem, locDom->ppHalfStep.data());
  HCC_ARRAY_STRUC(Real_t, sigxx, numElem, locDom->p_sigxx.data());
  HCC_ARRAY_STRUC(Real_t, sigyy, numElem, locDom->p_sigyy.data());
  HCC_ARRAY_STRUC(Real_t, sigzz, numElem, locDom->p_sigzz.data());
  HCC_ARRAY_STRUC(Real_t, determ, numElem, locDom->p_determ.data());
  HCC_ARRAY_STRUC(Real_t, dvdx, numElem*8, locDom->p_dvdx.data());
  HCC_ARRAY_STRUC(Real_t, dvdy, numElem*8, locDom->p_dvdy.data());
  HCC_ARRAY_STRUC(Real_t, dvdz, numElem*8, locDom->p_dvdz.data());
  HCC_ARRAY_STRUC(Real_t, x8n, numElem*8, locDom->p_x8n.data());
  HCC_ARRAY_STRUC(Real_t, y8n, numElem*8, locDom->p_y8n.data());
  HCC_ARRAY_STRUC(Real_t, z8n, numElem*8, locDom->p_z8n.data());
  HCC_ARRAY_STRUC(Real_t, fx_elem, numElem*8, locDom->p_fx_elem.data());
  HCC_ARRAY_STRUC(Real_t, fy_elem, numElem*8, locDom->p_fy_elem.data());
  HCC_ARRAY_STRUC(Real_t, fz_elem, numElem*8, locDom->p_fz_elem.data());

  struct MeshGPU meshGPU = MeshGPU(matElemlist_av,
                           ss_av, arealg_av,
                           vdov_av,
                           nodelist_av,
                           x_av,y_av,z_av,
                           xd_av,yd_av,zd_av,
                           fx_av,fy_av,fz_av,
                           elemMass_av,nodeElemCount_av,nodeElemCornerList_av,
                           v_av, volo_av, vnew_av, vnewc_av,
                           xdd_av,ydd_av,zdd_av,nodalMass_av,
                           symmX_av,symmY_av,symmZ_av,delv_av,
                           dxx_av,dyy_av,dzz_av,
                           
                           delx_zeta_av,delv_zeta_av,delx_xi_av,delv_xi_av,
                           delx_eta_av,delv_eta_av,elemBC_av,
                           lxim_av,lxip_av,letam_av,
                           letap_av,lzetam_av,lzetap_av,
                           qq_av,ql_av,e_av,p_av,q_av,
                           e_old,p_old,q_old,delvc,
                           compression,compHalfStep,
                           qq_old,ql_old,work,
                           p_new,e_new,q_new,
                           bvc,pbvc,
                           pHalfStep,
                           sigxx,sigyy,sigzz,
                           determ,
                           dvdx,dvdy,dvdz,
                           x8n,y8n,z8n,
                           fx_elem,fy_elem,fz_elem,
                           mindthydro,
                           mindtcourant);


   while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {

      TimeIncrement(*locDom) ;
      LagrangeLeapFrog(*locDom, &meshGPU);

      if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
         printf("cycle = %d, time = %e, dt=%e\n",
                locDom->cycle(),
		double(locDom->time()),
		double(locDom->deltatime()) ) ;
      }
   }

   HCC_SYNC(x_av,locDom->m_x.data());
   HCC_SYNC(e_av,locDom->m_e.data());
   // Use reduced max elapsed time
   elapsedTime = (getTime() - start);
   double elapsedTimeG = elapsedTime;
/*

   std::fstream file;
   file.open("x.asc", std::fstream::out);
   if (file.is_open()) {
       for (Index_t i=0; i<locDom->numElem(); i++)
           file << locDom->x(i) << std::endl;
       file.close();
   }

   file.open("e.asc", std::fstream::out);
   if (file.is_open()) {
       for (Index_t i=0; i<locDom->numElem(); i++)
           file << locDom->e(i) << std::endl;
       file.close();
   }
*/
   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(elapsedTimeG, *locDom, opts.nx, numRanks);
   }
   return 0 ;
}
