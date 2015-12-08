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
#include <math.h>
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

struct MeshGPU meshGPU;
int BLOCKSIZE;

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

#ifdef KERNELS_TIMING
typedef struct KernelStats_t {
    std::uint64_t time;
    std::uint64_t numCalls;
} KernelStats;
std::map<std::string, KernelStats> totalTime;
#endif

/*
 * getEventTime - gets time between startPoint and endPoint
 *                  for a particular OpenCL event
 * Arguments:
 * event      - cl_event, OpenCL event
 * startPoint - as endPoint
 * endPoint   - must be one of the following
 *      CL_PROFILING_COMMAND_QUEUED
 *      CL_PROFILING_COMMAND_SUBMIT
 *      CL_PROFILING_COMMAND_START
 *      CL_PROFILING_COMMAND_END
 *
 * Return value: time in nanoseconds
 */

cl_ulong getEventTime(cl_event& event, unsigned int startPoint, unsigned int endPoint) {
    if ((startPoint == CL_PROFILING_COMMAND_QUEUED || startPoint == CL_PROFILING_COMMAND_SUBMIT || startPoint == CL_PROFILING_COMMAND_START || startPoint == CL_PROFILING_COMMAND_END) &&
        (endPoint == CL_PROFILING_COMMAND_QUEUED || endPoint == CL_PROFILING_COMMAND_SUBMIT || endPoint == CL_PROFILING_COMMAND_START || endPoint == CL_PROFILING_COMMAND_END) &&
        startPoint < endPoint) {

        cl_ulong startTime, endTime;
        clGetEventProfilingInfo(event, startPoint, sizeof(cl_ulong), &startTime, NULL);
        clGetEventProfilingInfo(event, endPoint, sizeof(cl_ulong), &endTime, NULL);
        return endTime - startTime;
    }
    return 0;
}

inline void addArgs(cl_kernel &kernel, int i) {}

template<typename T, typename... Args>
inline void addArgs(cl_kernel &kernel, int i, const T &arg, const Args& ...restOfArgs) {
    CLsetup::err |= clSetKernelArg(kernel, i, sizeof(arg), &arg);
    CLsetup::checkErr(CLsetup::err, "clSetKernelArg()");

    addArgs(kernel, i+1, restOfArgs...);
}

template<typename... Args>
void callOpenCLKernel(const cl_program& program, const char* name, size_t numLocalThreads, size_t numGlobalThreads, Args... args) {
    cl_kernel kernel;
    const std::string temp_str(name);
    if(CLsetup::kernels.find(temp_str) == CLsetup::kernels.end())
    {
        kernel = clCreateKernel(program, name, &CLsetup::err);
        CLsetup::checkErr(CLsetup::err, "clCreateKernel()");
        CLsetup::kernels[temp_str] = kernel;
    }
    else
        kernel = CLsetup::kernels[temp_str];

    const size_t global_work_size[3] = {numGlobalThreads, 0, 0};
    const size_t local_work_size[3] = {numLocalThreads, 0, 0};

    addArgs(kernel, 0, args...);

#ifdef KERNELS_TIMING
    cl_event eventForTiming;
#endif
    CLsetup::err = clEnqueueNDRangeKernel(
            CLsetup::queue,       // cl_command_queue command_queue
            kernel,               // cl_kernel kernel
            1,                    // cl_uint work_dim
            NULL,                 // const size_t *global_work_offset
            global_work_size,     // const size_t *global_work_size
            local_work_size,      // const size_t *local_work_size
            0,                    // cl_uint num_events_in_wait_list
            NULL,                 // const cl_event *event_wait_list
#ifdef KERNELS_TIMING
            &eventForTiming);     // cl_event *event
#else
            NULL);
#endif
    CLsetup::checkErr(CLsetup::err, (std::string("Queue::enqueueNDRangeKernel() ") + std::string(name)).c_str());
    clFlush(CLsetup::queue);

#ifdef KERNELS_TIMING
    clFinish(CLsetup::queue);
    KernelStats& st = totalTime[std::string(name)];
    st.time += getEventTime(eventForTiming, CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END);
    st.numCalls ++;
#endif
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
void CollectDomainNodesToElemNodes(Domain &domain,
                                   const Index_t* elemToNode,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8])
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = domain.x(nd0i);
   elemX[1] = domain.x(nd1i);
   elemX[2] = domain.x(nd2i);
   elemX[3] = domain.x(nd3i);
   elemX[4] = domain.x(nd4i);
   elemX[5] = domain.x(nd5i);
   elemX[6] = domain.x(nd6i);
   elemX[7] = domain.x(nd7i);

   elemY[0] = domain.y(nd0i);
   elemY[1] = domain.y(nd1i);
   elemY[2] = domain.y(nd2i);
   elemY[3] = domain.y(nd3i);
   elemY[4] = domain.y(nd4i);
   elemY[5] = domain.y(nd5i);
   elemY[6] = domain.y(nd6i);
   elemY[7] = domain.y(nd7i);

   elemZ[0] = domain.z(nd0i);
   elemZ[1] = domain.z(nd1i);
   elemZ[2] = domain.z(nd2i);
   elemZ[3] = domain.z(nd3i);
   elemZ[4] = domain.z(nd4i);
   elemZ[5] = domain.z(nd5i);
   elemZ[6] = domain.z(nd6i);
   elemZ[7] = domain.z(nd7i);

}

/******************************************/

/*
static inline
void InitStressTermsForElems(Domain &domain,
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                             Index_t numElem)
*/
static inline
void InitStressTermsForElems(cl_mem sigxx, cl_mem sigyy, cl_mem sigzz,
                             Index_t numElem)
{
    callOpenCLKernel(CLsetup::program, "InitStressTermsForElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
            numElem, sigxx, sigyy, sigzz, meshGPU.m_p, meshGPU.m_q);
}

/******************************************/

static inline
void IntegrateStressForElems(Domain &mesh, 
                                  cl_mem sigxx, cl_mem sigyy, cl_mem sigzz,
                                  cl_mem determ, int& badvol, Index_t numElem)
{
    cl_mem fx_elem = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE, 
            numElem*8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at fx_elem");
    cl_mem fy_elem = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE, 
            numElem*8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at fy_elem");
    cl_mem fz_elem = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE, 
            numElem*8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at fz_elem");

    callOpenCLKernel(CLsetup::program, "IntegrateStressForElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    numElem, meshGPU.m_nodelist, meshGPU.m_x, meshGPU.m_y, meshGPU.m_z, fx_elem, fy_elem, fz_elem, sigxx, sigyy, sigzz, determ);

    // TODO: change work group size
    /*
    size_t size;
    CLsetup::err = kernel.getWorkGroupInfo<size_t>(CLsetup::device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &size);
    size_t preferred;
    CLsetup::err = kernel.getWorkGroupInfo<size_t>(CLsetup::device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferred);
    cout << setw(80) << "size " << size << endl;
    cout << setw(80) << "preferred " << preferred << endl;
    */

    callOpenCLKernel(CLsetup::program, "AddNodeForcesFromElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    mesh.numNode(), meshGPU.m_nodeElemCount, meshGPU.m_nodeElemCornerList, fx_elem, fy_elem, fz_elem, meshGPU.m_fx, meshGPU.m_fy, meshGPU.m_fz);

    // JDC -- need a reduction step to check for non-positive element volumes
    badvol=0; 

    clReleaseMemObject(fx_elem);
    clReleaseMemObject(fy_elem);
    clReleaseMemObject(fz_elem);
}

/******************************************/

static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

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

/******************************************/

static inline
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
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

/******************************************/

static inline
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Real_t hxx[4];
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

/*
static inline
void CalcFBHourglassForceForElems( Domain &domain,
                                   Real_t *determ,
                                   Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                   Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                   Real_t hourg, Index_t numElem,
                                   Index_t numNode)
*/
static inline
void CalcFBHourglassForceForElems(Domain &mesh, 
            cl_mem determ,
            cl_mem x8n,      cl_mem y8n,      cl_mem z8n,
            cl_mem dvdx,     cl_mem dvdy,     cl_mem dvdz,
            Real_t hourg)
{
    Index_t numElem = mesh.numElem();
    Index_t numNode = mesh.numNode();

    cl_mem fx_elem = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem*8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at fx_elem");
    cl_mem fy_elem = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem*8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at fy_elem");
    cl_mem fz_elem = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem*8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at fz_elem");

    callOpenCLKernel(CLsetup::program, "CalcFBHourglassForceForElems_kernel", 64, PAD(numElem, 64), 
                    determ, x8n, y8n, z8n, dvdx, dvdy, dvdz, hourg, numElem, meshGPU.m_nodelist,
                    meshGPU.m_ss, meshGPU.m_elemMass, meshGPU.m_xd, meshGPU.m_yd, meshGPU.m_zd, fx_elem, fy_elem, fz_elem);

    callOpenCLKernel(CLsetup::program, "AddNodeForcesFromElems2_kernel", 64, PAD(numNode, 64), 
                    mesh.numNode(), meshGPU.m_nodeElemCount, meshGPU.m_nodeElemCornerList,
                    fx_elem, fy_elem, fz_elem, meshGPU.m_fx, meshGPU.m_fy, meshGPU.m_fz);
    clReleaseMemObject(fx_elem);
    clReleaseMemObject(fy_elem);
    clReleaseMemObject(fz_elem);
}

/******************************************/

/*
static inline
void CalcHourglassControlForElems(Domain& domain,
                                  Real_t determ[], Real_t hgcoef)
*/
static inline
void CalcHourglassControlForElems(Domain& mesh, cl_mem determ, Real_t hgcoef)
{
   Index_t numElem = mesh.numElem() ;
   Index_t numElem8 = numElem * 8 ;

    cl_mem dvdx = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at dvdx");
    cl_mem dvdy = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at dvdy");
    cl_mem dvdz = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at dvdz");
    cl_mem x8n = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at x8n");
    cl_mem y8n = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at y8n");
    cl_mem z8n = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            numElem8*sizeof(Real_t),
            NULL,
            &CLsetup::err);
    CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at z8n");

    callOpenCLKernel(CLsetup::program, "CalcHourglassControlForElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    numElem, meshGPU.m_nodelist, meshGPU.m_x, meshGPU.m_y, meshGPU.m_z,
                    determ, meshGPU.m_volo, meshGPU.m_v, dvdx, dvdy, dvdz, x8n, y8n, z8n);
    // JDC -- need a reduction to check for negative volumes

   if ( hgcoef > Real_t(0.) ) {
       CalcFBHourglassForceForElems(mesh, determ, x8n, y8n, z8n, dvdx, dvdy,dvdz,hgcoef) ;
   }
    clReleaseMemObject(dvdx);
    clReleaseMemObject(dvdy);
    clReleaseMemObject(dvdz);
    clReleaseMemObject(x8n);
    clReleaseMemObject(y8n);
    clReleaseMemObject(z8n);
}

/******************************************/

/*
static inline
void CalcVolumeForceForElems(Domain& domain)
*/
static inline
void CalcVolumeForceForElems(Domain& mesh)
{
   Index_t numElem = mesh.numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = mesh.hgcoef() ;
      int badvol;
      
      cl_mem sigxx = clCreateBuffer(
              CLsetup::context,
              CL_MEM_READ_WRITE,
              numElem*sizeof(Real_t),
              NULL,
              &CLsetup::err);
      CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at sigxx");

      cl_mem sigyy = clCreateBuffer(
              CLsetup::context,
              CL_MEM_READ_WRITE,
              numElem*sizeof(Real_t),
              NULL,
              &CLsetup::err);
      CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at sigyy");

      cl_mem sigzz = clCreateBuffer(
              CLsetup::context,
              CL_MEM_READ_WRITE,
              numElem*sizeof(Real_t),
              NULL,
              &CLsetup::err);
      CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at sigzz");

      cl_mem determ = clCreateBuffer(
              CLsetup::context,
              CL_MEM_READ_WRITE,
              numElem*sizeof(Real_t),
              NULL,
              &CLsetup::err);
      CLsetup::checkErr(CLsetup::err, "Buffer::Buffer(), at determ");

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(sigxx, sigyy, sigzz, numElem);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems(mesh, sigxx, sigyy, sigzz, determ, badvol, numElem);

      // check for negative element volume
      if (badvol) exit(VolumeError) ;

      CalcHourglassControlForElems(mesh, determ, hgcoef) ;

      clReleaseMemObject(sigxx);
      clReleaseMemObject(sigyy);
      clReleaseMemObject(sigzz);
      clReleaseMemObject(determ);

   }
}

/******************************************/

/*
static inline void CalcForceForNodes(Domain& domain)
*/
static inline void CalcForceForNodes(Domain& domain)
{
  Index_t numNode = domain.numNode() ;

  for (Index_t i=0; i<numNode; ++i) {
     domain.fx(i) = Real_t(0.0) ;
     domain.fy(i) = Real_t(0.0) ;
     domain.fz(i) = Real_t(0.0) ;
  }

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;

}

/******************************************/

/*
static inline
void CalcAccelerationForNodes(Domain &domain, Index_t numNode)
*/
static inline
void CalcAccelerationForNodes(Domain &mesh, Index_t numNode)
{
    callOpenCLKernel(CLsetup::program, "CalcAccelerationForNodes_kernel", BLOCKSIZE, PAD(mesh.numNode(), BLOCKSIZE), 
                    mesh.numNode(), meshGPU.m_xdd, meshGPU.m_ydd, meshGPU.m_zdd,
                    meshGPU.m_fx, meshGPU.m_fy, meshGPU.m_fz, meshGPU.m_nodalMass);
}

/******************************************/

/*
static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain& domain)
*/
static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain& mesh)
{
    Index_t numNodeBC = (mesh.sizeX()+1)*(mesh.sizeX()+1) ;
    callOpenCLKernel(CLsetup::program, "ApplyAccelerationBoundaryConditionsForNodes_kernel", BLOCKSIZE, PAD(numNodeBC, BLOCKSIZE), 
                    numNodeBC, meshGPU.m_xdd, meshGPU.m_ydd, meshGPU.m_zdd,
                    meshGPU.m_symmX, meshGPU.m_symmY, meshGPU.m_symmZ);
}

/******************************************/

/*
static inline
void CalcVelocityForNodes(Domain &domain, const Real_t dt, const Real_t u_cut,
                          Index_t numNode)
*/
static inline
void CalcVelocityForNodes(Domain &mesh, const Real_t dt, const Real_t u_cut, Index_t numNode)
{
    callOpenCLKernel(CLsetup::program, "CalcVelocityForNodes_kernel", BLOCKSIZE, PAD(mesh.numNode(), BLOCKSIZE), 
                    mesh.numNode(), dt, u_cut, meshGPU.m_xd, meshGPU.m_yd, meshGPU.m_zd,
                    meshGPU.m_xdd, meshGPU.m_ydd, meshGPU.m_zdd);
}

/******************************************/

/*
static inline
void CalcPositionForNodes(Domain &domain, const Real_t dt, Index_t numNode)
*/
static inline
void CalcPositionForNodes(Domain &mesh, const Real_t dt, Index_t numNode)
{
    callOpenCLKernel(CLsetup::program, "CalcPositionForNodes_kernel", BLOCKSIZE, PAD(mesh.numNode(), BLOCKSIZE), 
                    mesh.numNode(), dt, meshGPU.m_x, meshGPU.m_y, meshGPU.m_z, meshGPU.m_xd, meshGPU.m_yd, meshGPU.m_zd);
}

/******************************************/

static inline
void LagrangeNodal(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   Domain_member fieldData[6] ;
#endif

   const Real_t delt = domain.deltatime() ;
   Real_t u_cut = domain.u_cut() ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

   CalcAccelerationForNodes(domain, domain.numNode());
   
   ApplyAccelerationBoundaryConditionsForNodes(domain);

   CalcVelocityForNodes( domain, delt, u_cut, domain.numNode()) ;

   CalcPositionForNodes( domain, delt, domain.numNode() );
   
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
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
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
                 const Real_t z2, const Real_t z3)
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
                                     const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = std::max(a,charLength) ;

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
                                Real_t* const d )
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
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

/******************************************/

/*
void CalcKinematicsForElems( Domain &domain, Real_t *vnew, 
                             Real_t deltaTime, Index_t numElem )
*/
static inline
void CalcKinematicsForElems( Index_t numElem, Real_t dt )
{
    callOpenCLKernel(CLsetup::program, "CalcKinematicsForElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    numElem, dt, meshGPU.m_nodelist, meshGPU.m_volo, meshGPU.m_v, 
                    meshGPU.m_x, meshGPU.m_y, meshGPU.m_z, meshGPU.m_xd, meshGPU.m_yd, meshGPU.m_zd, 
                    meshGPU.m_vnew, meshGPU.m_delv, meshGPU.m_arealg, meshGPU.m_dxx, meshGPU.m_dyy, meshGPU.m_dzz);
}

/******************************************/

/*
static inline
void CalcLagrangeElements(Domain& domain, Real_t* vnew)
*/
static inline
void CalcLagrangeElements(Domain& mesh)
{
   Index_t numElem = mesh.numElem() ;
   const Real_t deltatime = mesh.deltatime();
   if (numElem > 0) {
       CalcKinematicsForElems(numElem, deltatime);
       Index_t numElem = mesh.numElem();
       callOpenCLKernel(CLsetup::program, "CalcLagrangeElementsPart2_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    numElem, meshGPU.m_dxx, meshGPU.m_dyy, meshGPU.m_dzz, meshGPU.m_vdov);

   }
}

/******************************************/

/*
static inline
void CalcMonotonicQGradientsForElems(Domain& domain, Real_t vnew[])
*/
static inline
void CalcMonotonicQGradientsForElems(Domain& mesh)
{
    Index_t numElem = mesh.numElem();
    callOpenCLKernel(CLsetup::program, "CalcMonotonicQGradientsForElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    numElem, meshGPU.m_nodelist, meshGPU.m_x, meshGPU.m_y, meshGPU.m_z,
                    meshGPU.m_xd, meshGPU.m_yd, meshGPU.m_zd, meshGPU.m_volo,
                    meshGPU.m_vnew, meshGPU.m_delx_zeta, meshGPU.m_delv_zeta,
                    meshGPU.m_delx_xi, meshGPU.m_delv_xi, meshGPU.m_delx_eta, meshGPU.m_delv_eta);
}

/******************************************/

/*
static inline
void CalcMonotonicQRegionForElems(Domain &domain, Int_t r,
                                  Real_t vnew[], Real_t ptiny)
*/
static inline
void CalcMonotonicQRegionForElems(// parameters
                          Index_t regionStart,
                          Real_t qlc_monoq,
                          Real_t qqc_monoq,
                          Real_t monoq_limiter_mult,
                          Real_t monoq_max_slope,
                          Real_t ptiny,
                          // the elementset length
                          Index_t elength )
{
    callOpenCLKernel(CLsetup::program, "CalcMonotonicQRegionForElems_kernel", BLOCKSIZE, PAD(elength, BLOCKSIZE), 
                    regionStart, qlc_monoq, qqc_monoq, monoq_limiter_mult, monoq_max_slope, ptiny, elength, meshGPU.m_matElemlist,
                    meshGPU.m_elemBC, meshGPU.m_lxim, meshGPU.m_lxip, meshGPU.m_letam, meshGPU.m_letap, meshGPU.m_lzetam,
                    meshGPU.m_lzetap, meshGPU.m_delv_xi, meshGPU.m_delv_eta, meshGPU.m_delv_zeta, meshGPU.m_delx_xi, meshGPU.m_delx_eta,
                    meshGPU.m_delx_zeta, meshGPU.m_vdov, meshGPU.m_elemMass, meshGPU.m_volo, meshGPU.m_vnew, meshGPU.m_qq, meshGPU.m_ql);
}

/******************************************/

static inline
void CalcMonotonicQForElems(Domain& mesh)
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
          CalcMonotonicQRegionForElems(// parameters
                               regionStart,
                               qlc_monoq,
                               qqc_monoq,
                               monoq_limiter_mult,
                               monoq_max_slope,
                               ptiny,
                               // the elemset length
                               elength);
       }
   }
}

/******************************************/

static inline
void CalcQForElems(Domain& domain)
{
   //
   // MONOTONIC Q option
   //

   Index_t numElem = domain.numElem() ;

   if (numElem != 0) {
      /* Calculate velocity gradients */
      CalcMonotonicQGradientsForElems(domain);

      CalcMonotonicQForElems(domain) ;

   }
}

/******************************************/

/*
static inline
void CalcPressureForElems(Real_t* p_new, Real_t* bvc,
                          Real_t* pbvc, Real_t* e_old,
                          Real_t* compression, Real_t *vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          Index_t length, Index_t *regElemList)
*/
static inline
void CalcPressureForElems(Index_t regionStart, cl_mem p_new, cl_mem bvc,
                                cl_mem pbvc, cl_mem e_old,
                                cl_mem compression, cl_mem vnewc,
                                Real_t pmin,
                                Real_t p_cut, Real_t eosvmax,
                                Index_t length)
{
    Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
    callOpenCLKernel(CLsetup::program, "CalcPressureForElems_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    regionStart, meshGPU.m_matElemlist, p_new, bvc, pbvc, e_old, compression, vnewc, pmin, p_cut, eosvmax, length, c1s);
}

/******************************************/

/*
static inline
void CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new,
                        Real_t* bvc, Real_t* pbvc,
                        Real_t* p_old, Real_t* e_old, Real_t* q_old,
                        Real_t* compression, Real_t* compHalfStep,
                        Real_t* vnewc, Real_t* work, Real_t* delvc, Real_t pmin,
                        Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                        Real_t* qq_old, Real_t* ql_old,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length, Index_t *regElemList)
*/
static inline
void CalcEnergyForElems(Index_t regionStart, cl_mem p_new, cl_mem e_new, cl_mem q_new,
                        cl_mem bvc, cl_mem pbvc,
                        cl_mem p_old, cl_mem e_old,  cl_mem q_old,
                        cl_mem compression, cl_mem compHalfStep,
                        cl_mem vnewc, cl_mem work,  cl_mem delvc, Real_t pmin, 
                        Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin,
                        cl_mem qq, cl_mem ql,
                        Real_t rho0,
                        Real_t eosvmax,
                        Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
    cl_mem pHalfStep = clCreateBuffer(
            CLsetup::context,
            CL_MEM_READ_WRITE,
            length*sizeof(Real_t),
            NULL,
            &CLsetup::err);

    callOpenCLKernel(CLsetup::program, "CalcEnergyForElemsPart1_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    length, emin, e_old, delvc, p_old, q_old, work, e_new);

    CalcPressureForElems(regionStart, pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax, length);

    callOpenCLKernel(CLsetup::program, "CalcEnergyForElemsPart2_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    length, rho0, e_cut, emin, compHalfStep, delvc, pbvc, bvc, pHalfStep, ql, qq, p_old, q_old, work, e_new, q_new);
   
    CalcPressureForElems(regionStart, p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

    callOpenCLKernel(CLsetup::program, "CalcEnergyForElemsPart3_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    regionStart, meshGPU.m_matElemlist, length, rho0, sixth, e_cut, emin, pbvc, vnewc, bvc, p_new, ql, qq, p_old, q_old, pHalfStep, q_new, delvc, e_new);

    CalcPressureForElems(regionStart, p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);
   
    callOpenCLKernel(CLsetup::program, "CalcEnergyForElemsPart4_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    regionStart, meshGPU.m_matElemlist, length, rho0, q_cut, delvc, pbvc, e_new, vnewc, bvc, p_new, ql, qq, q_new);

    clReleaseMemObject(pHalfStep);
   return ;
}

/******************************************/

/*
static inline
void CalcSoundSpeedForElems(Domain &domain,
                            Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3,
                            Index_t len, Index_t *regElemList)
*/
static inline
void CalcSoundSpeedForElems(Index_t regionStart, cl_mem vnewc, Real_t rho0, cl_mem enewc,
        cl_mem pnewc, cl_mem pbvc, cl_mem bvc, Real_t ss4o3, Index_t nz)
{
    callOpenCLKernel(CLsetup::program, "CalcSoundSpeedForElems_kernel", BLOCKSIZE, PAD(nz, BLOCKSIZE), 
                    regionStart, vnewc, rho0, enewc, pnewc, pbvc, bvc, ss4o3, nz, meshGPU.m_matElemlist, meshGPU.m_ss);
}

/******************************************/

/*
static inline
void EvalEOSForElems(Domain& domain, Real_t *vnewc,
                     Int_t numElemReg, Index_t *regElemList, Int_t rep)
*/
static inline
void EvalEOSForElems(Index_t regionStart, Domain& mesh, cl_mem vnewc, Int_t numElemReg, Int_t rep)
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

    cl_mem e_old = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem delvc = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem p_old = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem q_old = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem compression = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem compHalfStep = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem qq = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem ql = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem work = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem p_new = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem e_new = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem q_new = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem bvc = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);
    cl_mem pbvc = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, numElemReg*sizeof(Real_t), NULL, &CLsetup::err);

    //loop to add load imbalance based on region number 
    for(Int_t j = 0; j < rep; j++) {
        callOpenCLKernel(CLsetup::program, "EvalEOSForElemsPart1_kernel", BLOCKSIZE, PAD(numElemReg, BLOCKSIZE), 
                        regionStart, numElemReg, eosvmin, eosvmax, meshGPU.m_matElemlist, meshGPU.m_e, meshGPU.m_delv, meshGPU.m_p, 
                        meshGPU.m_q, meshGPU.m_qq, meshGPU.m_ql, vnewc, e_old, delvc, p_old, q_old, compression, compHalfStep, qq, ql, work);

        CalcEnergyForElems(regionStart, p_new, e_new, q_new, bvc, pbvc,
                     p_old, e_old,  q_old, compression, compHalfStep,
                     vnewc, work,  delvc, pmin,
                     p_cut, e_cut, q_cut, emin,
                     qq, ql, rho0, eosvmax, numElemReg);
    }

    callOpenCLKernel(CLsetup::program, "EvalEOSForElemsPart2_kernel", BLOCKSIZE, PAD(numElemReg, BLOCKSIZE), 
                regionStart, numElemReg, meshGPU.m_matElemlist, p_new, e_new, q_new, meshGPU.m_p, meshGPU.m_e, meshGPU.m_q);

    CalcSoundSpeedForElems(regionStart, vnewc, rho0, e_new, p_new,
             pbvc, bvc, ss4o3, numElemReg) ;

	clReleaseMemObject(e_old);
	clReleaseMemObject(delvc);
	clReleaseMemObject(p_old);
	clReleaseMemObject(q_old);
	clReleaseMemObject(compression);
	clReleaseMemObject(compHalfStep);
	clReleaseMemObject(qq);
	clReleaseMemObject(ql);
	clReleaseMemObject(work);
	clReleaseMemObject(p_new);
	clReleaseMemObject(e_new);
	clReleaseMemObject(q_new);
	clReleaseMemObject(bvc);
	clReleaseMemObject(pbvc);
}

/******************************************/

/*
static inline
void ApplyMaterialPropertiesForElems(Domain& domain, Real_t vnew[])
*/
static inline
void ApplyMaterialPropertiesForElems(Domain& mesh)
{
  Index_t length = mesh.numElem() ;

  if (length != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = mesh.eosvmin() ;
    Real_t eosvmax = mesh.eosvmax() ;
    cl_mem vnewc = clCreateBuffer( CLsetup::context, CL_MEM_READ_WRITE, length*sizeof(Real_t), NULL, &CLsetup::err);

    callOpenCLKernel(CLsetup::program, "ApplyMaterialPropertiesForElemsPart1_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                length, eosvmin, eosvmax, meshGPU.m_matElemlist, meshGPU.m_vnew, vnewc);
    
    //TODO: add this check to a kernel
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
           EvalEOSForElems(regionStart, mesh, vnewc, numElemReg, rep);
       }
    }
    clReleaseMemObject(vnewc);
  }
}

/******************************************/

/*
static inline
void UpdateVolumesForElems(Domain &domain, Real_t *vnew,
                           Real_t v_cut, Index_t length)
*/
static inline
void UpdateVolumesForElems(Domain &mesh,
                            Real_t v_cut, Index_t length)
{
   Index_t numElem = mesh.numElem();
   if (numElem != 0) {
      Real_t v_cut = mesh.v_cut();
      callOpenCLKernel(CLsetup::program, "UpdateVolumesForElems_kernel", BLOCKSIZE, PAD(numElem, BLOCKSIZE), 
                    numElem, v_cut, meshGPU.m_vnew, meshGPU.m_v);
   }
}

/******************************************/

static inline
void LagrangeElements(Domain& domain, Index_t numElem)
{
  CalcLagrangeElements(domain) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain) ;

  ApplyMaterialPropertiesForElems(domain) ;

  UpdateVolumesForElems(domain, 
                        domain.v_cut(), numElem) ;

}

/******************************************/

/*
static inline
void CalcCourantConstraintForElems(Domain &domain, Index_t length,
                                   Index_t *regElemlist,
                                   Real_t qqc, Real_t& dtcourant)
*/
static inline
void CalcCourantConstraintForElems(Domain &mesh, Index_t regionStart, Index_t length,
                                   Index_t *regElemlist,
                                   Real_t qqc, Real_t& dtcourant)
{
//    Real_t qqc = mesh.qqc();
    Real_t qqc2 = Real_t(64.0) * qqc * qqc ;

    size_t localThreads = BLOCKSIZE;
    size_t globalThreads = PAD(length, localThreads);
    const unsigned int numBlocks = globalThreads/localThreads;

    cl_mem dev_mindtcourant = clCreateBuffer(CLsetup::context, CL_MEM_READ_WRITE, sizeof(Real_t)*numBlocks, NULL, &CLsetup::err);

    callOpenCLKernel(CLsetup::program, "CalcCourantConstraintForElems_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    regionStart, length, qqc2, meshGPU.m_matElemlist, meshGPU.m_ss, meshGPU.m_vdov, meshGPU.m_arealg, dev_mindtcourant);

    Real_t *mindtcourant = new Real_t[numBlocks];

    CLsetup::err = clEnqueueReadBuffer(
            CLsetup::queue,
            dev_mindtcourant,
            CL_TRUE,
            0,
            sizeof(Real_t)*numBlocks,
            mindtcourant,
            0,
            NULL,
            NULL);
    CLsetup::checkErr(CLsetup::err, "Command Queue::enqueueReadBuffer() - mindtcourant");
    
    clReleaseMemObject(dev_mindtcourant);

    // finish the MIN computation over the thread blocks
    for (unsigned int i=0; i<numBlocks; i++) {
        MINEQ(dtcourant,mindtcourant[i]);
    }
    delete []mindtcourant;

    if (dtcourant < Real_t(1.0e+20))
        mesh.dtcourant() = dtcourant ;
}

/******************************************/

/*
static inline
void CalcHydroConstraintForElems(Domain &domain, Index_t length,
                                 Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
*/
static inline
void CalcHydroConstraintForElems(Domain &mesh, Index_t regionStart, Index_t length,
                                 Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
{
//    Real_t dvovmax = mesh.dvovmax() ;
//    Index_t length = mesh.numElem() ;

    size_t localThreads = BLOCKSIZE;
    size_t globalThreads = PAD(length, localThreads);
    const unsigned int numBlocks = globalThreads/localThreads;

    cl_mem dev_mindthydro = clCreateBuffer(CLsetup::context, CL_MEM_READ_WRITE, sizeof(Real_t)*numBlocks, NULL, &CLsetup::err);

    callOpenCLKernel(CLsetup::program, "CalcHydroConstraintForElems_kernel", BLOCKSIZE, PAD(length, BLOCKSIZE), 
                    regionStart, length, dvovmax, meshGPU.m_matElemlist, meshGPU.m_vdov, dev_mindthydro);

    Real_t *mindthydro = new Real_t[numBlocks];
    CLsetup::err = clEnqueueReadBuffer(
            CLsetup::queue,
            dev_mindthydro,
            CL_TRUE,
            0,
            sizeof(Real_t)*numBlocks,
            mindthydro,
            0,
            NULL,
            NULL);
    CLsetup::checkErr(CLsetup::err, "Command Queue::enqueueReadBuffer() - mindthydro");

    clReleaseMemObject(dev_mindthydro);

    // finish the MIN computation over the thread blocks
    for (unsigned int i=0; i<numBlocks; i++) {
        MINEQ(dthydro,mindthydro[i]);
    }
    delete []mindthydro;
    
    if (dthydro < Real_t(1.0e+20))
        mesh.dthydro() = dthydro ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems(Domain& domain) {

   // Initialize conditions to a very large value
   domain.dtcourant() = 1.0e+20;
   domain.dthydro() = 1.0e+20;

   for (Index_t r=0 ; r < domain.numReg() ; ++r) {
      Index_t regionStart = domain.regStartPosition(r);
      Index_t numElemReg = domain.regElemSize(r);
      if (numElemReg > 0) {
          /* evaluate time constraint */
          CalcCourantConstraintForElems(domain, regionStart, domain.regElemSize(r),
                                        domain.regElemlist(r),
                                        domain.qqc(),
                                        domain.dtcourant()) ;

          /* check hydro constraint */
          CalcHydroConstraintForElems(domain, regionStart, domain.regElemSize(r),
                                      domain.regElemlist(r),
                                      domain.dvovmax(),
                                      domain.dthydro()) ;
      }
   }
}

/******************************************/

static inline
void LagrangeLeapFrog(Domain& domain)
{
#ifdef SEDOV_SYNC_POS_VEL_LATE
   Domain_member fieldData[6] ;
#endif

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);


#ifdef SEDOV_SYNC_POS_VEL_LATE
#endif

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain.numElem());

   CalcTimeConstraintsForElems(domain);

}

/******************************************/

int main(int argc, char *argv[])
{
   Domain *locDom ;
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

   //debug to see region sizes
//   for(Int_t i = 0; i < locDom->numReg(); i++)
//      std::cout << "region" << i + 1<< "size" << locDom->regElemSize(i) <<std::endl;

   /* initialize meshGPU */
   BLOCKSIZE=256;
   CLsetup::init("kernels.cl", 0, 0, BLOCKSIZE);
   meshGPU.init(locDom);
   meshGPU.freshenGPU();

   // BEGIN timestep to solution */
   Real_t start;
   Real_t elapsedTime;
   start = getTime();
   while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {

      TimeIncrement(*locDom) ;
      LagrangeLeapFrog(*locDom) ;

      if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
         printf("cycle = %d, time = %e, dt=%e\n",
                locDom->cycle(), double(locDom->time()), double(locDom->deltatime()) ) ;
      }
   }
   // Use reduced max elapsed time
   elapsedTime = (getTime() - start);
   double elapsedTimeG = elapsedTime;

   meshGPU.m_e_stale = CPU_STALE;
   meshGPU.m_x_stale = CPU_STALE;
   freshenCPU(locDom->m_e, meshGPU.m_e);
   freshenCPU(locDom->m_x, meshGPU.m_x);
   clFinish(CLsetup::queue);

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

   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(elapsedTimeG, *locDom, opts.nx, numRanks);
   }

#ifdef KERNELS_TIMING
   std::cout << std::setw(60) << "Kernel Name"
           << std::setw(30) << "Time(ns)"
           << std::setw(20) << "Num of Calls"
           << std::setw(20) << "Avg Time(ns)" << std::endl;
   for (auto it = totalTime.begin(); it != totalTime.end(); it++) {
       std::cout << std::setw(60) << it->first
           << std::setw(30) << it->second.time
           << std::setw(20) << it->second.numCalls
           << std::setw(20) << std::fixed << (double)it->second.time / (double)it->second.numCalls<< std::endl;
   }
#endif

   return 0 ;
}
