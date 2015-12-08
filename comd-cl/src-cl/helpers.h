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

#ifndef HELPER_H
#define HELPER_H

#include "clTypes.h"
#include "cl_utils.h"
#include "CoMDTypes.h"
#include "decomposition.h"
#include "eam.h"
#include "eamCL.h"
#include "constants.h"
//#include "ljCl.h"
#include "ljCL.h"
#include "atomsCL.h"
#include "linkCellsCL.h"
#include "timestepCL.h"
#include "decompositionCL.h"
#include "pfx_boxes.h"

#define NUMNEIGHBORS 27

#define DIAG_LEVEL 0
#define APPLE_OCL_10 0

typedef struct HostSpeciesDataSt
{
   char  name[3];   //!< element name
   int	 atomicNo;  //!< atomic number  
   real_t mass;     //!< mass in internal units
} HostSpeciesData;

typedef struct HostDeformationSt
{
   cl_real* stress;
   cl_real* strain;
   cl_real* invStrain;
   cl_real defGrad;
} HostDeformation;

/// Structure-of-arrays structs
/// the OpenCl sim struct holds both the host and device 
/// simulation structs, as well as the associated kernels.

typedef struct HostSimSoaSt 
{
   // boxes data 
   HostDomain* domain;
   HostBoxes* boxes;
   HostAtomsSoa* atoms;
   HostSpeciesData* species;
   cl_real* invMass; // per species
   // real scalars
   cl_real dt;
   cl_real ePotential;
   cl_real eKinetic;
   // integer values
   //int nLocalAtoms;
   // eam flag
   int eamFlag;
   HostEamPot eamPot;
   HostEamCh eamCh;
   HostLjPot ljPot;

} HostSimSoa;

typedef struct DevSimSoaSt 
{
   // boxes array values
   cl_mem invMass;
   // boxes data 
   DevAtomsSoa atoms;
   DevBoxesSoa boxes;
   DevDomain domain;
   // note all scalars can be passed directly to the kernels
   // real scalars
   cl_real dt;
   cl_real ePotential;
   cl_real eKinetic;
   // integer values
   DevEamPot eamPot;
   DevEamCh eamCh;
   DevLjPot ljPot;
} DevSimSoa;

typedef struct OclSimSoaSt
{
   HostSimSoa* hostSim;
   DevSimSoa* devSim;

   cl_int eamFlag;
   cl_int gpuFlag;

   // kernels
   cl_kernel* forceKernels;
   cl_kernel* advanceVelocity;
   cl_kernel* advancePosition;
   cl_kernel* pfxKernel;

   // events
   cl_event forceEvent;
   cl_event avEvent;
   cl_event apEvent;

   // work group sizes for each kernel
   size_t gForce[2];
   size_t lForce[2];
   size_t gVelocity[2];
   size_t lVelocity[2];
   size_t gPosition[2];
   size_t lPosition[2];

} OclSimSoa;

typedef struct HostSimAosSt 
{
   // boxes array values
   cl_real* invMass; // per species

   // boxes data 
   HostAtomsAos* atoms;
   HostBoxes* boxes;
   HostDomain* domain;
   HostSpeciesData* species;
   // real scalars
   cl_real dt;
   cl_real ePotential;
   cl_real eKinetic;
   // integer values
   int nLocalAtoms;
   // eam flag
   int eamFlag;
   HostEamPot eamPot;
   HostEamCh eamCh;
   HostLjPot ljPot;
} HostSimAos;

typedef struct DevSimAosSt 
{
   // boxes array values
   cl_mem invMass;
   // boxes data 
   DevAtomsAos atoms;
   DevBoxesAos boxes;
   DevDomain domain;
   // note all scalars can be passed directly to the kernels
   // real scalars
   cl_real dt;
   cl_real ePotential;
   cl_real eKinetic;
   // integer values
   DevEamPot eamPot;
   DevEamCh eamCh;
   DevLjPot ljPot;
} DevSimAos;


/* General helper utils */

void printArray(real_t* array, int n, char *name);

void printSim(SimFlat *s,FILE *fp);

void oclRunKernel(cl_kernel kernel, cl_event *event, size_t* nGlobal, size_t* nLocal);

void clGetElapsedTime(cl_event event, cl_real* elapsed_time, cl_real* enqueuedTime);

void computeForceOcl(size_t* nGlobal, size_t* nLocal, cl_real* t_kern, OclSimSoa* oclSim, SimFlat* sim);

void computeReductionBoxes(int *, int, cl_mem, cl_kernel);

/* SoA (default) variants) */

void getPrintStateSoa(DevSimSoa* simDevSoa, HostSimSoa* simHostSoa);

void sumLocalEnergySoa(DevSimSoa* simDevSoa, HostSimSoa* simHostSoa);

void sumLocalPrintEnergySoa(DevSimSoa* simDevSoa, HostSimSoa* simHostSoa);

void printStateSoa(HostSimSoa simHostSoa, int nCells);

void buildModulesSoa(OclSimSoa* oclSim, cl_kernel *Viz);

HostSimSoa* initHostSimSoa(SimFlat *sim, int eamFlag);

DevSimSoa* initDevSimSoa(HostSimSoa *simHostSoa);

void putSimSoa(HostSimSoa* simHostSoa, DevSimSoa* simDevSoa);

void FreeSimSoa(HostSimSoa* simHostSoa, DevSimSoa* simDevSoa);

/* AoS variants */

void getPrintStateAos(DevSimAos simDevSoa, HostSimAos simHostSoa);

void computePrintEnergyAos(DevSimAos simDevSoa, HostSimAos* simHostSoa);

void printStateAos(HostSimAos simHostSoa, int nCells);

void buildModulesAos(cl_kernel *forceKernels, cl_kernel *advancePosition, cl_kernel *AdvanceVelocity, cl_kernel *Viz, 
      HostSimAos simHostSoa, size_t *nLocal, size_t *nGlobal);

void initHostSimAos (HostSimAos *simHostSoa, SimFlat *sim);

void initDevSimAos(DevSimAos *simDevSoa, HostSimAos *simHostSoa);

void putSimAos(HostSimAos simHostSoa, DevSimAos simDevSoa);

void FreeSimAos(HostSimAos simHostSoa, DevSimAos simDevSoa);

/* Graphics kernels */

void oclGraphics(cl_kernel vizKernel, DevSimSoa simDevSoa, size_t* nGlobal, size_t* nLocal);

void oclRender();

void oclInitInterop(int ncells);

#endif
