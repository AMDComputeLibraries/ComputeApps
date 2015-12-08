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

/**
 * a simple md simulator
 **/
#include "computeOCL.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <pthread.h>

#include "mytype.h"
#include "helpers.h"
#include "yamlOutput.h"
#include "memUtils.h"
#include "timestep.h"
#include "performanceTimers.h"

cl_kernel vizSoa;

cl_kernel advanceVelocityAos;
cl_kernel advancePositionAos;
cl_kernel vizAos;
cl_kernel *forceKernelsAos;

real_t tExec;
real_t tOverall;
real_t tLocal;

int iter;
int nIter;

void printLine()
{
   printf("********************************************************************************\n");
}


OclSimSoa* initOclSimSoa(SimFlat* sim, Command cmd)
{
   OclSimSoa* oclSim = malloc(sizeof(OclSimSoa));

   oclSim->eamFlag = cmd.doeam;
   oclSim->gpuFlag = cmd.useGpu;

   if(oclSim->eamFlag) 
   {
      oclSim->forceKernels = malloc(sizeof(cl_kernel)*3);
   } 
   else 
   {
      oclSim->forceKernels = malloc(sizeof(cl_kernel)*1);
   }
   oclSim->advanceVelocity = malloc(sizeof(cl_kernel));
   oclSim->advancePosition = malloc(sizeof(cl_kernel));
   oclSim->pfxKernel = malloc(sizeof(cl_kernel));

   HostSimSoa* hostSim = initHostSimSoa(sim, oclSim->eamFlag);
   DevSimSoa* devSim = initDevSimSoa(hostSim);
   putSimSoa(hostSim, devSim);

   oclSim->hostSim = hostSim;
   oclSim->devSim = devSim;

   buildModulesSoa(oclSim, &vizSoa);

   cl_real dthalf = 0.5*hostSim->dt;
   cl_real dtminushalf = -0.5*hostSim->dt;

   if (oclSim->eamFlag) 
   {
      // set the arguments for all 3 EAM_Force kernels
      setEamArgsSoa(oclSim->forceKernels, devSim);
   } 
   else 
   {
      // set kernel arguments for ljForce
      setLjArgsSoa(oclSim->forceKernels[0], devSim);
   }

   // set kernel arguments for advanceVelocitySoa
   setAvArgsSoa(*oclSim->advanceVelocity, devSim, dthalf);

   // set kernel arguments for advancePosition
   setApArgsSoa(*oclSim->advancePosition, devSim, hostSim->dt);

	// set kernel arguments for pfxBoxes
	setPfxArgsSoa(*oclSim->pfxKernel, devSim);

   // Start the simulation here;
   printLine();
   printf("Starting SoA OpenCL simulation\n");

	//ts = timeNow();

	computeReductionBoxes(sim->boxes->nAtoms, sim->boxes->nLocalBoxes, oclSim->devSim->boxes.nAtomsPfx, *(oclSim->pfxKernel));

	cl_real tKern;
	computeForceOcl(oclSim->gForce, oclSim->lForce, &tKern, oclSim, sim);

	getVec(devSim->atoms.f, hostSim->atoms->f, hostSim->atoms->totalRealSize, 0);

	//if (DIAG_LEVEL > 1)
      getPrintStateSoa(devSim, hostSim);

   return oclSim;

}

void computeIterationSoa2(SimFlat* sim, OclSimSoa* oclSim)
{
   real_t tKern, tEnq;
   real_t tAcc = 0.0;
   int nSteps = sim->printRate;

   for (int iStep = 0;iStep < nSteps; iStep++) 
   {

      startTimer(oclTimestep);
      // advance velocity a half timestep
      startTimer(oclVelocity);
      oclRunKernel(*oclSim->advanceVelocity, &oclSim->avEvent, oclSim->gVelocity, oclSim->lVelocity);
      clWaitForEvents(1, &oclSim->avEvent);
      stopTimer(oclVelocity);
      clGetElapsedTime(oclSim->avEvent, &tKern, &tEnq);
      tAcc += tKern;

      if (DIAG_LEVEL > 1)
      {
         printf("After first velocity substep\n");
         getPrintStateSoa(oclSim->devSim, oclSim->hostSim);
      }

      // advance particle positions a full timestep
      startTimer(oclPosition);
      oclRunKernel(*oclSim->advancePosition, &oclSim->apEvent, oclSim->gPosition, oclSim->lPosition);
      clWaitForEvents(1, &oclSim->apEvent);
      stopTimer(oclPosition);
      clGetElapsedTime(oclSim->apEvent, &tKern, &tEnq);
      tAcc += tKern;

      if (DIAG_LEVEL > 1)
      {
         printf("After position substep\n");
         getPrintStateSoa(oclSim->devSim, oclSim->hostSim);
      }

      startTimer(oclCopy);
      // copy atom info back to base sim
      // note that atom ownership and link cell occupancy are unchanged
      getAtomsSoa(oclSim->hostSim->atoms, &oclSim->devSim->atoms);
      stopTimer(oclCopy);
      startTimer(oclRedistribute);
      atomsToSim(sim, oclSim->hostSim->atoms);

      // for now this uses the base sim for redistribution
      redistributeAtoms(sim);

      // need to update the atom counts for the link cells
      for (int iBox=0; iBox<sim->boxes->nTotalBoxes; iBox++)
      {
         oclSim->hostSim->boxes->nAtoms[iBox] = sim->boxes->nAtoms[iBox];
      }
      oclCopyToDevice(oclSim->hostSim->boxes->nAtoms, oclSim->devSim->boxes.nAtoms, oclSim->hostSim->boxes->nTotalBoxesIntSize, 0);

	  // copy the redistributed atom info to the device
	  atomsToSoa(sim, oclSim->hostSim->atoms);
	  stopTimer(oclRedistribute);
	  startTimer(oclCopy);
	  putAtomsSoa(oclSim->hostSim->atoms, &oclSim->devSim->atoms);
	  stopTimer(oclCopy);

	  computeReductionBoxes(sim->boxes->nAtoms, sim->boxes->nLocalBoxes, oclSim->devSim->boxes.nAtomsPfx, *(oclSim->pfxKernel));

	  // compute force
	  startTimer(oclForce);
	  computeForceOcl(oclSim->gForce, oclSim->lForce, &tKern, oclSim, sim);
	  stopTimer(oclForce);
	  tAcc += tKern;

      if (DIAG_LEVEL > 1)
         getPrintStateSoa(oclSim->devSim, oclSim->hostSim);

      // advance velocity a half timestep
      startTimer(oclVelocity);
      oclRunKernel(*oclSim->advanceVelocity, &oclSim->avEvent, oclSim->gVelocity, oclSim->lVelocity);
      clWaitForEvents(1, &oclSim->avEvent);
      stopTimer(oclVelocity);
      clGetElapsedTime(oclSim->avEvent, &tKern, &tEnq);
      tAcc += tKern;

      if (DIAG_LEVEL > 1)
      {
         printf("After second velocity substep\n");
         getPrintStateSoa(oclSim->devSim, oclSim->hostSim);
      }
      stopTimer(oclTimestep);
   }

#ifdef INTEROP_VIZ 
   oclGraphics(vizSoa, oclSim->devSim, nGlobal, nLocal);
#endif

   sumLocalEnergySoa(oclSim->devSim, oclSim->hostSim);

   sim->ePotential = oclSim->hostSim->ePotential;
   sim->eKinetic   = oclSim->hostSim->eKinetic  ;
}


void runRef()
{
   printLine();
   printf("Running reference simulation\n");
   // write initial state 
   //writeClsman(sim,(char *) "init.bin");

   // do the computation 
   //(void)doComputeWork(sim); 

   // write final configuration 
   //writeClsman(sim,(char *) "final.bin");

   // free memory 
   //destroySimulation(&sim);
}


#ifdef INTEROP_VIZ
void keyboard(unsigned char key, int x, int y) 
{ 
   glutPostRedisplay(); 
}

void idle() 
{ 
   glutPostRedisplay(); 
}

void mouse(int button, int state, int x, int y) 
{
   if (state == GLUT_DOWN) mouse_buttons |= 1<<button;
   else if (state == GLUT_UP) mouse_buttons = 0;

   mouse_old_x = x;
   mouse_old_y = y;
   glutPostRedisplay();
}

void motion(int x, int y) 
{
   float dx = x - mouse_old_x;
   float dy = y - mouse_old_y;

   if (mouse_buttons == 1)
   {
      Quaternion newRotX;
      QuaternionSetEulerAngles(&newRotX, -0.2*dx*3.14159/180.0, 0.0, 0.0);
      QuaternionMul(&q, q, newRotX);

      Quaternion newRotY;
      QuaternionSetEulerAngles(&newRotY, 0.0, 0.0, -0.2*dy*3.14159/180.0);
      QuaternionMul(&q, q, newRotY);
   }
   else if (mouse_buttons == 4)
   {
      cameraFOV += dy/25.0f;
   }

   mouse_old_x = x;
   mouse_old_y = y;
   glutPostRedisplay();
}

void renderData() 
{
   if (iter == 0) ts = timeNow();
   if (iter++ < nIter) computeIterationSoa();
   if (iter == nIter) 
   { 
      te = timeNow();
      finishOclSoa();
   }

   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   oclRender();
   glutSwapBuffers();
}
#endif

