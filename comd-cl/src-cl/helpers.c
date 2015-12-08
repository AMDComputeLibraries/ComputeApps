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

#include <string.h>
#include "computeOCL.h"
#include "atomsCL.h"
#include "performanceTimers.h"

// this should match the value in the LJ and/or EAM kernels

// Make sure PASS_1, PASS_2, PASS_3 flags are correctly set!
// Make sure PASS_2 matches that is eam_kernels.c

#define PASS_1 1
#define PASS_2 0
#define PASS_3 1

/// This chunk of code exists only because older OpenCL implementations
/// on Apple didn't treat vector types correctly. If you see errors during build time
/// about subscripted values, then you have a newer version of OpenCL, and these 'if'
/// clauses are no longer needed. They appear a couple more times in this file.
/// You can now set manually toggle these chunks of code using the APPLE_OCL_10 flag
/// below. If you have OpenCL 1.1 or newer, set it to 0
///

#define UNROLL 1
#define NSPECIES 1 

void dummyTest()
{
   cl_real4 dummy;

#if (defined (__APPLE__) || defined(MACOSX)) && (APPLE_OCL_10)
   dummy[0] = 1.0;
   dummy[1] = 1.0;
#else
   dummy.s[0] = 1.0;
   dummy.s[1] = 1.0;
#endif
}

void printArray(real_t* array, int n, char *name)
{
   printf("%s;\n", name);
   for (int i=0;i<n;i++)
   {
      printf("%d, %17.9e\n", i, array[i]);
   }
}

void sumLocalEnergySoa(DevSimSoa* devSim, HostSimSoa* hostSim)
{
   ///  Copy the array of particle energies from device to host. 
   ///  The total potential energy is summed and returned in the hostSim.ePotential variable

   oclCopyToHost(devSim->atoms.U, hostSim->atoms->U, hostSim->atoms->localRealSize, 0);
   getVec(devSim->atoms.p, hostSim->atoms->p, hostSim->atoms->localRealSize, 0);

   double ePot = 0.0;
   double eKin = 0.0;
   for (int iBox=0;iBox<hostSim->boxes->nLocalBoxes;iBox++)
   {
      for (int iAtom=0;iAtom<hostSim->boxes->nAtoms[iBox];iAtom++)
      {
         int iOff = iBox*MAXATOMS + iAtom;
         int iSpecies = hostSim->atoms->iSpecies[iOff];
         double invMass = 1.0/hostSim->species[iSpecies].mass;
         ePot += hostSim->atoms->U[iOff];
         eKin += (hostSim->atoms->p.x[iOff]*hostSim->atoms->p.x[iOff] +
                  hostSim->atoms->p.y[iOff]*hostSim->atoms->p.y[iOff] +
                  hostSim->atoms->p.z[iOff]*hostSim->atoms->p.z[iOff])*invMass;

      }
   }
   hostSim->ePotential = (cl_real)ePot;
   hostSim->eKinetic = (cl_real)eKin*0.5;
}

void sumLocalPrintEnergySoa(DevSimSoa* devSim, HostSimSoa* hostSim)
{
   ///  Copy the array of particle energies from device to host. 
   ///  The total potential energy is summed and returned in the hostSim.ePotential variable

   oclCopyToHost(devSim->atoms.U, hostSim->atoms->U, hostSim->atoms->localRealSize, 0);
   getVec(devSim->atoms.p, hostSim->atoms->p, hostSim->atoms->localRealSize, 0);

   double ePot = 0.0;
   double eKin = 0.0;
   for (int iBox=0;iBox<hostSim->boxes->nLocalBoxes;iBox++)
   {
      for (int iAtom=0;iAtom<hostSim->boxes->nAtoms[iBox];iAtom++)
      {
         int iOff = iBox*MAXATOMS + iAtom;
         int iSpecies = hostSim->atoms->iSpecies[iOff];
         double invMass = 1.0/hostSim->species[iSpecies].mass;
         ePot += hostSim->atoms->U[iOff];
         eKin += (hostSim->atoms->p.x[iOff]*hostSim->atoms->p.x[iOff] +
                  hostSim->atoms->p.y[iOff]*hostSim->atoms->p.y[iOff] +
                  hostSim->atoms->p.z[iOff]*hostSim->atoms->p.z[iOff])*invMass;

      }
   }
   hostSim->ePotential = (cl_real)ePot;
   hostSim->eKinetic = (cl_real)eKin*0.5;

   ePot = ePot/hostSim->atoms->nLocal;
   eKin = eKin*0.5/hostSim->atoms->nLocal;
   real_t eTotal = ePot + eKin;
   real_t Temp = eKin / (kB_eV * 1.5);

   printf(" %18.12f %18.12f %18.12f %12.4f", eTotal, ePot, eKin, Temp);
}

void computePrintEnergyAos(DevSimAos devSim, HostSimAos* hostSim)
{
   ///  Copy the array of particle energies from device to host. 
   ///  The total potential energy is summed and returned in the hostSim.ePotential variable

   double eLocal;

   oclCopyToHost(devSim.atoms.U, hostSim->atoms->U, hostSim->atoms->localRealSize, 0);

   eLocal = 0.0;
   for (int iBox=0;iBox<hostSim->boxes->nLocalBoxes;iBox++)
   {
      for (int iAtom=0;iAtom<hostSim->boxes->nAtoms[iBox];iAtom++)
      {
         eLocal += hostSim->atoms->U[iBox*MAXATOMS + iAtom];
      }
   }
   hostSim->ePotential = (cl_real)eLocal;

   //printf("System potential energy = %30.20f\n", hostSim->ePotential);
   printf(" %30.20e", hostSim->ePotential);

}

void printStateSoa(HostSimSoa hostSim, int nToPrint)
{
   /// Print the box index, atom index, position, momentum and force for the 
   /// first nCells boxes of the simulation

   int i, iBox;
   int atomCount = 0;
   iBox = 0;
   printf("System state:\n");
   while (atomCount < nToPrint)
   {
      for (int iAtom=0;iAtom<hostSim.boxes->nAtoms[iBox];iAtom++)
      {

         i = iBox*MAXATOMS + iAtom;

         printf("%02d, %02d, "
               "X=(%+020.12e %+020.12e %+020.12e) 1 "
               "P=(%+020.12e %+020.12e %+020.12e) "
               "F=(%+020.12e %+020.12e %+020.12e)\n",
               iBox, iAtom, 
               hostSim.atoms->r.x[i],hostSim.atoms->r.y[i],hostSim.atoms->r.z[i],
               hostSim.atoms->p.x[i],hostSim.atoms->p.y[i],hostSim.atoms->p.z[i],
               hostSim.atoms->f.x[i],hostSim.atoms->f.y[i],hostSim.atoms->f.z[i]);
         atomCount ++;

      }
      iBox ++;
   }

}

void printStateAos(HostSimAos hostSim, int nToPrint)
{
   /// Print the box index, atom index, position, momentum and force for the 
   /// first nCells boxes of the simulation

   int i, iBox;
   int atomCount = 0;
   iBox = 0;
   printf("System state:\n");
   while (atomCount < nToPrint)
   {
      for (int iAtom=0;iAtom<hostSim.boxes->nAtoms[iBox];iAtom++)
      {

         i = iBox*MAXATOMS + iAtom;
#if (defined (__APPLE__) || defined(MACOSX)) && (APPLE_OCL_10)
         printf("%02d, %02d, "
               "X=(%+020.12e %+020.12e %+020.12e) 1 "
               "P=(%+020.12e %+020.12e %+020.12e) "
               "F=(%+020.12e %+020.12e %+020.12e)\n",
               iBox, iAtom, 
               hostSim.atoms->r[i][0],hostSim.atoms->r[i][1],hostSim.atoms->r[i][2],
               hostSim.atoms->p[i][0],hostSim.atoms->p[i][1],hostSim.atoms->p[i][2],
               hostSim.atoms->f[i][0],hostSim.atoms->f[i][1],hostSim.atoms->f[i][2]);
#else
         printf("%02d, %02d, "
               "X=(%+020.12e %+020.12e %+020.12e) 1 "
               "P=(%+020.12e %+020.12e %+020.12e) "
               "F=(%+020.12e %+020.12e %+020.12e)\n",
               iBox, iAtom, 
               hostSim.atoms->r[i].s[0],hostSim.atoms->r[i].s[1],hostSim.atoms->r[i].s[2],
               hostSim.atoms->p[i].s[0],hostSim.atoms->p[i].s[1],hostSim.atoms->p[i].s[2],
               hostSim.atoms->f[i].s[0],hostSim.atoms->f[i].s[1],hostSim.atoms->f[i].s[2]);
#endif
         atomCount ++;
      }
      iBox ++;
   }

}

void putEamCh(HostEamCh eamChH, DevEamCh eamChD)
{
   oclCopyToDevice(eamChH.rho, eamChD.rho, eamChH.rhoChebSize, 0);
   oclCopyToDevice(eamChH.phi, eamChD.phi, eamChH.phiChebSize, 0);
   oclCopyToDevice(eamChH.F, eamChD.F, eamChH.fChebSize, 0);

   oclCopyToDevice(eamChH.nValues, eamChD.nValues, sizeof(cl_int)*3, 0);
}

void oclRunKernel(cl_kernel kernel, cl_event *event, size_t* nGlobal, size_t* nLocal)
{
   int err = clEnqueueNDRangeKernel(commandq, kernel, 2, NULL, nGlobal, nLocal, 0, NULL, event);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to enqueue kernel! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   clWaitForEvents(1, event);
}

void printSim(SimFlat *s,FILE *fp) 
{
   /// Print the base simulation data
   /// Note this is in a slightly different order than the OpenCL code returns

   for (int i=0; i<s->boxes->nLocalBoxes; i++)
   {
      int j;
      int *id;

      for (int iOff=i*MAXATOMS,j=0; j<s->boxes->nAtoms[i]; j++,iOff++)
      {
         if ( s->atoms->gid[iOff] < 10)
         {
            fprintf(fp,
                  "%02d %02d "
                  "X=(%+020.12e %+020.12e %+020.12e) 1 "
                  "P=(%+020.12e %+020.12e %+020.12e) "
                  "F=(%+020.12e %+020.12e %+020.12e)\n",
                  i,
                  s->atoms->gid[iOff]+1,
                  s->atoms->r[iOff][0],s->atoms->r[iOff][1],s->atoms->r[iOff][2],
                  s->atoms->p[iOff][0],s->atoms->p[iOff][1],s->atoms->p[iOff][2],
                  s->atoms->f[iOff][0],s->atoms->f[iOff][1],s->atoms->f[iOff][2]
                  );
         }
      }
   }
   return;
}

void clGetElapsedTime(cl_event event, cl_real* elapsed_time, cl_real* enqueuedTime)
{
   /// Helper routine to return the start-to-finish time (elapsed_time) 
   /// and the time from enqueueing to finish (enqueuedTime)

   cl_ulong t_start, t_end, t_enqueue;
   int err;
   size_t param_size;


   err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, &param_size);
   if (err != CL_SUCCESS)
   {
      printf("Error: %s\n", print_cl_errstring(err));
      printf("t_end = %llu\n", t_end);
      //exit(1);
   }

   err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, &param_size);
   if (err != CL_SUCCESS)
   {
      printf("Error: %s\n", print_cl_errstring(err));
      printf("t_start = %llu\n", t_start);
      //exit(1);
   }

   err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &t_enqueue, &param_size);

   *elapsed_time = (t_end - t_start)*1.0e-9 ;
   *enqueuedTime = (t_end - t_enqueue)*1.0e-9 ;
}

void getPrintStateSoa(DevSimSoa* devSim, HostSimSoa* hostSim) 
{
   getVec(devSim->atoms.r, hostSim->atoms->r, hostSim->atoms->localRealSize, 0);
   getVec(devSim->atoms.p, hostSim->atoms->p, hostSim->atoms->localRealSize, 0);
   getVec(devSim->atoms.f, hostSim->atoms->f, hostSim->atoms->localRealSize, 0);
   printStateSoa(*hostSim, 2);
}

void getPrintStateAos(DevSimAos devSim, HostSimAos hostSim) 
{
   getVecAos(devSim.atoms.r, hostSim.atoms->r, hostSim.atoms->localRealSize, 0);
   getVecAos(devSim.atoms.p, hostSim.atoms->p, hostSim.atoms->localRealSize, 0);
   getVecAos(devSim.atoms.f, hostSim.atoms->f, hostSim.atoms->localRealSize, 0);
   printStateAos(hostSim, 2);
}

#if USE_CHEBY
void initHostEamCh(HostEamCh *eamChH, SimFlat *sim) 
{
   /// Allocate and initialize all the EAM potential data needed

   int i;
   int n_v_rho;
   int n_v_phi;
   int n_v_F;

   // assign eam potential values
   printf("Using eam potential\n");
   EamPotential *new_pot;
   new_pot = (EamPotential*) sim->pot;
   EamCheby *chPot = sim->chPot;
   eamChH->cutoff  = new_pot->cutoff;
   printf("Cutoff = %e\n", eamChH->cutoff);

   /// The Chebyshev arrays are stored as:
   /// The interpolants for the potential
   /// The interpolants for the derivative
   /// The 3 values r0, rN and a dummy float to match the table 

   n_v_rho = chPot->rho->n;
   eamChH->rhoChebSize = (3 + 2*chPot->rho->n)*sizeof(cl_real);
   printf("rho potential size = %d\n", eamChH->rhoChebSize);

   n_v_phi = chPot->phi->n;
   eamChH->phiChebSize = (3 + 2*chPot->phi->n)*sizeof(cl_real);
   printf("phi potential size = %d\n", eamChH->phiChebSize);

   n_v_F = chPot->f->n;
   eamChH->fChebSize = (3 + 2*chPot->f->n)*sizeof(cl_real);
   printf("F potential size = %d\n", eamChH->fChebSize);

   eamChH->rho = malloc(eamChH->rhoChebSize);
   eamChH->phi = malloc(eamChH->phiChebSize);
   eamChH->F = malloc(eamChH->fChebSize);
   eamChH->nValues = malloc(3*sizeof(cl_int));

   eamChH->rho[2*n_v_rho+0] = chPot->rho->x0;
   eamChH->rho[2*n_v_rho+2] = chPot->rho->invDx;

   for (i=0;i<n_v_rho;i++)
   {
      eamChH->rho[i] = chPot->rho->values[i];
      eamChH->rho[n_v_rho+i] = chPot->drho->values[i];
   }

   eamChH->phi[2*n_v_phi+0] = chPot->phi->x0;
   eamChH->phi[2*n_v_phi+2] = chPot->phi->invDx;

   for (i=0;i<n_v_phi;i++)
   {
      eamChH->phi[i] = chPot->phi->values[i];
      eamChH->phi[n_v_phi+i] = chPot->dphi->values[i];
   }

   eamChH->F[2*n_v_F+0] = chPot->f->x0;
   eamChH->F[2*n_v_F+2] = chPot->f->invDx;

   for (i=0;i<n_v_F;i++)
   {
      eamChH->F[i] = chPot->f->values[i];
      eamChH->F[n_v_F+i] = chPot->df->values[i];
   }

   eamChH->nValues[0] = n_v_phi;
   eamChH->nValues[1] = n_v_rho;
   eamChH->nValues[2] = n_v_F;
}
#endif

/// Allocate and initialize all the host-side simulation data needed, 
///  including the appropriate potential data 
HostSimSoa* initHostSimSoa(SimFlat *sim, int eamFlag)
{
   HostSimSoa* hostSim = malloc(sizeof(HostSimSoa));
   hostSim->dt = sim->dt;
   printf("dt = %e\n", hostSim->dt);

   hostSim->eamFlag = eamFlag;

   hostSim->domain = initHostDomainSoa(sim);
   hostSim->boxes = initHostBoxesSoa(sim);
   hostSim->atoms = initHostAtomsSoa(sim);

   // only 1 species for now but coded for more 
   hostSim->species = malloc(sizeof(HostSpeciesData)*NSPECIES);
   hostSim->invMass = malloc(sizeof(cl_real)*NSPECIES);
   for (int iSpecies = 0; iSpecies < NSPECIES; iSpecies++)
   {
      strcpy(hostSim->species[iSpecies].name, sim->species->name);
      hostSim->species[iSpecies].atomicNo = sim->species->atomicNo;
      hostSim->species[iSpecies].mass = sim->species->mass;

      hostSim->invMass[iSpecies] = 1.0/hostSim->species[iSpecies].mass;
   }

   if(hostSim->eamFlag)
   {
      initHostEam(&hostSim->eamPot, sim);
#if USE_CHEBY
      initHostEamCh(&hostSim->eamCh, sim);
#endif
   }
   else
   {
      initHostLj(&hostSim->ljPot, sim);
   }

   printf("Host SoA simulation initialized\n");
   return hostSim;
}

/// Allocate and initialize all the host-side simulation data needed, 
///  including the appropriate potential data 
void initHostSimAos (HostSimAos *hostSim, SimFlat *sim)
{
   hostSim->dt = sim->dt;
   printf("dt = %e\n", hostSim->dt);

   hostSim->domain = initHostDomainAos(sim);
   hostSim->boxes  = initHostBoxesAos(sim);
   hostSim->atoms  = initHostAtomsAos(sim);

   hostSim->nLocalAtoms             = sim->atoms->nLocal;

   // only 1 species for now
   hostSim->invMass = malloc(sizeof(cl_real));

   hostSim->invMass[0] = 1.0/hostSim->species->mass;

   if(hostSim->eamFlag)
   {
      initHostEam(&hostSim->eamPot, sim);
#if USE_CHEBY
      initHostEamCh(&hostSim->eamCh, sim);
#endif
   }
   else
   {
      initHostLj(&hostSim->ljPot, sim);
   }

   printf("Host AoS simulation initialized\n");
}

DevSimSoa* initDevSimSoa(HostSimSoa *hostSim)
{
   /// Allocate all the device-side arrays needed for the simulation
   DevSimSoa* devSim = malloc(sizeof(DevSimSoa));

   // allocate memory buffer on device
   printf("Allocating device memory...");
   fflush(stdout);

   createDevBoxesSoa(&devSim->boxes, hostSim->boxes);
   createDevAtomsSoa(&devSim->atoms, hostSim->atoms);

   // particle mass
   oclCreateReadWriteBuffer(&devSim->invMass, sizeof(cl_real));

   if (hostSim->eamFlag)
   {
      initDevEam(&hostSim->eamPot, &devSim->eamPot);

#if USE_CHEBY
      //************************************************************************
      // EAM Chebychev coefficient data
      oclCreateReadWriteBuffer(&devSim->eamCh.rho, hostSim->eamCh.rhoChebSize);
      oclCreateReadWriteBuffer(&devSim->eamCh.phi, hostSim->eamCh.phiChebSize);
      oclCreateReadWriteBuffer(&devSim->eamCh.F, hostSim->eamCh.fChebSize);

      oclCreateReadWriteBuffer(&devSim->eamCh.nValues, sizeof(cl_int)*3);

      // add this here to make passing arguments to kernels easier
      devSim->eamCh.cutoff = hostSim->eamCh.cutoff;
#endif
   }
   else
   {
      devSim->ljPot.cutoff = hostSim->ljPot.cutoff;
      devSim->ljPot.sigma = hostSim->ljPot.sigma;
      devSim->ljPot.epsilon = hostSim->ljPot.epsilon;
   }

   printf("device memory allocated\n");
   return devSim;
}

void initDevSimAos(DevSimAos *devSim, HostSimAos *hostSim)
{
   /// Allocate all the device-side arrays needed for the simulation

   // allocate memory buffer on device
   printf("Allocating device memory (AoS)...");

   createDevAtomsAos(&devSim->atoms, hostSim->atoms);

   // particle mass
   oclCreateReadWriteBuffer(&devSim->invMass, sizeof(cl_real));


   createDevBoxesAos(&devSim->boxes, hostSim->boxes);


   if (hostSim->eamFlag)
   {
      initDevEam(&hostSim->eamPot, &devSim->eamPot);

#if USE_CHEBY
      //************************************************************************
      // EAM Chebychev coefficient data
      oclCreateReadWriteBuffer(&devSim->eamCh.rho, hostSim->eamCh.rhoChebSize);
      oclCreateReadWriteBuffer(&devSim->eamCh.phi, hostSim->eamCh.phiChebSize);
      oclCreateReadWriteBuffer(&devSim->eamCh.F, hostSim->eamCh.fChebSize);

      oclCreateReadWriteBuffer(&devSim->eamCh.nValues, sizeof(cl_int)*3);

      // add this here to make passing arguments to kernels easier
      devSim->eamCh.cutoff = hostSim->eamCh.cutoff;
#endif
   }
   else
   {
      devSim->ljPot.cutoff = hostSim->ljPot.cutoff;
      devSim->ljPot.sigma = hostSim->ljPot.sigma;
      devSim->ljPot.epsilon = hostSim->ljPot.epsilon;
   }
   printf("device memory allocated\n");

}

void putSimSoa(HostSimSoa* hostSim, DevSimSoa* devSim)
{
   /// Copy all the host-side simulation data to the corresponding device arrays

   printf("Copying data to device...");

   putBoxesSoa(hostSim->boxes, &devSim->boxes);

   // copy the input arrays to the device
   // positions
   putAtomsSoa(hostSim->atoms, &devSim->atoms);

   oclCopyToDevice(hostSim->invMass, devSim->invMass, sizeof(cl_real), 0);
   // simulation data

   if(hostSim->eamFlag)
   {
      putEamPot(hostSim->eamPot, devSim->eamPot);
      //   putEamCh(hostSim.eamCh, devSim.eamCh);
   }
   printf("data copied\n");

}

void putSimAos(HostSimAos hostSim, DevSimAos devSim)
{
   /// Copy all the host-side simulation data to the corresponding device arrays

   printf("Copying data to device (AoS)...");
   fflush(stdout);

   // copy the input arrays to the device
   // positions
   putAtomsAos(hostSim.atoms, &devSim.atoms);

   printf("positions...");
   fflush(stdout);

   // mass
   oclCopyToDevice(hostSim.invMass, devSim.invMass, sizeof(cl_real), 0);

   printf("mass...");
   fflush(stdout);

   // simulation data
   putBoxesAos(hostSim.boxes, &devSim.boxes);

   printf("boxes...");
   fflush(stdout);

   if(hostSim.eamFlag)
   {
      putEamPot(hostSim.eamPot, devSim.eamPot);
   }
   printf("data copied\n");
   fflush(stdout);

}

void FreeSimSoa(HostSimSoa* hostSim, DevSimSoa* devSim)
{
   /// clean up all the host and device memory objects

   freeAtomsSoa(hostSim->atoms, devSim->atoms);

   free(hostSim->invMass);

   clReleaseMemObject(devSim->invMass);

   if(hostSim->eamFlag)
   {
   }
}

void FreeSimAos(HostSimAos hostSim, DevSimAos devSim)
{
   /// clean up all the host and device memory objects

   freeAtomsAos(hostSim.atoms, devSim.atoms);

   free(hostSim.invMass);

   clReleaseMemObject(devSim.invMass);

   if(hostSim.eamFlag)
   {
   }
}

void buildModulesSoa(OclSimSoa* oclSim,
      cl_kernel *Viz) 
{
   /// Build the kernels to compute force, and advance position and velocity.
   /// Return the appropriate global and local sizes

   HostSimSoa* hostSim = oclSim->hostSim;

   cl_kernel* forceKernels = oclSim->forceKernels;
   cl_kernel* advancePosition = oclSim->advancePosition;
   cl_kernel* advanceVelocity = oclSim->advanceVelocity;
   cl_kernel* pfxKernel = oclSim->pfxKernel;

   cl_program timestepModule;
   cl_program ljModule;
   cl_program eamModule;
   cl_program vizModule;

	cl_program boxModule;

	buildProgramFromFile(&boxModule, "./src-cl/pfx_boxes.cl", context, deviceId);
   int err;

   // build the program from the kernel source file

   buildProgramFromFile(&timestepModule, "./src-cl/timestep_kernels.cl", context, deviceId);
   // only build the modules needed for the potential chosen
   if(hostSim->eamFlag)
   {
      buildProgramFromFile(&eamModule, "./src-cl/eam_kernels.cl", context, deviceId);
   }
   else
   {
      buildProgramFromFile(&ljModule, "./src-cl/lj_kernels.cl", context, deviceId);
   }
   printf("Program built\n");

   if(hostSim->eamFlag)
   {
      // create the EAM_Force_x kernels from the program
      forceKernels[0] = clCreateKernel(eamModule, "EAM_Force_1", &err);
      if (!forceKernels[0] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel EAM_Force_1!\n");
         exit(1);
      }
      else
      {
         printf("Kernel EAM_Force_1 built\n");
      }
      forceKernels[1] = clCreateKernel(eamModule, "EAM_Force_2", &err);
      if (!forceKernels[1] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel EAM_Force_2!\n");
         exit(1);
      }
      else
      {
         printf("Kernel EAM_Force_2 built\n");
      }
      forceKernels[2] = clCreateKernel(eamModule, "EAM_Force_3", &err);
      if (!forceKernels[2] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel EAM_Force_3!\n");
         exit(1);
      }
      else
      {
         printf("Kernel EAM_Force_3 built\n");
      }

   }
   else
   {
      // create the ljForce kernel from the program
      forceKernels[0] = clCreateKernel(ljModule, "ljForce", &err);
      if (!forceKernels[0] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel ljForce!\n");
         exit(1);
      }
      else
      {
         printf("Kernel ljForce built\n");
      }
   }
   // create the advanceVelocity kernel from the program
   *advanceVelocity = clCreateKernel(timestepModule, "advanceVelocitySoa", &err);
   if (!*advanceVelocity || err != CL_SUCCESS)
   {
      printf("Error: Failed to create compute kernel advanceVelocity!\n");
      exit(1);
   }
   else
   {
      printf("Kernel advanceVelocity built\n");
   }
   // create the advancePosition kernel from the program
   *advancePosition = clCreateKernel(timestepModule, "advancePositionSoa", &err);
   if (!*advancePosition || err != CL_SUCCESS)
   {
      printf("Error: Failed to create compute kernel advancePosition!\n");
      exit(1);
   }
   else
   {
      printf("Kernel advancePosition built\n");
   }

   *pfxKernel = clCreateKernel(boxModule, "pfxSumBoxes", &err);

	if(!*pfxKernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create kernel pfxSumBoxes \n");
		exit(1);
	}
	else
   {
      printf("Kernel prefix sum built\n");
   }

   // determine allowable local work sizes for the device we chose
   if(hostSim->eamFlag)
   {
      size_t lTemp = 0;
      err = clGetKernelWorkGroupInfo(forceKernels[0], deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(oclSim->lForce), oclSim->lForce, NULL);
      if (err != CL_SUCCESS)
      {
         printf("Error: Failed to retrieve eamForce work group info! %d\n", err);
         printf("Error: %s\n", print_cl_errstring(err));
         exit(1);
      }
      lTemp = oclSim->lForce[0];
      err = clGetKernelWorkGroupInfo(forceKernels[2], deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(oclSim->lForce), oclSim->lForce, NULL);
      if (err != CL_SUCCESS)
      {
         printf("Error: Failed to retrieve eamForce work group info! %d\n", err);
         printf("Error: %s\n", print_cl_errstring(err));
         exit(1);
      }
      // set local size to smallest for safety
      if (lTemp < oclSim->lForce[0]) oclSim->lForce[0] = lTemp;
      printf("Maximum local size for eamForce is (%lu, %lu)\n", oclSim->lForce[0], oclSim->lForce[1]);
   }
   else
   {
      err = clGetKernelWorkGroupInfo(forceKernels[0], deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(oclSim->lForce), oclSim->lForce, NULL);
      if (err != CL_SUCCESS)
      {
         printf("Error: Failed to retrieve ljForce work group info! %d\n", err);
         printf("Error: %s\n", print_cl_errstring(err));
         exit(1);
      }
      printf("Maximum local size for ljForce is (%lu, %lu)\n", oclSim->lForce[0], oclSim->lForce[1]);
   }

   err = clGetKernelWorkGroupInfo(*advanceVelocity, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(oclSim->lVelocity), oclSim->lVelocity, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to retrieve advanceVelocity work group info! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   printf("Maximum local size for advanceVelocity is (%lu, %lu)\n", oclSim->lVelocity[0], oclSim->lVelocity[1]);

   err = clGetKernelWorkGroupInfo(*advancePosition, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(oclSim->lPosition), oclSim->lPosition, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to retrieve advancePosition work group info! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   printf("Maximum local size for advancePosition is (%lu, %lu)\n", oclSim->lPosition[0], oclSim->lPosition[1]);

   oclSim->lForce[1] = 1;
   oclSim->lVelocity[1] = 1;
   oclSim->lPosition[1] = 1;

   // set the global size equal to the numer of atoms
   oclSim->gForce[0] = hostSim->atoms->nGlobal;// MAXATOMS;
   oclSim->gForce[1] = 1;//hostSim->boxes->nLocalBoxes;

   oclSim->gVelocity[0] = MAXATOMS;
   oclSim->gVelocity[1] = hostSim->boxes->nLocalBoxes;

   oclSim->gPosition[0] = MAXATOMS;
   oclSim->gPosition[1] = hostSim->boxes->nLocalBoxes;

   // fudge the loop unrollling here
//   oclSim->gForce[0] = oclSim->gForce[0]*UNROLL;
//   oclSim->gForce[1] = oclSim->gForce[1]/UNROLL;

//   oclSim->gVelocity[0] = oclSim->gVelocity[0]*UNROLL;
//   oclSim->gVelocity[1] = oclSim->gVelocity[1]/UNROLL;

//   oclSim->gPosition[0] = oclSim->gPosition[0]*UNROLL;
//   oclSim->gPosition[1] = oclSim->gPosition[1]/UNROLL;

   // if the local size is greater than the total size, set local size to global size
   int i;
   for (i=0;i<2;i++)
   {
      if (oclSim->gForce[i] < oclSim->lForce[i]) oclSim->lForce[i] = oclSim->gForce[i];
      if (oclSim->gVelocity[i] < oclSim->lVelocity[i]) oclSim->lVelocity[i] = oclSim->gVelocity[i];
      if (oclSim->gPosition[i] < oclSim->lPosition[i]) oclSim->lPosition[i] = oclSim->gPosition[i];
   }

   oclSim->lForce[0] = 256;

   printf("Global and local sizes are (%lu, %lu), (%lu, %lu)\n", 
      oclSim->gForce[0], oclSim->gForce[1], oclSim->lForce[0], oclSim->lForce[1]);
   printf("Global and local sizes are (%lu, %lu), (%lu, %lu)\n", 
      oclSim->gVelocity[0], oclSim->gVelocity[1], oclSim->lVelocity[0], oclSim->lVelocity[1]);
   printf("Global and local sizes are (%lu, %lu), (%lu, %lu)\n", 
      oclSim->gPosition[0], oclSim->gPosition[1], oclSim->lPosition[0], oclSim->lPosition[1]);
}
void buildModulesAos(cl_kernel *forceKernels, 
      cl_kernel *advancePosition, 
      cl_kernel *advanceVelocity,
      cl_kernel *Viz, 
      HostSimAos hostSim, 
      size_t *nLocal, 
      size_t *nGlobal)
{
   /// Build the kernels to compute force, and advance position and velocity.
   /// Return the appropriate global and local sizes

   cl_program timestepModule;
   cl_program ljModule;
   cl_program eamModule;
   cl_program vizModule;

   int err;

   // build the program from the kernel source file

   buildProgramFromFile(&timestepModule, "./src-cl/timestep_kernels.c", context, deviceId);
   // only build the modules needed for the potential chosen
   if(hostSim.eamFlag)
   {
      buildProgramFromFile(&eamModule, "./src-cl/eam_kernels.c", context, deviceId);
   }
   else
   {
      buildProgramFromFile(&ljModule, "./src-cl/lj_kernels_aos.c", context, deviceId);
   }

   printf("Program built\n");

   if(hostSim.eamFlag)
   {
      // create the EAM_Force_x kernels from the program
      forceKernels[0] = clCreateKernel(eamModule, "EAM_Force_1_AoS", &err);
      if (!forceKernels[0] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel EAM_Force_1!\n");
         exit(1);
      }
      else
      {
         printf("Kernel EAM_Force_1 built\n");
      }
      forceKernels[1] = clCreateKernel(eamModule, "EAM_Force_2_AoS", &err);
      if (!forceKernels[1] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel EAM_Force_2!\n");
         exit(1);
      }
      else
      {
         printf("Kernel EAM_Force_2 built\n");
      }
      forceKernels[2] = clCreateKernel(eamModule, "EAM_Force_3_AoS", &err);
      if (!forceKernels[2] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel EAM_Force_3!\n");
         exit(1);
      }
      else
      {
         printf("Kernel EAM_Force_3 built\n");
      }

   }
   else
   {
      // create the ljForce kernel from the program
      forceKernels[0] = clCreateKernel(ljModule, "ljForceAos", &err);
      if (!forceKernels[0] || err != CL_SUCCESS)
      {
         printf("Error: Failed to create compute kernel ljForceAos!\n");
         exit(1);
      }
      else
      {
         printf("Kernel ljForceAos built\n");
      }
   }
   // create the advanceVelocity kernel from the program
   *advanceVelocity = clCreateKernel(timestepModule, "advanceVelocityAos", &err);
   if (!*advanceVelocity || err != CL_SUCCESS)
   {
      printf("Error: Failed to create compute kernel advanceVelocityAos!\n");
      exit(1);
   }
   else
   {
      printf("Kernel advanceVelocity built\n");
   }
   // create the advancePosition kernel from the program
   *advancePosition = clCreateKernel(timestepModule, "advancePositionAos", &err);
   if (!*advancePosition || err != CL_SUCCESS)
   {
      printf("Error: Failed to create compute kernel advancePositionAos!\n");
      exit(1);
   }
   else
   {
      printf("Kernel advancePosition built\n");
   }

   // determine allowable local work sizes for the device we chose
   if(hostSim.eamFlag)
   {
   }
   else
   {
      err = clGetKernelWorkGroupInfo(forceKernels[0], deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(nLocal), nLocal, NULL);
      if (err != CL_SUCCESS)
      {
         printf("Error: Failed to retrieve ljForce work group info! %d\n", err);
         printf("Error: %s\n", print_cl_errstring(err));
         exit(1);
      }
      printf("Maximum local size for ljForce is (%lu, %lu)\n", nLocal[0], nLocal[1]);
   }

   err = clGetKernelWorkGroupInfo(*advanceVelocity, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(nLocal), nLocal, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to retrieve advanceVelocity work group info! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   printf("Maximum local size for advanceVelocity is (%lu, %lu)\n", nLocal[0], nLocal[1]);

   err = clGetKernelWorkGroupInfo(*advancePosition, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(nLocal), nLocal, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to retrieve advancePosition work group info! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   printf("Maximum local size for advancePosition is (%lu, %lu)\n", nLocal[0], nLocal[1]);

   //nLocal[1] += 1;

   // set the global size equal to the numer of atoms
   nGlobal[0] = MAXATOMS;
   nGlobal[1] = hostSim.boxes->nTotalBoxes;

   // if the local size is greater than the total size, set local size to global size
   int i;
   for (i=0;i<2;i++)
   {
      if (nGlobal[i] < nLocal[i])
      {
         nLocal[i] = nGlobal[i];
      }
   }
   printf("Global and local sizes are (%lu, %lu), (%lu, %lu)\n", nGlobal[0], nGlobal[1], nLocal[0], nLocal[1]);

}

void haloForceExchangeSoa(HostSimSoa* hostSim, DevSimSoa* devSim, SimFlat* sim)
{
      // here we need to do all the force copying stuff
      startTimer(eamHaloTimer);
      EamPotential* pot = (EamPotential*)sim->pot;
      oclCopyToHost(devSim->eamPot.dfEmbed, hostSim->eamPot.dfEmbed, hostSim->atoms->totalRealSize, 0);
      for (int iBox=0; iBox<hostSim->boxes->nTotalBoxes; iBox++)
      {
         for (int iAtom=0;iAtom<sim->boxes->nAtoms[iBox];iAtom++)
         {
            int iOff = iBox*MAXATOMS + iAtom;
            pot->dfEmbed[iOff] = hostSim->eamPot.dfEmbed[iOff];
         }
      }
      haloExchange(pot->forceExchange, pot->forceExchangeData);

      for (int iBox=0; iBox<hostSim->boxes->nTotalBoxes; iBox++)
      {
         for (int iAtom=0;iAtom<sim->boxes->nAtoms[iBox];iAtom++)
         {
            int iOff = iBox*MAXATOMS + iAtom;
            hostSim->eamPot.dfEmbed[iOff] = pot->dfEmbed[iOff];
         }
      }
      oclCopyToDevice(hostSim->eamPot.dfEmbed, devSim->eamPot.dfEmbed, hostSim->atoms->totalRealSize, 0);

      stopTimer(eamHaloTimer);
}

void computeForceOcl(
      size_t* nGlobal, 
      size_t* nLocal, 
      cl_real* tKern,
      OclSimSoa* oclSim,
      SimFlat* sim)
{
   /// Execute the appropriate force kernels 

   int nTot = oclSim->hostSim->atoms->nLocal;
   int err;
   cl_real tSimple, tOverall;
   cl_real tTotal = 0.0;
   if (oclSim->eamFlag)
   {
#if (PASS_1)
      if (DIAG_LEVEL > 1)
      {
         printf("Running EAM kernel 1..");
         fflush(stdout);
      }
      oclRunKernel(oclSim->forceKernels[0], &oclSim->forceEvent, nGlobal, nLocal);
      err = clWaitForEvents(1, &oclSim->forceEvent);
      if (DIAG_LEVEL > 1)
      {
         printf("done\n");
         fflush(stdout);
      }
      clGetElapsedTime(oclSim->forceEvent, &tSimple, &tOverall);
      tTotal += tSimple;
#endif
#if (PASS_2)
      if (DIAG_LEVEL > 1)
      {
         printf("Running EAM kernel 2..");
         fflush(stdout);
      }
      oclRunKernel(oclSim->forceKernels[1], &oclSim->forceEvent, nGlobal, nLocal);
      err = clWaitForEvents(1, &oclSim->forceEvent);
      if (DIAG_LEVEL > 1)
      {
         printf("done\n");
         fflush(stdout);
      }
      clGetElapsedTime(oclSim->forceEvent, &tSimple, &tOverall);
      tTotal += tSimple;
#endif
      haloForceExchangeSoa(oclSim->hostSim, oclSim->devSim, sim);

#if (PASS_3)
      if (DIAG_LEVEL > 1)
      {
         printf("Running EAM kernel 3..");
         fflush(stdout);
      }
      oclRunKernel(oclSim->forceKernels[2], &oclSim->forceEvent, nGlobal, nLocal);
      err = clWaitForEvents(1, &oclSim->forceEvent);
      if (DIAG_LEVEL > 1)
      {
         printf("done\n");
         fflush(stdout);
      }
      clGetElapsedTime(oclSim->forceEvent, &tSimple, &tOverall);
      tTotal += tSimple;
#endif
      *tKern = tTotal;

		printf("eam force time: %f\n", tTotal);

      if (DIAG_LEVEL > 0)
         printf("Kernel EAM_Force executed in %.3e secs. (%e us/atom for %d atoms)\n", 
               tTotal, 1.0e6*tTotal/nTot, nTot);

   }
   else
   {
      if (DIAG_LEVEL > 1)
      {
         printf("Running LJ kernel..");
      }
      oclRunKernel(oclSim->forceKernels[0], &oclSim->forceEvent, nGlobal, nLocal);
      err = clWaitForEvents(1, &oclSim->forceEvent);
      if (DIAG_LEVEL > 1)
      {
         printf("done\n");
      }
      clGetElapsedTime(oclSim->forceEvent, &tSimple, &tOverall);
      if (DIAG_LEVEL > 0)
         printf("Kernel ljForce executed in %.3e secs. (%e us/atom for %d atoms)\n", 
               tSimple, 1.0e6*tSimple/nTot, nTot);
      *tKern = tSimple;
//	  printf("force kernel time: %f\n", tSimple);
   }

}

void computeReductionBoxes(int *box_cnt, int nboxes, cl_mem boxes_d, cl_kernel pfxKernel)
{
	cl_int cerr = 0;

	// would not be required if redistribute atoms moves to GPU
	oclCopyToDevice(box_cnt, boxes_d, nboxes*sizeof(int), 0);

	// right now just using one workgroup
	size_t gt = 256, lt = 256;

	cl_event ev;

	cerr |= clEnqueueNDRangeKernel(commandq, pfxKernel, 1, NULL, &gt, &lt, 0, NULL, &ev);
	if(cerr != CL_SUCCESS)
	{
		printf("error enqueuing boxes\n");
		exit(1);
	}
	clFlush(commandq);
//	int64_t t1, t2;
//	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &t1, NULL);
//	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &t2, NULL);
//
//	printf("red time: %f\n", (float)(t2-t1)*1e-9);
}
