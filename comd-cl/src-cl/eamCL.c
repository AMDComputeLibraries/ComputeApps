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

void initHostEam(HostEamPot *hostEamPot, SimFlat *sim) 
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
   hostEamPot->cutoff  = new_pot->cutoff;
   printf("Cutoff = %e\n", hostEamPot->cutoff);

   n_v_rho = new_pot->rho->n;
   hostEamPot->rhoPotSize = (6 + new_pot->rho->n)*sizeof(cl_real);
   printf("rho potential size = %d\n", hostEamPot->rhoPotSize);

   n_v_phi = new_pot->phi->n;
   hostEamPot->phiPotSize = (6 + new_pot->phi->n)*sizeof(cl_real);
   printf("phi potential size = %d\n", hostEamPot->phiPotSize);

   n_v_F = new_pot->f->n;
   hostEamPot->fPotSize = (6 + new_pot->f->n)*sizeof(cl_real);
   printf("F potential size = %d\n", hostEamPot->fPotSize);

   hostEamPot->rho = malloc(hostEamPot->rhoPotSize);
   hostEamPot->phi = malloc(hostEamPot->phiPotSize);
   hostEamPot->F = malloc(hostEamPot->fPotSize);
   hostEamPot->nValues = malloc(3*sizeof(cl_int));

   // Note the EAM array has 3 extra values to account for over/under flow
   // We also add another 3 values to store x0, xn, invDx
   hostEamPot->rho[n_v_rho+3] = new_pot->rho->x0;
   hostEamPot->rho[n_v_rho+4] = new_pot->rho->x0+n_v_rho/new_pot->rho->invDx;
   hostEamPot->rho[n_v_rho+5] = new_pot->rho->invDx;

   for (i=0;i<n_v_rho+3;i++)
   {
      hostEamPot->rho[i] = new_pot->rho->values[i-1];
   }

   hostEamPot->phi[n_v_phi+3] = new_pot->phi->x0;
   hostEamPot->phi[n_v_phi+4] = new_pot->phi->x0+n_v_phi/new_pot->phi->invDx;
   hostEamPot->phi[n_v_phi+5] = new_pot->phi->invDx;

   for (i=0;i<n_v_phi+3;i++)
   {
      hostEamPot->phi[i] = new_pot->phi->values[i-1];
   }

   hostEamPot->F[n_v_F+3] = new_pot->f->x0;
   hostEamPot->F[n_v_F+4] = new_pot->f->x0+n_v_F/new_pot->f->invDx;
   hostEamPot->F[n_v_F+5] = new_pot->f->invDx;

   for (i=0;i<n_v_F+3;i++)
   {
      hostEamPot->F[i] = new_pot->f->values[i-1];
   }

   hostEamPot->nValues[0] = n_v_phi;
   hostEamPot->nValues[1] = n_v_rho;
   hostEamPot->nValues[2] = n_v_F;

   // extra atom arrays needed for EAM computation
   hostEamPot->totalRealSize = MAXATOMS*sim->boxes->nTotalBoxes*sizeof(cl_real);
   hostEamPot->rhobar = malloc(hostEamPot->totalRealSize);
   hostEamPot->dfEmbed = malloc(hostEamPot->totalRealSize);
}

void initDevEam(HostEamPot* hostEamPot, DevEamPot* devEamPot)
{
   oclCreateReadWriteBuffer(&devEamPot->rho, hostEamPot->rhoPotSize);
   oclCreateReadWriteBuffer(&devEamPot->phi, hostEamPot->phiPotSize);
   oclCreateReadWriteBuffer(&devEamPot->F, hostEamPot->fPotSize);

   oclCreateReadWriteBuffer(&devEamPot->nValues, sizeof(cl_int)*3);

   oclCreateReadWriteBuffer(&devEamPot->dfEmbed, hostEamPot->totalRealSize);
   oclCreateReadWriteBuffer(&devEamPot->rhobar, hostEamPot->totalRealSize);

   // add this here to make passing arguments to kernels easier
   devEamPot->cutoff = hostEamPot->cutoff;

}

void putEamPot(HostEamPot hostEamPot, DevEamPot devEamPot)
{
   oclCopyToDevice(hostEamPot.rho, devEamPot.rho, hostEamPot.rhoPotSize, 0);
   oclCopyToDevice(hostEamPot.phi, devEamPot.phi, hostEamPot.phiPotSize, 0);
   oclCopyToDevice(hostEamPot.F, devEamPot.F, hostEamPot.fPotSize, 0);

   oclCopyToDevice(hostEamPot.nValues, devEamPot.nValues, sizeof(cl_int)*3, 0);
}

void setEamArgsSoa(cl_kernel *forceKernels, DevSimSoa* devSim)
{ 
   /** Set the kernel arguments for the three EAM force computation kernels
    **/

#if (USE_CHEBY) 
   DevEamCh localPot;
   localPot = devSim->eamCh;
#else 
   DevEamPot localPot;
   localPot = devSim->eamPot;
#endif

   printf("Setting EAM kernel arguments\n");
   printf("Kernel 1...");
   fflush(stdout);
   // set kernel arguments for EAM_Force_1
   int err = 0;
   int nArg = 0;
   err  = clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.r.x);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.r.y);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.r.z);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.f.x);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.f.y);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.f.z);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.U);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &localPot.rhobar);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &localPot.dfEmbed);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.neighborList);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.nNeighbors);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &localPot.nValues);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &localPot.phi);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &localPot.rho);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &localPot.F);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_real), &localPot.cutoff);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_int), &devSim->boxes.nLocalBoxes);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.nAtomsPfx);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set EAM_Force_1 arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      fflush(stdout);
      exit(1);
   }
   else
   {
      printf("done\n");
      fflush(stdout);
   }

   // set kernel arguments for EAM_Force_2
   printf("Kernel 2...");
   fflush(stdout);
   err = 0;
   nArg = 0;
   err  = clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &localPot.dfEmbed);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->atoms.U);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &localPot.rhobar);

   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);

   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &localPot.F);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &localPot.nValues);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_int), &devSim->boxes.nLocalBoxes);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->boxes.nAtomsPfx);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set EAM_Force_2 arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      fflush(stdout);
      exit(1);
   }
   else
   {
      printf("done\n");
      fflush(stdout);
   }

   // set kernel arguments for EAM_Force_3
   printf("Kernel 3...");
   err = 0;
   nArg = 0;
   err  = clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.r.x);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.r.y);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.r.z);

   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.f.x);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.f.y);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.f.z);

   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &localPot.dfEmbed);

   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.neighborList);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.nNeighbors);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);

   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &localPot.nValues);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &localPot.rho);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_real), &localPot.cutoff);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_int), &devSim->boxes.nLocalBoxes);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.nAtomsPfx);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set EAM_Force_3 arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("done\n");
   }

}

void setEamArgsAos( cl_kernel *forceKernels, DevSimAos* devSim)
{ 
   /** Set the kernel arguments for the three EAM force computation kernels
    **/

   int err, nArg;

   printf("Setting EAM kernel arguments\n");
   printf("Kernel 1...");
   fflush(stdout);
   // set kernel arguments for EAM_Force_1
   err = 0;
   nArg = 0;
   err  = clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.r);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.f);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->atoms.U);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->eamPot.rhobar);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->eamPot.dfEmbed);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->domain.localMin);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->domain.localMax);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.neighborList);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.nNeighbors);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->eamPot.nValues);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->eamPot.phi);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->eamPot.rho);
   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_mem), &devSim->eamPot.F);

   err |= clSetKernelArg(forceKernels[0], nArg++, sizeof(cl_real), &devSim->eamPot.cutoff);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set EAM_Force_1 arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
      fflush(stdout);
   }
   else
   {
      printf("done\n");
      fflush(stdout);
   }

   // set kernel arguments for EAM_Force_2
   printf("Kernel 2...");
   err = 0;
   nArg = 0;
   err  = clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->eamPot.dfEmbed);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->atoms.U);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->eamPot.rhobar);

   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);

   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->eamPot.F);
   err |= clSetKernelArg(forceKernels[1], nArg++, sizeof(cl_mem), &devSim->eamPot.nValues);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set EAM_Force_2 arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
      fflush(stdout);
   }
   else
   {
      printf("done\n");
      fflush(stdout);
   }

   // set kernel arguments for EAM_Force_3
   printf("Kernel 3...");
   err = 0;
   nArg = 0;
   // field arrays
   err  = clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.r);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->atoms.f);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->eamPot.dfEmbed);
   // boxes data
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->domain.localMin);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->domain.localMax);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.neighborList);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.nNeighbors);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   //potential data
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->eamPot.nValues);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_mem), &devSim->eamPot.rho);
   err |= clSetKernelArg(forceKernels[2], nArg++, sizeof(cl_real), &devSim->eamPot.cutoff);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set EAM_Force_3 arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      fflush(stdout);
      exit(1);
   }
   else
   {
      printf("done\n");
      fflush(stdout);
   }

}

