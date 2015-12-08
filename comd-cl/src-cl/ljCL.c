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

#include "ljCL.h"

#include "ljForce.h"
#include "helpers.h"

void initHostLj(HostLjPot *ljPot_H, SimFlat *sim)
{
   /** Allocate and initialize all the LJ potential data needed **/

   LjPotential *new_pot;
   new_pot = (LjPotential*) sim->pot;
   ljPot_H->sigma  = new_pot->sigma;
   ljPot_H->epsilon  = new_pot->epsilon;
   ljPot_H->cutoff  = new_pot->cutoff;

   printf("Using Lennard-Jones potential\n");
   printf("Epsilon = %f eV\n", ljPot_H->epsilon);
   printf("Sigma = %f Angstroms\n", ljPot_H->sigma);
   printf("Cutoff = %f Angstroms\n", ljPot_H->cutoff);

}

void setLjArgsSoa(cl_kernel ljForce,
      DevSimSoa* devSim)
{
   /** Set the kernel arguments for the LJ force kernel **/

   printf("Setting LJ kernel arguments\n");
   // set kernel arguments for ljForce
   int err = 0;
   int nArg = 0;
   // field arrays
   err  = clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.r.x);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.r.y);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.r.z);

   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.f.x);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.f.y);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.f.z);

   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.U);
   // boxes data
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.neighborList);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.nNeighbors);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   // potential data
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_real), &devSim->ljPot.sigma);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_real), &devSim->ljPot.epsilon);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_real), &devSim->ljPot.cutoff);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_int), &devSim->boxes.nLocalBoxes);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.nAtomsPfx);

   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set ljForce arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("ljForce arguments set\n");
   }
}

void setLjArgsAos(cl_kernel ljForce,
      DevSimAos* devSim)
{
   /** Set the kernel arguments for the LJ force kernel **/

   printf("Setting LJ kernel arguments\n");
   // set kernel arguments for ljForce
   int err = 0;
   int nArg = 0;
   // field arrays
   err  = clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.r);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.f);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->atoms.U);
   // boxes data
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.neighborList);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.nNeighbors);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   // potential data
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_real), &devSim->ljPot.sigma);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_real), &devSim->ljPot.epsilon);
   err |= clSetKernelArg(ljForce, nArg++, sizeof(cl_real), &devSim->ljPot.cutoff);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set ljForceAos arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("ljForceAos arguments set\n");
   }
}

