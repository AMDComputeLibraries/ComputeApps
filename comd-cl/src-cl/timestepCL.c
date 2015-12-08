#include "helpers.h"

void setAvArgsSoa(cl_kernel advanceVelocity, DevSimSoa* devSim, cl_real dt)
{
   /** Set the arguments for the advanceVelocity kernel.
     Because of the Verlet timestepping scheme we keep the timestep as a separate argument
    **/

   int err = 0;
   int nArg = 0;
   // field arrays
   err  = clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.p.x);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.p.y);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.p.z);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.f.x);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.f.y);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.f.z);
   // boxes data
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   // timestep
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_real), &dt);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set advanceVelocity arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("advanceVelocity arguments set\n");
      printf("dt = %e\n", dt);
   }
}

void setApArgsSoa(cl_kernel advancePosition, DevSimSoa* devSim, cl_real dt)
{
   /** Set the arguments for the advancePosition kernel.
     Because of the Verlet timestepping scheme we keep the timestep as a separate argument
    **/

   int err = 0;
   int nArg = 0;
   // field arrays
   err  = clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.p.x);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.p.y);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.p.z);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.r.x);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.r.y);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.r.z);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.iSpecies);
   // boxes data
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   // timestep
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->invMass);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_real), &dt);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set advancePosition arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("advancePosition arguments set\n");
      printf("dt = %e\n", dt);
   }
}

void setAvArgsAos(cl_kernel advanceVelocity, DevSimAos* devSim, cl_real dt)
{
   /** Set the arguments for the advanceVelocity kernel.
     Because of the Verlet timestepping scheme we keep the timestep as a separate argument
    **/

   int err = 0;
   int nArg = 0;
   // field arrays
   err  = clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.p);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->atoms.f);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   err |= clSetKernelArg(advanceVelocity, nArg++, sizeof(cl_real), &dt);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set advanceVelocityAos arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("advanceVelocityAos arguments set\n");
      printf("dt = %e\n", dt);
   }
}

void setApArgsAos(cl_kernel advancePosition, DevSimAos* devSim, cl_real dt)
{
   /** Set the arguments for the advancePosition kernel.
     Because of the Verlet timestepping scheme we keep the timestep as a separate argument
    **/

   int err = 0;
   int nArg = 0;
   // field arrays
   err  = clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.p);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->atoms.r);
   //err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->m);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_mem), &devSim->boxes.nAtoms);
   err |= clSetKernelArg(advancePosition, nArg++, sizeof(cl_real), &dt);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set advancePositionAos arguments! %d\n", err);
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
   else
   {
      printf("advancePositionAos arguments set\n");
      printf("dt = %e\n", dt);
   }
}

