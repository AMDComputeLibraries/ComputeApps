#include "helpers.h"

HostAtomsSoa* initHostAtomsSoa(SimFlat* sim)
{
   HostAtomsSoa* atoms;

   atoms = malloc(sizeof(HostAtomsSoa));

   atoms->nLocal = sim->atoms->nLocal;
   atoms->nGlobal = sim->atoms->nGlobal;

   atoms->totalRealSize = MAXATOMS*sim->boxes->nTotalBoxes*sizeof(cl_real);
   atoms->localRealSize = MAXATOMS*sim->boxes->nLocalBoxes*sizeof(cl_real);
   atoms->haloRealSize  = MAXATOMS*sim->boxes->nHaloBoxes*sizeof(cl_real);

   atoms->totalIntSize  = MAXATOMS*sim->boxes->nTotalBoxes*sizeof(cl_int);
   atoms->localIntSize  = MAXATOMS*sim->boxes->nLocalBoxes*sizeof(cl_int);
   atoms->haloIntSize   = MAXATOMS*sim->boxes->nHaloBoxes*sizeof(cl_int);

   atoms->gid      = malloc(atoms->totalIntSize);
   atoms->iSpecies = malloc(atoms->totalIntSize);

   // location
   atoms->r.x = malloc(atoms->totalRealSize);
   atoms->r.y = malloc(atoms->totalRealSize);
   atoms->r.z = malloc(atoms->totalRealSize);

   // momenta
   atoms->p.x = malloc(atoms->totalRealSize);
   atoms->p.y = malloc(atoms->totalRealSize);
   atoms->p.z = malloc(atoms->totalRealSize);

   // forces
   atoms->f.x = malloc(atoms->totalRealSize);
   atoms->f.y = malloc(atoms->totalRealSize);
   atoms->f.z = malloc(atoms->totalRealSize);

   // energy
   atoms->U = malloc(atoms->totalRealSize);

   // at initialization time, copy all the atoms (local and halo)
   // into the hostSim->atoms struct
   atomsToSoa(sim, atoms);

   return atoms;
}

void atomsToSoa(SimFlat* sim, HostAtomsSoa* atoms)
{
   atoms->nLocal = sim->atoms->nLocal;
   atoms->nGlobal = sim->atoms->nGlobal;

   for (int iBox=0;iBox<sim->boxes->nTotalBoxes;iBox++)
   {
      for (int iAtom=0;iAtom<sim->boxes->nAtoms[iBox];iAtom++)
      {
         int iOff = iBox*MAXATOMS + iAtom;

         atoms->gid[iOff]   = sim->atoms->gid[iOff];
         atoms->iSpecies[iOff] = sim->atoms->iSpecies[iOff];
         
         atoms->r.x[iOff] = sim->atoms->r[iOff][0];
         atoms->r.y[iOff] = sim->atoms->r[iOff][1];
         atoms->r.z[iOff] = sim->atoms->r[iOff][2];

         atoms->p.x[iOff] = sim->atoms->p[iOff][0];
         atoms->p.y[iOff] = sim->atoms->p[iOff][1];
         atoms->p.z[iOff] = sim->atoms->p[iOff][2];

         //atoms->f.x[iOff] = sim->atoms->f[iOff][0];
         //atoms->f.y[iOff] = sim->atoms->f[iOff][1];
         //atoms->f.z[iOff] = sim->atoms->f[iOff][2];

         atoms->U[iOff]   = sim->atoms->U[iOff];
      }
   }
}

void atomsToSim(SimFlat* sim, HostAtomsSoa* atoms)
{
   for (int iBox=0;iBox<sim->boxes->nTotalBoxes;iBox++)
   {
      for (int iAtom=0;iAtom<sim->boxes->nAtoms[iBox];iAtom++)
      {
         int iOff = iBox*MAXATOMS + iAtom;

         sim->atoms->gid[iOff]    = atoms->gid[iOff];
         sim->atoms->iSpecies[iOff] = atoms->iSpecies[iOff];

         sim->atoms->r[iOff][0] = atoms->r.x[iOff];
         sim->atoms->r[iOff][1] = atoms->r.y[iOff];
         sim->atoms->r[iOff][2] = atoms->r.z[iOff];

         sim->atoms->p[iOff][0] = atoms->p.x[iOff];
         sim->atoms->p[iOff][1] = atoms->p.y[iOff];
         sim->atoms->p[iOff][2] = atoms->p.z[iOff];

         //sim->atoms->f[iOff][0] = atoms->f.x[iOff];
         //sim->atoms->f[iOff][1] = atoms->f.y[iOff];
         //sim->atoms->f[iOff][2] = atoms->f.z[iOff];

         sim->atoms->U[iOff]    = atoms->U[iOff];
      }
   }
}

void createDevAtomsSoa(DevAtomsSoa* atoms, HostAtomsSoa* hostAtoms)
{
   atoms->nLocal = hostAtoms->nLocal;
   atoms->nGlobal = hostAtoms->nGlobal;

   oclCreateReadWriteBuffer(&atoms->gid, hostAtoms->totalIntSize);
   oclCreateReadWriteBuffer(&atoms->iSpecies, hostAtoms->totalIntSize);

   // positions, momenta, force
   createDevVec(&atoms->r, hostAtoms->totalRealSize);
   createDevVec(&atoms->p, hostAtoms->totalRealSize);
   createDevVec(&atoms->f, hostAtoms->totalRealSize);

   // particle energy
   oclCreateReadWriteBuffer(&atoms->U, hostAtoms->totalRealSize);

}

void putAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa* devAtoms)
{
   putVec(hostAtoms->r, devAtoms->r, hostAtoms->totalRealSize, 0);
   putVec(hostAtoms->p, devAtoms->p, hostAtoms->totalRealSize, 0);
   putVec(hostAtoms->f, devAtoms->f, hostAtoms->totalRealSize, 0);

   oclCopyToDevice(hostAtoms->gid, devAtoms->gid, hostAtoms->totalIntSize, 0);
   oclCopyToDevice(hostAtoms->iSpecies, devAtoms->iSpecies, hostAtoms->totalIntSize, 0);
}

void getAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa* devAtoms)
{
   getVec(devAtoms->r, hostAtoms->r, hostAtoms->totalRealSize, 0);
   getVec(devAtoms->p, hostAtoms->p, hostAtoms->totalRealSize, 0);
   getVec(devAtoms->f, hostAtoms->f, hostAtoms->totalRealSize, 0);

   oclCopyToHost(devAtoms->gid, hostAtoms->gid, hostAtoms->totalIntSize, 0);
   oclCopyToHost(devAtoms->iSpecies, hostAtoms->iSpecies, hostAtoms->totalIntSize, 0);
}

HostAtomsAos* initHostAtomsAos(SimFlat* sim)
{
   HostAtomsAos* atoms;

   atoms = malloc(sizeof(HostAtomsAos));

   atoms->nLocal = sim->atoms->nLocal;
   atoms->nGlobal = sim->atoms->nGlobal;

   atoms->totalRealSize = MAXATOMS*sim->boxes->nTotalBoxes*sizeof(cl_real);
   atoms->localRealSize = MAXATOMS*sim->boxes->nLocalBoxes*sizeof(cl_real);

   // location
   atoms->r = malloc(atoms->totalRealSize*r3);
   // momenta
   atoms->p = malloc(atoms->totalRealSize*r3);
   // forces
   atoms->f = malloc(atoms->totalRealSize*r3);
   // energy
   atoms->U = malloc(atoms->totalRealSize);

   for (int iBox=0;iBox<sim->boxes->nLocalBoxes;iBox++)
   {
      for (int iAtom=0;iAtom<sim->boxes->nAtoms[iBox];iAtom++)
      {
         int iOff = iBox*MAXATOMS + iAtom;
#if (defined (__APPLE__) || defined(MACOSX)) && (APPLE_OCL_10)
         atoms->r[iOff][0] = sim->atoms->r[iOff][0];
         atoms->r[iOff][1] = sim->atoms->r[iOff][1];
         atoms->r[iOff][2] = sim->atoms->r[iOff][2];

         atoms->p[iOff][0] = sim->atoms->p[iOff][0];
         atoms->p[iOff][1] = sim->atoms->p[iOff][1];
         atoms->p[iOff][2] = sim->atoms->p[iOff][2];

         atoms->f[iOff][0] = sim->atoms->f[iOff][0];
         atoms->f[iOff][1] = sim->atoms->f[iOff][1];
         atoms->f[iOff][2] = sim->atoms->f[iOff][2];
#else
         atoms->r[iOff].s[0] = sim->atoms->r[iOff][0];
         atoms->r[iOff].s[1] = sim->atoms->r[iOff][1];
         atoms->r[iOff].s[2] = sim->atoms->r[iOff][2];

         atoms->p[iOff].s[0] = sim->atoms->p[iOff][0];
         atoms->p[iOff].s[1] = sim->atoms->p[iOff][1];
         atoms->p[iOff].s[2] = sim->atoms->p[iOff][2];

         atoms->f[iOff].s[0] = sim->atoms->f[iOff][0];
         atoms->f[iOff].s[1] = sim->atoms->f[iOff][1];
         atoms->f[iOff].s[2] = sim->atoms->f[iOff][2];
#endif
      }
   }
   return atoms;
}

void createDevAtomsAos(DevAtomsAos* atoms, HostAtomsAos* hostAtoms)
{
   atoms->nLocal = hostAtoms->nLocal;
   atoms->nGlobal = hostAtoms->nGlobal;

   // positions, momenta, force
   oclCreateReadWriteBuffer(&atoms->r, hostAtoms->totalRealSize*r3);
   oclCreateReadWriteBuffer(&atoms->p, hostAtoms->totalRealSize*r3);
   oclCreateReadWriteBuffer(&atoms->f, hostAtoms->totalRealSize*r3);

   // particle energy
   oclCreateReadWriteBuffer(&atoms->U, hostAtoms->totalRealSize);

   oclCreateReadWriteBuffer(&atoms->gid, hostAtoms->totalIntSize);
   oclCreateReadWriteBuffer(&atoms->iSpecies, hostAtoms->totalIntSize);

}

void putAtomsAos(HostAtomsAos* hostAtoms, DevAtomsAos* devAtoms)
{
   oclCopyToDevice(hostAtoms->r, devAtoms->r, hostAtoms->totalRealSize*r3, 0);
   oclCopyToDevice(hostAtoms->p, devAtoms->p, hostAtoms->totalRealSize*r3, 0);
   oclCopyToDevice(hostAtoms->f, devAtoms->f, hostAtoms->totalRealSize*r3, 0);
}

void freeAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa devAtoms)
{
   free(hostAtoms->r.x);
   free(hostAtoms->r.y);
   free(hostAtoms->r.z);

   free(hostAtoms->p.x);
   free(hostAtoms->p.y);
   free(hostAtoms->p.z);

   free(hostAtoms->f.x);
   free(hostAtoms->f.y);
   free(hostAtoms->f.z);

   free(hostAtoms->U);
   free(hostAtoms->gid);
   free(hostAtoms->iSpecies);

   clReleaseMemObject(devAtoms.r.x);
   clReleaseMemObject(devAtoms.r.y);
   clReleaseMemObject(devAtoms.r.z);

   clReleaseMemObject(devAtoms.p.x);
   clReleaseMemObject(devAtoms.p.y);
   clReleaseMemObject(devAtoms.p.z);

   clReleaseMemObject(devAtoms.f.x);
   clReleaseMemObject(devAtoms.f.y);
   clReleaseMemObject(devAtoms.f.z);

   clReleaseMemObject(devAtoms.U);
   clReleaseMemObject(devAtoms.gid);
   clReleaseMemObject(devAtoms.iSpecies);

}

void freeAtomsAos(HostAtomsAos* hostAtoms, DevAtomsAos devAtoms)
{
   free(hostAtoms->r);
   free(hostAtoms->p);
   free(hostAtoms->f);
   free(hostAtoms->U);
   free(hostAtoms->gid);
   free(hostAtoms->iSpecies);

   clReleaseMemObject(devAtoms.r);
   clReleaseMemObject(devAtoms.p);
   clReleaseMemObject(devAtoms.f);
   clReleaseMemObject(devAtoms.U);
   clReleaseMemObject(devAtoms.gid);
   clReleaseMemObject(devAtoms.iSpecies);
}

void putHaloAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa* devAtoms)
{
   putVec(hostAtoms->r, devAtoms->r, hostAtoms->haloRealSize, hostAtoms->localRealSize);
   putVec(hostAtoms->p, devAtoms->p, hostAtoms->haloRealSize, hostAtoms->localRealSize);
}

void putHaloAtomsAos(HostAtomsAos* hostAtoms, DevAtomsAos* devAtoms)
{
   oclCopyToDevice(hostAtoms->r, devAtoms->r, hostAtoms->haloRealSize*r3, hostAtoms->localRealSize*r3);
   oclCopyToDevice(hostAtoms->p, devAtoms->p, hostAtoms->haloRealSize*r3, hostAtoms->localRealSize*r3);
}
