#ifndef ATOMSCL_H
#define ATOMSCL_H

#include "clTypes.h"
#include "cl_utils.h"
//#include "ComDTypes.h"
#include "CoMDTypes.h"

typedef struct HostAtomsSoaSt
{
   cl_int nLocal;
   cl_int nGlobal;

   cl_int* gid;
   cl_int* iSpecies;

   HostVec r;
   HostVec p;
   HostVec f;
   cl_real* U;

   int totalRealSize;
   int localRealSize;
   int haloRealSize;

   int totalIntSize;
   int localIntSize;
   int haloIntSize;

} HostAtomsSoa;

typedef struct HostAtomsAosSt
{
   cl_int nLocal;
   cl_int nGlobal;

   cl_int* gid;
   cl_int* iSpecies;

   cl_real4* r;
   cl_real4* p;
   cl_real4* f;
   cl_real* U;

   int totalRealSize;
   int localRealSize;
   int haloRealSize;

   int totalIntSize;
   int localIntSize;
   int haloIntSize;

} HostAtomsAos;

typedef struct DevAtomsSoaSt
{
   cl_int nLocal;
   cl_int nGlobal;

   cl_mem gid;
   cl_mem iSpecies;

   DevVec r;
   DevVec p;
   DevVec f;
   cl_mem U;

} DevAtomsSoa;

typedef struct DevAtomsAosSt
{
   cl_int nLocal;
   cl_int nGlobal;

   cl_mem gid;
   cl_mem iSpecies;

   cl_mem r;
   cl_mem p;
   cl_mem f;
   cl_mem U;

} DevAtomsAos;

HostAtomsSoa* initHostAtomsSoa(SimFlat* sim);

void createDevAtomsSoa(DevAtomsSoa* atoms, HostAtomsSoa* hostAtoms);

void putAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa* devAtoms);

void getAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa* devAtoms);

void freeAtomsAos(HostAtomsAos* hostAtoms, DevAtomsAos devAtoms);

void putHaloAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa* devAtoms);

void atomsToSoa(SimFlat* sim, HostAtomsSoa* atoms);

void atomsToSim(SimFlat* sim, HostAtomsSoa* atoms);

HostAtomsAos* initHostAtomsAos(SimFlat* sim);

void createDevAtomsAos(DevAtomsAos* atoms, HostAtomsAos* hostAtoms);

void putAtomsAos(HostAtomsAos* hostAtoms, DevAtomsAos* devAtoms);

void freeAtomsSoa(HostAtomsSoa* hostAtoms, DevAtomsSoa devAtoms);

void putHaloAtomsAos(HostAtomsAos* hostAtoms, DevAtomsAos* devAtoms);

#endif

