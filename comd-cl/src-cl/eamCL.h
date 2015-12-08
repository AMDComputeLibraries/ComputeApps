#ifndef EAMCL_H
#define EAMCL_H

struct DevSimSoaSt;
struct DevSimAosSt;

typedef struct HostEamPotSt 
{
   // table values
   cl_real* rho;
   cl_real* phi;
   cl_real* F;
   cl_int* nValues;
   // extra atom arrays for EAM
   cl_real* rhobar;
   cl_real* dfEmbed;
   cl_real cutoff;
   int rhoPotSize;
   int phiPotSize;
   int fPotSize;
   // size of the atom arrays 
   int totalRealSize;
} HostEamPot;

typedef struct DevEamPotSt 
{
   // table values
   cl_mem rho;
   cl_mem phi;
   cl_mem F;
   cl_mem nValues;
   // extra atom arrays for EAM
   cl_mem rhobar;
   cl_mem dfEmbed;
   cl_real cutoff;
} DevEamPot;

typedef struct HostEamChSt 
{
   cl_real* rho;
   cl_real* phi;
   cl_real* F;
   cl_int* nValues;
   cl_real cutoff;
   int rhoChebSize;
   int phiChebSize;
   int fChebSize;
} HostEamCh;

typedef struct DevEamChSt 
{
   cl_mem rho;
   cl_mem phi;
   cl_mem F;
   cl_mem nValues;
   cl_real cutoff;
} DevEamCh;

void initHostEam(HostEamPot *hostEamPot, SimFlat *sim);

void initDevEam(HostEamPot* hostEamPot, DevEamPot* devEamPot);

void putEamPot(HostEamPot hostEamPot, DevEamPot devEamPot);

void setEamArgsSoa(cl_kernel *forceKernels, struct DevSimSoaSt* devSimSoa);

void setEamArgsAos(cl_kernel *forceKernels, struct DevSimAosSt* devSimSoa);

#endif
