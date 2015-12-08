#ifndef DECOMPOSITIONCL_H
#define DECOMPOSITIONCL_H

typedef struct HostDomainSt
{
   int procGrid[3];
   int procCoord[3];

   cl_real* globalMin;
   cl_real* globalMax;
   cl_real* globalExtent;

   cl_real* localMin;
   cl_real* localMax;
   cl_real* localExtent;

} HostDomain;

typedef struct DevDomainSt
{
   cl_mem procGrid;
   cl_mem procCoord;

   cl_mem globalMin;
   cl_mem globalMax;
   cl_mem globalExtent;

   cl_mem localMin;
   cl_mem localMax;
   cl_mem localExtent;

} DevDomain;

HostDomain* initHostDomainSoa(SimFlat* sim);

HostDomain* initHostDomainAos(SimFlat* sim);

void createDevDomainSoa(DevDomain* devDomain);

void createDevDomainAos(DevDomain* devDomain);

#endif
