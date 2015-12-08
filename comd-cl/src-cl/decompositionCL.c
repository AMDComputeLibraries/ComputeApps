#include "helpers.h"

HostDomain* initHostDomainSoa(SimFlat* sim)
{
   HostDomain* domain;

   domain = malloc(sizeof(HostDomain));
   domain->localMin    = malloc(3*sizeof(cl_real));
   domain->localMax    = malloc(3*sizeof(cl_real));
   for (int j=0;j<3;j++)
   {
      domain->localMin[j]  = sim->domain->localMin[j];
      domain->localMax[j]  = sim->domain->localMax[j];
   }
   return domain;
}

HostDomain* initHostDomainAos(SimFlat* sim)
{
   HostDomain* domain;

   domain = malloc(sizeof(HostDomain));
   domain->localMin     = malloc(sizeof(cl_real4));
   domain->localMax     = malloc(sizeof(cl_real4));
   for (int j=0;j<3;j++)
   {
      domain->localMin[j]  = sim->domain->localMin[j];
      domain->localMax[j]  = sim->domain->localMax[j];
   }
   return domain;
}

void createDevDomainSoa(DevDomain* devDomain)
{
   oclCreateReadWriteBuffer(&devDomain->localMin, sizeof(cl_real)*3);
   oclCreateReadWriteBuffer(&devDomain->localMax, sizeof(cl_real)*3);
}

void createDevDomainAos(DevDomain* devDomain)
{
   oclCreateReadWriteBuffer(&devDomain->localMin, sizeof(cl_real4));
   oclCreateReadWriteBuffer(&devDomain->localMax, sizeof(cl_real4));
}


