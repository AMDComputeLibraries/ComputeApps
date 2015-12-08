#ifndef LJCL_H
#define LJCL_H

#include "cl_utils.h"
#include "CoMDTypes.h"

struct DevSimSoaSt;
struct DevSimAosSt;

typedef struct HostLjPotSt 
{
   cl_real cutoff;
   cl_real sigma;
   cl_real epsilon;
} HostLjPot;

typedef struct DevLjPotSt 
{
   cl_real cutoff;
   cl_real sigma;
   cl_real epsilon;
} DevLjPot;

void setLjArgsSoa(cl_kernel ljForce, struct DevSimSoaSt* simDev);

void setLjArgsAos(cl_kernel ljForce, struct DevSimAosSt* simDev);

void initHostLj(HostLjPot *ljPot_H, SimFlat *sim);

#endif
