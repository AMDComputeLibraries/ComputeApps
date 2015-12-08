#ifndef TIMESTEPCL_H
#define TIMESTEPCL_H

#include "cl_utils.h"
#include "CoMDTypes.h"

struct DevSimSoaSt;
struct DevSimAosSt;

void setAvArgsSoa(cl_kernel AdvanceVelocity, struct DevSimSoaSt* simDevSoa, cl_real dt);

void setApArgsSoa(cl_kernel advancePosition, struct DevSimSoaSt* simDevSoa, cl_real dt);

void setAvArgsAos(cl_kernel AdvanceVelocity, struct DevSimAosSt* simDevSoa, cl_real dt);

void setApArgsAos(cl_kernel advancePosition, struct DevSimAosSt* simDevSoa, cl_real dt);

#endif
