#ifndef CLTYPES_H
#define CLTYPES_H

#include "cl_utils.h"
#include "mytype.h"

typedef struct HostVecSt 
{
   cl_real* x;
   cl_real* y;
   cl_real* z;
} HostVec;

typedef struct DevVecSt 
{
   cl_mem x;
   cl_mem y;
   cl_mem z;
} DevVec;

void createDevVec(DevVec *a_D, int arraySize);

void getVector(cl_mem ax_D, cl_mem ay_D, cl_mem az_D,
      cl_real* ax_H, cl_real* ay_H, cl_real* az_H,
      int arraySize, int offset);

void putVec(HostVec a_H, DevVec a_D, int arraySize, int offset);

void putVector(cl_real* ax_H, cl_real* ay_H, cl_real* az_H, cl_mem ax_D, cl_mem ay_D, cl_mem az_D, int arraySize, int offset);

void getVec(DevVec a_D, HostVec a_H, int arraySize, int offset);

void getVecAos(cl_mem a_D, cl_real4* a_H, int arraySize, int offset);

#endif
