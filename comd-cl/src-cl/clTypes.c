#include "helpers.h"

void createDevVec(DevVec *aDev, int arraySize) 
{
   oclCreateReadWriteBuffer(&aDev->x, arraySize);
   oclCreateReadWriteBuffer(&aDev->y, arraySize);
   oclCreateReadWriteBuffer(&aDev->z, arraySize);
}

void getVecAos(cl_mem aDev,
      cl_real4* aHost,
      int arraySize,
      int offset)
{
   oclCopyToHost(aDev, aHost, r3*arraySize, offset);
}

void getVec(DevVec aDev,
      HostVec aHost,
      int arraySize,
      int offset)
{
   oclCopyToHost(aDev.x, aHost.x, arraySize, offset);
   oclCopyToHost(aDev.y, aHost.y, arraySize, offset);
   oclCopyToHost(aDev.z, aHost.z, arraySize, offset);
}

void getVector(cl_mem axDev, cl_mem ayDev, cl_mem azDev,
      cl_real* axHost, cl_real* ayHost, cl_real* azHost,
      int arraySize,
      int offset)
{
   oclCopyToHost(axDev, axHost, arraySize, offset);
   oclCopyToHost(ayDev, ayHost, arraySize, offset);
   oclCopyToHost(azDev, azHost, arraySize, offset);
}

void putVector(
      cl_real* axHost, cl_real* ayHost, cl_real* azHost,
      cl_mem axDev, cl_mem ayDev, cl_mem azDev,
      int arraySize,
      int offset)
{
   oclCopyToDevice(axHost, axDev, arraySize, offset);
   oclCopyToDevice(ayHost, ayDev, arraySize, offset);
   oclCopyToDevice(azHost, azDev, arraySize, offset);
}

void putVec(
      HostVec aHost,
      DevVec aDev,
      int arraySize,
      int offset)
{
   oclCopyToDevice(aHost.x, aDev.x, arraySize, offset);
   oclCopyToDevice(aHost.y, aDev.y, arraySize, offset);
   oclCopyToDevice(aHost.z, aDev.z, arraySize, offset);
}

