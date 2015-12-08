#ifndef __DEFORMATION_H_
#define __DEFORMATION_H_

#include "CoMDTypes.h"

FILE* stressOut;

void printTensor(int step, real_t* mat9);
void matVec3(real_t *mat, real_t *vec);
void matInv3x3 (real_t *in, real_t *out);
void forwardDeformation(SimFlat *s);
void reverseDeformation(SimFlat *s);

Deformation* initDeformation(SimFlat *sim, real_t defGrad);

#endif
