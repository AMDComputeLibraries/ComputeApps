#include "deformation.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>

#include "CoMDTypes.h"
#include "memUtils.h"

void printTensor(int step, real_t* mat9)
{
   //fprintf(stressOut, "[ %g %g %g] \n [%g %g %g] \n [%g %g %g] \n",
   fprintf(stressOut, "%6d % 12.6g % 12.6g % 12.6g % 12.6g % 12.6g % 12.6g % 12.6g % 12.6g % 12.6g \n",
         step,
         mat9[0], mat9[1], mat9[2],
         mat9[3], mat9[4], mat9[5],
         mat9[6], mat9[7], mat9[8]);
}

// form the matrix-vector product in place
void matVec3 (real_t *mat, real_t *vec)
{
   real_t prod[3];

   prod[0] = mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2];
   prod[1] = mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2];
   prod[2] = mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2];

   vec[0] = prod[0];
   vec[1] = prod[1];
   vec[2] = prod[2];
}


void matInv3x3 (real_t *in, real_t *out)
{
   out[0] = in[8]*in[4] - in[7]*in[5];
   out[1] = in[7]*in[2] - in[8]*in[1];
   out[2] = in[5]*in[1] - in[4]*in[2];
   out[3] = in[6]*in[5] - in[8]*in[3];
   out[4] = in[8]*in[0] - in[6]*in[2];
   out[5] = in[3]*in[2] - in[5]*in[0];
   out[6] = in[7]*in[3] - in[6]*in[4];
   out[7] = in[6]*in[1] - in[7]*in[0];
   out[8] = in[4]*in[0] - in[3]*in[1];

   real_t det = in[0]*(in[8]*in[4] - in[7]*in[5])
      - in[3]*(in[8]*in[1] - in[7]*in[2])
      + in[6]*(in[5]*in[1] - in[4]*in[2]);

   // add the determinant to the tensor 
   in[9] = det;

   real_t invDet = 1.0/det;

   printf("Determinant, inverse; %g, %g\n", det, invDet);

   for (int m = 0; m<9; m++) 
   {
      out[m] = out[m]*invDet;
   }
   // add the determinant to the inverse
   out[9] = invDet;

   real_t test[9];

   for (int i = 0;i<3;i++) 
   {
      for (int j = 0;j<3;j++) 
      {
         int m = 3*i + j;
         test[m] = 0.0;
         for (int k = 0; k<3; k++) 
         {
            test[m] +=in[3*i + k]*out[3*k + j];
         }
      }

   }
   printf("Input matrix:\n");
   printTensor(0, in);

   printf("Output matrix:\n");
   printTensor(0, out);

   printf("Test matrix:\n");
   printTensor(0, test);
}

// initialize the strain, inverse strain tensors
Deformation* initDeformation(SimFlat* sim, real_t defGrad)
{
   Deformation* defInfo = comdMalloc(sizeof(Deformation));

   defInfo->defGrad = defGrad;

   real_t testMat[10];
   real_t testInv[10];
   for (int m = 0; m<10;m++) 
   {
      testMat[m] = 0.0;
   }
   // diagonal terms
   // 1,1 term is deformation gradient
   testMat[0] = defGrad;
   testMat[4] = 1.0;
   testMat[8] = 1.0;
   //
   testMat[1] = 0.0;
   testMat[3] = 0.0;

   matInv3x3(testMat, testInv);

   // transfer to sim structs, including determinant values
   printf("Writing to strain tensors\n");
   for (int m = 0; m<10; m ++) 
   {
      defInfo->strain[m] = testMat[m];
      defInfo->invStrain[m] = testInv[m];
      defInfo->stress[m] = 0.0;
   }
   defInfo->globalVolume = 
      sim->domain->globalExtent[0]*
      sim->domain->globalExtent[1]*
      sim->domain->globalExtent[2];
   printf("done\n");
   return defInfo;
}

void forwardDeformation(SimFlat *s)
{
   for (int iBox=0; iBox < s->boxes->nTotalBoxes; iBox++)
   {
      // apply deformation to all particles
      for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++)
      {
	 /* loop over atoms in iBox */
	 real_t atomPos[3];
	 for (int m=0; m<3; m++)
	 {
	    atomPos[m] = s->atoms->r[iOff][m];
	 }
	 matVec3(s->defInfo->strain, atomPos);
	 for (int m=0; m<3; m++)
	 {
	    s->atoms->r[iOff][m] = atomPos[m];
	 }
      }
   }

}

void reverseDeformation(SimFlat *s)
{
   for (int iBox=0; iBox < s->boxes->nTotalBoxes; iBox++)
   {
      // apply deformation to all particles
      for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++)
      {
	 /* loop over atoms in iBox */
	 real_t atomPos[3];
	 for (int m=0; m<3; m++)
	 {
	    atomPos[m] = s->atoms->r[iOff][m];
	 }
	 matVec3(s->defInfo->invStrain, atomPos);
	 for (int m=0; m<3; m++)
	 {
	    s->atoms->r[iOff][m] = atomPos[m];
	 }
      }
   }

}
