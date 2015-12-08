/*******************************************************************************
Copyright (c) 2015 Advanced Micro Devices, Inc. 

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/** Kernels for computing the EAM potential
  Since we can only block on kernel completion, the 3 sweeps done in the original code 
  need to be implemented as 3 separate kernels. 
  Note also that the potential arrays are large enough to require accessing them from 
  global memory.
  Since OpenCL doesn't pick up #include properly, we need to manually switch real_t from 
  float to double in each kernel file individually.

  Note: More careful analysis shows we can consolidate kernels 1 and 2 into a single pass;
  there is a flag PASS_2 in this file which should be set to match the flag PASS_2 in 
  helpers.c. Switching this flag allows you to test that a) the results match and 2) evaluate
  the overhead of adding the extra kernel which only loops over all particles.
 **/

//Initial implementation of the MD code
#define N_MAX_ATOMS 64
#define N_MAX_NEIGHBORS 27
#define PERIODIC 1

#define KERN_DIAG 0
#define USE_SPLINE 0
#define PASS_2 0 // this should match the setting in helpers.c

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* CL_REAL_T is set to single or double depending on compile time flags */
typedef CL_REAL_T real_t; 
typedef CL_REAL4_T cl_real4;

int binSearch(global int *boxes, int n, int nboxes, int tot_atoms)
{
	int l = 0, r = nboxes - 1, mid = 0, box = 0;

	while(l <= r)
	{
		mid = (l + r)/2;
		
		if(n >= boxes[mid-1] && n < boxes[mid])
		{
			box = mid;
			break;
		}
		else if(mid == (nboxes - 1) && n >= boxes[mid] && n < tot_atoms)
		{
			box = mid + 1;
			break;
		}
		else if(boxes[mid] > n)
			r = mid - 1;
		else
			l = mid + 1;
	}

	return (box == 0) ? 0 : box - 1;
}

#if (USE_CHEBY)
// given a list of Chebyshev coefficients c, compute the value at x
// x must be in the range [a, b]
// call signature might change to put it in line with the regular EAM call
real_t chebev(
        real_t a, 
        real_t b, 
        __global const real_t *c,
        int m, 
        real_t x) 
{
   real_t d, dd, sv, y, y2;
   real_t ch;
#if(KERN_DIAG > 0) 
   if ((x-a)*(x-b) > 0.0)
   {
      printf("x not in range in chebev, %f\n", x);
   }
#endif
   d=0.0;
   dd=0.0;
   y=(2.0*x-a-b)/(b-a);
   y2=2.0*y;
   for (int j=m-1;j>0;j--)
   {
      sv=d;
      d=y2*d-dd+c[j];
      dd=sv;
   }
   ch=y*d-dd+0.5*c[0];
   return ch;
}
/* Chebyshev interpolation routine modified to match the table lookup signature.
 * Returns the value and derivative, ch and dch, at the position x
 */
void eamCheby(
	real_t x,
        __global const real_t* c,
        const int m, 
	real_t *ch,
	real_t *dch) 
{
    // coefficients array c has Chebyshev coeffs, derivative coeffs and range limits
   real_t a = c[2*m + 0];
   real_t b = c[2*m + 1];

   real_t d, dd, sv, y, y2;
#if(KERN_DIAG > 0) 
   if ((x-a)*(x-b) > 0.0)
   {
      printf("x not in range in chebev, %f\n", x);
   }
#endif
   // compute the value
   d=0.0;
   dd=0.0;
   y=(2.0*x-a-b)/(b-a);
   y2=2.0*y;
   for (int j=m-1;j>0;j--)
   {
      sv=d;
      d=y2*d-dd+c[j];
      dd=sv;
   }
   *ch=y*d-dd+0.5*c[0];

    // compute the derivative
    // identical to above but c values offset by m
   d=0.0;
   dd=0.0;
   y=(2.0*x-a-b)/(b-a);
   y2=2.0*y;
   for (int j=m-1;j>0;j--)
   {
      sv=d;
      d=y2*d-dd+c[j+m];
      dd=sv;
   }
   *dch=y*d-dd+0.5*c[0+m];
}

#endif

#if (USE_SPLINE)
void PhiSpline(real_t r, real_t *f, real_t *df)
{
    // values for copper
    /* 
    // original values from 87 paper
    real_t a_k[6] = { 29.059214 , -140.05681 , 130.07331 , -17.48135 , 31.82546 , 71.58749};
    real_t r_k[6] = { 1.2247449 , 1.1547054 , 1.1180065 , 1.0000000 , 0.8660254 , 0.7071068};
    */
    // new values for smoother potential
    real_t a_k[6] = {61.73525861, -108.18467800, 57.00053948,-12.88796578, 39.16381901, 0.0};
    real_t r_k[6] = {1.225, 1.202, 1.154, 1.050, 0.866, 0.707};
    real_t az = 1.0*3.615;
    real_t az3 = az*az*az;
    real_t az2 = az*az;
    // set output values to zero
    *f=0.0;
    *df=0.0;
    //r = 1.0; //3.615;
    r = r/3.615;
    //printf("r = %e\n", r);
    // sum over all coefficients
    for (int k=0;k<6;k++)
    {
	r_k[k] = r_k[k]*3.615;
	if (r < r_k[k])
	{
	    *f += (r_k[k]-r)*(r_k[k]-r)*(r_k[k]-r)*a_k[k]/az3;
	    *df -= 3.0*(r_k[k]-r)*(r_k[k]-r)*a_k[k]/az2;
	}
    }
}

void RhoSpline(real_t r, real_t *f, real_t *df)
{
    // values for copper
    /*
    // original values from 87 paper
    real_t R_k[2] = { 1.2247449 , 1.0000000 };
    real_t A_k[2] = { 9.806694 , 16.774638 };
    */
    // new values for smoother potential
    real_t R_k[2] = { 1.225, 0.990 };
    real_t A_k[2] = { 10.03718305, 17.06363299 };
    real_t az = 1.0*3.615;
    real_t az3 = az*az*az;
    real_t az2 = az*az;
    // set output values to zero
    *f=0.0;
    *df=0.0;
    //r = 1.0; //3.615;
    r = r/3.615;
    // sum over all coefficients
    for (int k=0;k<2;k++)
    {
	R_k[k] = R_k[k]*3.615;
	if (r < R_k[k])
	{
	    *f += (R_k[k]-r)*(R_k[k]-r)*(R_k[k]-r)*A_k[k]/az3;
	    *df -= 3.0*(R_k[k]-r)*(R_k[k]-r)*A_k[k]/az2;
	}
    }
}

void FSpline(real_t rho, real_t *f, real_t *df)
{
    *f = -1.0*sqrt(rho);
    *df = -0.5/(*f);
}

#endif

void eamInterpolateDeriv(real_t r,
	__global const real_t* values,
	const int nValues,
	real_t *value1, 
	real_t *f1)
{
    int i1;
    int i;
    real_t gi, gi1;

    // extract values from potential 'struct'
    real_t x0 = values[nValues+3];
    real_t xn = values[nValues+4];
    real_t invDx = values[nValues+5];

    // identical to Sriram's loop in eam.c
    if ( r<x0) r = x0;
    else if (r>xn) r = xn;

    r = (r-x0)*(invDx) ;
    i1 = (int)trunc((float)r);

    /* reset r to fractional distance */
    r = r - (int)trunc((float)r);

    // all indices shifted up by one compared to the original code
    gi  = values[i1+2] - values[i1];
    gi1 = values[i1+3] - values[i1+1];

    // Note the shift removes [i1-1] as a possibility
    // values[i1-1] is guaranteed(?) inbounds because 
    // a->x0 = x0 + (xn-x0)/(double)n; 
    // appears in allocPotentialArray
    *value1 = values[i1+1] + 0.5*r*(
	    r*(values[i1+2]+ values[i1] -2.0*values[i1+1]) +
	    gi
	    );
    if(i1<=1) 
	*f1 = 0.0;
    else 
	*f1 = 0.5*(gi + r*(gi1-gi))*invDx;

    return;

}

// Simple version without local blocking to check for correctness
__kernel void EAM_Force_1(
	__global real_t* xPos,
	__global real_t* yPos,
	__global real_t* zPos,

	__global real_t* fx,
	__global real_t* fy,
	__global real_t* fz,

	__global real_t* energy,
	__global real_t* rho,
	__global real_t* rhobar,

	__global const int* neighborList,
	__global const int* nNeighbors,
	__global const int* nAtoms,

	__global const int* nValues,

	__global const real_t* phi_pot, // the potentials are passed in as real arrays: x0, xn, invDx, values[?]
	__global const real_t* rho_pot,
	__global const real_t* F_pot,

	const real_t cutoff,
	const int boxSize,
	global int *boxes
	) 
{
#if USE_SPLINE
    // values for copper
    real_t a_k[6] = { 29.059214 , -140.05681 , 130.07331 , -17.48135 , 31.82546 , 71.58749};
    real_t r_k[6] = { 1.2247449 , 1.1547054 , 1.1180065 , 1.0000000 , 0.8660254 , 0.7071068};

    real_t R_k[2] = { 1.2247449 , 1.0000000 };
    real_t A_k[2] = { 9.806694 , 16.774638 };
#endif

//    int iAtom = get_global_id(0);
//    int iBox = get_global_id(1);

	 // prefix sum array to compute iBox
	int tid = get_global_id(0);
	int iBox = binSearch(boxes, tid, boxSize, get_global_size(0));

	int iAtom = tid - boxes[iBox];

    real_t dx, dy, dz;
    real_t r, r2, r6;
    real_t fr, e_i;
    real_t rho_i;

    // accumulate local force value
    real_t fx_i, fy_i, fz_i;

    real_t rCut = cutoff;
    real_t rCut2 = rCut*rCut;
    real_t rhoTmp;
    real_t phiTmp;
    real_t dPhi, dRho;
    real_t fi, fiprime;

    int i;
    int j_local;

    int i_offset;
    int iParticle;

    // zero out forces on particles
    i_offset = iBox*N_MAX_ATOMS;
    iParticle = i_offset + iAtom;

    fx_i = 0.0;
    fy_i = 0.0;
    fz_i = 0.0;

    rho_i = 0.0;

    e_i = 0.0;

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box

#if(KERN_DIAG > 0) 
	printf("i = %d, %f, %f, %f\n", iParticle, xPos[iParticle], yPos[iParticle], zPos[iParticle]);

	printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
#endif

	for (int j = 0; j<nNeighbors[iBox]; j++)
	{// loop over neighbor cells
	    int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
	    int jOffset = jBox*N_MAX_ATOMS;

	    for (int jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
	    {// loop over all groups in neighbor cell 

		int jParticle = jOffset + jAtom; // global offset of particle

		dx = xPos[iParticle] - xPos[jParticle];
		dy = yPos[iParticle] - yPos[jParticle];
		dz = zPos[iParticle] - zPos[jParticle];

#if(KERN_DIAG > 0) 
		printf("dx, dy, dz = %f, %f, %f\n", dx, dy, dz);
		printf("i = %d, j = %d, %f, %f, %f\n", iParticle, jParticle, xPos[jParticle], yPos[jParticle], zPos[jParticle]);
#endif

		r2 = dx*dx + dy*dy + dz*dz;

		if ( r2 <= rCut2 && r2 > 0.0)
		{// no divide by zero

#if(KERN_DIAG > 0) 
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
		    printf("r2, rCut2 = %f, %f\n", r2, rCut2);
#endif

		    r = sqrt(r2);
		    //r = 1.0;

/*
#if(USE_SPLINE)
		    //r = r*r_k[0]/cutoff;
		    //r = r/3.615;
		    PhiSpline(r, &phiTmp, &dPhi);
		    RhoSpline(r, &rhoTmp, &dRho);
		    */
#if (USE_CHEBY)
		    eamCheby(r, phi_pot, nValues[0], &phiTmp, &dPhi);
		    eamCheby(r, rho_pot, nValues[1], &rhoTmp, &dRho);
#else
		    eamInterpolateDeriv(r, phi_pot, nValues[0], &phiTmp, &dPhi);
		    eamInterpolateDeriv(r, rho_pot, nValues[1], &rhoTmp, &dRho);
#endif

#if(KERN_DIAG > 0) 
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
		    printf("iParticle = %d, jParticle = %d, phiTmp = %f, dPhi = %f\n", iParticle, jParticle, phiTmp, dPhi);
		    printf("iParticle = %d, jParticle = %d, rhoTmp = %f, dRho = %f\n", iParticle, jParticle, rhoTmp, dRho);
#endif

#if(USE_SPLINE)
		    fx_i += (dRho+dPhi)*dx/r;
		    fy_i += (dRho+dPhi)*dy/r;
		    fz_i += (dRho+dPhi)*dz/r;
#else
		    fx_i += dPhi*dx/r;
		    fy_i += dPhi*dy/r;
		    fz_i += dPhi*dz/r;
#endif

		    e_i += phiTmp;

		    rho_i += rhoTmp;

		} else {
		}


	    } // loop over all atoms
	} // loop over neighbor cells

	fx[iParticle] = fx_i;
	fy[iParticle] = fy_i;
	fz[iParticle] = fz_i;

	// since we loop over all particles, each particle contributes 1/2 the pair energy to the total
	//energy[iParticle] = e_i*0.5;

	rho[iParticle] = rho_i;

	 // we can actually include the Force_2 kernel here, to save some time!
	 // skip the next 4 lines if PASS_2 = 1 in helpers.c
#if(PASS_2 == 0)
#if (USE_CHEBY)
	eamCheby(rho_i,F_pot,nValues[2],&fi,&fiprime);
#else
	eamInterpolateDeriv(rho_i,F_pot,nValues[2],&fi,&fiprime);
#endif
	rhobar[iParticle] = fiprime; // update rhoprime 
	//update energy terms 
	energy[iParticle] = e_i*0.5 + fi;
#else
	energy[iParticle] = e_i*0.5;
#endif
    } else { // zero out the energy of the other particles for safety
	energy[iParticle] = 0.0;
    }
}

__kernel void EAM_Force_2(
	__global real_t* rhobar,
	__global real_t* energy,
	__global real_t* rho,
	__global const int* nAtoms,
	__global const real_t* F_pot,
	__global const int* nValues,
	const int boxSize,
	global int *boxes
	)
{

 //   int iAtom = get_global_id(0);
//    int iBox = get_global_id(1);

	 // prefix sum array to compute iBox
	int tid = get_global_id(0);
	int iBox = binSearch(boxes, tid, boxSize, get_global_size(0));

	int iAtom = tid - boxes[iBox];

    real_t fi, fiprime;

    int i_offset;
    int iParticle;

    i_offset = iBox*N_MAX_ATOMS;
    iParticle = i_offset + iAtom;

    /*
    // local copy of F potential
    real_t F_local[nValues+3];

    // load values into local potentials
    for (int i=0;i<nValues+3;i++)
    {
    F_local[i] = F_pot[i];
    }
     */

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box


	//printf("iBox = %d, iAtom = %d\n", iBox, iAtom);

        /*
#if(USE_SPLINE)
	FSpline(rho[iParticle], &fi, &fiprime);
        */
#if (USE_CHEBY)
	eamCheby(rho[iParticle],F_pot,nValues[2],&fi,&fiprime);
#else
	eamInterpolateDeriv(rho[iParticle],F_pot,nValues[2],&fi,&fiprime);
#endif
	rhobar[iParticle] = fiprime; // update rhoprime 
	//update energy terms 
	energy[iParticle] += fi;

    }

}

__kernel void EAM_Force_3(
	__global real_t* xPos,
	__global real_t* yPos,
	__global real_t* zPos,

	__global real_t* fx,
	__global real_t* fy,
	__global real_t* fz,

	__global real_t* fi,

	__global const int* neighborList,
	__global const int* nNeighbors,
	__global const int* nAtoms,
	__global const int* nValues,

	__global const real_t* rho_pot,
	const real_t cutoff,
	const int boxSize,
	global int *boxes
	) 
{

//    int iAtom = get_global_id(0);
//    int iBox = get_global_id(1);

	 // prefix sum array to compute iBox
	int tid = get_global_id(0);
	int iBox = binSearch(boxes, tid, boxSize, get_global_size(0));

	int iAtom = tid - boxes[iBox];

    real_t dx, dy, dz;
    real_t r, r2;

    // accumulate local force value
    real_t fx_i, fy_i, fz_i;

    real_t rCut = cutoff;
    real_t rCut2 = rCut*rCut;
    real_t rhoTmp, dRho;

    int i;
    int j_local;

    real_t rTmp, rhoijprime;

    // global offset of local thread
    int i_offset = iBox*N_MAX_ATOMS;
    int iParticle = i_offset + iAtom;

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box

	// zero out forces on particles
	fx_i = fx[iParticle];
	fy_i = fy[iParticle];
	fz_i = fz[iParticle];

#if(KERN_DIAG > 0) 

	printf("i = %d, %f, %f, %f\n", iParticle, xPos[iParticle], yPos[iParticle], zPos[iParticle]);

	printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
#endif

	for (int j = 0; j<nNeighbors[iBox]; j++)
	{// loop over neighbor cells
	    int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
	    int jOffset = jBox*N_MAX_ATOMS;

	    for (int jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
	    {// loop over all groups in neighbor cell 

		int jParticle = jOffset + jAtom; // global offset of particle

		dx = xPos[iParticle] - xPos[jParticle];
		dy = yPos[iParticle] - yPos[jParticle];
		dz = zPos[iParticle] - zPos[jParticle];

#if(KERN_DIAG > 0) 
		printf("dx, dy, dz = %f, %f, %f\n", dx, dy, dz);
		printf("i = %d, j = %d, %f, %f, %f\n", iParticle, jParticle, xPos[jParticle], yPos[jParticle], zPos[jParticle]);
#endif

		r2 = dx*dx + dy*dy + dz*dz;

		if ( r2 < rCut2 && r2 > 0.0)
		{// no divide by zero

#if(KERN_DIAG > 0) 
		    printf("r2, rCut2 = %f, %f\n", r2, rCut2);
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
#endif

		    r = sqrt(r2);

/*
#if (USE_SPLINE)
		    RhoSpline(r, &rhoTmp, &dRho);
#else
		    */
#if (USE_CHEBY)
		    eamCheby(r, rho_pot, nValues[1], &rhoTmp, &dRho);
#else
		    eamInterpolateDeriv(r, rho_pot, nValues[1], &rhoTmp, &dRho);
#endif
		    rhoijprime = dRho;

#if(KERN_DIAG > 0) 
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
#endif

		    rTmp = (fi[iParticle]+fi[jParticle])*rhoijprime/r;

		    fx_i += (fi[iParticle]+fi[jParticle])*rhoijprime*dx/r;
		    fy_i += (fi[iParticle]+fi[jParticle])*rhoijprime*dy/r;
		    fz_i += (fi[iParticle]+fi[jParticle])*rhoijprime*dz/r;
		    /*
		       fx_i += rTmp*dx;
		       fy_i += rTmp*dy;
		       fz_i += rTmp*dz;
		     */

		} else {
		}


	    } // loop over all atoms in jBox
	} // loop over neighbor cells

#if (USE_SPLINE)
#else
	fx[iParticle] = fx_i;
	fy[iParticle] = fy_i;
	fz[iParticle] = fz_i;
#endif

    } // loop over all atoms in iBox
}

// AoS Versions
// Simple version without local blocking to check for correctness
__kernel void EAM_Force_1_AoS(
	__global cl_real4* pos,

	__global cl_real4* f,

	__global real_t* energy,
	__global real_t* rho,
	__global real_t* rhobar,

	__global const int* neighborList,
	__global const int* nNeighbors,
	__global const int* nAtoms,

	__global const int* nValues,

	__global const real_t* phi_pot, // the potentials are passed in as real arrays: x0, xn, invDx, values[?]
	__global const real_t* rho_pot,
	__global const real_t* F_pot,

	const real_t cutoff)
{

    int iAtom = get_global_id(0);
    int iBox = get_global_id(1);

    real_t r, r2, r6;
    real_t fr, e_i;
    real_t rho_i;

    cl_real4 dr;
    cl_real4 f_i;

    real_t rCut = cutoff;
    real_t rCut2 = rCut*rCut;
    real_t rhoTmp;
    real_t phiTmp;
    real_t dPhi, dRho;
    real_t fi, fiprime;

    int i;
    int j_local;

    int i_offset;
    int iParticle;

    i_offset = iBox*N_MAX_ATOMS;
    iParticle = i_offset + iAtom;

    // zero out forces on particles
    f_i.x = 0.0;
    f_i.y = 0.0;
    f_i.z = 0.0;

    rho_i = 0.0;

    e_i = 0.0;


    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box

#if(KERN_DIAG > 0) 
	printf("i = %d, %f, %f, %f\n", iParticle, pos[iParticle].x, pos[iParticle].y, pos[iParticle].z);

	printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
#endif

	for (int j = 0; j<nNeighbors[iBox]; j++)
	{// loop over neighbor cells
	    int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
	    int jOffset = jBox*N_MAX_ATOMS;

	    for (int jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
	    {// loop over all groups in neighbor cell 

		int jParticle = jOffset + jAtom; // global offset of particle

		dr = pos[iParticle] - pos[jParticle];

#if(KERN_DIAG > 0) 
		printf("dx, dy, dz = %f, %f, %f\n", dr.x, dr.y, dr.z);
		printf("i = %d, j = %d, %f, %f, %f\n", 
		iParticle, jParticle, pos[jParticle].x, pos[jParticle].y, pos[jParticle].z);
#endif

		r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

		if ( r2 <= rCut2 && r2 > 0.0)
		{// no divide by zero

#if(KERN_DIAG > 0) 
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
		    printf("r2, rCut2 = %f, %f\n", r2, rCut2);
#endif

		    r = sqrt(r2);

		    eamInterpolateDeriv(r, phi_pot, nValues[0], &phiTmp, &dPhi);
		    eamInterpolateDeriv(r, rho_pot, nValues[1], &rhoTmp, &dRho);

#if(KERN_DIAG > 0) 
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
		    printf("iParticle = %d, jParticle = %d, phiTmp = %f, dPhi = %f\n", iParticle, jParticle, phiTmp, dPhi);
		    printf("iParticle = %d, jParticle = %d, rhoTmp = %f, dRho = %f\n", iParticle, jParticle, rhoTmp, dRho);
#endif

		    f_i.x += dPhi*dr.x/r;
		    f_i.y += dPhi*dr.y/r;
		    f_i.z += dPhi*dr.z/r;

		    e_i += phiTmp;

		    rho_i += rhoTmp;

		} else {
		}


	    } // loop over all atoms
	} // loop over neighbor cells

	f[iParticle] = f_i;

	// since we loop over all particles, each particle contributes 1/2 the pair energy to the total
	//energy[iParticle] = e_i*0.5;

	rho[iParticle] = rho_i;

#if(PASS_2 == 0)
	eamInterpolateDeriv(rho_i,F_pot,nValues[2],&fi,&fiprime);
	rhobar[iParticle] = fiprime; // update rhoprime 
	//update energy terms 
	energy[iParticle] = e_i*0.5 + fi;
#else
	energy[iParticle] = e_i*0.5;
#endif
    }
}

__kernel void EAM_Force_2_AoS(
	__global real_t* rhobar,
	__global real_t* energy,
	__global real_t* rho,
	__global const int* nAtoms,
	__global const real_t* F_pot,
	__global const int* nValues
	)
{

    int iAtom = get_global_id(0);
    int iBox = get_global_id(1);

    real_t fi, fiprime;


    int i;

    int i_offset;
    int iParticle;

    i_offset = iBox*N_MAX_ATOMS;
    iParticle = i_offset + iAtom;

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box


#if(KERN_DIAG > 0) 
	printf("iBox = %d, iAtom = %d\n", iBox, iAtom);
#endif

	eamInterpolateDeriv(rho[iParticle],F_pot,nValues[2],&fi,&fiprime);
	rhobar[iParticle] = fiprime; // update rhoprime 
	//update energy terms 
	energy[iParticle] += fi;

    }

}
__kernel void EAM_Force_3_AoS(
	__global cl_real4* pos,

	__global cl_real4* f,

	__global real_t* fi,

	__global const int* neighborList,
	__global const int* nNeighbors,
	__global const int* nAtoms,

	__global const int* nValues,
	__global const real_t* rho_pot,
	const real_t cutoff) 
{

    int iAtom = get_global_id(0);
    int iBox = get_global_id(1);

    cl_real4 dr;
    real_t r, r2;

    cl_real4 drBox;

    // accumulate local force value
    cl_real4 f_i;

    real_t rCut = cutoff;
    real_t rCut2 = rCut*rCut;
    real_t rhoTmp, dRho;

    int i;
    int j_local;

    real_t rTmp, rhoijprime;

    // global offset of local thread
    int i_offset = iBox*N_MAX_ATOMS;
    int iParticle = i_offset + iAtom;

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box

	// zero out forces on particles
	f_i.x = f[iParticle].x;
	f_i.y = f[iParticle].y;
	f_i.z = f[iParticle].z;

#if(KERN_DIAG > 0) 
	printf("i = %d, %f, %f, %f\n", iParticle, pos[iParticle].x, pos[iParticle].y, pos[iParticle].z);
	printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
#endif

	for (int j = 0; j<nNeighbors[iBox]; j++)
	{// loop over neighbor cells
	    int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
	    int jOffset = jBox*N_MAX_ATOMS;

	    for (int jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
	    {// loop over all groups in neighbor cell 

		int jParticle = jOffset + jAtom; // global offset of particle

		dr = pos[iParticle] - pos[jParticle];

#if(KERN_DIAG > 0) 
		printf("dx, dy, dz = %f, %f, %f\n", dr.x, dr.y, dr.z);
		printf("i = %d, j = %d, %f, %f, %f\n", iParticle, jParticle, pos[jParticle].x, pos[jParticle].y, pos[jParticle].z);
#endif

		r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

		if ( r2 < rCut2 && r2 > 0.0)
		{// no divide by zero

#if(KERN_DIAG > 0) 
		    printf("r2, rCut = %f, %f\n", r2, rCut);
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
#endif

		    r = sqrt(r2);

		    eamInterpolateDeriv(r, rho_pot, nValues[1], &rhoTmp, &dRho);
		    rhoijprime = dRho;

#if(KERN_DIAG > 0) 
		    printf("%d, %d, %f\n", iParticle, jParticle, r2);
#endif

		    rTmp = (fi[iParticle]+fi[jParticle])*rhoijprime/r;

		    /*
		    f_i.x += (fi[iParticle]+fi[jParticle])*rhoijprime*dr.x/r;
		    f_i.y += (fi[iParticle]+fi[jParticle])*rhoijprime*dr.y/r;
		    f_i.z += (fi[iParticle]+fi[jParticle])*rhoijprime*dr.z/r;
		     */
		       f_i.x += rTmp*dr.x;
		       f_i.y += rTmp*dr.y;
		       f_i.z += rTmp*dr.z;

		} else {
		}


	    } // loop over all atoms in jBox
	} // loop over neighbor cells

	f[iParticle] = f_i;

    } // loop over all atoms in iBox
}

