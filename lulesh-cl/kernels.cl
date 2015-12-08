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

#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
//#define XI_M        0x003
//#define XI_M_SYMM   0x001
//#define XI_M_FREE   0x002
//
//#define XI_P        0x00c
//#define XI_P_SYMM   0x004
//#define XI_P_FREE   0x008
//
//#define ETA_M       0x030
//#define ETA_M_SYMM  0x010
//#define ETA_M_FREE  0x020
//
//#define ETA_P       0x0c0
//#define ETA_P_SYMM  0x040
//#define ETA_P_FREE  0x080
//
//#define ZETA_M      0x300
//#define ZETA_M_SYMM 0x100
//#define ZETA_M_FREE 0x200
//
//#define ZETA_P      0xc00
//#define ZETA_P_SYMM 0x400
//#define ZETA_P_FREE 0x800
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000

#define X 0
#define Y 1
#define Z 2

#define MINEQ(a,b) (a)=(((a)<(b))?(a):(b))

/* Could also support fixed point and interval arithmetic types */
typedef float        real4 ;
#ifdef SINGLE
typedef real4		 real8 ;
#else
typedef double       real8 ;
#endif

typedef int    Index_t ; /* array subscript and loop index */
typedef real8  Real_t ;  /* floating point representation */
typedef int    Int_t ;   /* integer representation */

inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline real8  FABS(real8  arg) { return fabs(arg) ; }
inline real8  FMAX(real8  arg1,real8  arg2) { return fmax(arg1,arg2) ; }

#define VOLUDER(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)		\
{									\
  const Real_t twelfth = (Real_t)(1.0) / (Real_t)(12.0) ;			\
									\
   *dvdc= 								\
     ((a1) + (a2)) * ((b0) + (b1)) - ((a0) + (a1)) * ((b1) + (b2)) +	\
     ((a0) + (a4)) * ((b3) + (b4)) - ((a3) + (a4)) * ((b0) + (b4)) -	\
     ((a2) + (a5)) * ((b3) + (b5)) + ((a3) + (a5)) * ((b2) + (b5));	\
   *dvdc *= twelfth;							\
}

__constant Real_t gamma[4][8] =
{
    { +1, +1, -1, -1, -1, -1, +1, +1 }, 
    { +1, -1, -1, +1, -1, +1, +1, -1 }, 
    { +1, -1, +1, -1, +1, -1, +1, -1 }, 
    { -1, +1, -1, +1, +1, -1, +1, -1 }
};

static inline
void CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = (Real_t)(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = (Real_t)(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = (Real_t)(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = (Real_t)(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = (Real_t)(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = (Real_t)(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = (Real_t)(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = (Real_t)(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = (Real_t)(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */

  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = (Real_t)(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static inline
void CalcElemShapeFunctionDerivatives_g( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       __global Real_t* const volume )
{
	/*
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;
  */

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = (Real_t)(.125) * ( (x[6]-x[0]) + (x[5]-x[3]) - (x[7]-x[1]) - (x[4]-x[2]) );
  fjxet = (Real_t)(.125) * ( (x[6]-x[0]) - (x[5]-x[3]) + (x[7]-x[1]) - (x[4]-x[2]) );
  fjxze = (Real_t)(.125) * ( (x[6]-x[0]) + (x[5]-x[3]) + (x[7]-x[1]) + (x[4]-x[2]) );

  fjyxi = (Real_t)(.125) * ( (y[6]-y[0]) + (y[5]-y[3]) - (y[7]-y[1]) - (y[4]-y[2]) );
  fjyet = (Real_t)(.125) * ( (y[6]-y[0]) - (y[5]-y[3]) + (y[7]-y[1]) - (y[4]-y[2]) );
  fjyze = (Real_t)(.125) * ( (y[6]-y[0]) + (y[5]-y[3]) + (y[7]-y[1]) + (y[4]-y[2]) );

  fjzxi = (Real_t)(.125) * ( (z[6]-z[0]) + (z[5]-z[3]) - (z[7]-z[1]) - (z[4]-z[2]) );
  fjzet = (Real_t)(.125) * ( (z[6]-z[0]) - (z[5]-z[3]) + (z[7]-z[1]) - (z[4]-z[2]) );
  fjzze = (Real_t)(.125) * ( (z[6]-z[0]) + (z[5]-z[3]) + (z[7]-z[1]) + (z[4]-z[2]) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = (Real_t)(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static inline
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = (Real_t)(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = (Real_t)(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = (Real_t)(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = (Real_t)(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = (Real_t)(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = (Real_t)(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = (Real_t)(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = (Real_t)(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = (Real_t)(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

static inline
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   /*
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = (Real_t)(0.0);
      pfy[i] = (Real_t)(0.0);
      pfz[i] = (Real_t)(0.0);
   }
   */
   pfx[0] = (Real_t)(0.0); pfx[1] = (Real_t)(0.0);
   pfx[2] = (Real_t)(0.0); pfx[3] = (Real_t)(0.0);
   pfx[4] = (Real_t)(0.0); pfx[5] = (Real_t)(0.0);
   pfx[6] = (Real_t)(0.0); pfx[7] = (Real_t)(0.0);
   pfy[0] = (Real_t)(0.0); pfy[1] = (Real_t)(0.0);
   pfy[2] = (Real_t)(0.0); pfy[3] = (Real_t)(0.0);
   pfy[4] = (Real_t)(0.0); pfy[5] = (Real_t)(0.0);
   pfy[6] = (Real_t)(0.0); pfy[7] = (Real_t)(0.0);
   pfz[0] = (Real_t)(0.0); pfz[1] = (Real_t)(0.0);
   pfz[2] = (Real_t)(0.0); pfz[3] = (Real_t)(0.0);
   pfz[4] = (Real_t)(0.0); pfz[5] = (Real_t)(0.0);
   pfz[6] = (Real_t)(0.0); pfz[7] = (Real_t)(0.0);
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

static inline
void SumElemStressesToNodeForces( Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  __global Real_t* const fx,
                                  __global Real_t* const fy,
                                  __global Real_t* const fz,
                                  int stride)
{
  Real_t pfx0 = B[0][0] ;   Real_t pfx1 = B[0][1] ;
  Real_t pfx2 = B[0][2] ;   Real_t pfx3 = B[0][3] ;
  Real_t pfx4 = B[0][4] ;   Real_t pfx5 = B[0][5] ;
  Real_t pfx6 = B[0][6] ;   Real_t pfx7 = B[0][7] ;

  Real_t pfy0 = B[1][0] ;   Real_t pfy1 = B[1][1] ;
  Real_t pfy2 = B[1][2] ;   Real_t pfy3 = B[1][3] ;
  Real_t pfy4 = B[1][4] ;   Real_t pfy5 = B[1][5] ;
  Real_t pfy6 = B[1][6] ;   Real_t pfy7 = B[1][7] ;

  Real_t pfz0 = B[2][0] ;   Real_t pfz1 = B[2][1] ;
  Real_t pfz2 = B[2][2] ;   Real_t pfz3 = B[2][3] ;
  Real_t pfz4 = B[2][4] ;   Real_t pfz5 = B[2][5] ;
  Real_t pfz6 = B[2][6] ;   Real_t pfz7 = B[2][7] ;

  fx[0*stride] = -( stress_xx * pfx0 );
  fx[1*stride] = -( stress_xx * pfx1 );
  fx[2*stride] = -( stress_xx * pfx2 );
  fx[3*stride] = -( stress_xx * pfx3 );
  fx[4*stride] = -( stress_xx * pfx4 );
  fx[5*stride] = -( stress_xx * pfx5 );
  fx[6*stride] = -( stress_xx * pfx6 );
  fx[7*stride] = -( stress_xx * pfx7 );

  fy[0*stride] = -( stress_yy * pfy0  );
  fy[1*stride] = -( stress_yy * pfy1  );
  fy[2*stride] = -( stress_yy * pfy2  );
  fy[3*stride] = -( stress_yy * pfy3  );
  fy[4*stride] = -( stress_yy * pfy4  );
  fy[5*stride] = -( stress_yy * pfy5  );
  fy[6*stride] = -( stress_yy * pfy6  );
  fy[7*stride] = -( stress_yy * pfy7  );

  fz[0*stride] = -( stress_zz * pfz0 );
  fz[1*stride] = -( stress_zz * pfz1 );
  fz[2*stride] = -( stress_zz * pfz2 );
  fz[3*stride] = -( stress_zz * pfz3 );
  fz[4*stride] = -( stress_zz * pfz4 );
  fz[5*stride] = -( stress_zz * pfz5 );
  fz[6*stride] = -( stress_zz * pfz6 );
  fz[7*stride] = -( stress_zz * pfz7 );
}

static inline
Real_t CalcElemVolumeD( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = (Real_t)(1.0)/(Real_t)(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

static inline
Real_t CalcElemVolume(const Real_t x[8],
                const Real_t y[8],
                const Real_t z[8] )
{
return CalcElemVolumeD( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = (Real_t)(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = FMAX(a,charLength) ;

   charLength = (Real_t)(4.0) * volume / SQRT(charLength);

   return charLength;
}

static inline
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = (Real_t)(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = (Real_t)( .5) * ( dxddy + dyddx );
  d[4]  = (Real_t)( .5) * ( dxddz + dzddx );
  d[3]  = (Real_t)( .5) * ( dzddy + dyddz );
}

   /* More general version of reduceInPlacePOT (this works for arbitrary
    * numThreadsPerBlock <= 1024). Again, conditionals on
    * numThreadsPerBlock are evaluated at compile time.
    */

void
reduceSum(Real_t *sresult, int numThreadsPerBlock, const int threadID)
{
    /* If number of threads is not a power of two, first add the ones
       after the last power of two into the beginning. At most one of
       these conditionals will be true for a given NPOT block size. */
    if (numThreadsPerBlock > 512 && numThreadsPerBlock <= 1024)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-512)
            sresult[threadID] += sresult[threadID + 512];
    }
    
    if (numThreadsPerBlock > 256 && numThreadsPerBlock < 512)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-256)
            sresult[threadID] += sresult[threadID + 256];
    }
    
    if (numThreadsPerBlock > 128 && numThreadsPerBlock < 256)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-128)
            sresult[threadID] += sresult[threadID + 128];
    }
    
    if (numThreadsPerBlock > 64 && numThreadsPerBlock < 128)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-64)
            sresult[threadID] += sresult[threadID + 64];
    }
    
    if (numThreadsPerBlock > 32 && numThreadsPerBlock < 64)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-32)
            sresult[threadID] += sresult[threadID + 32];
    }
    
    if (numThreadsPerBlock > 16 && numThreadsPerBlock < 32)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-16)
            sresult[threadID] += sresult[threadID + 16];
    }
    
    if (numThreadsPerBlock > 8 && numThreadsPerBlock < 16)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-8)
            sresult[threadID] += sresult[threadID + 8];
    }
    
    if (numThreadsPerBlock > 4 && numThreadsPerBlock < 8)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-4)
            sresult[threadID] += sresult[threadID + 4];
    }
    
    if (numThreadsPerBlock > 2 && numThreadsPerBlock < 4)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-2)
            sresult[threadID] += sresult[threadID + 2];
    }
    
    if (numThreadsPerBlock >= 512) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 256)
            sresult[threadID] += sresult[threadID + 256];
    }
    
    if (numThreadsPerBlock >= 256) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 128)
            sresult[threadID] += sresult[threadID + 128];
    }
    if (numThreadsPerBlock >= 128) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 64)
            sresult[threadID] += sresult[threadID + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef _DEVICEEMU
    if (numThreadsPerBlock >= 64) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 32)
            sresult[threadID] += sresult[threadID + 32];
    }
    if (numThreadsPerBlock >= 32) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 16)
            sresult[threadID] += sresult[threadID + 16];
    }
    if (numThreadsPerBlock >= 16) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 8)
            sresult[threadID] += sresult[threadID + 8];
    }
    if (numThreadsPerBlock >= 8) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 4)
            sresult[threadID] += sresult[threadID + 4];
    }
    if (numThreadsPerBlock >= 4) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 2)
            sresult[threadID] += sresult[threadID + 2];
    }
    if (numThreadsPerBlock >= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 1)
            sresult[threadID] += sresult[threadID + 1];
    }
#else
    if (threadID < 32) {
        volatile Real_t *vol = sresult;
        if (numThreadsPerBlock >= 64) vol[threadID] += vol[threadID + 32];
        if (numThreadsPerBlock >= 32) vol[threadID] += vol[threadID + 16];
        if (numThreadsPerBlock >= 16) vol[threadID] += vol[threadID + 8];
        if (numThreadsPerBlock >= 8) vol[threadID] += vol[threadID + 4];
        if (numThreadsPerBlock >= 4) vol[threadID] += vol[threadID + 2];
        if (numThreadsPerBlock >= 2) vol[threadID] += vol[threadID + 1];
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}

void
reduceMin(__local Real_t *sresult, int numThreadsPerBlock, const int threadID)
{
    /* If number of threads is not a power of two, first add the ones
       after the last power of two into the beginning. At most one of
       these conditionals will be true for a given NPOT block size. */
    if (numThreadsPerBlock > 512 && numThreadsPerBlock <= 1024)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-512)
            MINEQ(sresult[threadID],sresult[threadID + 512]);
    }
    
    if (numThreadsPerBlock > 256 && numThreadsPerBlock < 512)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-256)
            MINEQ(sresult[threadID],sresult[threadID + 256]);
    }
    
    if (numThreadsPerBlock > 128 && numThreadsPerBlock < 256)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-128)
            MINEQ(sresult[threadID],sresult[threadID + 128]);
    }
    
    if (numThreadsPerBlock > 64 && numThreadsPerBlock < 128)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-64)
            MINEQ(sresult[threadID],sresult[threadID + 64]);
    }
    
    if (numThreadsPerBlock > 32 && numThreadsPerBlock < 64)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-32)
            MINEQ(sresult[threadID],sresult[threadID + 32]);
    }
    
    if (numThreadsPerBlock > 16 && numThreadsPerBlock < 32)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-16)
            MINEQ(sresult[threadID],sresult[threadID + 16]);
    }
    
    if (numThreadsPerBlock > 8 && numThreadsPerBlock < 16)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-8)
            MINEQ(sresult[threadID],sresult[threadID + 8]);
    }
    
    if (numThreadsPerBlock > 4 && numThreadsPerBlock < 8)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-4)
            MINEQ(sresult[threadID],sresult[threadID + 4]);
    }
    
    if (numThreadsPerBlock > 2 && numThreadsPerBlock < 4)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < numThreadsPerBlock-2)
            MINEQ(sresult[threadID],sresult[threadID + 2]);
    }
    
    if (numThreadsPerBlock >= 512) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 256)
            MINEQ(sresult[threadID],sresult[threadID + 256]);
    }
    
    if (numThreadsPerBlock >= 256) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 128)
            MINEQ(sresult[threadID],sresult[threadID + 128]);
    }
    if (numThreadsPerBlock >= 128) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (threadID < 64)
            MINEQ(sresult[threadID],sresult[threadID + 64]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadID < 32) {
        volatile __local Real_t *vol = sresult;
        if (numThreadsPerBlock >= 64) MINEQ(vol[threadID],vol[threadID + 32]);
        if (numThreadsPerBlock >= 32) MINEQ(vol[threadID],vol[threadID + 16]);
        if (numThreadsPerBlock >= 16) MINEQ(vol[threadID],vol[threadID + 8]);
        if (numThreadsPerBlock >= 8)  MINEQ(vol[threadID],vol[threadID + 4]);
        if (numThreadsPerBlock >= 4)  MINEQ(vol[threadID],vol[threadID + 2]);
        if (numThreadsPerBlock >= 2)  MINEQ(vol[threadID],vol[threadID + 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel
void InitStressTermsForElems_kernel(
    int numElem,__global Real_t *sigxx, __global Real_t *sigyy, __global Real_t *sigzz, __global Real_t *p, __global Real_t *q)
{
    int i = get_global_id(X);
    if (i<numElem)
        sigxx[i] = sigyy[i] = sigzz[i] =  - p[i] - q[i] ;
}

__kernel
void IntegrateStressForElems_kernel( Index_t numElem, __global Index_t *nodelist,
                                     __global Real_t *x, __global Real_t *y, __global Real_t *z,
                                     __global Real_t *fx_elem, __global Real_t *fy_elem, __global Real_t *fz_elem,
                                     __global Real_t *sigxx, __global Real_t *sigyy, __global Real_t *sigzz,
                                     __global Real_t *determ)
{
  Real_t B[3][8] ;// shape function derivatives
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;

  int k=get_global_id(X);
  if (k<numElem) {
    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*numElem];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode]; 
	}

    /* Volume calculation involves extra work for numerical consistency. */
    /*
     * computation is done per node and per element
     * the per element part can be spread across 
     * the entire work-group
     */
    CalcElemShapeFunctionDerivatives_g(x_local, y_local, z_local,
                                         B, &determ[k]);

    /* computation is done per element faces (each element has 6 faces)
     * each element has 8 nodes and 6 faces 
     * if one thread is spawned for each node and these
     * threads are used for the element's faces
     * 2 of the threads will be remaining idle
     */
    CalcElemNodeNormals( B[0] , B[1], B[2],
                         x_local, y_local, z_local );

    /* computation is done per nodes */
    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                 &fx_elem[k], &fy_elem[k], &fz_elem[k], numElem ) ;
  }
}

__kernel
void AddNodeForcesFromElems_kernel( Index_t numNode,
                                    __global Int_t *nodeElemCount, __global Index_t *nodeElemCornerList,
                                    __global Real_t *fx_elem, __global Real_t *fy_elem, __global Real_t *fz_elem,
                                    __global Real_t *fx_node, __global Real_t *fy_node, __global Real_t *fz_node)
{
    int i=get_global_id(X);
    if (i<numNode) {
        Int_t count=nodeElemCount[i];
        Real_t fx,fy,fz;
        fx=fy=fz=(Real_t)(0.0);
        for (int j=0;j<count;j++) {
            Index_t elem=nodeElemCornerList[i+numNode*j];
            fx+=fx_elem[elem]; fy+=fy_elem[elem]; fz+=fz_elem[elem];
        }
        fx_node[i]=fx; fy_node[i]=fy; fz_node[i]=fz;
    }
}

__kernel
void AddNodeForcesFromElems2_kernel( Index_t numNode,
                                    __global Int_t *nodeElemCount, __global Index_t *nodeElemCornerList,
                                    __global Real_t *fx_elem, __global Real_t *fy_elem, __global Real_t *fz_elem,
                                    __global Real_t *fx_node, __global Real_t *fy_node, __global Real_t *fz_node)
{
    int i=get_global_id(X);
    if (i<numNode) {
        Int_t count=nodeElemCount[i];
        Real_t fx,fy,fz;
        fx=fy=fz=(Real_t)(0.0);
        for (int j=0;j<count;j++) {
            Index_t elem=nodeElemCornerList[i+numNode*j];
            fx+=fx_elem[elem]; fy+=fy_elem[elem]; fz+=fz_elem[elem];
        }
        fx_node[i]+=fx; fy_node[i]+=fy; fz_node[i]+=fz;
    }
}

static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
	const Real_t twelfth = (Real_t)1.0 / (Real_t)12.0 ;
	
	*dvdx =
	(y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
	(y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
	(y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
	*dvdy =
	- (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
	(x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
	(x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);
	
	*dvdz =
	- (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
	(y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
	(y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);
	
	*dvdx *= twelfth;
	*dvdy *= twelfth;
	*dvdz *= twelfth;
}

static inline
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
	VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
			y[1], y[2], y[3], y[4], y[5], y[7],
			z[1], z[2], z[3], z[4], z[5], z[7],
			&dvdx[0], &dvdy[0], &dvdz[0]);
	VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
			y[0], y[1], y[2], y[7], y[4], y[6],
			z[0], z[1], z[2], z[7], z[4], z[6],
			&dvdx[3], &dvdy[3], &dvdz[3]);
	VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
			y[3], y[0], y[1], y[6], y[7], y[5],
			z[3], z[0], z[1], z[6], z[7], z[5],
			&dvdx[2], &dvdy[2], &dvdz[2]);
	VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
			y[2], y[3], y[0], y[5], y[6], y[4],
			z[2], z[3], z[0], z[5], z[6], z[4],
			&dvdx[1], &dvdy[1], &dvdz[1]);
	VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
			y[7], y[6], y[5], y[0], y[3], y[1],
			z[7], z[6], z[5], z[0], z[3], z[1],
			&dvdx[4], &dvdy[4], &dvdz[4]);
	VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
			y[4], y[7], y[6], y[1], y[0], y[2],
			z[4], z[7], z[6], z[1], z[0], z[2],
			&dvdx[5], &dvdy[5], &dvdz[5]);
	VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
			y[5], y[4], y[7], y[2], y[1], y[3],
			z[5], z[4], z[7], z[2], z[1], z[3],
			&dvdx[6], &dvdy[6], &dvdz[6]);
	VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
			y[6], y[5], y[4], y[3], y[2], y[0],
			z[6], z[5], z[4], z[3], z[2], z[0],
			&dvdx[7], &dvdy[7], &dvdz[7]);
}

__kernel
void CalcHourglassControlForElems_kernel(Int_t numElem,__global Index_t *nodelist,
                                         __global Real_t *x,__global Real_t *y,__global Real_t *z,
                                         __global Real_t *determ,__global Real_t *volo,__global Real_t *v,
                                         __global Real_t *dvdx,__global Real_t *dvdy,__global Real_t *dvdz,
                                         __global Real_t *x8n,__global Real_t *y8n,__global Real_t *z8n)
{
    Real_t x1[8],y1[8],z1[8];
    Real_t pfx[8],pfy[8],pfz[8];

    unsigned int elem = get_global_id(X);

    if (elem>=numElem) elem = numElem - 1; // don't return -- need thread to participate in sync operations

	for (int node = 0; node < 8; node++) {
		Index_t idx = elem + numElem * node;
		Index_t ni = nodelist[idx];

		x1[node] = x[ni];
		y1[node] = y[ni];
		z1[node] = z[ni];
	}
	CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1); 

	for (int node = 0; node < 8; node++) {
		Index_t idx = elem + numElem * node;
		dvdx[idx] = pfx[node];
		dvdy[idx] = pfy[node];
		dvdz[idx] = pfz[node];
		x8n[idx]  = x1[node];
		y8n[idx]  = y1[node];
		z8n[idx]  = z1[node];
	}
    determ[elem] = volo[elem] * v[elem];
}

__kernel
void CalcFBHourglassForceForElems_kernel(
    const __global Real_t *determ,
    const __global Real_t *x8n,      const __global Real_t *y8n,      const __global Real_t *z8n,
    const __global Real_t *dvdx,     const __global Real_t *dvdy,     const __global Real_t *dvdz,
    Real_t hourg,
    Index_t numElem, const __global Index_t *nodelist,
    const __global Real_t *ss, const __global Real_t *elemMass,
    const __global Real_t *xd, const __global Real_t *yd, const __global Real_t *zd,
    __global Real_t *fx_elem, __global Real_t *fy_elem, __global Real_t *fz_elem)
{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/
    Real_t coefficient;
    Real_t hgfx, hgfy, hgfz;
    Real_t hourgam[4][8];
    
	/*************************************************/
	/*    compute the hourglass modes */
    
    uint gid = get_global_id(X);
    unsigned int elem=gid;

    if (elem>=numElem) elem=numElem-1; // don't return -- need thread to participate in sync operations

    Real_t volinv = (Real_t)(1.0)/determ[elem];

    Real_t hourmodx[4];
    Real_t hourmody[4];
    Real_t hourmodz[4];

    for (int i = 0; i < 4; i++)
    {
       hourmodx[i]=0;
	   hourmodx[i] += x8n[elem+numElem*0] * gamma[i][0];
	   hourmodx[i] += x8n[elem+numElem*1] * gamma[i][1];
	   hourmodx[i] += x8n[elem+numElem*2] * gamma[i][2];
	   hourmodx[i] += x8n[elem+numElem*3] * gamma[i][3];
	   hourmodx[i] += x8n[elem+numElem*4] * gamma[i][4];
	   hourmodx[i] += x8n[elem+numElem*5] * gamma[i][5];
	   hourmodx[i] += x8n[elem+numElem*6] * gamma[i][6];
	   hourmodx[i] += x8n[elem+numElem*7] * gamma[i][7];
	}

    for (int i = 0; i < 4; i++)
    {
       hourmody[i]=0;
	   hourmody[i] += y8n[elem+numElem*0] * gamma[i][0];
	   hourmody[i] += y8n[elem+numElem*1] * gamma[i][1];
	   hourmody[i] += y8n[elem+numElem*2] * gamma[i][2];
	   hourmody[i] += y8n[elem+numElem*3] * gamma[i][3];
	   hourmody[i] += y8n[elem+numElem*4] * gamma[i][4];
	   hourmody[i] += y8n[elem+numElem*5] * gamma[i][5];
	   hourmody[i] += y8n[elem+numElem*6] * gamma[i][6];
	   hourmody[i] += y8n[elem+numElem*7] * gamma[i][7];
	}

    for (int i = 0; i < 4; i++)
    {
       hourmodz[i]=0;
	   hourmodz[i] += z8n[elem+numElem*0] * gamma[i][0];
	   hourmodz[i] += z8n[elem+numElem*1] * gamma[i][1];
	   hourmodz[i] += z8n[elem+numElem*2] * gamma[i][2];
	   hourmodz[i] += z8n[elem+numElem*3] * gamma[i][3];
	   hourmodz[i] += z8n[elem+numElem*4] * gamma[i][4];
	   hourmodz[i] += z8n[elem+numElem*5] * gamma[i][5];
	   hourmodz[i] += z8n[elem+numElem*6] * gamma[i][6];
	   hourmodz[i] += z8n[elem+numElem*7] * gamma[i][7];
	}

    for (int i = 0; i < 4; i++)
	{
	  hourgam[i][0] = gamma[i][0] - volinv*(dvdx[elem+numElem*0]*hourmodx[i] +
													dvdy[elem+numElem*0]*hourmody[i] +
													dvdz[elem+numElem*0]*hourmodz[i]);
	  hourgam[i][1] = gamma[i][1] - volinv*(dvdx[elem+numElem*1]*hourmodx[i] +
													dvdy[elem+numElem*1]*hourmody[i] +
													dvdz[elem+numElem*1]*hourmodz[i]);
	  hourgam[i][2] = gamma[i][2] - volinv*(dvdx[elem+numElem*2]*hourmodx[i] +
													dvdy[elem+numElem*2]*hourmody[i] +
													dvdz[elem+numElem*2]*hourmodz[i]);
	  hourgam[i][3] = gamma[i][3] - volinv*(dvdx[elem+numElem*3]*hourmodx[i] +
													dvdy[elem+numElem*3]*hourmody[i] +
													dvdz[elem+numElem*3]*hourmodz[i]);
	  hourgam[i][4] = gamma[i][4] - volinv*(dvdx[elem+numElem*4]*hourmodx[i] +
													dvdy[elem+numElem*4]*hourmody[i] +
													dvdz[elem+numElem*4]*hourmodz[i]);
	  hourgam[i][5] = gamma[i][5] - volinv*(dvdx[elem+numElem*5]*hourmodx[i] +
													dvdy[elem+numElem*5]*hourmody[i] +
													dvdz[elem+numElem*5]*hourmodz[i]);
	  hourgam[i][6] = gamma[i][6] - volinv*(dvdx[elem+numElem*6]*hourmodx[i] +
													dvdy[elem+numElem*6]*hourmody[i] +
													dvdz[elem+numElem*6]*hourmodz[i]);
	  hourgam[i][7] = gamma[i][7] - volinv*(dvdx[elem+numElem*7]*hourmodx[i] +
													dvdy[elem+numElem*7]*hourmody[i] +
													dvdz[elem+numElem*7]*hourmodz[i]);
    }

    coefficient = - hourg * (Real_t)(0.01) * ss[elem] * elemMass[elem] / CBRT(determ[elem]);  
      
    Index_t ni[8];
	ni[0] = nodelist[elem+numElem*0];
	ni[1] = nodelist[elem+numElem*1];
	ni[2] = nodelist[elem+numElem*2];
	ni[3] = nodelist[elem+numElem*3];
	ni[4] = nodelist[elem+numElem*4];
	ni[5] = nodelist[elem+numElem*5];
	ni[6] = nodelist[elem+numElem*6];
	ni[7] = nodelist[elem+numElem*7];

    Real_t h[4];   
    for (int i=0;i<4;i++)
    {      
        h[i] = 0;
		h[i]+=hourgam[i][0]*xd[ni[0]];
		h[i]+=hourgam[i][1]*xd[ni[1]];
		h[i]+=hourgam[i][2]*xd[ni[2]];
		h[i]+=hourgam[i][3]*xd[ni[3]];
		h[i]+=hourgam[i][4]*xd[ni[4]];
		h[i]+=hourgam[i][5]*xd[ni[5]];
		h[i]+=hourgam[i][6]*xd[ni[6]];
		h[i]+=hourgam[i][7]*xd[ni[7]];
    }
    
    for( int node = 0; node < 8; node++)
    {
	    hgfx = 0;
		hgfx += hourgam[0][node] * h[0];
		hgfx += hourgam[1][node] * h[1];
		hgfx += hourgam[2][node] * h[2];
		hgfx += hourgam[3][node] * h[3];
        fx_elem[elem+numElem*node]=hgfx * coefficient;
    }       

    for (int i = 0; i < 4; i++)
    {      
        h[i] = 0;
		h[i]+=hourgam[i][0]*yd[ni[0]];
		h[i]+=hourgam[i][1]*yd[ni[1]];
		h[i]+=hourgam[i][2]*yd[ni[2]];
		h[i]+=hourgam[i][3]*yd[ni[3]];
		h[i]+=hourgam[i][4]*yd[ni[4]];
		h[i]+=hourgam[i][5]*yd[ni[5]];
		h[i]+=hourgam[i][6]*yd[ni[6]];
		h[i]+=hourgam[i][7]*yd[ni[7]];
    }   
    
    for( int node = 0; node < 8; node++)
    {
   
        hgfy = 0;
		hgfy += hourgam[0][node] * h[0];
		hgfy += hourgam[1][node] * h[1];
		hgfy += hourgam[2][node] * h[2];
		hgfy += hourgam[3][node] * h[3];
        fy_elem[elem+numElem*node]=hgfy * coefficient;
    }       

    for (int i=0;i<4;i++)
    {      
        h[i] = 0;
		h[i]+=hourgam[i][0]*zd[ni[0]];
		h[i]+=hourgam[i][1]*zd[ni[1]];
		h[i]+=hourgam[i][2]*zd[ni[2]];
		h[i]+=hourgam[i][3]*zd[ni[3]];
		h[i]+=hourgam[i][4]*zd[ni[4]];
		h[i]+=hourgam[i][5]*zd[ni[5]];
		h[i]+=hourgam[i][6]*zd[ni[6]];
		h[i]+=hourgam[i][7]*zd[ni[7]];
    }   
    
    for( int node = 0; node < 8; node++)
    {
	    hgfz = 0;
		hgfz += hourgam[0][node] * h[0];
		hgfz += hourgam[1][node] * h[1];
		hgfz += hourgam[2][node] * h[2];
		hgfz += hourgam[3][node] * h[3];
        fz_elem[elem+numElem*node]=hgfz * coefficient;
    }       
}

__kernel
void CalcAccelerationForNodes_kernel(int numNode,
                                     __global Real_t *xdd, __global Real_t *ydd, __global Real_t *zdd,
                                     __global Real_t *fx, __global Real_t *fy, __global Real_t *fz,
                                     __global Real_t *nodalMass)
{
    int i=get_global_id(X);
    if (i<numNode) {
        xdd[i]=fx[i]/nodalMass[i];
        ydd[i]=fy[i]/nodalMass[i];
        zdd[i]=fz[i]/nodalMass[i];
    }
}

__kernel
void ApplyAccelerationBoundaryConditionsForNodes_kernel(
    int numNodeBC, __global Real_t *xdd, __global Real_t *ydd, __global Real_t *zdd,
    __global Index_t *symmX, __global Index_t *symmY, __global Index_t *symmZ)
{
    int i=get_global_id(X);
    if (i<numNodeBC) {
        xdd[symmX[i]] = (Real_t)(0.0) ;
        ydd[symmY[i]] = (Real_t)(0.0) ;
        zdd[symmZ[i]] = (Real_t)(0.0) ;
    }
}

__kernel
void CalcVelocityForNodes_kernel(int numNode, const Real_t dt, const Real_t u_cut,
                                 __global Real_t *xd, __global Real_t *yd, __global Real_t *zd,
                                 __global Real_t *xdd, __global Real_t *ydd, __global Real_t *zdd)
{
    int i = get_global_id(X);
    if (i<numNode) {
        Real_t xdtmp, ydtmp, zdtmp ;
        
        xdtmp = xd[i] + xdd[i] * dt ;
        if( FABS(xdtmp) < u_cut ) xdtmp = (Real_t)(0.0);
        xd[i] = xdtmp ;
        
        ydtmp = yd[i] + ydd[i] * dt ;
        if( FABS(ydtmp) < u_cut ) ydtmp = (Real_t)(0.0);
        yd[i] = ydtmp ;
        
        zdtmp = zd[i] + zdd[i] * dt ;
        if( FABS(zdtmp) < u_cut ) zdtmp = (Real_t)(0.0);
        zd[i] = zdtmp ;
    }
}

__kernel
void CalcPositionForNodes_kernel(int numNode, Real_t dt,
                                 __global Real_t *x, __global Real_t *y, __global Real_t *z,
                                 __global Real_t *xd, __global Real_t *yd, __global Real_t *zd)
{
    int i = get_global_id(X);
    if (i<numNode) {
        x[i] += xd[i] * dt;
        y[i] += yd[i] * dt;
        z[i] += zd[i] * dt;
    }
}

__kernel
void CalcKinematicsForElems_kernel(
    Index_t numElem, Real_t dt,
    __global Index_t *nodelist,__global Real_t *volo,__global Real_t *v,
    __global Real_t *x,__global Real_t *y,__global Real_t *z,__global Real_t *xd,__global Real_t *yd,__global Real_t *zd,
    __global Real_t *vnew,__global Real_t *delv,__global Real_t *arealg,__global Real_t *dxx,__global Real_t *dyy,__global Real_t *dzz
    )
{
  Real_t B[3][8] ; /** shape function derivatives */
  Real_t D[6] ;
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  Real_t xd_local[8] ;
  Real_t yd_local[8] ;
  Real_t zd_local[8] ;
  Real_t detJ = (Real_t)(0.0) ;

  int k=get_global_id(X);
  if (k<numElem) {

    Real_t volume ;
    Real_t relativeVolume ;

    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*numElem];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode];
    }

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / volo[k] ;
    vnew[k] = relativeVolume ;
    delv[k] = relativeVolume - v[k] ;

    // set characteristic length
    arealg[k] = CalcElemCharacteristicLength(x_local,y_local,z_local,volume);

    // get nodal velocities from global array and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*numElem];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
    }

    Real_t dt2 = (Real_t)(0.5) * dt;
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives(x_local,y_local,z_local,B,&detJ);

    CalcElemVelocityGradient(xd_local,yd_local,zd_local,B,detJ,D);
    // put velocity gradient quantities into their global arrays.
    dxx[k] = D[0];
    dyy[k] = D[1];
    dzz[k] = D[2];
  }
}

__kernel
void CalcLagrangeElementsPart2_kernel(
    Index_t numElem,
    __global Real_t *dxx,__global Real_t *dyy, __global Real_t *dzz,
    __global Real_t *vdov
    )
{
    int k=get_global_id(X);
    if (k<numElem) {

        // calc strain rate and apply as constraint (only done in FB element)
        Real_t vdovNew = dxx[k] + dyy[k] + dzz[k] ;
        Real_t vdovthird = vdovNew/(Real_t)(3.0) ;
        
        // make the rate of deformation tensor deviatoric
        vdov[k] = vdovNew ;
        dxx[k] -= vdovthird ;
        dyy[k] -= vdovthird ;
        dzz[k] -= vdovthird ;
        
        // See if any volumes are negative, and take appropriate action.
        //if (mesh.vnew(k) <= (Real_t)(0.0))
        //{
        //    exit(VolumeError) ;
        //}
    }
}

__kernel
void CalcMonotonicQGradientsForElems_kernel(
    Index_t numElem,
    __global Index_t *nodelist,
    __global Real_t *x,__global Real_t *y,__global Real_t *z,__global Real_t *xd,__global Real_t *yd,__global Real_t *zd,
    __global Real_t *volo,__global Real_t *vnew,
    __global Real_t *delx_zeta,__global Real_t *delv_zeta,
    __global Real_t *delx_xi,__global Real_t *delv_xi,
    __global Real_t *delx_eta,__global Real_t *delv_eta
    )
{
#define SUM4(a,b,c,d) (a + b + c + d)
   const Real_t ptiny = (Real_t)(1.e-36) ;

   int i=get_global_id(X);
   if (i<numElem) {
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      Index_t n0 = nodelist[i+0*numElem] ;
      Index_t n1 = nodelist[i+1*numElem] ;
      Index_t n2 = nodelist[i+2*numElem] ;
      Index_t n3 = nodelist[i+3*numElem] ;
      Index_t n4 = nodelist[i+4*numElem] ;
      Index_t n5 = nodelist[i+5*numElem] ;
      Index_t n6 = nodelist[i+6*numElem] ;
      Index_t n7 = nodelist[i+7*numElem] ;

      Real_t x0 = x[n0] ; Real_t y0 = y[n0] ; Real_t z0 = z[n0] ;
      Real_t x1 = x[n1] ; Real_t y1 = y[n1] ; Real_t z1 = z[n1] ;
      Real_t x2 = x[n2] ; Real_t y2 = y[n2] ; Real_t z2 = z[n2] ;
      Real_t x3 = x[n3] ; Real_t y3 = y[n3] ; Real_t z3 = z[n3] ;
      Real_t x4 = x[n4] ; Real_t y4 = y[n4] ; Real_t z4 = z[n4] ;
      Real_t x5 = x[n5] ; Real_t y5 = y[n5] ; Real_t z5 = z[n5] ;
      Real_t x6 = x[n6] ; Real_t y6 = y[n6] ; Real_t z6 = z[n6] ;
      Real_t x7 = x[n7] ; Real_t y7 = y[n7] ; Real_t z7 = z[n7] ;

      Real_t xv0 = xd[n0] ; Real_t yv0 = yd[n0] ; Real_t zv0 = zd[n0] ;
      Real_t xv1 = xd[n1] ; Real_t yv1 = yd[n1] ; Real_t zv1 = zd[n1] ;
      Real_t xv2 = xd[n2] ; Real_t yv2 = yd[n2] ; Real_t zv2 = zd[n2] ;
      Real_t xv3 = xd[n3] ; Real_t yv3 = yd[n3] ; Real_t zv3 = zd[n3] ;
      Real_t xv4 = xd[n4] ; Real_t yv4 = yd[n4] ; Real_t zv4 = zd[n4] ;
      Real_t xv5 = xd[n5] ; Real_t yv5 = yd[n5] ; Real_t zv5 = zd[n5] ;
      Real_t xv6 = xd[n6] ; Real_t yv6 = yd[n6] ; Real_t zv6 = zd[n6] ;
      Real_t xv7 = xd[n7] ; Real_t yv7 = yd[n7] ; Real_t zv7 = zd[n7] ;

      Real_t vol = volo[i]*vnew[i] ;
      Real_t norm = (Real_t)(1.0) / ( vol + ptiny ) ;

      Real_t dxj = (Real_t)(-0.25)*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
      Real_t dyj = (Real_t)(-0.25)*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
      Real_t dzj = (Real_t)(-0.25)*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

      Real_t dxi = (Real_t)( 0.25)*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
      Real_t dyi = (Real_t)( 0.25)*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
      Real_t dzi = (Real_t)( 0.25)*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

      Real_t dxk = (Real_t)( 0.25)*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
      Real_t dyk = (Real_t)( 0.25)*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
      Real_t dzk = (Real_t)( 0.25)*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      delx_zeta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = (Real_t)(0.25)*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
      dyv = (Real_t)(0.25)*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
      dzv = (Real_t)(0.25)*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

      delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      delx_xi[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = (Real_t)(0.25)*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
      dyv = (Real_t)(0.25)*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
      dzv = (Real_t)(0.25)*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

      delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      delx_eta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = (Real_t)(-0.25)*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
      dyv = (Real_t)(-0.25)*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
      dzv = (Real_t)(-0.25)*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

      delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
   }
#undef SUM4
}

__kernel
void CalcMonotonicQRegionForElems_kernel(
    Index_t regionStart,
    Real_t qlc_monoq,
    Real_t qqc_monoq,
    Real_t monoq_limiter_mult,
    Real_t monoq_max_slope,
    Real_t ptiny,
    
    // the elementset length
    Index_t elength,
    
    __global Index_t *matElemlist,__global Index_t *elemBC,
    __global Index_t *lxim,__global Index_t *lxip,
    __global Index_t *letam,__global Index_t *letap,
    __global Index_t *lzetam,__global Index_t *lzetap,
    __global Real_t *delv_xi,__global Real_t *delv_eta,__global Real_t *delv_zeta,
    __global Real_t *delx_xi,__global Real_t *delx_eta,__global Real_t *delx_zeta,
    __global Real_t *vdov,__global Real_t *elemMass,__global Real_t *volo,__global Real_t *vnew,
    __global Real_t *qq,__global Real_t *ql
    )
{
    int ielem=get_global_id(X);
    if (ielem<elength) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = matElemlist[regionStart + ielem];

      Int_t bcMask = elemBC[i] ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = (Real_t)(1.) / ( delv_xi[i] + ptiny ) ;

      switch (bcMask & XI_M) {
         case 0:         delvm = delv_xi[lxim[i]] ; break ; 
         case XI_M_SYMM: delvm = delv_xi[i]       ; break ; 
         case XI_M_FREE: delvm = (Real_t)(0.0)    ; break ; 
         default:        /* ERROR */              ; break ; 
      }
      switch (bcMask & XI_P) {
         case 0:         delvp = delv_xi[lxip[i]] ; break ; 
         case XI_P_SYMM: delvp = delv_xi[i]       ; break ; 
         case XI_P_FREE: delvp = (Real_t)(0.0)    ; break ; 
         default:        /* ERROR */              ; break ; 
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = (Real_t)(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < (Real_t)(0.)) phixi = (Real_t)(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = (Real_t)(1.) / ( delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
         case 0:          delvm = delv_eta[letam[i]] ; break ; 
         case ETA_M_SYMM: delvm = delv_eta[i]        ; break ; 
         case ETA_M_FREE: delvm = (Real_t)(0.0)      ; break ; 
         default:         /* ERROR */                ; break ; 
      }
      switch (bcMask & ETA_P) {
         case 0:          delvp = delv_eta[letap[i]] ; break ; 
         case ETA_P_SYMM: delvp = delv_eta[i]        ; break ; 
         case ETA_P_FREE: delvp = (Real_t)(0.0)      ; break ; 
         default:         /* ERROR */                ; break ; 
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = (Real_t)(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < (Real_t)(0.)) phieta = (Real_t)(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = (Real_t)(1.) / ( delv_zeta[i] + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case 0:           delvm = delv_zeta[lzetam[i]] ; break ; 
         case ZETA_M_SYMM: delvm = delv_zeta[i]         ; break ; 
         case ZETA_M_FREE: delvm = (Real_t)(0.0)        ; break ; 
         default:          /* ERROR */                  ; break ; 
      }
      switch (bcMask & ZETA_P) {
         case 0:           delvp = delv_zeta[lzetap[i]] ; break ; 
         case ZETA_P_SYMM: delvp = delv_zeta[i]         ; break ; 
         case ZETA_P_FREE: delvp = (Real_t)(0.0)        ; break ; 
         default:          /* ERROR */                  ; break ; 
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = (Real_t)(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < (Real_t)(0.)) phizeta = (Real_t)(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( vdov[i] > (Real_t)(0.) )  {
         qlin  = (Real_t)(0.) ;
         qquad = (Real_t)(0.) ;
      }
      else {
         Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
         Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
         Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

         if ( delvxxi   > (Real_t)(0.) ) delvxxi   = (Real_t)(0.) ;
         if ( delvxeta  > (Real_t)(0.) ) delvxeta  = (Real_t)(0.) ;
         if ( delvxzeta > (Real_t)(0.) ) delvxzeta = (Real_t)(0.) ;

         Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * ((Real_t)(1.) - phixi) +
               delvxeta  * ((Real_t)(1.) - phieta) +
               delvxzeta * ((Real_t)(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * ((Real_t)(1.) - phixi*phixi) +
               delvxeta*delvxeta   * ((Real_t)(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * ((Real_t)(1.) - phizeta*phizeta)  ) ;
      }

      qq[i] = qquad ;
      ql[i] = qlin  ;
   }
}

__kernel
void CalcPressureForElems_kernel(Index_t regionStart,
                                __global Index_t *matElemlist,
                                __global Real_t* p_new, __global Real_t* bvc,
                                 __global Real_t* pbvc, __global Real_t* e_old,
                                 __global Real_t* compression, __global Real_t *vnewc,
                                 Real_t pmin,
                                 Real_t p_cut, Real_t eosvmax,
                                 Index_t length, Real_t c1s)
{
   int i=get_global_id(X);
   if (i<length) {
       
      bvc[i] = c1s * (compression[i] + (Real_t)(1.));
      pbvc[i] = c1s;

      p_new[i] = bvc[i] * e_old[i] ;

      if (FABS(p_new[i]) < p_cut)
         p_new[i] = (Real_t)(0.0) ;
      
      int elem = matElemlist[regionStart+i];
      if ( vnewc[elem] >= eosvmax ) /* impossible condition here? */
         p_new[i] = (Real_t)(0.0) ;

      if (p_new[i] < pmin)
         p_new[i]   = pmin ;
   }
}

__kernel
void CalcEnergyForElemsPart1_kernel(
    Index_t length,Real_t emin,
    __global Real_t *e_old,__global Real_t *delvc,__global Real_t *p_old,__global Real_t *q_old,__global Real_t *work,
    __global Real_t *e_new)
{
    int i=get_global_id(X);
    if (i < length) {
        e_new[i] = e_old[i] - (Real_t)(0.5) * delvc[i] * (p_old[i] + q_old[i])
            + (Real_t)(0.5) * work[i];
        
        if (e_new[i] < emin) {
            e_new[i] = emin ;
        }
    }
}

__kernel
void CalcEnergyForElemsPart2_kernel(
    Index_t length,Real_t rho0,Real_t e_cut,Real_t emin,
    __global Real_t *compHalfStep,__global Real_t *delvc,__global Real_t *pbvc,__global Real_t *bvc,
    __global Real_t *pHalfStep,__global Real_t *ql,__global Real_t *qq,__global Real_t *p_old,__global Real_t *q_old,__global Real_t *work,
    __global Real_t *e_new,
    __global Real_t *q_new
    )
{
    int i=get_global_id(X);
    if (i<length) {

      Real_t vhalf = (Real_t)(1.) / ((Real_t)(1.) + compHalfStep[i]) ;

      if ( delvc[i] > (Real_t)(0.) ) {
         q_new[i] /* = qq[i] = ql[i] */ = (Real_t)(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i] + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;
//            Real_t ssc = 1.0;

//         if ( ssc <= (Real_t)(.1111111e-36) ) {
//            ssc = (Real_t)(.3333333e-18) ;
         if ( ssc <= (Real_t)(0.) ) {
            ssc = (Real_t)(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] + (Real_t)(0.5) * delvc[i]
         * (  (Real_t)(3.0)*(p_old[i]     + q_old[i])
              - (Real_t)(4.0)*(pHalfStep[i] + q_new[i])) ;

      e_new[i] += (Real_t)(0.5) * work[i];

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = (Real_t)(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }
}

__kernel
void CalcEnergyForElemsPart3_kernel(
    Index_t regionStart, __global Index_t *matElemlist, Index_t length,Real_t rho0,Real_t sixth,Real_t e_cut,Real_t emin,
    __global Real_t *pbvc,__global Real_t *vnewc,__global Real_t *bvc,__global Real_t *p_new,__global Real_t *ql,__global Real_t *qq,
    __global Real_t *p_old,__global Real_t *q_old,__global Real_t *pHalfStep,__global Real_t *q_new,__global Real_t *delvc,
    __global Real_t *e_new)
{
    int i=get_global_id(X);
    if (i<length) {
      Real_t q_tilde ;

      if (delvc[i] > (Real_t)(0.)) {
         q_tilde = (Real_t)(0.) ;
      }
      else {
		 Index_t elem = matElemlist[regionStart+i];
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i] ) / rho0 ;

//         if ( ssc <= (Real_t)(.1111111e-36) ) {
//            ssc = (Real_t)(.3333333e-18) ;
         if ( ssc <= (Real_t)(0.) ) {
            ssc = (Real_t)(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] - (  (Real_t)(7.0)*(p_old[i]     + q_old[i])
                               - (Real_t)(8.0)*(pHalfStep[i] + q_new[i])
                               + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = (Real_t)(0.)  ;
      }
      if ( e_new[i] < emin ) {
         e_new[i] = emin ;
      }
   }
}

__kernel
void CalcEnergyForElemsPart4_kernel(
    Index_t regionStart, __global Index_t *matElemlist, Index_t length,Real_t rho0,Real_t q_cut,
    __global Real_t *delvc,__global Real_t *pbvc,__global Real_t *e_new,__global Real_t *vnewc,__global Real_t *bvc,
    __global Real_t *p_new,__global Real_t *ql,__global Real_t *qq,
    __global Real_t *q_new)
{
    int i=get_global_id(X);
    if (i<length) {

      if ( delvc[i] <= (Real_t)(0.) ) {
		 Index_t elem = matElemlist[regionStart+i];
         Real_t ssc = ( pbvc[i] * e_new[i] + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i] ) / rho0 ;

//         if ( ssc <= (Real_t)(.1111111e-36) ) {
//            ssc = (Real_t)(.3333333e-18) ;
         if ( ssc <= (Real_t)(0.) ) {
            ssc = (Real_t)(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;

         if (FABS(q_new[i]) < q_cut) q_new[i] = (Real_t)(0.) ;
      }
   }
}

__kernel
void CalcSoundSpeedForElems_kernel(Index_t regionStart, __global Real_t *vnewc, Real_t rho0, __global Real_t *enewc,
                            __global Real_t *pnewc, __global Real_t *pbvc,
                            __global Real_t *bvc, Real_t ss4o3, Index_t nz,__global Index_t *matElemlist,
                            __global Real_t *ss)
{
    int i=get_global_id(X);
    if (i<nz) {
      Index_t iz = matElemlist[regionStart+i];
      Real_t ssTmp = (pbvc[i] * enewc[i] + vnewc[iz] * vnewc[iz] * bvc[i] * pnewc[i]) / rho0;
      if (ssTmp <= (Real_t)(.1111111e-36)) {
         ssTmp = (Real_t)(.3333333e-18);
      }
      ss[iz] = SQRT(ssTmp);
   }
}

__kernel
void EvalEOSForElemsPart1_kernel(
    Index_t regionStart,
    Index_t length,Real_t eosvmin,Real_t eosvmax,
    __global Index_t *matElemlist,
    __global Real_t *e,__global Real_t *delv,__global Real_t *p,__global Real_t *q,__global Real_t *qq,__global Real_t *ql,
    __global Real_t *vnewc,
    __global Real_t *e_old,__global Real_t *delvc,__global Real_t *p_old,__global Real_t *q_old,
    __global Real_t *compression,__global Real_t *compHalfStep,
    __global Real_t *qq_old,__global Real_t *ql_old,__global Real_t *work)
{
    int i=get_global_id(X);
    if (i<length) {
        Index_t zidx = matElemlist[regionStart+i];
        //TODO: solve this issue
//        printf("EvalEOS: %d %d %d\n", regionStart, i, zidx);
        e_old[i] = e[zidx];
        delvc[i] = delv[zidx];
        p_old[i] = p[zidx];
        q_old[i] = q[zidx];

        Real_t vchalf ;
        compression[i] = (Real_t)(1.) / vnewc[zidx] - (Real_t)(1.);
        vchalf = vnewc[zidx] - delvc[i] * (Real_t)(.5);
        compHalfStep[i] = (Real_t)(1.) / vchalf - (Real_t)(1.);

        if ( eosvmin != (Real_t)(0.) ) {
            if (vnewc[zidx] <= eosvmin) { /* impossible due to calling func? */
                compHalfStep[i] = compression[i] ;
            }
        }
        if ( eosvmax != (Real_t)(0.) ) {
            if (vnewc[zidx] >= eosvmax) { /* impossible due to calling func? */
                p_old[i]        = (Real_t)(0.) ;
                compression[i]  = (Real_t)(0.) ;
                compHalfStep[i] = (Real_t)(0.) ;
            }
        }

        qq_old[i] = qq[zidx] ;
        ql_old[i] = ql[zidx] ;
        work[i] = (Real_t)(0.) ; 
    }
}

__kernel
void EvalEOSForElemsPart2_kernel(
    Index_t regionStart,
    Index_t length,
    __global Index_t *matElemlist,__global Real_t *p_new,__global Real_t *e_new,__global Real_t *q_new,
    __global Real_t *p,__global Real_t *e,__global Real_t *q)
{
    int i=get_global_id(X);
    if (i<length) {
        Index_t zidx = matElemlist[regionStart+i] ;
        p[zidx] = p_new[i];
        e[zidx] = e_new[i];
        q[zidx] = q_new[i];
    }
}

__kernel
void ApplyMaterialPropertiesForElemsPart1_kernel(
    Index_t length,Real_t eosvmin,Real_t eosvmax,
    __global Index_t *matElemlist,__global Real_t *vnew,
    __global Real_t *vnewc)
{
    int i=get_global_id(X);
    if (i<length) {
        Index_t zn = matElemlist[i] ;
		//TODO: figure this out
//        vnewc[i] = vnew[zn] ;
        vnewc[i] = vnew[i] ;

        if (eosvmin != (Real_t)(0.)) {
            if (vnewc[i] < eosvmin)
                vnewc[i] = eosvmin ;
        }

        if (eosvmax != (Real_t)(0.)) {
            if (vnewc[i] > eosvmax)
                vnewc[i] = eosvmax ;
        }
    }
}

__kernel
void UpdateVolumesForElems_kernel(Index_t numElem,Real_t v_cut,
                                  __global Real_t *vnew,
                                  __global Real_t *v)
{
    int i=get_global_id(X);
    if (i<numElem) {
         Real_t tmpV ;
         tmpV = vnew[i] ;

         if ( FABS(tmpV - (Real_t)(1.0)) < v_cut )
            tmpV = (Real_t)(1.0) ;
         v[i] = tmpV ;
    }
}

__kernel
void CalcCourantConstraintForElems_kernel(
    Index_t regionStart, Index_t length,Real_t qqc2,
    __global Index_t *matElemlist,__global Real_t *ss,__global Real_t *vdov,__global Real_t *arealg,
    __global Real_t *mindtcourant)
{
    __local Real_t minArray[BLOCKSIZE];
    int i=get_global_id(X);

    Real_t dtcourant = (Real_t)(1.0e+20) ;
    if (i<length) {
        Index_t indx = matElemlist[regionStart+i] ;
        Real_t dtf = ss[indx] * ss[indx] ;
        if ( vdov[indx] < (Real_t)(0.) ) {
            dtf = dtf
                + qqc2 * arealg[indx] * arealg[indx]
                * vdov[indx] * vdov[indx] ;
        }
        dtf = SQRT(dtf) ;
        dtf = arealg[indx] / dtf ;

        /* determine minimum timestep with its corresponding elem */
        if (vdov[indx] != (Real_t)(0.)) {
            if ( dtf < dtcourant ) {
                dtcourant = dtf ;
            }
        }
    }
	int tid = get_local_id(X);
    minArray[tid] = dtcourant;
//    reduceMin(minArray, BLOCKSIZE, get_local_id(X));

	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < 128) {
		if (minArray[tid] > minArray[tid + 128])
			minArray[tid] = minArray[tid + 128];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < 64) {
		if (minArray[tid] > minArray[tid + 64])
			minArray[tid] = minArray[tid + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < 32) {
		if (minArray[tid] > minArray[tid + 32])
			minArray[tid] = minArray[tid + 32];
	}
	if (tid < 16) {
		if (minArray[tid] > minArray[tid + 16])
			minArray[tid] = minArray[tid + 16];
	}
	if (tid < 8) {
		if (minArray[tid] > minArray[tid + 8])
			minArray[tid] = minArray[tid + 8];
	}
	if (tid < 4) {
		if (minArray[tid] > minArray[tid + 4])
			minArray[tid] = minArray[tid + 4];
	}
	if (tid < 2) {
		if (minArray[tid] > minArray[tid + 2])
			minArray[tid] = minArray[tid + 2];
	}
	if (tid == 0) {
		if (minArray[tid] > minArray[tid + 1])
			minArray[tid] = minArray[tid + 1];
	}

    if (tid == 0)
        mindtcourant[get_group_id(X)] = minArray[tid];
}

__kernel
void CalcHydroConstraintForElems_kernel(
    Index_t regionStart, Index_t length,Real_t dvovmax,
    __global Index_t *matElemlist,__global Real_t *vdov,
    __global Real_t *mindthydro)
{
    __local Real_t minArray[BLOCKSIZE];

    int i = get_global_id(X);

    Real_t dthydro = (Real_t)(1.0e+20) ;
    if (i<length) {
      Index_t indx = matElemlist[regionStart+i] ;
      if (vdov[indx] != (Real_t)(0.)) {
         Real_t dtdvov = dvovmax / (FABS(vdov[indx])+(Real_t)(1.e-20)) ;
         if ( dthydro > dtdvov ) {
            dthydro = dtdvov ;
         }
      }
    }
    minArray[get_local_id(X)]=dthydro;
//    reduceMin(minArray, BLOCKSIZE, get_local_id(X));
	int tid = get_local_id(X);

	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < 128) {
		if (minArray[tid] > minArray[tid + 128])
			minArray[tid] = minArray[tid + 128];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < 64) {
		if (minArray[tid] > minArray[tid + 64])
			minArray[tid] = minArray[tid + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < 32) {
		if (minArray[tid] > minArray[tid + 32])
			minArray[tid] = minArray[tid + 32];
	}
	if (tid < 16) {
		if (minArray[tid] > minArray[tid + 16])
			minArray[tid] = minArray[tid + 16];
	}
	if (tid < 8) {
		if (minArray[tid] > minArray[tid + 8])
			minArray[tid] = minArray[tid + 8];
	}
	if (tid < 4) {
		if (minArray[tid] > minArray[tid + 4])
			minArray[tid] = minArray[tid + 4];
	}
	if (tid < 2) {
		if (minArray[tid] > minArray[tid + 2])
			minArray[tid] = minArray[tid + 2];
	}
	if (tid == 0) {
		if (minArray[tid] > minArray[tid + 1])
			minArray[tid] = minArray[tid + 1];
	}

    if (get_local_id(X)==0)
        mindthydro[get_group_id(X)]=minArray[0];
}

