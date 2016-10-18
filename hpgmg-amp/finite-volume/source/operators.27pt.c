/*******************************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. 

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
//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifdef USE_GPU
  #if defined(__KALMAR_AMP__) || defined(__HCC_AMP__)
    #include <amp.h>
    #include <amp_math.h>
  #elif defined(__KALMAR_HC__) || defined(__HCC_HC__)
    #include <hc.hpp>
    #include <hc_math.hpp>
  #else
    #error Neither *_AMP__ nor *_HC__ defined; has compiler changed?
  #endif
#endif // USE_GPU

//------------------------------------------------------------------------------------------------------------------------------
#ifdef _OPENMP
#include <omp.h>
#endif
//------------------------------------------------------------------------------------------------------------------------------
#include "timers.h"
#include "defines.h"
#include "level.h"
#include "operators.h"
//------------------------------------------------------------------------------------------------------------------------------
#define MyPragma(a) _Pragma(#a)
//------------------------------------------------------------------------------------------------------------------------------
#if (_OPENMP>=201107) // OpenMP 3.1 supports max reductions...
  // XL C/C++ 12.01.0000.0009 sets _OPENMP to 201107, but does not support the max clause within a _Pragma().  
  // This issue was fixed by XL C/C++ 12.01.0000.0011
  // If you do not have this version of XL C/C++ and run into this bug, uncomment these macros...
  //#warning not threading norm() calculations due to issue with XL/C, _Pragma, and reduction(max:bmax)
  //#define PRAGMA_THREAD_ACROSS_BLOCKS(    level,b,nb     )    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1)                     )
  //#define PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,b,nb,bsum)    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1) reduction(  +:bsum) )
  //#define PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,b,nb,bmax)    
  #define PRAGMA_THREAD_ACROSS_BLOCKS(    level,b,nb     )    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1)                     )
  #define PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,b,nb,bsum)    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1) reduction(  +:bsum) )
  #define PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,b,nb,bmax)    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1) reduction(max:bmax) )
#elif _OPENMP // older OpenMP versions don't support the max reduction clause
  #warning Threading max reductions requires OpenMP 3.1 (July 2011).  Please upgrade your compiler.                                                           
  #define PRAGMA_THREAD_ACROSS_BLOCKS(    level,b,nb     )    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1)                     )
  #define PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,b,nb,bsum)    MyPragma(omp parallel for private(b) if(nb>1) schedule(static,1) reduction(  +:bsum) )
  #define PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,b,nb,bmax)    
#else // flat MPI should not define any threading...
  #define PRAGMA_THREAD_ACROSS_BLOCKS(    level,b,nb     )    
  #define PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,b,nb,bsum)    
  #define PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,b,nb,bmax)    
#endif

#ifdef USE_GPU
#define GPU_RESTRICT
  #if defined(GPU_TILE_BLOCKS)
    #define BBS GPU_TILE_BLOCKS
  #else
    #define BBS 1
  #endif
  #if defined(GPU_TILE_K)
    #define KBS GPU_TILE_K
  #else
    #define KBS 2
  #endif
  #if defined(GPU_TILE_J)
    #define JBS GPU_TILE_J
  #else
    #define JBS 8
  #endif
  #if defined(GPU_TILE_I)
    #define IBS GPU_TILE_I
  #else
    #define IBS 32
  #endif
#else // USE_GPU
#define GPU_RESTRICT __restrict__
#endif // USE_GPU

#if defined(PRINT_DETAILS)
void print_smooth_details(void)
{
  fprintf(stderr, "  Smooth:\n");
  fprintf(stderr, "   27pt operator\n");

  #ifdef USE_HELMHOLTZ
    fprintf(stderr, "   solving HELMHOLTZ: alpha is used\n");
  #else
    fprintf(stderr, "   solving POISSON: alpha is not used\n");
  #endif // USE_HELMHOLTZ

  #if defined(USE_CHEBY)
    fprintf(stderr, "   CHEBY smoother\n");
  #elif defined(USE_GSRB)
    fprintf(stderr, "   GSRB smoother (not recommended)");
    #if defined(GSRB_FP)
      fprintf(stderr,", FP");
    #elif defined(GSRB_STRIDE2)
      fprintf(stderr,", STRIDE2");
    #elif defined(GSRB_BRANCH)
      fprintf(stderr,", BRANCH");
    #else // GSRB_type
      fprintf(stderr,", UNKNOWN");
    #endif // GSRB_type
    #if defined(GSRB_OOP)
      fprintf(stderr, ", out-of-place\n");
    #else
      fprintf(stderr, ", in-place\n");
    #endif
  #elif defined(USE_JACOBI)
    fprintf(stderr, "   JACOBI smoother\n");
  #elif defined(USE_L1JACOBI)
    fprintf(stderr, "   L1 JACOBI smoother\n");
  #elif defined(USE_SYMGS)
    fprintf(stderr, "   SYMGS smoother\n");
    #ifdef USE_GPU_FOR_SMOOTH
    #error SYMGS smoother not implemented on GPU
    #endif // USE_GPU_FOR_SMOOTH
  #else
    #error Unknown smoother not implemented
  #endif
  #if defined(USE_GPU_FOR_SMOOTH)
    fprintf(stderr, "    GPU_THRESHOLD is %d\n", GPU_THRESHOLD);
    fprintf(stderr, "    Not using LDS\n");
  #endif

}
void print_smooth_info(void)
{
  static int printSmoothInfo = 1;
  if (printSmoothInfo) {
    print_smooth_details();
    printSmoothInfo = 0;
  }
}

#endif // PRINT_DETAILS

//------------------------------------------------------------------------------------------------------------------------------
void apply_BCs(level_type * level, int x_id, int shape){apply_BCs_p2(level,x_id,shape);} // 27pt uses cell centered, not cell averaged
//void apply_BCs(level_type * level, int x_id, int shape){apply_BCs_v2(level,x_id,shape);}
//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_COEF0 (-4.2666666666666666666)  // -128.0/30.0;
#define STENCIL_COEF1 ( 0.4666666666666666666)  //   14.0/30.0;
#define STENCIL_COEF2 ( 0.1000000000000000000)  //    3.0/30.0;
#define STENCIL_COEF3 ( 0.0333333333333333333)  //    1.0/30.0;
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_VARIABLE_COEFFICIENT
  #error This implementation does not support variable-coefficient operators
#endif
#ifdef STENCIL_FUSE_BC
  #error This implementation does not support fusion of the boundary conditions with the operator
#endif
//------------------------------------------------------------------------------------------------------------------------------
#define Dinv_ijk() Dinv[ijk]        // simply retrieve it rather than recalculating it
//------------------------------------------------------------------------------------------------------------------------------
#define apply_op_ijk(x)				\
(						\
  a*x[ijk] - b*h2inv*(				\
    STENCIL_COEF3*(x[ijk-kStride-jStride-1] +	\
                   x[ijk-kStride-jStride+1] +	\
                   x[ijk-kStride+jStride-1] +	\
                   x[ijk-kStride+jStride+1] +	\
                   x[ijk+kStride-jStride-1] +	\
                   x[ijk+kStride-jStride+1] +	\
                   x[ijk+kStride+jStride-1] +	\
                   x[ijk+kStride+jStride+1] ) +	\
    STENCIL_COEF2*(x[ijk-kStride-jStride  ] +	\
                   x[ijk-kStride        -1] +	\
                   x[ijk-kStride        +1] +	\
                   x[ijk-kStride+jStride  ] +	\
                   x[ijk        -jStride-1] +	\
                   x[ijk        -jStride+1] +	\
                   x[ijk        +jStride-1] +	\
                   x[ijk        +jStride+1] +	\
                   x[ijk+kStride-jStride  ] +	\
                   x[ijk+kStride        -1] +	\
                   x[ijk+kStride        +1] +	\
                   x[ijk+kStride+jStride  ] ) +	\
    STENCIL_COEF1*(x[ijk-kStride          ] +	\
                   x[ijk        -jStride  ] +	\
                   x[ijk                -1] +	\
                   x[ijk                +1] +	\
                   x[ijk        +jStride  ] +	\
                   x[ijk+kStride          ] ) +	\
    STENCIL_COEF0*(x[ijk                  ] )	\
  )						\
)
#define apply_op_ijk_gpu                        \
(						\
  a*lxn(0,0,0) - b*h2inv*(				\
    STENCIL_COEF3*(lxn(-1,-1,-1) + lxn(-1,-1, 1) + lxn(-1, 1,-1) + lxn(-1, 1, 1) + \
                   lxn( 1,-1,-1) + lxn( 1,-1, 1) + lxn( 1, 1,-1) + lxn( 1, 1, 1))+ \
    STENCIL_COEF2*(lxn(-1,-1, 0) + lxn(-1, 0,-1) + lxn(-1, 0, 1) + lxn(-1, 1, 0) + \
                   lxn( 0,-1,-1) + lxn( 0,-1, 1) + lxn( 0, 1,-1) + lxn( 0, 1, 1) + \
                   lxn( 1,-1, 0) + lxn( 1, 0,-1) + lxn( 1, 0, 1) + lxn( 1, 1, 0))+ \
    STENCIL_COEF1*(lxn(-1, 0, 0) + lxn( 0,-1, 0) + lxn( 0, 0,-1) + lxn( 0, 0, 1) + \
                   lxn( 0, 1, 0) + lxn( 1, 0, 0)) +                                \
    STENCIL_COEF0*(lxn( 0, 0, 0))                                                  \
  )						                                   \
)

//------------------------------------------------------------------------------------------------------------------------------
int stencil_get_radius(){return(1);} // 27pt = dense 3^3
int stencil_get_shape(){return(STENCIL_SHAPE_BOX);} // needs faces, edges, and corners
//------------------------------------------------------------------------------------------------------------------------------
void rebuild_operator(level_type * level, level_type *fromLevel, double a, double b){
  // form restriction of alpha[], beta_*[] coefficients from fromLevel
  if(fromLevel != NULL){
    restriction(level,VECTOR_ALPHA ,fromLevel,VECTOR_ALPHA ,RESTRICT_CELL  );
    restriction(level,VECTOR_BETA_I,fromLevel,VECTOR_BETA_I,RESTRICT_FACE_I);
    restriction(level,VECTOR_BETA_J,fromLevel,VECTOR_BETA_J,RESTRICT_FACE_J);
    restriction(level,VECTOR_BETA_K,fromLevel,VECTOR_BETA_K,RESTRICT_FACE_K);
  } // else case assumes alpha/beta have been set

  // exchange alpha/beta/...  (must be done before calculating Dinv)
  exchange_boundary(level,VECTOR_ALPHA ,STENCIL_SHAPE_BOX); // safe
  exchange_boundary(level,VECTOR_BETA_I,STENCIL_SHAPE_BOX);
  exchange_boundary(level,VECTOR_BETA_J,STENCIL_SHAPE_BOX);
  exchange_boundary(level,VECTOR_BETA_K,STENCIL_SHAPE_BOX);

  // black box rebuild of D^{-1}, l1^{-1}, dominant eigenvalue, ...
  rebuild_operator_blackbox(level,a,b,2);

  // exchange Dinv/L1inv/...
  exchange_boundary(level,VECTOR_DINV ,STENCIL_SHAPE_BOX); // safe
  exchange_boundary(level,VECTOR_L1INV,STENCIL_SHAPE_BOX);
}


//------------------------------------------------------------------------------------------------------------------------------
#ifdef  USE_GSRB
#warning GSRB is not recommended for the 27pt operator
#ifndef GSRB_IN_PLACE
#define GSRB_OOP
#endif
#define NUM_SMOOTHS      2 // RBRB
#include "operators/gsrb.c"
#elif   USE_CHEBY
#define NUM_SMOOTHS      1
#define CHEBYSHEV_DEGREE 4 // i.e. one degree-4 polynomial smoother
#include "operators/chebyshev.c"
#elif   USE_JACOBI
#define NUM_SMOOTHS      6
#include "operators/jacobi.c"
#elif   USE_L1JACOBI
#define NUM_SMOOTHS      6
#include "operators/jacobi.c"
#elif   USE_SYMGS
#define NUM_SMOOTHS      2 // FBFB
#include "operators/symgs.c"
#else
#error You must compile with either -DUSE_GSRB, -DUSE_CHEBY, -DUSE_JACOBI, -DUSE_L1JACOBI, or -DUSE_SYMGS
#endif
#include "operators/residual.c"
#include "operators/apply_op.c"
#include "operators/rebuild.c"
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/blockCopy.c"
#include "operators/misc.c"
#include "operators/exchange_boundary.c"
#include "operators/boundary_fd.c" // 27pt uses cell centered, not cell averaged
//#include "operators/boundary_fv.c"
#include "operators/restriction.c"
#include "operators/interpolation_p2.c"
//#include "operators/interpolation_v2.c"
//------------------------------------------------------------------------------------------------------------------------------
void interpolation_vcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_p2(level_f,id_f,prescale_f,level_c,id_c);} // 27pt uses cell centered, not cell averaged
void interpolation_fcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_p2(level_f,id_f,prescale_f,level_c,id_c);}
//void interpolation_vcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_v2(level_f,id_f,prescale_f,level_c,id_c);}
//void interpolation_fcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_v2(level_f,id_f,prescale_f,level_c,id_c);}
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/problem.p6.c"
//------------------------------------------------------------------------------------------------------------------------------
