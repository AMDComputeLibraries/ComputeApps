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
  #if defined(__KALMAR_AMP__)
    #include <amp.h>
    #include <amp_math.h>
  #elif defined(__KALMAR_HC__)
    #include <hc.hpp>
    #include <hc_math.hpp>
  #else
    #error Neither __KALMAR_AMP__ nor __KALMAR_HC__ defined; has compiler changed?
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
#ifndef STENCIL_CONSTANT_COEFFICIENT
#define STENCIL_VARIABLE_COEFFICIENT
#endif
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
#else
#define GPU_RESTRICT __restrict__
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC
  #error This implementation does not support fusion of the boundary conditions with the operator
#endif
//------------------------------------------------------------------------------------------------------------------------------
void apply_BCs(level_type * level, int x_id, int shape){apply_BCs_v4(level,x_id,shape);}
//------------------------------------------------------------------------------------------------------------------------------
#define Dinv_ijk() Dinv[ijk]        // simply retrieve it rather than recalculating it
//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_TWELFTH ( 0.0833333333333333333)  // 1.0/12.0;
//------------------------------------------------------------------------------------------------------------------------------
#if defined(USE_GPU_FOR_SMOOTH)
#ifndef GPU_ARRAY_VIEW

  #ifndef GPU_TILE_DIM
    #define GPU_TILE_DIM 0
  #endif // GPU_TILE_DIM


  #ifndef USE_LDS
  #undef USElxn
  #undef USElbk
  #undef USElbj
  #undef USElbi
  #endif
  #if defined(USE_LDS)
    #error LDS use not (yet?) implemented for fv4
    #undef USE_LDS
  #endif // USE_LDS
/*

    #ifdef USElxn
      // The use of local_x is much more complicated for fv4 than fv2
      // because of the extra ghost zones and more complicated usage patterns
      #define lxn(x,inck,incj,inci) (local_x[l_k+inck+2][l_j+incj+2][l_i+inci+2])
      #define LXN_SIZE ((KBS+4)*(JBS+4)*(IBS+4))
      #define DECLARE_LXN tile_static double local_x[KBS+4][JBS+4][IBS+4]
      #warning Initialization of local_x is not yet correct
      #define INITIALIZE_LXN(x)                             \
      {                                                     \
        int iadj = (l_i == 0) ? -1 : 1;                     \
        int jadj = (l_j == 0) ? -1 : 1;                     \
        int kadj = (l_k == 0) ? -1 : 1;                     \
        lxn(x,0,0,iadj) = x[ijk+iadj];                        \
        lxn(x,0,jadj,0) = x[ijk+jadj*jStride];                \
        lxn(x,kadj,0,0) = x[ijk+kadj*kStride];                \
        lxn(x,0,0,0) = x[ijk];                                \
      }
    #else // USElxn
      #define lxn(x,inck,incj,inci) (x[ijk+inci+incj*jStride+inck*kStride])
      #define LXN_SIZE (0)
      #define DECLARE_LXN
      #define INITIALIZE_LXN(x)
    #endif // USElxn

    // The use of local_beta_k, local_beta_j, and local_i is also more
    // complicated than fv2, in this case because there are some 
    // more complicated references.
    #ifdef USElbk
      #define lbk(x,inck,incj,inci) (local_beta_k[l_k+inck+1][l_j+incj+1][l_i+inci+1])
      #define LBK_SIZE ((KBS+2)*(JBS+2)*(IBS+2))
      #define DECLARE_LBK tile_static double local_beta_k[KBS+2][JBS+2][IBS+2]
      #warning Initialization of local_beta_k is not yet correct
      #define INITIALIZE_LBK()                              \
      {                                                     \
        int iadj = (l_i == 0) ? -1 : 1;                     \
        int jadj = (l_j == 0) ? -1 : 1;                     \
        int kadj = (l_k == 0) ? -1 : 1;                     \
        lbk(0,0,iadj) = beta_k[ijk+iadj];                   \
        lbk(0,jadj,0) = beta_k[ijk+jadj*jStride];           \
        lbk(kadj,0,0) = beta_k[ijk+kadj*kStride];           \
        lbk(0,0,0) = beta_k[ijk];                           \
      }
    #else // USElbk
      #define lbk(x,inck,incj,inci) (beta_k[ijk+inci+incj*jStride+inck*kStride])
      #define LBK_SIZE (0)
      #define DECLARE_LBK
      #define INITIALIZE_LBK(x)
    #endif // USElbk

    #ifdef USElbj
      #define lbj(x,inck,incj,inci) (local_beta_j[l_k+inck+1][l_j+incj+1][l_i+inci+1])
      #define LBJ_SIZE ((KBS+2)*(JBS+2)*(IBS+2))
      #define DECLARE_LBJ tile_static double local_beta_j[KBS+2][JBS+2][IBS+2]
      #warning Initialization of local_beta_j is not yet correct
      #define INITIALIZE_LBJ()                              \
      {                                                     \
        int iadj = (l_i == 0) ? -1 : 1;                     \
        int jadj = (l_j == 0) ? -1 : 1;                     \
        int kadj = (l_k == 0) ? -1 : 1;                     \
        lbj(0,0,iadj) = beta_j[ijk+iadj];                   \
        lbj(0,jadj,0) = beta_j[ijk+jadj*jStride];           \
        lbj(kadj,0,0) = beta_j[ijk+kadj*kStride];           \
        lbj(0,0,0) = beta_j[ijk];                           \
      }
    #else // USElbj
      #define lbj(x,inck,incj,inci) (beta_j[ijk+inci+incj*jStride+inck*kStride])
      #define LBJ_SIZE (0)
      #define DECLARE_LBJ
      #define INITIALIZE_LBJ(x)
    #endif // USElbj

    #ifdef USElbi
      #define lbi(x,inck,incj,inci) (local_beta_i[l_k+inck+1][l_j+incj+1][l_i+inci+1])
      #define LBI_SIZE ((KBS+2)*(JBS+2)*(IBS+2))
      #define DECLARE_LBI tile_static double local_beta_i[KBS+2][JBS+2][IBS+2]
      #warning Initialization of local_beta_i is not yet correct
      #define INITIALIZE_LBI()                              \
      {                                                     \
        int iadj = (l_i == 0) ? -1 : 1;                     \
        int jadj = (l_j == 0) ? -1 : 1;                     \
        int kadj = (l_k == 0) ? -1 : 1;                     \
        lbi(0,0,iadj) = beta_i[ijk+iadj];                   \
        lbi(0,jadj,0) = beta_i[ijk+jadj*jStride];           \
        lbi(kadj,0,0) = beta_i[ijk+kadj*kStride];           \
        lbi(0,0,0) = beta_i[ijk];                           \
      }
    #else // USElbi
      #define lbi(x,inck,incj,inci) (beta_j[ijk+inci+incj*jStride+inck*kStride])
      #define LBI_SIZE (0)
      #define DECLARE_LBI
      #define INITIALIZE_LBI(x)
    #endif // USElbi
*/
#endif // GPU_ARRAY_VIEW

    #ifdef STENCIL_VARIABLE_COEFFICIENT
      #define apply_op_ijk_poisson                                                                           \
      (                                                                                                  \
       -b*h2inv*(                                                                                        \
          STENCIL_TWELFTH*(                                                                              \
            + lbi( 0, 0, 0)*( 15.0*(lxn( 0, 0,-1)-lxn(0,0,0)) - (lxn( 0, 0,-2)-lxn( 0, 0,+1)) )          \
            + lbi( 0, 0,+1)*( 15.0*(lxn( 0, 0,+1)-lxn(0,0,0)) - (lxn( 0, 0,+2)-lxn( 0, 0,-1)) )          \
            + lbj( 0, 0, 0)*( 15.0*(lxn( 0,-1, 0)-lxn(0,0,0)) - (lxn( 0,-2, 0)-lxn( 0,+1, 0)) )          \
            + lbj( 0,+1, 0)*( 15.0*(lxn( 0,+1, 0)-lxn(0,0,0)) - (lxn( 0,+2, 0)-lxn( 0,-1, 0)) )          \
            + lbk( 0, 0, 0)*( 15.0*(lxn(-1, 0, 0)-lxn(0,0,0)) - (lxn(-2, 0, 0)-lxn(+1, 0, 0)) )          \
            + lbk(+1, 0, 0)*( 15.0*(lxn(+1, 0, 0)-lxn(0,0,0)) - (lxn(+2, 0, 0)-lxn(-1, 0, 0)) )          \
          )                                                                                              \
          + 0.25*STENCIL_TWELFTH*(                                                                       \
            + (lbi( 0,+1, 0)-lbi( 0,-1, 0)) * (lxn( 0,+1,-1)-lxn( 0,+1, 0)-lxn( 0,-1,-1)+lxn( 0,-1, 0))  \
            + (lbi(+1, 0, 0)-lbi(-1, 0, 0)) * (lxn(+1, 0,-1)-lxn(+1, 0, 0)-lxn(-1, 0,-1)+lxn(-1, 0, 0))  \
            + (lbj( 0, 0,+1)-lbj( 0, 0,-1)) * (lxn( 0,-1,+1)-lxn( 0, 0,+1)-lxn( 0,-1,-1)+lxn( 0, 0,-1))  \
            + (lbj(+1, 0, 0)-lbj(-1, 0, 0)) * (lxn(+1,-1, 0)-lxn(+1, 0, 0)-lxn(-1,-1, 0)+lxn(-1, 0, 0))  \
            + (lbk( 0, 0,+1)-lbk( 0, 0,-1)) * (lxn(-1, 0,+1)-lxn( 0, 0,+1)-lxn(-1, 0,-1)+lxn( 0, 0,-1))  \
            + (lbk( 0,+1, 0)-lbk( 0,-1, 0)) * (lxn(-1,+1, 0)-lxn( 0,+1, 0)-lxn(-1,-1, 0)+lxn( 0,-1, 0))  \
                                                                                                         \
            + (lbi( 0,+1,+1)-lbi( 0,-1,+1)) * (lxn( 0,+1,+1)-lxn( 0,+1, 0)-lxn( 0,-1,+1)+lxn( 0,-1, 0))  \
            + (lbi(+1, 0,+1)-lbi(-1, 0,+1)) * (lxn(+1, 0,+1)-lxn(+1, 0, 0)-lxn(-1, 0,+1)+lxn(-1, 0, 0))  \
            + (lbj( 0,+1,+1)-lbj( 0,+1,-1)) * (lxn( 0,+1,+1)-lxn( 0, 0,+1)-lxn( 0,+1,-1)+lxn( 0, 0,-1))  \
            + (lbj(+1,+1, 0)-lbj(-1,+1, 0)) * (lxn(+1,+1, 0)-lxn(+1, 0, 0)-lxn(-1,+1, 0)+lxn(-1, 0, 0))  \
            + (lbk(+1, 0,+1)-lbk(+1, 0,-1)) * (lxn(+1, 0,+1)-lxn( 0, 0,+1)-lxn(+1, 0,-1)+lxn( 0, 0,-1))  \
            + (lbk(+1,+1, 0)-lbk(+1,-1, 0)) * (lxn(+1,+1, 0)-lxn( 0,+1, 0)-lxn(+1,-1, 0)+lxn( 0,-1, 0))  \
          )                                                                                              \
        )                                                                                                \
      )
      #ifdef USE_HELMHOLTZ
        #define apply_op_ijk_gpu  (a*lalpha(ijk)*lxn(0,0,0) + apply_op_ijk_poisson)
      #else
        #define apply_op_ijk_gpu  (                           apply_op_ijk_poisson)
      #endif

    #else // constant coefficient (don't bother differentiating between Poisson and Helmholtz)...
      #define apply_op_ijk_gpu      \
      (                             \
        a*lxn(0,0,0)                \
        - b*h2inv*STENCIL_TWELFTH*( \
       - 1.0*(lxn(-2, 0, 0) +       \
              lxn( 0,-2, 0) +       \
              lxn( 0, 0,-2) +       \
              lxn( 0, 0,+2) +       \
              lxn( 0,+2, 0) +       \
              lxn(+2, 0, 0))        \
       +16.0*(lxn(-1, 0, 0) +       \
              lxn( 0,-1, 0) +       \
              lxn( 0, 0,-1) +       \
              lxn( 0, 0,+1) +       \
              lxn( 0,+1, 0) +       \
              lxn(+1, 0, 0))        \
       -90.0*(lxn( 0, 0, 0))        \
      )                             \
    )
    #endif // variable/constant coefficient

#define LDS_USE (LXN_SIZE + LBK_SIZE + LBJ_SIZE + LBI_SIZE)
#endif // USE_GPU_FOR_SMOOTH

#ifdef PRINT_DETAILS
void print_smooth_details(void)
{
  fprintf(stderr, "  Smooth:\n");
  fprintf(stderr, "   fv4 operator\n");

  #ifdef USE_HELMHOLTZ
    fprintf(stderr, "   solving HELMHOLTZ\n");
    fprintf(stderr, "   alpha is used\n");
  #else // USE_HELMHOLTZ
    fprintf(stderr, "   solving POISSON\n");
    fprintf(stderr, "   alpha is not used\n");
  #endif // USE_HELMHOLTZ


  #if defined(USE_CHEBY)
    fprintf(stderr, "   CHEBY smoother\n");
    fprintf(stderr, "   WARNING: The Chebyshev smoother may underperform for 4th order.\n");
    fprintf(stderr, "   If so, please use GSRB or JACOBI\n");
  #elif defined(USE_GSRB)
    fprintf(stderr, "   GSRB smoother");
    #if defined(GSRB_FP)
      fprintf(stderr,", FP");
    #elif defined(GSRB_STRIDE2)
      fprintf(stderr,", STRIDE2");
    #elif defined(GSRB_BRANCH)
      fprintf(stderr,", BRANCH");
    #else // GSRB_type
      fprintf(stderr,", UNKNOWN");
    #endif // GSRB_type
    #if defined(GSRB_IN_PLACE)
      fprintf(stderr, ", in-place\n");
    #else
      fprintf(stderr, ", out-of-place\n");
    #endif
  #elif defined(USE_JACOBI)
    fprintf(stderr, "   JACOBI smoother\n");
  #elif defined(USE_L1JACOBI)
    fprintf(stderr, "   L1 JACOBI smoother\n");
  #elif defined(USE_SYMGS)
    #error SYMGS smoother not implemented on GPU
    fprintf(stderr, "   SYMGS smoother not implemented on GPU\n");
  #else
    #error Unknown smoother not implemented on GPU
    fprintf(stderr, "   Unknown smoother\n");
  #endif

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    fprintf(stderr, "   variable coefficients\n");
    fprintf(stderr, "   using beta_k, beta_j, beta_i\n");
  #else // STENCIL_VARIABLE_COEFFICIENT
    fprintf(stderr, "   fixed coefficients\n");
    fprintf(stderr, "   not using beta_k, beta_j, beta_i\n");
  #endif // STENCIL_VARIABLE_COEFFICIENT

  #ifdef USE_GPU_FOR_SMOOTH
  fprintf(stderr, "    GPU_THRESHOLD is %d\n", GPU_THRESHOLD);
  #if defined(USE_LDS)
  #if defined(GPU_TILE_DIM) && (GPU_TILE_DIM != 0)
  #ifdef USElxn
    fprintf(stderr, "   using LDS for x_n\n");
  #else // USElxn
    fprintf(stderr, "   not using LDS for x_n\n");
  #endif // USElxn

  #ifdef USElbk
          fprintf(stderr, "   using LDS for beta_k\n");
    #if !defined(STENCIL_VARIABLE_COEFFICIENT)
          fprintf(stderr, "   Wasting LDS for beta_k, which is not used in this case\n");
    #endif // STENCIL_VARIABLE_COEFFICIENT
  #else // USElbk
          fprintf(stderr, "   not using LDS for beta_k\n");
  #endif // USElbk

  #ifdef USElbj
    fprintf(stderr, "   using LDS for beta_j\n");
    #if !defined(STENCIL_VARIABLE_COEFFICIENT)
          fprintf(stderr, "   Wasting LDS for beta_j, which is not used in this case\n");
    #endif // STENCIL_VARIABLE_COEFFICIENT
  #else // USElbj
    fprintf(stderr, "   not using LDS for beta_j\n");
  #endif // USElbj

  #ifdef USElbi
    fprintf(stderr, "   using LDS for beta_i\n");
    #if !defined(STENCIL_VARIABLE_COEFFICIENT)
      fprintf(stderr, "   Wasting LDS for beta_i, which is not used in this case\n");
    #endif // STENCIL_VARIABLE_COEFFICIENT
  #else // USElbi
    fprintf(stderr, "   not using LDS for beta_i\n");
  #endif // USElbi


  fprintf(stderr, "   using %d local doubles per large tile\n", LDS_USE);
  if (LDS_USE > 4096) {
    fflush(stdout);
    fflush(stderr);
    fprintf(stderr, "   Too many local doubles\n");
    fflush(stderr);
    exit(0);
  }
  #else // GPU_TILE_DIM
  fprintf(stderr, "   No GPU tiling, so no LDS use\n");
  #endif // GPU_TILE_DIM
  #else // USE_LDS
  fprintf(stderr, "   Not using LDS\n");
  #endif // USE_LDS
  #endif // USE_GPU_FOR_SMOOTH
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



#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ
  #define apply_op_ijk(x)                                                                                                                            \
  (                                                                                                                                                  \
    a*alpha[ijk]*x[ijk]                                                                                                                              \
   -b*h2inv*(                                                                                                                                        \
      STENCIL_TWELFTH*(                                                                                                                              \
        + beta_i[ijk        ]*( 15.0*(x[ijk-1      ]-x[ijk]) - (x[ijk-2        ]-x[ijk+1      ]) )                                                   \
        + beta_i[ijk+1      ]*( 15.0*(x[ijk+1      ]-x[ijk]) - (x[ijk+2        ]-x[ijk-1      ]) )                                                   \
        + beta_j[ijk        ]*( 15.0*(x[ijk-jStride]-x[ijk]) - (x[ijk-2*jStride]-x[ijk+jStride]) )                                                   \
        + beta_j[ijk+jStride]*( 15.0*(x[ijk+jStride]-x[ijk]) - (x[ijk+2*jStride]-x[ijk-jStride]) )                                                   \
        + beta_k[ijk        ]*( 15.0*(x[ijk-kStride]-x[ijk]) - (x[ijk-2*kStride]-x[ijk+kStride]) )                                                   \
        + beta_k[ijk+kStride]*( 15.0*(x[ijk+kStride]-x[ijk]) - (x[ijk+2*kStride]-x[ijk-kStride]) )                                                   \
      )                                                                                                                                              \
      + 0.25*STENCIL_TWELFTH*(                                                                                                                       \
        + (beta_i[ijk        +jStride]-beta_i[ijk        -jStride]) * (x[ijk-1      +jStride]-x[ijk+jStride]-x[ijk-1      -jStride]+x[ijk-jStride])  \
        + (beta_i[ijk        +kStride]-beta_i[ijk        -kStride]) * (x[ijk-1      +kStride]-x[ijk+kStride]-x[ijk-1      -kStride]+x[ijk-kStride])  \
        + (beta_j[ijk        +1      ]-beta_j[ijk        -1      ]) * (x[ijk-jStride+1      ]-x[ijk+1      ]-x[ijk-jStride-1      ]+x[ijk-1      ])  \
        + (beta_j[ijk        +kStride]-beta_j[ijk        -kStride]) * (x[ijk-jStride+kStride]-x[ijk+kStride]-x[ijk-jStride-kStride]+x[ijk-kStride])  \
        + (beta_k[ijk        +1      ]-beta_k[ijk        -1      ]) * (x[ijk-kStride+1      ]-x[ijk+1      ]-x[ijk-kStride-1      ]+x[ijk-1      ])  \
        + (beta_k[ijk        +jStride]-beta_k[ijk        -jStride]) * (x[ijk-kStride+jStride]-x[ijk+jStride]-x[ijk-kStride-jStride]+x[ijk-jStride])  \
                                                                                                                                                     \
        + (beta_i[ijk+1      +jStride]-beta_i[ijk+1      -jStride]) * (x[ijk+1      +jStride]-x[ijk+jStride]-x[ijk+1      -jStride]+x[ijk-jStride])  \
        + (beta_i[ijk+1      +kStride]-beta_i[ijk+1      -kStride]) * (x[ijk+1      +kStride]-x[ijk+kStride]-x[ijk+1      -kStride]+x[ijk-kStride])  \
        + (beta_j[ijk+jStride+1      ]-beta_j[ijk+jStride-1      ]) * (x[ijk+jStride+1      ]-x[ijk+1      ]-x[ijk+jStride-1      ]+x[ijk-1      ])  \
        + (beta_j[ijk+jStride+kStride]-beta_j[ijk+jStride-kStride]) * (x[ijk+jStride+kStride]-x[ijk+kStride]-x[ijk+jStride-kStride]+x[ijk-kStride])  \
        + (beta_k[ijk+kStride+1      ]-beta_k[ijk+kStride-1      ]) * (x[ijk+kStride+1      ]-x[ijk+1      ]-x[ijk+kStride-1      ]+x[ijk-1      ])  \
        + (beta_k[ijk+kStride+jStride]-beta_k[ijk+kStride-jStride]) * (x[ijk+kStride+jStride]-x[ijk+jStride]-x[ijk+kStride-jStride]+x[ijk-jStride])  \
      )                                                                                                                                              \
    )                                                                                                                                                \
  )
  #else // Poisson...
  #define apply_op_ijk(x)                                                                                                                            \
  (                                                                                                                                                  \
   -b*h2inv*(                                                                                                                                        \
      STENCIL_TWELFTH*(                                                                                                                              \
        + beta_i[ijk        ]*( 15.0*(x[ijk-1      ]-x[ijk]) - (x[ijk-2        ]-x[ijk+1      ]) )                                                   \
        + beta_i[ijk+1      ]*( 15.0*(x[ijk+1      ]-x[ijk]) - (x[ijk+2        ]-x[ijk-1      ]) )                                                   \
        + beta_j[ijk        ]*( 15.0*(x[ijk-jStride]-x[ijk]) - (x[ijk-2*jStride]-x[ijk+jStride]) )                                                   \
        + beta_j[ijk+jStride]*( 15.0*(x[ijk+jStride]-x[ijk]) - (x[ijk+2*jStride]-x[ijk-jStride]) )                                                   \
        + beta_k[ijk        ]*( 15.0*(x[ijk-kStride]-x[ijk]) - (x[ijk-2*kStride]-x[ijk+kStride]) )                                                   \
        + beta_k[ijk+kStride]*( 15.0*(x[ijk+kStride]-x[ijk]) - (x[ijk+2*kStride]-x[ijk-kStride]) )                                                   \
      )                                                                                                                                              \
      + 0.25*STENCIL_TWELFTH*(                                                                                                                       \
        + (beta_i[ijk        +jStride]-beta_i[ijk        -jStride]) * (x[ijk-1      +jStride]-x[ijk+jStride]-x[ijk-1      -jStride]+x[ijk-jStride])  \
        + (beta_i[ijk        +kStride]-beta_i[ijk        -kStride]) * (x[ijk-1      +kStride]-x[ijk+kStride]-x[ijk-1      -kStride]+x[ijk-kStride])  \
        + (beta_j[ijk        +1      ]-beta_j[ijk        -1      ]) * (x[ijk-jStride+1      ]-x[ijk+1      ]-x[ijk-jStride-1      ]+x[ijk-1      ])  \
        + (beta_j[ijk        +kStride]-beta_j[ijk        -kStride]) * (x[ijk-jStride+kStride]-x[ijk+kStride]-x[ijk-jStride-kStride]+x[ijk-kStride])  \
        + (beta_k[ijk        +1      ]-beta_k[ijk        -1      ]) * (x[ijk-kStride+1      ]-x[ijk+1      ]-x[ijk-kStride-1      ]+x[ijk-1      ])  \
        + (beta_k[ijk        +jStride]-beta_k[ijk        -jStride]) * (x[ijk-kStride+jStride]-x[ijk+jStride]-x[ijk-kStride-jStride]+x[ijk-jStride])  \
                                                                                                                                                     \
        + (beta_i[ijk+1      +jStride]-beta_i[ijk+1      -jStride]) * (x[ijk+1      +jStride]-x[ijk+jStride]-x[ijk+1      -jStride]+x[ijk-jStride])  \
        + (beta_i[ijk+1      +kStride]-beta_i[ijk+1      -kStride]) * (x[ijk+1      +kStride]-x[ijk+kStride]-x[ijk+1      -kStride]+x[ijk-kStride])  \
        + (beta_j[ijk+jStride+1      ]-beta_j[ijk+jStride-1      ]) * (x[ijk+jStride+1      ]-x[ijk+1      ]-x[ijk+jStride-1      ]+x[ijk-1      ])  \
        + (beta_j[ijk+jStride+kStride]-beta_j[ijk+jStride-kStride]) * (x[ijk+jStride+kStride]-x[ijk+kStride]-x[ijk+jStride-kStride]+x[ijk-kStride])  \
        + (beta_k[ijk+kStride+1      ]-beta_k[ijk+kStride-1      ]) * (x[ijk+kStride+1      ]-x[ijk+1      ]-x[ijk+kStride-1      ]+x[ijk-1      ])  \
        + (beta_k[ijk+kStride+jStride]-beta_k[ijk+kStride-jStride]) * (x[ijk+kStride+jStride]-x[ijk+jStride]-x[ijk+kStride-jStride]+x[ijk-jStride])  \
      )                                                                                                                                              \
    )                                                                                                                                                \
  )
  #endif
#else // constant coefficient (don't bother differentiating between Poisson and Helmholtz)...
  #define apply_op_ijk(x)                 \
  (                                       \
    a*x[ijk] - b*h2inv*STENCIL_TWELFTH*(  \
       - 1.0*(x[ijk-2*kStride] +          \
              x[ijk-2*jStride] +          \
              x[ijk-2        ] +          \
              x[ijk+2        ] +          \
              x[ijk+2*jStride] +          \
              x[ijk+2*kStride] )          \
       +16.0*(x[ijk  -kStride] +          \
              x[ijk  -jStride] +          \
              x[ijk  -1      ] +          \
              x[ijk  +1      ] +          \
              x[ijk  +jStride] +          \
              x[ijk  +kStride] )          \
       -90.0*(x[ijk          ] )          \
    )                                     \
  )
#endif

//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_VARIABLE_COEFFICIENT
int stencil_get_radius(){return(2);} // stencil reaches out 2 cells
int stencil_get_shape(){return(STENCIL_SHAPE_NO_CORNERS);} // needs faces and edges, but not corners
#else
int stencil_get_radius(){return(2);} // stencil reaches out 2 cells
int stencil_get_shape(){return(STENCIL_SHAPE_STAR);} // needs just faces
#endif
//------------------------------------------------------------------------------------------------------------------------------
void rebuild_operator(level_type * level, level_type *fromLevel, double a, double b){
  // form restriction of alpha[], beta_*[] coefficients from fromLevel
  if(fromLevel != NULL){
    restriction(level,VECTOR_ALPHA ,fromLevel,VECTOR_ALPHA ,RESTRICT_CELL  );
    restriction(level,VECTOR_BETA_I,fromLevel,VECTOR_BETA_I,RESTRICT_FACE_I);
    restriction(level,VECTOR_BETA_J,fromLevel,VECTOR_BETA_J,RESTRICT_FACE_J);
    restriction(level,VECTOR_BETA_K,fromLevel,VECTOR_BETA_K,RESTRICT_FACE_K);
  } // else case assumes alpha/beta have been set

  // extrapolate the beta's into the ghost zones (needed for mixed derivatives)
  extrapolate_betas(level);
  //initialize_problem(level,level->h,a,b); // approach used for testing smooth beta's; destroys the black box nature of the solver

  // exchange alpha/beta/...  (must be done before calculating Dinv)
  exchange_boundary(level,VECTOR_ALPHA ,STENCIL_SHAPE_BOX); // safe
  exchange_boundary(level,VECTOR_BETA_I,STENCIL_SHAPE_BOX);
  exchange_boundary(level,VECTOR_BETA_J,STENCIL_SHAPE_BOX);
  exchange_boundary(level,VECTOR_BETA_K,STENCIL_SHAPE_BOX);

  // black box rebuild of D^{-1}, l1^{-1}, dominant eigenvalue, ...
  rebuild_operator_blackbox(level,a,b,4);

  // exchange Dinv/L1inv/...
  exchange_boundary(level,VECTOR_DINV ,STENCIL_SHAPE_BOX); // safe
  exchange_boundary(level,VECTOR_L1INV,STENCIL_SHAPE_BOX);
}


//------------------------------------------------------------------------------------------------------------------------------
#ifdef  USE_GSRB
#ifndef GSRB_IN_PLACE
#define GSRB_OOP
#endif
#define NUM_SMOOTHS      3 // RBRBRB
#include "operators/gsrb.c"
#elif   USE_CHEBY
#warning The Chebyshev smoother is currently underperforming for 4th order.  Please use -DUSE_GSRB or -DUSE_JACOBI
#define NUM_SMOOTHS      1
#define CHEBYSHEV_DEGREE 6 // i.e. one degree-6 polynomial smoother
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
#include "operators/boundary_fv.c"
#include "operators/restriction.c"
#include "operators/interpolation_v2.c"
#include "operators/interpolation_v4.c"
//------------------------------------------------------------------------------------------------------------------------------
void interpolation_vcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_v2(level_f,id_f,prescale_f,level_c,id_c);}
void interpolation_fcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_v4(level_f,id_f,prescale_f,level_c,id_c);}
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/problem.fv.c"
//------------------------------------------------------------------------------------------------------------------------------
