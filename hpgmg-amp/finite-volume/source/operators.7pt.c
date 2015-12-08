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
#ifdef USE_AMP
#include <amp.h>
#endif
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

#ifdef USE_AMP
#define AMP_RESTRICT
#else
#define AMP_RESTRICT __restrict__
#endif
//------------------------------------------------------------------------------------------------------------------------------
void apply_BCs(level_type * level, int x_id, int shape){
  #ifndef STENCIL_FUSE_BC
  // This is a failure mode if (trying to do communication-avoiding) && (BC!=BC_PERIODIC)
  apply_BCs_p1(level,x_id,shape);
  #endif
}
//------------------------------------------------------------------------------------------------------------------------------
// calculate Dinv?
#ifdef USE_AMP //<---------------------------------------------------------
  #pragma message "USE_AMP is defined"

  #define BBS AMP_TILE_BLOCKS
  #define KBS AMP_TILE_K
  #define JBS AMP_TILE_J
  #define IBS AMP_TILE_I

  #ifndef AMP_TILE_DIM
    #define AMP_TILE_DIM 0
  #endif

  #ifndef USE_LDS
  #pragma message "USE_LDS not defined; undefining USEl*"
  #undef USElxn
  #undef USElbk
  #undef USElbj
  #undef USElbi
  #undef USElval
  #endif // USE_LDS
  #if (BBS>1) && (AMP_DIM==4)
    #error LDS use is not yet implemented for 4 loops with BBS > 1
  #endif


  // All of this LDS initialization is based on STENCIL_SHAPE_STAR.
  // TODO: This could be generalized based on STENCIL_SHAPE and radius,
  // and moved out of the individual operators.*.c
  #ifdef USElxn
----------
    #pragma message "USElxn is defined"
    #if (AMP_DIM==4)&&(AMP_TILE_DIM==3)
      #define lxn(x,inck,incj,inci) (local_x[l_b][l_k+inck+1][l_j+incj+1][l_i+inci+1])
      #define LXN_SIZE (BBS*(KBS+2)*(JBS+2)*(IBS+2))
      #define DECLARE_LXN tile_static double local_x[BBS][KBS+2][JBS+2][IBS+2]
    #else
      #define lxn(x,inck,incj,inci) (local_x[l_k+inck+1][l_j+incj+1][l_i+inci+1])
      #define LXN_SIZE ((KBS+2)*(JBS+2)*(IBS+2))
      #define DECLARE_LXN tile_static double local_x[KBS+2][JBS+2][IBS+2]
      #define BBS 1
    #endif
    #if (BBS<1)||(KBS<1)||(JBS<1)||(IBS<8)
      #error BBS, KBS, and JBS must be >=1, IBS >= 8 for LDS use.
    #elif (KBS >=2)&&(JBS>=2)
      // In theory  this case can be done with 3 stores.
      // If both KBS and JBS are >= 2 and they are not both 4,
      // it can in theory be done in 2 stores.
      // I'm using 4 because it's easier.
      #define INITIALIZE_LXN(x)                             \
      {                                                     \
        int iadj = (l_i == 0) ? -1 : 1;                     \
        int jadj = (l_j == 0) ? -1 : 1;                     \
        int kadj = (l_k == 0) ? -1 : 1;                     \
        lxn(x,0,0,iadj) = x[ijk+iadj];                      \
        lxn(x,0,jadj,0) = x[ijk+jadj*jStride];              \
        lxn(x,kadj,0,0) = x[ijk+kadj*kStride];              \
        lxn(x,0,0,0) = x[ijk];                              \
      }
    #elif (KBS==1)&&(JBS==1)
      // Hardest case:
      // We only have IBS threads with which to initialize 5*IBS+2 locations
      // 6 is the minimum number of stores in this case.
      #define INITIALIZE_LXN(x)               \
      {                                       \
        int iadj = (l_i == 0) ? -1 : 1;       \
        lxn(x, 1,  0,   0) = x[ijk+kStride];  \
        lxn(x,-1,  0,   0) = x[ijk-kStride];  \
        lxn(x, 0,  1,   0) = x[ijk+jStride];  \
        lxn(x, 0, -1,   0) = x[ijk-jStride];  \
        lxn(x, 0,  0,iadj) = x[ijk+iadj];     \
        lxn(x, 0,  0,   0) = x[ijk];          \
      }
    #elif (KBS==1)
      // If one of KBS and JBS is 1 and the other 2, this requires a
      // minimum of 5 stores.  If one is 1 and the other >=4, it can
      // be done in 4 stores.
      #define INITIALIZE_LXN(x)                     \
      {                                             \
        int iadj = (l_i == 0) ? -1 : 1;             \
        int jadj = (l_j == 0) ? -1 : 1;             \
        lxn(x, 1,   0,   0) = x[ijk+kStride];       \
        lxn(x,-1,   0,   0) = x[ijk-kStride];       \
        lxn(x, 0,jadj,   0) = x[ijk+jadj*jStride];  \
        lxn(x, 0,   0,iadj) = x[ijk+iadj];          \
        lxn(x, 0,   0,   0) = x[ijk];               \
      }
    #elif (JBS==1)
      #define INITIALIZE_LXN(x)                     \
      {                                             \
        int iadj = (l_i == 0) ? -1 : 1;             \
        int kadj = (l_k == 0) ? -1 : 1;             \
        lxn(x,kadj,  0,   0) = x[ijk+kadj*kStride]; \
        lxn(x,   0,  1,   0) = x[ijk+jStride];      \
        lxn(x,   0, -1,   0) = x[ijk-jStride];      \
        lxn(x,   0,  0,iadj) = x[ijk+iadj];         \
        lxn(x,   0,  0,   0) = x[ijk];              \
      }
    #else
      #error Error in logic of LDS allocation
    #endif
  #else
    #pragma message "USElxn is not defined"
    #define lxn(x,inck,incj,inci) (x[ijk+inci+incj*jStride+inck*kStride])
    #define LXN_SIZE (0)
    #define DECLARE_LXN
    #define INITIALIZE_LXN(x)
  #endif

  // Note that, while the use of lxn reaches out +-1 in 3 dimensions,
  // the lb* only reach out +1 in 1 dimension.  This simplifies initialization.
  #ifdef USElbk
    #if (AMP_DIM==4)&&(AMP_TILE_DIM==3)
      #define lbk(inck) (local_beta_k[l_b][l_k+inck][l_j][l_i])
      #define LBK_SIZE (BBS*(KBS+1)*JBS*IBS)
      #define DECLARE_LBK tile_static double local_beta_k[BBS][KBS+1][JBS][IBS]
    #else
      #define lbk(inck) (local_beta_k[l_k+inck][l_j][l_i])
      #define LBK_SIZE ((KBS+1)*JBS*IBS)
      #define DECLARE_LBK tile_static double local_beta_k[KBS+1][JBS][IBS]
    #endif
    #define INITIALIZE_LBK()                              \
    {                                                     \
      lbk(1) = beta_k[ijk+kStride];                       \
      lbk(0) = beta_k[ijk];                               \
    }
  #else
    #pragma message "USElbk is not defined"
    #define lbk(inck) (beta_k[ijk+inck*kStride])
    #define LBK_SIZE (0)
    #define DECLARE_LBK
    #define INITIALIZE_LBK()
  #endif

  #ifdef USElbj
    #pragma message "USElbj is defined"
    #define lbj(incj) (local_beta_j[l_k][l_j+incj][l_i])
    #define LBJ_SIZE (KBS*(JBS+1)*IBS)
    #define DECLARE_LBJ tile_static double local_beta_j[KBS][JBS+1][IBS]
    #define INITIALIZE_LBJ()                              \
    {                                                     \
      lbj(1) = beta_j[ijk+jStride];                       \
      lbj(0) = beta_j[ijk];                                \
    }
  #else
    #pragma message "USElbj is not defined"
    #define lbj(incj) (beta_j[ijk+incj*jStride])
    #define LBJ_SIZE (0)
    #define DECLARE_LBJ
    #define INITIALIZE_LBJ()
  #endif

  #ifdef USElbi
    #pragma message "USElbi is defined"
    #define lbi(inci) (local_beta_i[l_k][l_j][l_i+inci])
    #define LBI_SIZE (KBS*JBS*(IBS+1))
    #define DECLARE_LBI tile_static double local_beta_i[KBS][JBS][IBS+1]
    #define INITIALIZE_LBI()                              \
    {                                                     \
      lbi(1) = beta_i[ijk+1];                             \
      lbi(0) = beta_i[ijk];                               \
    }
  #else
    #pragma message "USElbi is not defined"
    #define lbi(inci) (beta_i[ijk+inci])
    #define LBI_SIZE (0)
    #define DECLARE_LBI
    #define INITIALIZE_LBI()
  #endif

  #ifdef USElval
    #pragma message "USElval is defined"
    #if (KBS==1)||(JBS==1)
      #error lval initialization needs to be fixed for KBS or JBS == 1.
    #endif
    #define lval(inck,incj,inci) (local_valid[l_k+inck+1][l_j+incj+1][l_i+inci+1])
    #define LVAL_SIZE ((KBS+2)*(JBS+2)*(IBS+2))
    #define DECLARE_LVAL tile_static double local_valid[KBS+2][JBS+2][IBS+2]
    #define INITIALIZE_LVAL()                             \
    {                                                     \
      int iadj = (l_i == 0) ? -1 : 1;                     \
      int jadj = (l_j == 0) ? -1 : 1;                     \
      int kadj = (l_k == 0) ? -1 : 1;                     \
      lval(0,0,iadj) = valid[ijk+iadj];                        \
      lval(0,jadj,0) = valid[ijk+jadj*jStride];                \
      lval(kadj,0,0) = valid[ijk+kadj*kStride];                \
      lval(0,0,0) = valid[ijk];                                \
    }
  #else
    #pragma message "USElval is not defined"
    #define lval(inck,incj,inci) (valid[ijk+inci+incj*jStride+inck*kStride])
    #define LVAL_SIZE (0)
    #define DECLARE_LVAL
    #define INITIALIZE_LVAL()
  #endif


#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
  #define calculate_Dinv_amp()                 \
  (                                            \
    1.0 / (a*alpha[ijk] - b*h2inv*(            \
             + lbi(0)*( lval( 0, 0,-1) - 2.0 ) \
             + lbj(0)*( lval( 0,-1, 0) - 2.0 ) \
             + lbk(0)*( lval(-1, 0, 0) - 2.0 ) \
             + lbi(1)*( lval( 0, 0, 1) - 2.0 ) \
             + lbj(1)*( lval( 0, 1, 0) - 2.0 ) \
             + lbk(1)*( lval( 1, 0, 0) - 2.0 ) \
          ))                                   \
  )
  #else // variable coefficient Poisson ...
  #define calculate_Dinv_amp()                 \
  (                                            \
    1.0 / ( -b*h2inv*(                         \
             + lbi(0)*( lval( 0, 0,-1) - 2.0 ) \
             + lbj(0)*( lval( 0,-1, 0) - 2.0 ) \
             + lbk(0)*( lval(-1, 0, 0) - 2.0 ) \
             + lbi(1)*( lval( 0, 0, 1) - 2.0 ) \
             + lbj(1)*( lval( 0, 1, 0) - 2.0 ) \
             + lbk(1)*( lval( 1, 0, 0) - 2.0 ) \
          ))                                   \
  )
  #endif
#else // constant coefficient case... 
  #define calculate_Dinv_amp() \
  (                            \
    1.0 / (a - b*h2inv*(       \
             + lval( 0, 0,-1)  \
             + lval( 0,-1, 0)  \
             + lval(-1, 0, 0)  \
             + lval( 0, 0, 1)  \
             + lval( 0, 1, 0)  \
             + lval( 1, 0, 0)  \
             - 12.0            \
          ))                   \
  )
#endif // STENCIL_VARIABLE_COEFFICIENT

#if defined(STENCIL_FUSE_DINV) && defined(STENCIL_FUSE_BC)
#define Dinv_ijk_amp() calculate_Dinv_amp() // recalculate it
#else
#define Dinv_ijk_amp() Dinv[ijk]        // simply retrieve it rather than recalculating it
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
    #define apply_op_ijk_amp(x)                                                          \
    (                                                                                    \
      a*alpha[ijk]*lxn(x,0,0,0)                                                          \
      -b*h2inv*(                                                                         \
        + lbi(0)*( lval( 0, 0,-1)*( lxn(x,0,0,0) + lxn(x, 0, 0,-1)) - 2.0*lxn(x,0,0,0) ) \
        + lbj(0)*( lval( 0,-1, 0)*( lxn(x,0,0,0) + lxn(x, 0,-1, 0)) - 2.0*lxn(x,0,0,0) ) \
        + lbk(0)*( lval(-1, 0, 0)*( lxn(x,0,0,0) + lxn(x,-1, 0, 0)) - 2.0*lxn(x,0,0,0) ) \
        + lbi(1)*( lval( 0, 0, 1)*( lxn(x,0,0,0) + lxn(x, 0, 0, 1)) - 2.0*lxn(x,0,0,0) ) \
        + lbj(1)*( lval( 0, 1, 0)*( lxn(x,0,0,0) + lxn(x, 0, 1, 0)) - 2.0*lxn(x,0,0,0) ) \
        + lbk(1)*( lval( 1, 0, 0)*( lxn(x,0,0,0) + lxn(x, 1, 0, 0)) - 2.0*lxn(x,0,0,0) ) \
      )                                                                                  \
    )
    #else // variable coefficient Poisson ...
    #define apply_op_ijk_amp(x)                                                          \
    (                                                                                    \
      -b*h2inv*(                                                                         \
        + lbi(0)*( lval( 0, 0,-1)*( lxn(x,0,0,0) + lxn(x, 0, 0,-1)) - 2.0*lxn(x,0,0,0) ) \
        + lbj(0)*( lval( 0,-1, 0)*( lxn(x,0,0,0) + lxn(x, 0,-1, 0)) - 2.0*lxn(x,0,0,0) ) \
        + lbk(0)*( lval(-1, 0, 0)*( lxn(x,0,0,0) + lxn(x,-1, 0, 0)) - 2.0*lxn(x,0,0,0) ) \
        + lbi(1)*( lval( 0, 0, 1)*( lxn(x,0,0,0) + lxn(x, 0, 0, 1)) - 2.0*lxn(x,0,0,0) ) \
        + lbj(1)*( lval( 0, 1, 0)*( lxn(x,0,0,0) + lxn(x, 0, 1, 0)) - 2.0*lxn(x,0,0,0) ) \
        + lbk(1)*( lval( 1, 0, 0)*( lxn(x,0,0,0) + lxn(x, 1, 0, 0)) - 2.0*lxn(x,0,0,0) ) \
      )                                                                                  \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk_amp(x)                            \
    (                                                      \
      a*lxn(x,0,0,0) - b*h2inv*(                           \
        + lval( 0, 0,-1)*( lxn(x,0,0,0) + lxn(x, 0, 0,-1)) \
        + lval( 0,-1, 0)*( lxn(x,0,0,0) + lxn(x, 0,-1, 0)) \
        + lval(-1, 0, 0)*( lxn(x,0,0,0) + lxn(x,-1, 0, 0)) \
        + lval( 0, 0, 1)*( lxn(x,0,0,0) + lxn(x, 0, 0, 1)) \
        + lval( 0, 1, 0)*( lxn(x,0,0,0) + lxn(x, 0, 1, 0)) \
        + lval( 1, 0, 0)*( lxn(x,0,0,0) + lxn(x, 1, 0, 0)) \
                       -12.0*( lxn(x,0,0,0)              ) \
      )                                                    \
    )
  #endif // variable/constant coefficient

#else // STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz...
    #define apply_op_ijk_amp(x)                       \
    (                                             \
      a*alpha[ijk]*lxn(x,0,0,0)                   \
     -b*h2inv*(                                   \
        + lbi(1)*( lxn(x,0,0, 1) - lxn(x,0,0,0) ) \
        + lbi(0)*( lxn(x,0,0,-1) - lxn(x,0,0,0) ) \
        + lbj(1)*( lxn(x,0, 1,0) - lxn(x,0,0,0) ) \
        + lbj(0)*( lxn(x,0,-1,0) - lxn(x,0,0,0) ) \
        + lbk(1)*( lxn(x, 1,0,0) - lxn(x,0,0,0) ) \
        + lbk(0)*( lxn(x,-1,0,0) - lxn(x,0,0,0) ) \
      )                                           \
    )
    #else // variable coefficient Poisson...
    #define apply_op_ijk_amp(x)                       \
    (                                             \
      -b*h2inv*(                                  \
        + lbi(1)*( lxn(x,0,0, 1) - lxn(x,0,0,0) ) \
        + lbi(0)*( lxn(x,0,0,-1) - lxn(x,0,0,0) ) \
        + lbj(1)*( lxn(x,0, 1,0) - lxn(x,0,0,0) ) \
        + lbj(0)*( lxn(x,0,-1,0) - lxn(x,0,0,0) ) \
        + lbk(1)*( lxn(x, 1,0,0) - lxn(x,0,0,0) ) \
        + lbk(0)*( lxn(x,-1,0,0) - lxn(x,0,0,0) ) \
      )                                           \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk_amp(x)            \
    (                            \
      a*lxn(x,0,0,0) - b*h2inv*(  \
        + lxn(x,0,0,1)           \
        + lxn(x,0,0,-1)          \
        + lxn(x,0,1,0)           \
        + lxn(x,0,-1,0)          \
        + lxn(x,1,0,0)           \
        + lxn(x,-1,0,0)          \
        - lxn(x,0,0,0)*6.0       \
      )                          \
    )
  #endif // variable/constant coefficient

#endif // STENCIL_FUSE_BC

#define LDS_USE (LXN_SIZE + LBK_SIZE + LBJ_SIZE + LBI_SIZE + LVAL_SIZE)

void print_amp_details(void)
{
  fprintf(stderr, "Using GPU (C++ AMP/Kalmar)\n");
  fprintf(stderr, "  to parallelize %d loops with %d tile dimensions",
          AMP_DIM, AMP_TILE_DIM);

  switch(AMP_TILE_DIM) {
      case 0:
          fprintf(stderr, "<>\n");
          break;
      case 1:
          fprintf(stderr, "<%d>\n", IBS);
          break;
      case 2:
          fprintf(stderr, "<%d,%d>\n", JBS, IBS);
          break;
      case 3:
          if (AMP_DIM==4) {
              fprintf(stderr, "<%d*%d,%d,%d>\n", BBS, KBS, JBS, IBS);
          } else {
              fprintf(stderr, "<%d,%d,%d>\n", KBS, JBS, IBS);
          }
          break;
      default:
          fprintf(stderr, "AMP_TILE_DIM must be 0,1,2, or 3.\n");
          exit(1);
  }

  #ifdef USE_HELMHOLTZ
    fprintf(stderr, "   solving HELMHOLTZ\n");
    fprintf(stderr, "   alpha is used, but probably not worth putting in LDS\n");
  #else // USE_HELMHOLTZ
    fprintf(stderr, "   solving POISSON\n");
    fprintf(stderr, "   alpha is not used\n");
  #endif // USE_HELMHOLTZ

  fprintf(stderr, "   7pt operator\n");

  #if defined(USE_CHEBY)
    fprintf(stderr, "   CHEBY smoother\n");
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

  #ifdef STENCIL_FUSE_BC
    fprintf(stderr, "   fusing boundary conditions\n");
    fprintf(stderr, "   using valid[] for apply_op\n");
  #else // STENCIL_FUSE_BC
    fprintf(stderr, "   not fusing boundary conditions\n");
    fprintf(stderr, "   not using valid[] for apply_op\n");
  #endif // STENCIL_FUSE_BC

  #ifdef STENCIL_FUSE_DINV
    fprintf(stderr, "   fusing Dinv\n");
    #ifdef STENCIL_FUSE_BC
      fprintf(stderr, "   Dinv will be calculated on the fly\n");
      fprintf(stderr, "   using valid[] for Dinv\n");
    #else // STENCIL_FUSE_BC
      fprintf(stderr, "   Odd: Dinv is fused but not BC???\n");
    #endif // STENCIL_FUSE_BC
  #else // STENCIL_FUSE_DINV
    fprintf(stderr, "   not fusing Dinv\n");
    fprintf(stderr, "   Dinv will be looked up\n");
  #endif // STENCIL_FUSE_DINV

  #ifdef USE_LDS
  #if defined(AMP_TILE_DIM) && (AMP_TILE_DIM != 0)
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

  #ifdef USElval
    fprintf(stderr, "   using LDS for valid\n");
    #if !defined(STENCIL_FUSE_BC) && (!defined(STENCIL_FUSE_DINV) || !defined(STENCIL_FUSE_BC))
      fprintf(stderr, "   Wasting LDS for lval, which is not used in this case\n");
    #endif
  #else // USElval
    fprintf(stderr, "   not using LDS for valid\n");
  #endif // USElval

  fprintf(stderr, "   using %d local doubles per large tile\n", LDS_USE);
  if (LDS_USE > 4096) {
    fflush(stdout);
    fflush(stderr);
    fprintf(stderr, "   Too many local doubles\n");
    fflush(stderr);
    exit(0);
  }
  #else // AMP_TILE_DIM
  fprintf(stderr, "   No GPU tiling, so no LDS use; THRESHOLD is %d\n", AMP_INNER_THRESHOLD);
  #endif // AMP_TILE_DIM
  #else // USE_LDS
    fprintf(stderr, "   Not using LDS\n");
  #endif // USE_LDS

}

inline void print_amp_info(void)
{
  static bool printTileInfo = true;
  if (printTileInfo) {
    print_amp_details();
    printTileInfo = false;
  }
}
#else // USE_AMP
inline void print_amp_info(void) {}
#endif // USE_AMP <-----------------------------------------------

#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / (a*alpha[ijk] - b*h2inv*(                             \
             + beta_i[ijk        ]*( valid[ijk-1      ] - 2.0 ) \
             + beta_j[ijk        ]*( valid[ijk-jStride] - 2.0 ) \
             + beta_k[ijk        ]*( valid[ijk-kStride] - 2.0 ) \
             + beta_i[ijk+1      ]*( valid[ijk+1      ] - 2.0 ) \
             + beta_j[ijk+jStride]*( valid[ijk+jStride] - 2.0 ) \
             + beta_k[ijk+kStride]*( valid[ijk+kStride] - 2.0 ) \
          ))                                                    \
  )
  #else // variable coefficient Poisson ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / ( -b*h2inv*(                                          \
             + beta_i[ijk        ]*( valid[ijk-1      ] - 2.0 ) \
             + beta_j[ijk        ]*( valid[ijk-jStride] - 2.0 ) \
             + beta_k[ijk        ]*( valid[ijk-kStride] - 2.0 ) \
             + beta_i[ijk+1      ]*( valid[ijk+1      ] - 2.0 ) \
             + beta_j[ijk+jStride]*( valid[ijk+jStride] - 2.0 ) \
             + beta_k[ijk+kStride]*( valid[ijk+kStride] - 2.0 ) \
          ))                                                    \
  )
  #endif
#else // constant coefficient case... 
  #define calculate_Dinv()          \
  (                                 \
    1.0 / (a - b*h2inv*(            \
             + valid[ijk-1      ]   \
             + valid[ijk-jStride]   \
             + valid[ijk-kStride]   \
             + valid[ijk+1      ]   \
             + valid[ijk+jStride]   \
             + valid[ijk+kStride]   \
             - 12.0                 \
          ))                        \
  )
#endif

#if defined(STENCIL_FUSE_DINV) && defined(STENCIL_FUSE_BC)
#define Dinv_ijk() calculate_Dinv() // recalculate it
#else
#define Dinv_ijk() Dinv[ijk]        // simply retrieve it rather than recalculating it
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      a*alpha[ijk]*x[ijk]                                                                     \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk        ]*( valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk        ]*( valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) - 2.0*x[ijk] ) \
        + beta_i[ijk+1      ]*( valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk+jStride]*( valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk+kStride]*( valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) - 2.0*x[ijk] ) \
      )                                                                                       \
    )
    #else // variable coefficient Poisson ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk        ]*( valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk        ]*( valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) - 2.0*x[ijk] ) \
        + beta_i[ijk+1      ]*( valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk+jStride]*( valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk+kStride]*( valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) - 2.0*x[ijk] ) \
      )                                                                                       \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)                                \
    (                                                    \
      a*x[ijk] - b*h2inv*(                               \
        + valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) \
        + valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) \
        + valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) \
        + valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) \
        + valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) \
        + valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) \
                       -12.0*( x[ijk]                  ) \
      )                                                  \
    )
  #endif // variable/constant coefficient

#endif


//------------------------------------------------------------------------------------------------------------------------------
#ifndef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz...
    #define apply_op_ijk(x)                               \
    (                                                     \
      a*alpha[ijk]*x[ijk]                                 \
     -b*h2inv*(                                           \
        + beta_i[ijk+1      ]*( x[ijk+1      ] - x[ijk] ) \
        + beta_i[ijk        ]*( x[ijk-1      ] - x[ijk] ) \
        + beta_j[ijk+jStride]*( x[ijk+jStride] - x[ijk] ) \
        + beta_j[ijk        ]*( x[ijk-jStride] - x[ijk] ) \
        + beta_k[ijk+kStride]*( x[ijk+kStride] - x[ijk] ) \
        + beta_k[ijk        ]*( x[ijk-kStride] - x[ijk] ) \
      )                                                   \
    )
    #else // variable coefficient Poisson...
    #define apply_op_ijk(x)                               \
    (                                                     \
      -b*h2inv*(                                          \
        + beta_i[ijk+1      ]*( x[ijk+1      ] - x[ijk] ) \
        + beta_i[ijk        ]*( x[ijk-1      ] - x[ijk] ) \
        + beta_j[ijk+jStride]*( x[ijk+jStride] - x[ijk] ) \
        + beta_j[ijk        ]*( x[ijk-jStride] - x[ijk] ) \
        + beta_k[ijk+kStride]*( x[ijk+kStride] - x[ijk] ) \
        + beta_k[ijk        ]*( x[ijk-kStride] - x[ijk] ) \
      )                                                   \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)            \
    (                                \
      a*x[ijk] - b*h2inv*(           \
        + x[ijk+1      ]             \
        + x[ijk-1      ]             \
        + x[ijk+jStride]             \
        + x[ijk-jStride]             \
        + x[ijk+kStride]             \
        + x[ijk-kStride]             \
        - x[ijk        ]*6.0         \
      )                              \
    )
  #endif // variable/constant coefficient

#endif // BCs


//------------------------------------------------------------------------------------------------------------------------------
int stencil_get_radius(){return(1);} // 7pt reaches out 1 point
int stencil_get_shape(){return(STENCIL_SHAPE_STAR);} // needs just faces
//------------------------------------------------------------------------------------------------------------------------------
void rebuild_operator(level_type * level, level_type *fromLevel, double a, double b){
  if(level->my_rank==0){fprintf(stdout,"  rebuilding operator for level...  h=%e  ",level->h);}

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // form restriction of alpha[], beta_*[] coefficients from fromLevel
  if(fromLevel != NULL){
    restriction(level,VECTOR_ALPHA ,fromLevel,VECTOR_ALPHA ,RESTRICT_CELL  );
    restriction(level,VECTOR_BETA_I,fromLevel,VECTOR_BETA_I,RESTRICT_FACE_I);
    restriction(level,VECTOR_BETA_J,fromLevel,VECTOR_BETA_J,RESTRICT_FACE_J);
    restriction(level,VECTOR_BETA_K,fromLevel,VECTOR_BETA_K,RESTRICT_FACE_K);
  } // else case assumes alpha/beta have been set


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // exchange alpha/beta/...  (must be done before calculating Dinv)
  exchange_boundary(level,VECTOR_ALPHA ,STENCIL_SHAPE_BOX); // safe
  exchange_boundary(level,VECTOR_BETA_I,STENCIL_SHAPE_BOX);
  exchange_boundary(level,VECTOR_BETA_J,STENCIL_SHAPE_BOX);
  exchange_boundary(level,VECTOR_BETA_K,STENCIL_SHAPE_BOX);


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // calculate Dinv, L1inv, and estimate the dominant Eigenvalue
  double _timeStart = getTime();
  int block;

  double dominant_eigenvalue = -1e9;

  PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,block,level->num_my_blocks,dominant_eigenvalue)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double h2inv = 1.0/(level->h*level->h);
    double * __restrict__ alpha  = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
    double * __restrict__ beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
    double * __restrict__ beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
    double * __restrict__ beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
    double * __restrict__   Dinv = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
    double * __restrict__  L1inv = level->my_boxes[box].vectors[VECTOR_L1INV ] + ghosts*(1+jStride+kStride);
    double * __restrict__  valid = level->my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride);
    double block_eigenvalue = -1e9;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){ 
      int ijk = i + j*jStride + k*kStride;

      #ifdef STENCIL_VARIABLE_COEFFICIENT
      // radius of Gershgorin disc is the sum of the absolute values of the off-diagonal elements...
      double sumAbsAij = fabs(b*h2inv) * (
                           fabs( beta_i[ijk        ]*valid[ijk-1      ] )+
                           fabs( beta_j[ijk        ]*valid[ijk-jStride] )+
                           fabs( beta_k[ijk        ]*valid[ijk-kStride] )+
                           fabs( beta_i[ijk+1      ]*valid[ijk+1      ] )+
                           fabs( beta_j[ijk+jStride]*valid[ijk+jStride] )+
                           fabs( beta_k[ijk+kStride]*valid[ijk+kStride] )
                         );

      // center of Gershgorin disc is the diagonal element...
      double    Aii = a*alpha[ijk] - b*h2inv*(
                        beta_i[ijk        ]*( valid[ijk-1      ]-2.0 )+
                        beta_j[ijk        ]*( valid[ijk-jStride]-2.0 )+
                        beta_k[ijk        ]*( valid[ijk-kStride]-2.0 )+
                        beta_i[ijk+1      ]*( valid[ijk+1      ]-2.0 )+
                        beta_j[ijk+jStride]*( valid[ijk+jStride]-2.0 )+
                        beta_k[ijk+kStride]*( valid[ijk+kStride]-2.0 ) 
                      );
      #else // Constant coefficient versions with fused BC's...
      // radius of Gershgorin disc is the sum of the absolute values of the off-diagonal elements...
      double sumAbsAij = fabs(b*h2inv) * (
                           valid[ijk-1      ] +
                           valid[ijk-jStride] +
                           valid[ijk-kStride] +
                           valid[ijk+1      ] +
                           valid[ijk+jStride] +
                           valid[ijk+kStride] 
                         );

      // center of Gershgorin disc is the diagonal element...
      double    Aii = a - b*h2inv*(
                         valid[ijk-1      ] +
                         valid[ijk-jStride] +
                         valid[ijk-kStride] +
                         valid[ijk+1      ] +
                         valid[ijk+jStride] +
                         valid[ijk+kStride] - 12.0
                      );
      #endif

      // calculate Dinv = D^{-1}, L1inv = ( D+D^{L1} )^{-1}, and the dominant eigenvalue...
                             Dinv[ijk] = 1.0/Aii;				// inverse of the diagonal Aii
                          //L1inv[ijk] = 1.0/(Aii+sumAbsAij);			// inverse of the L1 row norm... L1inv = ( D+D^{L1} )^{-1}
      if(Aii>=1.5*sumAbsAij)L1inv[ijk] = 1.0/(Aii              ); 		// as suggested by eq 6.5 in Baker et al, "Multigrid smoothers for ultra-parallel computing: additional theory and discussion"...
                       else L1inv[ijk] = 1.0/(Aii+0.5*sumAbsAij);		// 
      double Di = (Aii + sumAbsAij)/Aii;if(Di>block_eigenvalue)block_eigenvalue=Di;	// upper limit to Gershgorin disc == bound on dominant eigenvalue
    }}}
    if(block_eigenvalue>dominant_eigenvalue){dominant_eigenvalue = block_eigenvalue;}
  }
  level->timers.blas1 += (double)(getTime()-_timeStart);


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // Reduce the local estimates dominant eigenvalue to a global estimate
  #ifdef USE_MPI
  double _timeStartAllReduce = getTime();
  double send = dominant_eigenvalue;
  MPI_Allreduce(&send,&dominant_eigenvalue,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  double _timeEndAllReduce = getTime();
  level->timers.collectives   += (double)(_timeEndAllReduce-_timeStartAllReduce);
  #endif
  if(level->my_rank==0){fprintf(stdout,"eigenvalue_max<%e\n",dominant_eigenvalue);}
  level->dominant_eigenvalue_of_DinvA = dominant_eigenvalue;


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // exchange Dinv/L1inv/...
  exchange_boundary(level,VECTOR_DINV ,STENCIL_SHAPE_BOX); // safe
  exchange_boundary(level,VECTOR_L1INV,STENCIL_SHAPE_BOX);
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
}


//------------------------------------------------------------------------------------------------------------------------------
#ifdef  USE_GSRB
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
#define NUM_SMOOTHS      2
#include "operators/symgs.c"
#else
#error You must compile with either -DUSE_GSRB, -DUSE_CHEBY, -DUSE_JACOBI, -DUSE_L1JACOBI, or -DUSE_SYMGS
#endif
#include "operators/residual.c"
#include "operators/apply_op.c"
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/blockCopy.c"
#include "operators/misc.c"
#include "operators/exchange_boundary.c"
#include "operators/boundary_fd.c"
#include "operators/matmul.c"
#include "operators/restriction.c"
#include "operators/interpolation_p0.c"
#include "operators/interpolation_p1.c"
//------------------------------------------------------------------------------------------------------------------------------
void interpolation_vcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_p0(level_f,id_f,prescale_f,level_c,id_c);}
void interpolation_fcycle(level_type * level_f, int id_f, double prescale_f, level_type *level_c, int id_c){interpolation_p1(level_f,id_f,prescale_f,level_c,id_c);}
//------------------------------------------------------------------------------------------------------------------------------
#include "operators/problem.p6.c"
//------------------------------------------------------------------------------------------------------------------------------
