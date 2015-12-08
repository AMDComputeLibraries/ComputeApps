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
#if   defined(GSRB_FP)
  #warning Overriding default GSRB implementation and using pre-computed 1.0/0.0 FP array for Red-Black to facilitate vectorization...
#elif defined(GSRB_STRIDE2)
  #if defined(GSRB_OOP)
  #warning Overriding default GSRB implementation and using out-of-place and stride-2 accesses to minimize the number of flops
  #else
  #warning Overriding default GSRB implementation and using stride-2 accesses to minimize the number of flops
  #endif
#elif defined(GSRB_BRANCH)
  #if defined(GSRB_OOP)
  #warning Overriding default GSRB implementation and using out-of-place implementation with an if-then-else on loop indices...
  #else
  #warning Overriding default GSRB implementation and using if-then-else on loop indices...
  #endif
#else
#define GSRB_STRIDE2 // default implementation
#endif
//------------------------------------------------------------------------------------------------------------------------------
//
#define INITIALIZE_CONSTS \
        const int box = level->my_blocks[block].read.box; \
        const int klo = level->my_blocks[block].read.k; \
        const int jlo = level->my_blocks[block].read.j; \
        const int ilo = level->my_blocks[block].read.i; \
        const int idim = level->my_blocks[block].dim.i ; \
        const int jdim = level->my_blocks[block].dim.j ; \
        const int kdim = level->my_blocks[block].dim.k ; \
        const int ihi = idim + ilo; \
        const int jhi = jdim + jlo; \
        const int khi = kdim + klo; \
        const double h2inv = 1.0/(level->h*level->h); \
        const int ghosts =  level->box_ghosts; \
        const int jStride = level->my_boxes[box].jStride; \
        const int kStride = level->my_boxes[box].kStride; \
        const double * AMP_RESTRICT rhs      = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT alpha    = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT beta_i   = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT beta_j   = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT beta_k   = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT Dinv     = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT valid    = level->my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); \
        const int color000 = (level->my_boxes[box].low.i^level->my_boxes[box].low.j^level->my_boxes[box].low.k^s)&1;

#ifdef GSRB_OOP
#define INITIALIZE_x_n                                                                           \
        const double * AMP_RESTRICT x_n; double * AMP_RESTRICT x_np1;                                                      \
          if((s&1)==0) {                                                                         \
            x_n      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); \
            x_np1    = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride); \
          } else {                                                                               \
            x_n      = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride); \
            x_np1    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); \
          }
#else // GSRB_OOP
#define INITIALIZE_x_n                                                                            \
        const double * AMP_RESTRICT x_n    = level->my_boxes[box].vectors[x_id] + ghosts*(1+jStride+kStride);  \
              double * AMP_RESTRICT x_np1  = level->my_boxes[box].vectors[x_id] + ghosts*(1+jStride+kStride);
#endif // GSRB_OOP

#if defined(GSRB_FP)

#define LDS_LOOP_BODY const double * RedBlack =                                               \
                        level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1); \
                      const int ij  = i + j*jStride;                                          \
                      const int ijk = i + j*jStride + k*kStride;                              \
                      DECLARE_LXN; INITIALIZE_LXN(x_n);                                       \
                      DECLARE_LBK; INITIALIZE_LBK();                                          \
                      DECLARE_LBJ; INITIALIZE_LBJ();                                          \
                      DECLARE_LBI; INITIALIZE_LBI();                                          \
                      tidx.barrier.wait_with_tile_static_memory_fence();                      \
                      double Ax     = apply_op_ijk_amp(x_n);                                  \
                      double lambda =     Dinv_ijk_amp();                                     \
                      x_np1[ijk] = lxn(x_n,0,0,0) + RedBlack[ij]*lambda*(rhs[ijk]-Ax);

// If the compiler does its job well, it will pull the initialization of RedBlack
// outside the two inner loops
#define LOOP_BODY const double * RedBlack =                                               \
                    level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1); \
                  int ij  = i + j*jStride;                                                \
                  int ijk = i + j*jStride + k*kStride;                                    \
                  double Ax     = apply_op_ijk(x_n);                                      \
                  double lambda =     Dinv_ijk();                                         \
                  x_np1[ijk] = x_n[ijk] + RedBlack[ij]*lambda*(rhs[ijk]-Ax);

// But we'll define this separately so as not to disadvantage the non-amp implementation
#define SERIAL_LOOP for(int k=klo;k<khi;k++) {                                                    \
                      const double * __restrict__ RedBlack =                                  \
                        level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1); \
                      for(int j=jlo;j<jhi;j++){                                                   \
                        for(int i=ilo;i<ihi;i++){                                                 \
                          int ij  = i + j*jStride;                                            \
                          int ijk = i + j*jStride + k*kStride;                                \
                          double Ax     = apply_op_ijk(x_n);                                  \
                          double lambda =     Dinv_ijk();                                     \
                          x_np1[ijk] = x_n[ijk] + RedBlack[ij]*lambda*(rhs[ijk]-Ax);          \
                    }}}
#elif defined(GSRB_STRIDE2)
#if defined(GSRB_OOP)
#define LDS_LOOP_BODY NOT IMPLEMENTED YET
#define LOOP_BODY NOT IMPLEMENTED YET
#define SERIAL_LOOP for(int k=klo;k<khi;k++){ \
                      for(int j=jlo;j<jhi;j++){ \
                        for(int i=ilo;i<ihi;i++){                           \
                          int ijk = i + j*jStride + k*kStride;              \
                          x_np1[ijk] = x_n[ijk];                            \
                        }                                                   \
                        for(int i=ilo+((ilo^j^k^color000)&1);i<ihi;i+=2){   \
                          int ijk = i + j*jStride + k*kStride;              \
                          double Ax     = apply_op_ijk(x_n);                \
                          double lambda =     Dinv_ijk();                   \
                          x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);     \
                        }                                                   \
                      }}
#else // GSRB_OOP
#define LDS_LOOP_BODY NOT IMPLEMENTED YET
#define LOOP_BODY NOT IMPLEMENTED YET
#define SERIAL_LOOP for(int k=klo;k<khi;k++){                               \
                      for(int j=jlo;j<jhi;j++){                             \
                        for(int i=ilo+((ilo^j^k^color000)&1);i<ihi;i+=2){   \
                          int ijk = i + j*jStride + k*kStride;          \
                          double Ax     = apply_op_ijk(x_n);            \
                          double lambda =     Dinv_ijk();               \
                          x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax); \
                        }                                               \
                    }}
#endif // GSRB_OOP

#elif defined(GSRB_BRANCH)
#if defined(GSRB_OOP)
#define LDS_LOOP_BODY const int ijk = i + j*jStride + k*kStride;            \
                      DECLARE_LXN; INITIALIZE_LXN(x_n);                     \
                      DECLARE_LBK; INITIALIZE_LBK();                        \
                      DECLARE_LBJ; INITIALIZE_LBJ();                        \
                      DECLARE_LBI; INITIALIZE_LBI();                        \
                      tidx.barrier.wait_with_tile_static_memory_fence();    \
                      if((i^j^k^color000^1)&1){                             \
                        double Ax   = apply_op_ijk_amp(x_n);                \
                        double lambda =     Dinv_ijk_amp();                 \
                        x_np1[ijk] = lxn(x_n,0,0,0) + lambda*(rhs[ijk]-Ax); \
                      }else{                                                \
                          x_np1[ijk] =  lxn(x_n,0,0,0);                     \
                      }

// The ?: form seems friendlier to AMP than the if/then form
// Make sure this doesn't break the compiler -- DEBUG
#define LOOP_BODY int ijk = i + j*jStride + k*kStride;            \
                  if((i^j^k^color000^1)&1){                       \
                    double Ax   = apply_op_ijk(x_n);              \
                    double lambda =     Dinv_ijk();               \
                    x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax); \
                  }else{                                          \
                      x_np1[ijk] =  x_n[ijk];                     \
                  }
#else // GSRB_OOP
#define LDS_LOOP_BODY const int ijk = i + j*jStride + k*kStride;            \
                      DECLARE_LXN; INITIALIZE_LXN(x_n);                     \
                      DECLARE_LBK; INITIALIZE_LBK();                        \
                      DECLARE_LBJ; INITIALIZE_LBJ();                        \
                      DECLARE_LBI; INITIALIZE_LBI();                        \
                      tidx.barrier.wait_with_tile_static_memory_fence();    \
                      if((i^j^k^color000^1)&1){                             \
                        double Ax   = apply_op_ijk_amp(x_n);                \
                        double lambda =     Dinv_ijk_amp();                 \
                        x_np1[ijk] = lxn(x_n,0,0,0) + lambda*(rhs[ijk]-Ax); \
                      }
#define LOOP_BODY int ijk = i + j*jStride + k*kStride;            \
                  if((i^j^k^color000^1)&1) {                      \
                    double Ax     = apply_op_ijk(x_n);            \
                    double lambda =     Dinv_ijk();               \
                    x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax); \
                  }

#endif // GSRB_OOP
#define SERIAL_LOOP for(int k=klo;k<khi;k++){                                 \
                      for(int j=jlo;j<jhi;j++){                               \
                        for(int i=ilo;i<ihi;i++){                             \
                          LOOP_BODY                                       \
                    }}}

#else // GSRB_*
#error no GSRB implementation was specified
#endif

#if defined(USE_AMP)
  #pragma message "Using GPU for smooth"
  using namespace Concurrency;
  #define BBS AMP_TILE_BLOCKS
  #define KBS AMP_TILE_K
  #define JBS AMP_TILE_J
  #define IBS AMP_TILE_I
  #define MAX_AMP_TILES 1024

#if defined(GSRB_STRIDE2)
  #error GSRB_STRIDE2 not (yet?) implemented for use with AMP.
#endif // GSRB_STRIDE2

void smooth(level_type * level, int x_id, int rhs_id, double a, double b) {
  int block,s;

  print_amp_info();

  for(s=0;s<2*NUM_SMOOTHS;s++) { // there are two sweeps per GSRB smooth

    // exchange the ghost zone...
    #ifdef GSRB_OOP // out-of-place GSRB ping pongs between x and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
    #else // in-place GSRB only operates on x
                 exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,        x_id,stencil_get_shape());
    #endif


    const int nBlocks = level->num_my_blocks;
    const int kDim = level->my_blocks[0].dim.k;
    const int jDim = level->my_blocks[0].dim.j;
    const int iDim = level->my_blocks[0].dim.i;


    // apply the smoother...
    double _timeStart = getTime();

    #ifdef AMP_ASSERT
    for(block=0;block<nBlocks;block++) {
      int kdim = level->my_blocks[block].dim.k;
      int jdim = level->my_blocks[block].dim.j;
      int idim = level->my_blocks[block].dim.i;
      if ((kdim != kDim) || (jdim != jDim) || (idim != iDim)) {
        fprintf(stderr, "s = %d, block = %d, kdim = %d, jdim = %d, idim = %d\n",
                s, block, level->my_blocks[block].dim.i, level->my_blocks[block].dim.j, level->my_blocks[block].dim.k);
        exit(1);
      }
    }
    #endif // AMP_ASSERT

    #if defined(AMP_DIM) && (AMP_DIM==4)
    #if !defined(AMP_TILE_DIM) || (AMP_TILE_DIM==0)
    #pragma message "Running 4 loops on GPU, no tiling"
    if (iDim*jDim*kDim*nBlocks >= AMP_INNER_THRESHOLD) {
      extent<1> e(iDim*jDim*kDim*nBlocks);
      parallel_for_each(e, [=] (index<1> idx) restrict(amp) {
        const int block =  idx[0]/(iDim*jDim*kDim);
        const int kIndex = (idx[0] / (iDim*jDim)) % kDim;
        const int jIndex = (idx[0] / iDim) % jDim;
        const int iIndex = idx[0] % iDim;

        INITIALIZE_CONSTS
        INITIALIZE_x_n

        const int k = kIndex + klo;
        const int j = jIndex + jlo;
        const int i = iIndex + ilo;

        LOOP_BODY

      }); // end pfe
    } else { // need full serial implementation here
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS
        INITIALIZE_x_n
        SERIAL_LOOP
      } // blocks
    } // end serial
    #elif AMP_TILE_DIM==3
    #pragma message "Running 4 loops on GPU, 3D tiling"
    // Note that we can only tile 3 dimensions.  To do 4, we combine block and k
    if ((nBlocks*kDim % (BBS*KBS) == 0) &&(jDim % JBS == 0) && (iDim % IBS == 0) &&
                                          (BBS*KBS*JBS*IBS <= MAX_AMP_TILES)) {
      extent<3> e(nBlocks*kDim,jDim,iDim);
      parallel_for_each(e.tile<BBS*KBS,JBS,IBS>(),
      [=] (tiled_index<BBS*KBS,JBS,IBS> tidx) restrict(amp) {
        const int block =  tidx.global[0] / kDim;
        const int kIndex = tidx.global[0] % kDim;
        const int jIndex = tidx.global[1];
        const int iIndex = tidx.global[2];

        INITIALIZE_CONSTS
        INITIALIZE_x_n

        const int k = kIndex + klo;
        const int j = jIndex + jlo;
        const int i = iIndex + ilo;

        #ifdef USE_LDS
        const int l_b = tidx.local[0] / KBS;
        const int l_k = tidx.local[0] % KBS;
        const int l_j = tidx.local[1];
        const int l_i = tidx.local[2];
        LDS_LOOP_BODY
        #else // USE_LDS
        LOOP_BODY
        #endif // USE_LDS
      }); // end pfe
    } else { // need full serial implementation here
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS
        INITIALIZE_x_n
        SERIAL_LOOP
      } // blocks
    } // end serial
    #else // AMP_TILE_DIM
    #error Unrecognized combination of AMP_DIM and AMP_TILE_DIM
    #endif // AMP_TILE_DIM

    #elif defined(AMP_DIM) && (AMP_DIM==3)

    // Don't parallelize the block loop
    // loop over all block/tiles this process owns...
    for(block=0;block<level->num_my_blocks;block++) {
//      const int kIndex = 0;
//      const int jIndex = 0;
//      const int iIndex = 0;

      INITIALIZE_CONSTS
      INITIALIZE_x_n

      #if !defined(AMP_TILE_DIM) || (AMP_TILE_DIM==0)
      #pragma message "Running 3 loops on GPU, no tiling"
      if (idim*jdim*kdim >= AMP_INNER_THRESHOLD) {
        extent<1> e(idim*jdim*kdim);
        parallel_for_each(e, [=] (index<1> idx) restrict(amp) {
          const int kIndex = idx[0] / (idim*jdim);
          const int jIndex = (idx[0] / idim) % jdim;
          const int iIndex = idx[0] % idim;
          const int k = kIndex + klo;
          const int j = jIndex + jlo;
          const int i = iIndex + ilo;
          LOOP_BODY
        });
      } else {
        SERIAL_LOOP
      }
      #elif (AMP_TILE_DIM==3)
      #pragma message "Running 3 loops on GPU, 3D tiling"
      if (((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {

        extent<3> e(kdim,jdim,idim);

        parallel_for_each(
          e.tile<KBS,JBS,IBS>(), [=] (tiled_index<KBS,JBS,IBS> tidx) restrict(amp) {
            const int k = tidx.global[0] + klo;
            const int j = tidx.global[1] + jlo;
            const int i = tidx.global[2] + ilo;

            #ifdef USE_LDS
            const int l_k = tidx.local[0];
            const int l_j = tidx.local[1];
            const int l_i = tidx.local[2];

            LDS_LOOP_BODY
            #else // USE_LDS
            LOOP_BODY
            #endif // USE_LDS
          }
        );
      } else {
            SERIAL_LOOP
      }
      #else // AMP_TILE_DIM
      #error Unknown AMP_DIM/AMP_TILE_DIM combination
      #endif
    } // block loop
    #else // AMP_DIM
    #error Unsupported AMP_DIM
    #endif // AMP_DIM
    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
}
#else // USE_AMP
#pragma message "Not using GPU for smooth"
void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
  int block,s;
  for(s=0;s<2*NUM_SMOOTHS;s++){ // there are two sweeps per GSRB smooth

    // exchange the ghost zone...
    #ifdef GSRB_OOP // out-of-place GSRB ping pongs between x and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
    #else // in-place GSRB only operates on x
                 exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,        x_id,stencil_get_shape());
    #endif

    // apply the smoother...
    double _timeStart = getTime();

    // loop over all block/tiles this process owns...
    PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
    for(block=0;block<level->num_my_blocks;block++){
      INITIALIZE_CONSTS
      INITIALIZE_x_n
      SERIAL_LOOP
    } // boxes
    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
}
#endif // USE_AMP


//------------------------------------------------------------------------------------------------------------------------------
