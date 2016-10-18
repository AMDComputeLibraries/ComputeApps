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

// This is really ugly and messes up syntax highlighting, but it does save typing
#if defined(__KALMAR_AMP__) || defined(__HCC_AMP__)
  #define PFE_TILE3(e,KBS,JBS,IBS) parallel_for_each(e.tile<KBS,JBS,IBS>(), [=] (tiled_index<KBS,JBS,IBS> tidx) restrict(amp) {
  #define PFE_TILE1(e,TILE_SIZE) parallel_for_each(e.tile<TILE_SIZE>(), [=] (tiled_index<TILE_SIZE> tidx) restrict(amp) {
#elif defined(__KALMAR_HC__) || defined(__HCC_HC__)
  #define PFE_TILE3(e,KBS,JBS,IBS) parallel_for_each(e.tile(KBS,JBS,IBS), [=] (tiled_index<3> tidx) restrict(amp) {
  #define PFE_TILE1(e,TILE_SIZE) parallel_for_each(e.tile(TILE_SIZE), [=] (tiled_index<1> tidx) restrict(amp) {
#endif
#define PFE_END })


#if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  #define HC_WAIT av.wait()
#else
  #define HC_WAIT
#endif

#define INITIALIZE_CONSTS(rstr)                                                                            \
  const int box = level->my_blocks[block].read.box;                                                        \
  const int klo = level->my_blocks[block].read.k;                                                          \
  const int jlo = level->my_blocks[block].read.j;                                                          \
  const int ilo = level->my_blocks[block].read.i;                                                          \
  const int idim = level->my_blocks[block].dim.i ;                                                         \
  const int jdim = level->my_blocks[block].dim.j ;                                                         \
  const int kdim = level->my_blocks[block].dim.k ;                                                         \
  const int ihi = idim + ilo;                                                                              \
  const int jhi = jdim + jlo;                                                                              \
  const int khi = kdim + klo;                                                                              \
  const double h2inv = 1.0/(level->h*level->h);                                                            \
  const int ghosts =  level->box_ghosts;                                                                   \
  const int jStride = level->my_boxes[box].jStride;                                                        \
  const int kStride = level->my_boxes[box].kStride;                                                        \
  const double * rstr rhs      = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride); \
  const double * rstr alpha   = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);  \
  const double * rstr beta_i   = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride); \
  const double * rstr beta_j   = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride); \
  const double * rstr beta_k   = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride); \
  const double * rstr Dinv     = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride); \
  const int color000 = (level->my_boxes[box].low.i^level->my_boxes[box].low.j^level->my_boxes[box].low.k^s)&1;

#ifdef GSRB_OOP
  #define INITIALIZE_x_n(rstr)                                                                         \
    const int x_n_base =   ((s&1)==0) ? x_id : VECTOR_TEMP;                                            \
    const int x_np1_base = ((s&1)==0) ? VECTOR_TEMP : x_id;                                            \
    const double * rstr x_n   = level->my_boxes[box].vectors[x_n_base  ] + ghosts*(1+jStride+kStride); \
          double * rstr x_np1 = level->my_boxes[box].vectors[x_np1_base] + ghosts*(1+jStride+kStride);
#else // GSRB_OOP
  #define INITIALIZE_x_n(rstr)                                                                     \
    const double * rstr x_n    = level->my_boxes[box].vectors[x_id] + ghosts*(1+jStride+kStride);  \
          double * rstr x_np1  = level->my_boxes[box].vectors[x_id] + ghosts*(1+jStride+kStride);
#endif // GSRB_OOP

#if defined(GSRB_FP)
  #define LDS_LOOP_BODY                                                                             \
    const double * RedBlack = level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1); \
    const int ij  = i + j*jStride;                                                                  \
    const int ijk = i + j*jStride + k*kStride;                                                      \
    DECLARE_LXN; INITIALIZE_LXN(x_n);                                                               \
    DECLARE_LBK; INITIALIZE_LBK();                                                                  \
    DECLARE_LBJ; INITIALIZE_LBJ();                                                                  \
    DECLARE_LBI; INITIALIZE_LBI();                                                                  \
    tidx.barrier.wait_with_tile_static_memory_fence();                                              \
    double Ax     = apply_op_ijk_gpu;                                                               \
    double lambda =     Dinv_ijk();                                                                 \
    x_np1[ijk] = lxn(0,0,0) + RedBlack[ij]*lambda*(rhs[ijk]-Ax);

  // If the compiler does its job well, it will pull the initialization of RedBlack
  // outside the two inner loops
  #define LOOP_BODY                                                                                 \
    const double * RedBlack = level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1); \
    int ij  = i + j*jStride;                                                                        \
    int ijk = i + j*jStride + k*kStride;                                                            \
    double Ax     = apply_op_ijk(x_n);                                                              \
    double lambda =     Dinv_ijk();                                                                 \
    x_np1[ijk] = x_n[ijk] + RedBlack[ij]*lambda*(rhs[ijk]-Ax);

  // But we'll define this separately so as not to disadvantage the non-gpu implementation
  #define SERIAL_LOOP \
    for(int k=klo;k<khi;k++) {                                                                                     \
      const double * __restrict__ RedBlack = level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1); \
      for(int j=jlo;j<jhi;j++){                                                                                    \
        for(int i=ilo;i<ihi;i++){                                                                                  \
          int ij  = i + j*jStride;                                                                                 \
          int ijk = i + j*jStride + k*kStride;                                                                     \
          double Ax     = apply_op_ijk(x_n);                                                                       \
          double lambda =     Dinv_ijk();                                                                          \
          x_np1[ijk] = x_n[ijk] + RedBlack[ij]*lambda*(rhs[ijk]-Ax);                                               \
        }                                                                                                          \
      }                                                                                                            \
    }

#elif defined(GSRB_STRIDE2)
  #if defined(GSRB_OOP)
    #define LDS_LOOP_BODY NOT IMPLEMENTED YET
    #define LOOP_BODY NOT IMPLEMENTED YET
    #define SERIAL_LOOP \
       for(int k=klo;k<khi;k++){                               \
         for(int j=jlo;j<jhi;j++){                             \
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
         }                                                     \
       }
  #else // GSRB_OOP
    #define LDS_LOOP_BODY NOT IMPLEMENTED YET
    #define LOOP_BODY NOT IMPLEMENTED YET
    #define SERIAL_LOOP \
      for(int k=klo;k<khi;k++){                               \
        for(int j=jlo;j<jhi;j++){                             \
          for(int i=ilo+((ilo^j^k^color000)&1);i<ihi;i+=2){   \
            int ijk = i + j*jStride + k*kStride;              \
            double Ax     = apply_op_ijk(x_n);                \
            double lambda =     Dinv_ijk();                   \
            x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);     \
          }                                                   \
        }                                                     \
      }
  #endif // GSRB_OOP

#elif defined(GSRB_BRANCH)
  #if defined(GSRB_OOP)
    #define LDS_LOOP_BODY                                \
      const int ijk = i + j*jStride + k*kStride;         \
      DECLARE_LXN; INITIALIZE_LXN(x_n);                  \
      DECLARE_LBK; INITIALIZE_LBK();                     \
      DECLARE_LBJ; INITIALIZE_LBJ();                     \
      DECLARE_LBI; INITIALIZE_LBI();                     \
      tidx.barrier.wait_with_tile_static_memory_fence(); \
      if((i^j^k^color000^1)&1){                          \
        double Ax   = apply_op_ijk_gpu;                  \
        double lambda =     Dinv_ijk();                  \
        x_np1[ijk] = lxn(0,0,0) + lambda*(rhs[ijk]-Ax);  \
      }else{                                             \
        x_np1[ijk] =  lxn(0,0,0);                        \
      }

    // The ?: form seems friendlier to GPU than the if/then form
    // but tends to break the compiler
    #define LOOP_BODY                                 \
      int ijk = i + j*jStride + k*kStride;            \
      if((i^j^k^color000^1)&1){                       \
        double Ax   = apply_op_ijk(x_n);              \
        double lambda =     Dinv_ijk();               \
        x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax); \
      }else{                                          \
        x_np1[ijk] =  x_n[ijk];                       \
      }
  #else // GSRB_OOP
    #define LDS_LOOP_BODY                                   \
      const int ijk = i + j*jStride + k*kStride;            \
      DECLARE_LXN; INITIALIZE_LXN(x_n);                     \
      DECLARE_LBK; INITIALIZE_LBK();                        \
      DECLARE_LBJ; INITIALIZE_LBJ();                        \
      DECLARE_LBI; INITIALIZE_LBI();                        \
      tidx.barrier.wait_with_tile_static_memory_fence();    \
      if((i^j^k^color000^1)&1){                             \
        double Ax   = apply_op_ijk_gpu;                     \
        double lambda =     Dinv_ijk();                     \
        x_np1[ijk] = lxn(0,0,0) + lambda*(rhs[ijk]-Ax);     \
      }
    #define LOOP_BODY                                 \
      int ijk = i + j*jStride + k*kStride;            \
      if((i^j^k^color000^1)&1) {                      \
        double Ax     = apply_op_ijk(x_n);            \
        double lambda =     Dinv_ijk();               \
        x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax); \
      }

  #endif // GSRB_OOP

  #define SERIAL_LOOP             \
    for(int k=klo;k<khi;k++){     \
      for(int j=jlo;j<jhi;j++){   \
        for(int i=ilo;i<ihi;i++){ \
          LOOP_BODY               \
        }                         \
      }                           \
    }                             \

#else // GSRB_*
  #error no GSRB implementation was specified
#endif // GSRB_*

#ifdef USE_GPU_FOR_SMOOTH

#if defined(__KALMAR_AMP__) || defined(__HCC_AMP__)
  using namespace Concurrency;
#elif defined(__KALMAR_HC__) || defined(__HCC_HC__)
  using namespace hc;
#else
  #error Missing predefine when USE_GPU_FOR_SMOOTH is defined.  Has compiler changed?
#endif // predefines

// We have a few too many thresholds around here.
#ifndef GPU_THRESHOLD
#define GPU_THRESHOLD 4096
#endif

#define to_ijk(k,j,i) ((i)+(j)*jStride+(k)*kStride)

#if defined GPU_ARRAY_VIEW
bool run_av_loop_on_gpu(const int nBlocks, const int kDim, const int jDim, const int iDim)
{
  switch(GPU_DIM) {
    case 4:
      switch(GPU_TILE_DIM) {
        case -1:
          return (kDim*jDim*iDim*nBlocks >= GPU_THRESHOLD) & (nBlocks %BBS == 0) & (kDim % KBS == 0) & (jDim % JBS == 0);
        case 3:
          return (kDim*jDim*iDim*nBlocks >= GPU_THRESHOLD) & ((nBlocks*kDim)%(BBS*KBS) == 0) & (jDim%JBS==0) && (iDim%IBS==0);
        case 1:
          return (kDim*jDim*iDim*nBlocks >= GPU_THRESHOLD) & ((kDim*jDim*iDim*nBlocks)%IBS==0);
        case 0:
          return (kDim*jDim*iDim*nBlocks >= GPU_THRESHOLD);
        default:
          break;
      }
    case 3:
      switch(GPU_TILE_DIM) {
        case -1:
          break;
        case 3:
          return (kDim*jDim*iDim >= GPU_THRESHOLD) & (kDim%KBS==0) & (jDim%JBS==0) && (iDim%IBS==0);
        case 1:
          return (kDim*jDim*iDim >= GPU_THRESHOLD) & ((kDim*jDim*iDim)%IBS==0);
        case 0:
          return (kDim*jDim*iDim >= GPU_THRESHOLD);
        default:
          break;
      }
    default:
      break;
  }
  fprintf(stderr, "Warning: Unsupported GPU_DIM (%d)/GPU_TILE_DIM (%d), running on CPU\n", GPU_DIM, GPU_TILE_DIM);
  return false;
}
  
template <int GHOSTS> void av_smooth(level_type * level, int x_id, int rhs_id, double a, double b) {
  int block,s;
  const int ghosts=GHOSTS; // this plus template to make compiler happy

  #ifdef PRINT_DETAILS
  if (level->my_rank == 0) print_smooth_info();
  #endif // PRINT_DETAILS

  #if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
  #endif // *_HC_

  for(s=0;s<2*NUM_SMOOTHS;s++) { // there are two sweeps per GSRB smooth

    // exchange the ghost zone...
    #ifdef GSRB_OOP // out-of-place GSRB ping pongs between x and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
    #else // in-place GSRB only operates on x
                 exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,        x_id,stencil_get_shape());
    #endif // GSRB_OOP


    // apply the smoother...
    double _timeStart = getTime();

    // In order to parallelize the block loop, we need to know (assume) these
    // don't change from block to block
    const int nBlocks = level->num_my_blocks;
    const int nBoxes = level->num_my_boxes;
    const int kDim = level->my_blocks[0].dim.k;
    const int jDim = level->my_blocks[0].dim.j;
    const int iDim = level->my_blocks[0].dim.i;
    #if defined(GPU_DIM) && (GPU_DIM==4)
      /*
       *  In this code we do a parallel_for_each over blocks and define the array_views outside that, based on box
       *  That means that the code in the inner loops needs i = ilo + tidx.global[0], etc.
       */
      if (run_av_loop_on_gpu(nBlocks,kDim,jDim,iDim)) {
        #ifdef GSRB_OOP
          int    xn_base_for_this_s  = ((s&1) == 0) ? x_id : VECTOR_TEMP;
          int  xnp1_base_for_this_s  = ((s&1) == 0) ? VECTOR_TEMP : x_id;
        #else // GSRB_OOP
          const int    xn_base_for_this_s  = x_id;
          // The compiler doesn't seem to like having two array views
          // point to the same place and update each other.
          //        const int  xnp1_base_for_this_s  = x_id;
        #endif // GSRB_OOP
        const double h2inv = 1.0/(level->h*level->h);

        array_view<double, 2> x_n_av(level->num_my_boxes,level->box_volume,level->vectors[xn_base_for_this_s]);
        #define lxn(inck,incj,inci)    (x_n_av  (box,to_ijk(k+ghosts+inck,j+ghosts+incj,i+ghosts+inci)))

        #ifdef GSRB_OOP
          array_view<double, 2> x_np1_av(level->num_my_boxes,level->box_volume,level->vectors[xnp1_base_for_this_s]);
          #define lxnp1(inck,incj,inci)  (x_np1_av(box,to_ijk(k+ghosts+inck,j+ghosts+incj,i+ghosts+inci)))
        #endif

        array_view<const double, 2> rhs_av (level->num_my_boxes,level->box_volume,level->vectors[rhs_id]);
        array_view<const double, 2> Dinv_av(level->num_my_boxes,level->box_volume,level->vectors[VECTOR_DINV]);
        #define lrhs(index)   (rhs_av   (box,index+ghosts*(1+jStride+kStride)))
        #define lDinv(index)  (Dinv_av  (box,index+ghosts*(1+jStride+kStride)))

        #ifdef USE_HELMHOLTZ
          array_view<const double, 2> alpha_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_ALPHA]);
          #define lalpha(index) (alpha_av(box,index+ghosts*(1+jStride+kStride)))
        #endif // USE_HELMHOLTZ
        #ifdef STENCIL_VARIABLE_COEFFICIENT
          array_view<const double, 2> beta_i_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_BETA_I]);
          array_view<const double, 2> beta_j_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_BETA_J]);
          array_view<const double, 2> beta_k_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_BETA_K]);

          #define lbi(inck,incj,inci) (beta_i_av(box, to_ijk(k+ghosts+inck, j+ghosts+incj, i+ghosts+inci)))
          #define lbj(inck,incj,inci) (beta_j_av(box, to_ijk(k+ghosts+inck, j+ghosts+incj, i+ghosts+inci)))
          #define lbk(inck,incj,inci) (beta_k_av(box, to_ijk(k+ghosts+inck, j+ghosts+incj, i+ghosts+inci)))
        #endif // STENCIL_VARIABLE_COEFFICIENT

        #define ILO 0
        #define JLO 1
        #define KLO 2
        #define BOX 3
        #define IDIM 4
        #define JDIM 5
        #define KDIM 6
        #define N_BLOCK_FIELDS 7
        int  local_block_info[N_BLOCK_FIELDS][nBlocks];
        for (int ib=0; ib<nBlocks; ++ib) {
          local_block_info[ILO][ib]  = level->my_blocks[ib].read.i;
          local_block_info[JLO][ib]  = level->my_blocks[ib].read.j;
          local_block_info[KLO][ib]  = level->my_blocks[ib].read.k;
          local_block_info[BOX][ib]  = level->my_blocks[ib].read.box;
          local_block_info[IDIM][ib] = level->my_blocks[ib].dim.i;
          local_block_info[JDIM][ib] = level->my_blocks[ib].dim.j;
          local_block_info[KDIM][ib] = level->my_blocks[ib].dim.k;
        }
        array_view<const int, 2> block_info_av(N_BLOCK_FIELDS,nBlocks,&local_block_info[0][0]);

//      #define ILO 0
//      #define JLO 1
//      #define KLO 2
        #define JSTRIDE 3
        #define KSTRIDE 4
        #define N_BOX_FIELDS 5
        int local_box_info[N_BOX_FIELDS][nBoxes];
        // I think I actually have to get box information, for the bottom corner
        for (int ib=0; ib<nBoxes; ++ib) {
          local_box_info[ILO][ib] = level->my_boxes[ib].low.i;
          local_box_info[JLO][ib] = level->my_boxes[ib].low.j;
          local_box_info[KLO][ib] = level->my_boxes[ib].low.k;
          local_box_info[JSTRIDE][ib] = level->my_boxes[ib].jStride;
          local_box_info[KSTRIDE][ib] = level->my_boxes[ib].kStride;
        }
        array_view<const int, 2> box_info_av(N_BOX_FIELDS,nBoxes,&local_box_info[0][0]);


        #if defined(GPU_TILE_DIM)  && (GPU_TILE_DIM==3)
          extent<3> e(nBlocks*kDim,jDim,iDim);
          PFE_TILE3(e,BBS*KBS,JBS,IBS)
            #if defined(USE_lxn) && (defined(OPERATOR_fv4) || defined(OPERATOR_27pt))
//            tile_static double local_x[BBS*(KBS+2*GHOSTS)][JBS+2*GHOSTS][IBS+2*GHOSTS];
              #error This initialization needs to be redone for fv4 and 27pt
              const int l_b = tidx.local[0] / KBS;
              const int l_k = tidx.local[0] % KBS;
              const int l_j = tidx.local[1];
              const int l_i = tidx.local[2];

              int iadj = (l_i == 0) ? -1 : 1;
              int jadj = (l_j == 0) ? -1 : 1;
              int kadj = (l_k == 0) ? -1 : 1;
            #endif // USElxn

            const int block =  tidx.global[0] / kDim;
            const int kIndex = tidx.global[0] % kDim;
            const int jIndex = tidx.global[1];
            const int iIndex = tidx.global[2];

            const int ilo = block_info_av(ILO,block);
            const int jlo = block_info_av(JLO,block);
            const int klo = block_info_av(KLO,block);

            const int idim = block_info_av(IDIM,block);
            const int jdim = block_info_av(JDIM,block);
            const int kdim = block_info_av(KDIM,block);

            const int box = block_info_av(BOX,block);
            const int jStride = box_info_av(JSTRIDE,box);
            const int kStride = box_info_av(KSTRIDE,box);

            const int ihi = idim + ilo;
            const int jhi = jdim + jlo;
            const int khi = kdim + klo;

            const int k = kIndex + klo;
            const int j = jIndex + jlo;
            const int i = iIndex + ilo;
            const int ijk = to_ijk(k,j,i);

            const int lowi = box_info_av(ILO,box);
            const int lowj = box_info_av(JLO,box);
            const int lowk = box_info_av(KLO,box);
            const int color000 = (lowi^lowj^lowk^s)&1;

            #ifdef GSRB_BRANCH
              #if defined(GSRB_OOP)
                if((i^j^k^color000^1)&1) {
                  double Ax     = apply_op_ijk_gpu;
                  double lambda = lDinv(ijk);
                  lxnp1(0,0,0)  = lxn(0,0,0) + lambda*(lrhs(ijk)-Ax);
                } else {
                  lxnp1(0,0,0) =  lxn(0,0,0);
                }
              #else // GSRB_OOP
                if((i^j^k^color000^1)&1) {
                  double Ax     = apply_op_ijk_gpu;
                  double lambda = lDinv(ijk);
                  lxn(0,0,0)    = lxn(0,0,0) + lambda*(lrhs(ijk)-Ax);
                }
              #endif // GSRB_OOP
            #else // GSRB_BRANCH
              #error Only know how to do GSRB_BRANCH for array_view version
            #endif // GSRB_BRANCH
          PFE_END;
        #elif defined(GPU_TILE_DIM) && (GPU_TILE_DIM==1)
          extent<1> e(nBlocks*kDim*jDim*iDim);

          PFE_TILE1(e,IBS)
            const int block =  tidx.global[0]/(iDim*jDim*kDim);
            const int kIndex = (tidx.global[0] / (iDim*jDim)) % kDim;
            const int jIndex = (tidx.global[0] / iDim) % jDim;
            const int iIndex = tidx.global[0] % iDim;


            const int ilo = block_info_av(ILO,block);
            const int jlo = block_info_av(JLO,block);
            const int klo = block_info_av(KLO,block);
            const int box = block_info_av(BOX,block);
            const int idim = block_info_av(IDIM,block);
            const int jdim = block_info_av(JDIM,block);
            const int kdim = block_info_av(KDIM,block);
            const int jStride = box_info_av(JSTRIDE,box);
            const int kStride = box_info_av(KSTRIDE,box);
            const int ihi = idim + ilo;
            const int jhi = jdim + jlo;
            const int khi = kdim + klo;
            const int k = kIndex + klo;
            const int j = jIndex + jlo;
            const int i = iIndex + ilo;
            const int ijk = to_ijk(k,j,i);
            const int lowi = box_info_av(ILO,box);
            const int lowj = box_info_av(JLO,box);
            const int lowk = box_info_av(KLO,box);
            const int color000 = (lowi^lowj^lowk^s)&1;
            #ifdef GSRB_BRANCH
              // It seems this should be faster with ?:, but it is not.  May even be a little slower.
              #if defined(GSRB_OOP)
                if((i^j^k^color000^1)&1) {
                  double Ax     = apply_op_ijk_gpu;
                  double lambda = lDinv(ijk);
                  lxnp1(0,0,0)  = lxn(0,0,0) + lambda*(lrhs(ijk)-Ax);
                } else {
                  lxnp1(0,0,0) =  lxn(0,0,0);
                }
              #else // GSRB_OOP
                if((i^j^k^color000^1)&1) {
                  double Ax     = apply_op_ijk_gpu;
                  double lambda = lDinv(ijk);
                  lxn(0,0,0)    = lxn(0,0,0) + lambda*(lrhs(ijk)-Ax);
                }
              #endif // GSRB_OOP
            #else // GSRB_BRANCH
              #error Currently only know how to do GSRB_BRANCH for array_view version
            #endif // GSRB_BRANCH
          PFE_END;
        #elif !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
          extent<1> e(nBlocks*kDim*jDim*iDim);
          parallel_for_each(e, [=] (index<1> idx) restrict(amp) {
            const int block =  idx[0]/(iDim*jDim*kDim);
            const int kIndex = (idx[0] / (iDim*jDim)) % kDim;
            const int jIndex = (idx[0] / iDim) % jDim;
            const int iIndex = idx[0] % iDim;

            const int ilo = block_info_av(ILO,block);
            const int jlo = block_info_av(JLO,block);
            const int klo = block_info_av(KLO,block);
            const int box = block_info_av(BOX,block);
            const int idim = block_info_av(IDIM,block);
            const int jdim = block_info_av(JDIM,block);
            const int kdim = block_info_av(KDIM,block);
            const int jStride = box_info_av(JSTRIDE,box);
            const int kStride = box_info_av(KSTRIDE,box);
            const int ihi = idim + ilo;
            const int jhi = jdim + jlo;
            const int khi = kdim + klo;
            const int k = kIndex + klo;
            const int j = jIndex + jlo;
            const int i = iIndex + ilo;
            const int ijk = to_ijk(k,j,i);

            const int lowi = box_info_av(ILO,box);
            const int lowj = box_info_av(JLO,box);
            const int lowk = box_info_av(KLO,box);
            const int color000 = (lowi^lowj^lowk^s)&1;
            #ifdef GSRB_BRANCH
            // It seems this should be faster with ?:, but it is not.  May even be a little slower.
              #if defined(GSRB_OOP)
                if((i^j^k^color000^1)&1) {
                  double Ax     = apply_op_ijk_gpu;
                  double lambda = lDinv(ijk);
                  lxnp1(0,0,0)  = lxn(0,0,0) + lambda*(lrhs(ijk)-Ax);
                } else {
                  lxnp1(0,0,0) =  lxn(0,0,0);
                }
              #else // GSRB_OOP
                if((i^j^k^color000^1)&1) {
                  double Ax     = apply_op_ijk_gpu;
                  double lambda = lDinv(ijk);
                  lxn(0,0,0)    = lxn(0,0,0) + lambda*(lrhs(ijk)-Ax);
                }
              #endif // GSRB_OOP
            #else // GSRB_BRANCH
              #error Currently only know how to do GSRB_BRANCH for array_view version
            #endif // GSRB_BRANCH
          });
        #else // GPU_TILE_DIM
          #error Unsupported GPU_DIM/GPU_TILE_DIM combination
        #endif // GPU_TILE_DIM
      } else {
        PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
        for(block=0;block<nBlocks;block++){
          INITIALIZE_CONSTS(__restrict__)
          INITIALIZE_x_n(__restrict__)
          SERIAL_LOOP
        }
      }
    #elif defined(GPU_DIM) && (GPU_DIM==3)
      /*
       *  In this code we loop serially over blocks and define the array_views within each block.
       *  That means that the code in the inner loops needs i = tidx.global[0], etc. with no adjustment for ilo,jlo,klo.
       */
      level->timers.smooth += (double)(getTime() - _timeStart);
      for(block=0;block<level->num_my_blocks;block++){
          const int box = level->my_blocks[block].read.box;
          const int ilo = level->my_blocks[block].read.i;
          const int jlo = level->my_blocks[block].read.j;
          const int klo = level->my_blocks[block].read.k;
          const int idim = level->my_blocks[block].dim.i;
          const int jdim = level->my_blocks[block].dim.j;
          const int kdim = level->my_blocks[block].dim.k;
          const int ihi = idim + ilo;
          const int jhi = jdim + jlo;
          const int khi = kdim + klo;
          const double h2inv = 1.0/(level->h*level->h);
          const int jStride = level->my_boxes[box].jStride;
          const int kStride = level->my_boxes[box].kStride;
          const int color000 = (level->my_boxes[box].low.i^level->my_boxes[box].low.j^level->my_boxes[box].low.k^s)&1;

          // This complicated index arithmetic is here because the compiler doesn't like negative indexes for array_views.
          const double *rhs = level->my_boxes[box].vectors[rhs_id]+ghosts*(1+jStride+kStride);
          array_view<const double, 1> rhs_av (to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts),  &rhs[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
          #define lrhs(index) (rhs_av    (index+ghosts*(1+jStride+kStride)))

          #ifdef USE_HELMHOLTZ
            const double *valpha =  level->my_boxes[box].vectors[VECTOR_ALPHA] + ghosts*(1+jStride+kStride);
            array_view<const double, 1> alpha_av(to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts), &valpha[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
            #define lalpha (alpha_av(to_ijk(k+ghosts,j+ghosts,i+ghosts)))
          #endif // USE_HELMHOLTZ

          #if defined(STENCIL_VARIABLE_COEFFICIENT)
            const double *beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K]+ghosts*(1+jStride+kStride);
            const double *beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J]+ghosts*(1+jStride+kStride);
            const double *beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I]+ghosts*(1+jStride+kStride);
            array_view<const double, 1> beta_k_av(to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts), &beta_k[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
            array_view<const double, 1> beta_j_av(to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts), &beta_j[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
            array_view<const double, 1> beta_i_av(to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts), &beta_i[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
            #define lbk(inck,incj,inci) (beta_k_av(to_ijk(k+ghosts+inck,j+ghosts+incj,i+ghosts+inci)))
            #define lbj(inck,incj,inci) (beta_j_av(to_ijk(k+ghosts+inck,j+ghosts+incj,i+ghosts+inci)))
            #define lbi(inck,incj,inci) (beta_i_av(to_ijk(k+ghosts+inck,j+ghosts+incj,i+ghosts+inci)))
          #endif // STENCIL_VARIABLE_COEFFICIENT

          const double *Dinv = level->my_boxes[box].vectors[VECTOR_DINV]+ghosts*(1+jStride+kStride);
          array_view<const double, 1> Dinv_av(to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts), &Dinv[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
          #define lDinv(index) (Dinv_av(index+ghosts*(1+jStride+kStride)))

          #ifdef GSRB_OOP
            const int x_n_base =   ((s&1)==0) ? x_id : VECTOR_TEMP;
            const int x_np1_base = ((s&1)==0) ? VECTOR_TEMP : x_id;
          #else // GSRB_OOP
            const int x_n_base = x_id;
            const int x_np1_base = x_id;
          #endif // GSRB_OOP
          double * x_n   = level->my_boxes[box].vectors[x_n_base] + ghosts*(1+jStride+kStride);
          double * x_np1 = level->my_boxes[box].vectors[x_np1_base] + ghosts*(1+jStride+kStride);
          array_view<double, 1> x_n_av   (to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts),  &x_n[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
          array_view<double, 1> x_np1_av  (to_ijk(kdim+2*ghosts,jdim+2*ghosts,idim+2*ghosts),&x_np1[to_ijk(klo-ghosts,jlo-ghosts,ilo-ghosts)]);
          #define lxn(inck, incj, inci)   (x_n_av    (to_ijk(k+inck+ghosts,j+incj+ghosts,i+inci+ghosts)))
          #ifdef GSRB_OOP
            #define lxnp1(index)            (x_np1_av  (index+ghosts*(1+jStride+kStride)))
          #else // GSRB_OOP
            #define lxnp1(index)            (lxn(0,0,0))
          #endif // GSRB_OOP

        if (run_av_loop_on_gpu(nBlocks,kDim,jDim,iDim)) {
          #if !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
            extent<3> e(kdim,jdim,idim);
            parallel_for_each(e, [=] (index<3> idx) restrict(amp) {
              const int k = idx[0];
              const int j = idx[1];
              const int i = idx[2];
              const int ijk = to_ijk(k,j,i);
              #if defined(GSRB_BRANCH)
                #if defined(GSRB_OOP)
                  if((i^j^k^color000^1)&1) {
                    double Ax   = apply_op_ijk_gpu;
                    lxnp1(ijk) = lxn(0,0,0) + lDinv(ijk)*(lrhs(ijk)-Ax);
                  } else {
                    lxnp1(ijk) =  lxn(0,0,0);
                  }
                #else // GSRB_OOP
                  if((i^j^k^color000^1)&1) {
                    double Ax     = apply_op_ijk_gpu;
                    lxn(0,0,0) = lxn(0,0,0) + lDinv(ijk)*(lrhs(ijk)-Ax);
                  }
                #endif // GSRB_OOP
              #else // GSRB_BRANCH
                #error Only GSRB_BRANCH is currently implemented with array_view
              #endif // GSRB_BRANCH
            });
          #elif (GPU_TILE_DIM==1)
            extent<1> e(kdim*jdim*idim);
            PFE_TILE1(e,IBS)
              const int k = (tidx.global[0] / (idim*jdim));
              const int j = (tidx.global[0] / idim) % jdim;
              const int i = tidx.global[0] % idim;
              const int ijk = to_ijk(k,j,i);
              #if defined(GSRB_BRANCH)
                #if defined(GSRB_OOP)
                  if((i^j^k^color000^1)&1) {
                    double Ax   = apply_op_ijk_gpu;
                    lxnp1(ijk) = lxn(0,0,0) + lDinv(ijk)*(lrhs(ijk)-Ax);
                  } else {
                    lxnp1(ijk) =  lxn(0,0,0);
                  }
                #else // GSRB_OOP
                  if((i^j^k^color000^1)&1) {
                    double Ax     = apply_op_ijk_gpu;
                    lxn(0,0,0) = lxn(0,0,0) + lDinv(ijk)*(lrhs(ijk)-Ax);
                  }
                #endif // GSRB_OOP
              #else // GSRB_BRANCH
                #error Only GSRB_BRANCH is currently implemented with array_view
              #endif // GSRB_BRANCH
            PFE_END;
          #elif (GPU_TILE_DIM==3)
            extent<3>e(kdim,jdim,idim);
            PFE_TILE3(e,KBS,JBS,IBS)
              const int k =  tidx.global[0];
              const int j =  tidx.global[1];
              const int i =  tidx.global[2];
              const int ijk = to_ijk(k,j,i);
              #if defined(GSRB_BRANCH)
                #if defined(GSRB_OOP)
                  if((i^j^k^color000^1)&1) {
                    double Ax   = apply_op_ijk_gpu;
                    lxnp1(ijk) = lxn(0,0,0) + lDinv(ijk)*(lrhs(ijk)-Ax);
                  } else {
                    lxnp1(ijk) =  lxn(0,0,0);
                  }
                #else // GSRB_OOP
                  if((i^j^k^color000^1)&1) {
                    double Ax     = apply_op_ijk_gpu;
                    lxn(0,0,0) = lxn(0,0,0) + lDinv(ijk)*(lrhs(ijk)-Ax);
                  }
                #endif // GSRB_OOP
              #else // GSRB_BRANCH
                #error Only GSRB_BRANCH is currently implemented with array_view
              #endif // GSRB_BRANCH
            PFE_END;
          #else
            SERIAL_LOOP
          #endif // GPU_TILE_DIM
          } else {
            SERIAL_LOOP
          }
        } // block loop
    #else // GPU_DIM
      #error Unsupported GPU_DIM
    #endif // GPU_DIM

    level->timers.smooth += (double)(getTime() - _timeStart);
  } // s-loop
}
#else // GPU_ARRAY_VIEW
void gpu_smooth(level_type * level, int x_id, int rhs_id, double a, double b) {
  int block,s;

  #ifdef PRINT_DETAILS
  if (level->my_rank == 0) print_smooth_info();
  #endif // PRINT_DETAILS

  #if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
  #endif // *_HC_

  for(s=0;s<2*NUM_SMOOTHS;s++) { // there are two sweeps per GSRB smooth

    // exchange the ghost zone...
    #ifdef GSRB_OOP // out-of-place GSRB ping pongs between x and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
    #else // in-place GSRB only operates on x
                 exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,        x_id,stencil_get_shape());
    #endif


    // apply the smoother...
    double _timeStart = getTime();

    // In order to parallelize the block loop, we need to know (assume) these
    // don't change from block to block
    const int nBlocks = level->num_my_blocks;
    const int nBoxes = level->num_my_boxes;
    const int kDim = level->my_blocks[0].dim.k;
    const int jDim = level->my_blocks[0].dim.j;
    const int iDim = level->my_blocks[0].dim.i;
    #if defined(GPU_DIM) && (GPU_DIM == 4)

    #if !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
    if (iDim*jDim*kDim*nBlocks >= GPU_THRESHOLD) {
      extent<1> e(iDim*jDim*kDim*nBlocks);
      parallel_for_each(e, [=] (index<1> idx) restrict(amp)
      {
        const int block =  idx[0]/(iDim*jDim*kDim);
        INITIALIZE_CONSTS()
        INITIALIZE_x_n()
        const int kIndex = (idx[0] / (iDim*jDim)) % kDim;
        const int jIndex = (idx[0] / iDim) % jDim;
        const int iIndex = idx[0] % iDim;
        const int k = kIndex + klo;
        const int j = jIndex + jlo;
        const int i = iIndex + ilo;

        LOOP_BODY
      });
      HC_WAIT;
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS(__restrict__)
        INITIALIZE_x_n(__restrict__)
        SERIAL_LOOP
      }
    }
    #elif GPU_TILE_DIM==3
    // Running 4 loops on GPU, 3D tiling
    // Note that we can only tile 3 dimensions.  To do 4, we combine block and k
    if ((nBlocks*kDim % (BBS*KBS) == 0) &&(jDim % JBS == 0) && (iDim % IBS == 0)) {
      extent<3> e(nBlocks*kDim,jDim,iDim);
      PFE_TILE3(e,BBS*KBS,JBS,IBS)
        const int block =  tidx.global[0] / kDim;
        INITIALIZE_CONSTS()
        INITIALIZE_x_n()

        const int kIndex = tidx.global[0] % kDim;
        const int jIndex = tidx.global[1];
        const int iIndex = tidx.global[2];

        const int k = kIndex + klo;
        const int j = jIndex + jlo;
        const int i = iIndex + ilo;

        #ifdef USE_LDS
        const int l_b = tidx.local[0] / KBS;
        const int l_k = tidx.local[0] % KBS;
        const int l_j = tidx.local[1];
        const int l_i = tidx.local[2];
        LDS_LOOP_BODY
        #else
        LOOP_BODY
        #endif
      PFE_END;
      HC_WAIT;
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS(__restrict__)
        INITIALIZE_x_n(__restrict__)
        SERIAL_LOOP
      }
    } // end if
    #else // GPU_TILE_DIM
    #error Unknown GPU_DIM/GPU_TILE_DIM combination
    #endif // GPU_TILE_DIM

    #elif defined(GPU_DIM) && (GPU_DIM == 3)
    if (level->box_volume >= GPU_THRESHOLD) {
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS()
        INITIALIZE_x_n()

        #if !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
        extent<3> e(kdim,jdim,idim);
        parallel_for_each(e, [=] (index<3> idx) restrict(amp) {
          const int k = idx[0] + klo;
          const int j = idx[1] + jlo;
          const int i = idx[2] + ilo;

          LOOP_BODY
        });
        HC_WAIT;
        #else // GPU_TILE_DIM
        if (((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {
          extent<3> e(kdim, jdim, idim);
          PFE_TILE3(e,KBS,JBS,IBS)
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
          PFE_END;
          HC_WAIT;
        } else {
          #pragma omp parallel for collapse(2) if (khi > 1)
          SERIAL_LOOP
        }
        #endif // GPU_TILE_DIM
      }
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS(__restrict__)
        INITIALIZE_x_n(__restrict__)
        SERIAL_LOOP
      }
    }
    #else // GPU_DIM
    #error Unsupported GPU_DIM combination
    #endif // GPU_DIM

    level->timers.smooth += (double)(getTime() - _timeStart);
  } // s-loop
}
#endif // GPU_ARRAY_VIEW
#endif // USE_GPU_FOR_SMOOTH
void smooth(level_type * level, int x_id, int rhs_id, double a, double b) {
  int block,s;
#ifdef USE_GPU_FOR_SMOOTH
#ifdef GPU_ARRAY_VIEW
  switch(level->box_ghosts) {
    case 1:
      av_smooth<1>(level, x_id, rhs_id, a, b);
      break;
    case 2:
      av_smooth<2>(level, x_id, rhs_id, a, b);
      break;
    default:
      fprintf(stderr, "Can't yet deal with %d ghosts in array_view code\n", level->box_ghosts);
      exit(1);
  }
#else
  gpu_smooth(level, x_id, rhs_id, a, b);
#endif
#else // USE_GPU_FOR_SMOOTH

  #ifdef PRINT_DETAILS
  if (level->my_rank == 0) print_smooth_info();
  #endif // PRINT_DETAILS

  for(s=0;s<2*NUM_SMOOTHS;s++) { // there are two sweeps per GSRB smooth

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
      const int box = level->my_blocks[block].read.box;
      const int ilo = level->my_blocks[block].read.i;
      const int jlo = level->my_blocks[block].read.j;
      const int klo = level->my_blocks[block].read.k;
      const int ihi = level->my_blocks[block].dim.i + ilo;
      const int jhi = level->my_blocks[block].dim.j + jlo;
      const int khi = level->my_blocks[block].dim.k + klo;

      int i,j,k;
      const double h2inv = 1.0/(level->h*level->h);
      const int ghosts =  level->box_ghosts;
      const int jStride = level->my_boxes[box].jStride;
      const int kStride = level->my_boxes[box].kStride;
      const int color000 = (level->my_boxes[box].low.i^level->my_boxes[box].low.j^level->my_boxes[box].low.k^s)&1;  // is element 000 red or black on *THIS* sweep

      const double * __restrict__ rhs      = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
      const double * __restrict__ alpha    = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_i   = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_j   = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_k   = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
      const double * __restrict__ Dinv     = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
      #ifdef GSRB_OOP
      const double * __restrict__ x_n;
            double * __restrict__ x_np1;
                     if((s&1)==0){x_n      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                  x_np1    = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                             else{x_n      = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                  x_np1    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}
      #else
      const double * __restrict__ x_n      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
           double * __restrict__ x_np1    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
      #endif


      #if defined(GSRB_FP)
      for(k=klo;k<khi;k++){const double * __restrict__ RedBlack = level->RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1);
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
            int ij  = i + j*jStride;
            int ijk = i + j*jStride + k*kStride;
            double Ax     = apply_op_ijk(x_n);
            double lambda =     Dinv_ijk();
            x_np1[ijk] = x_n[ijk] + RedBlack[ij]*lambda*(rhs[ijk]-Ax);
            //x_np1[ijk] = ((i^j^k^color000)&1) ? x_n[ijk] : x_n[ijk] + lambda*(rhs[ijk]-Ax);
      }}}


      #elif defined(GSRB_STRIDE2)
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
        #ifdef GSRB_OOP
        // out-of-place must copy old value...
        for(i=ilo;i<ihi;i++){
          int ijk = i + j*jStride + k*kStride;
          x_np1[ijk] = x_n[ijk];
        }
        #endif
        for(i=ilo+((ilo^j^k^color000)&1);i<ihi;i+=2){ // stride-2 GSRB
          int ijk = i + j*jStride + k*kStride;
          double Ax     = apply_op_ijk(x_n);
          double lambda =     Dinv_ijk();
          x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);
        }
      }}


      #elif defined(GSRB_BRANCH)
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        if((i^j^k^color000^1)&1){ // looks very clean when [0] is i,j,k=0,0,0 
          double Ax     = apply_op_ijk(x_n);
          double lambda =     Dinv_ijk();
          x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);
        #ifdef GSRB_OOP
        }else{
          x_np1[ijk] = x_n[ijk]; // copy old value when sweep color != cell color
        #endif
        }
      }}}


      #else
      #error no GSRB implementation was specified
      #endif

    } // boxes
    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
#endif // USE_GPU_FOR_SMOOTH
}
//------------------------------------------------------------------------------------------------------------------------------

