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
#include <stdint.h>
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

#ifndef GPU_THRESHOLD
#define GPU_THRESHOLD 32768
#endif

#ifdef USE_L1JACOBI
  #define LAMBDA_BASE (VECTOR_L1INV)
#else
  #define LAMBDA_BASE (VECTOR_DINV)
#endif

#define INITIALIZE_CONSTS(rstr) \
      const int box = level->my_blocks[block].read.box;    \
      const int ilo = level->my_blocks[block].read.i;      \
      const int jlo = level->my_blocks[block].read.j;      \
      const int klo = level->my_blocks[block].read.k;      \
      const int idim = level->my_blocks[block].dim.i;      \
      const int jdim = level->my_blocks[block].dim.j;      \
      const int kdim = level->my_blocks[block].dim.k;      \
      const int ihi = idim + ilo;                          \
      const int jhi = jdim + jlo;                          \
      const int khi = kdim + klo;                          \
      const int ghosts = level->box_ghosts;                \
      const int jStride = level->my_boxes[box].jStride;    \
      const int kStride = level->my_boxes[box].kStride;    \
      const double h2inv = 1.0/(level->h*level->h);        \
      const double * rstr rhs    = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride); \
      const double * rstr alpha  = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride); \
      const double * rstr beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride); \
      const double * rstr beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride); \
      const double * rstr beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride); \
      const double * rstr lambda = level->my_boxes[box].vectors[LAMBDA_BASE  ] + ghosts*(1+jStride+kStride); \
        const double * rstr x_n;   \
              double * rstr x_np1; \
      if((s&1)==0){x_n   = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);  \
                   x_np1 = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);} \
              else{x_n   = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);  \
                   x_np1 = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}


#define LDS_LOOP_BODY const int ijk = i + j*jStride + k*kStride; \
                      DECLARE_LXN; INITIALIZE_LXN(x_n); \
                      DECLARE_LBK; INITIALIZE_LBK(); \
                      DECLARE_LBJ; INITIALIZE_LBJ(); \
                      DECLARE_LBI; INITIALIZE_LBI(); \
                      tidx.barrier.wait_with_tile_static_memory_fence(); \
                      double Ax_n = apply_op_ijk_gpu(x_n); \
                      x_np1[ijk] = lxn(0,0,0) + weight*lambda[ijk]*(rhs[ijk]-Ax_n);
#define LOOP_BODY const int ijk = i + j*jStride + k*kStride; \
                  double Ax_n = apply_op_ijk(x_n); \
                  x_np1[ijk] = x_n[ijk] + weight*lambda[ijk]*(rhs[ijk]-Ax_n);

#define SERIAL_LOOP for(int k=klo;k<khi;k++){ \
                    for(int j=jlo;j<jhi;j++){ \
                    for(int i=ilo;i<ihi;i++){ \
                        LOOP_BODY \
                    }}}

#ifdef USE_GPU_FOR_SMOOTH

#if defined(__KALMAR_AMP__) || defined(__HCC_AMP__)
  using namespace Concurrency;
#elif defined(__KALMAR_HC__) || defined(__HCC_HC__)
  using namespace hc;
#else
  #error Missing predefine when USE_GPU_FOR_SMOOTH is defined.  Has compiler changed?
#endif // predefines
#define to_ijk(k,j,i) ((i)+(j)*jStride+(k)*kStride)

#if defined(GPU_ARRAY_VIEW)
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

// The templating is needed for dimensioning tile_static arrays in clamp mode
template <int GHOSTS>void av_smooth(level_type * level, int x_id, int rhs_id, double a, double b){

  if(NUM_SMOOTHS&1){
    fprintf(stderr,"error - NUM_SMOOTHS must be even...\n");
    exit(0);
  }

  #ifdef USE_L1JACOBI
  double weight = 1.0;
  #else // USE_L1JACOBI
  double weight = 2.0/3.0;
  #endif // USE_L1JACOBI

  int block,s;

  #ifdef PRINT_DETAILS
  if (level->my_rank == 0) print_smooth_info();
  #endif // PRINT_DETAILS

  #if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
  #endif // *_HC_

  for(s=0;s<NUM_SMOOTHS;s++){
    // exchange ghost zone data... Jacobi ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}


    // apply the smoother... Jacobi ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

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
      // Note that we are inside the s loop, and for a fixed s these bases don't change.
      const int xn_base_for_this_smooth    = ((s&1) == 0) ? x_id : VECTOR_TEMP;
      const int xnp1_base_for_this_smooth  = ((s&1) == 0) ? VECTOR_TEMP : x_id;
      const double h2inv = 1.0/(level->h*level->h);

      array_view<double, 2> x_n_av(level->num_my_boxes,level->box_volume,level->vectors[xn_base_for_this_smooth]);
      array_view<double, 2> x_np1_av(level->num_my_boxes,level->box_volume,level->vectors[xnp1_base_for_this_smooth]);
      array_view<const double, 2> rhs_av(level->num_my_boxes,level->box_volume,level->vectors[rhs_id]);
      array_view<const double, 2> lambda_av(level->num_my_boxes,level->box_volume,level->vectors[LAMBDA_BASE]);
      #define lrhs(index) (rhs_av(box,index+GHOSTS*(1+jStride+kStride)))
      #define llambda(index)  (lambda_av(box,index+GHOSTS*(1+jStride+kStride)))

      #ifdef USE_HELMHOLTZ
      array_view<const double, 2> alpha_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_ALPHA]);
      #define lalpha(index) (alpha_av(box,index+GHOSTS*(1+jStride+kStride)))
      #endif // USE_HELMHOLTZ

      #ifdef STENCIL_VARIABLE_COEFFICIENT
      array_view<const double, 2> beta_i_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_BETA_I]);
      array_view<const double, 2> beta_j_av(level->num_my_boxes,level->box_volume, level->vectors[VECTOR_BETA_J]);
      array_view<const double, 2> beta_k_av(level->num_my_boxes,level->box_volume,  level->vectors[VECTOR_BETA_K]);
      #define lbi(inck,incj,inci) (beta_i_av(box, to_ijk(k+GHOSTS+inck, j+GHOSTS+incj, i+GHOSTS+inci)))
      #define lbj(inck,incj,inci) (beta_j_av(box, to_ijk(k+GHOSTS+inck, j+GHOSTS+incj, i+GHOSTS+inci)))
      #define lbk(inck,incj,inci) (beta_k_av(box, to_ijk(k+GHOSTS+inck, j+GHOSTS+incj, i+GHOSTS+inci)))
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
        local_block_info[ILO][ib] = level->my_blocks[ib].read.i;
        local_block_info[JLO][ib] = level->my_blocks[ib].read.j;
        local_block_info[KLO][ib] = level->my_blocks[ib].read.k;
        local_block_info[BOX][ib] = level->my_blocks[ib].read.box;
        local_block_info[IDIM][ib] = level->my_blocks[ib].dim.i;
        local_block_info[JDIM][ib] = level->my_blocks[ib].dim.j;
        local_block_info[KDIM][ib] = level->my_blocks[ib].dim.k;
      }
      array_view<const int, 2> block_info_av(N_BLOCK_FIELDS,nBlocks,&local_block_info[0][0]);

      #define JSTRIDE 0
      #define KSTRIDE 1
      #define N_BOX_FIELDS 2
      int  local_box_info[N_BOX_FIELDS][nBoxes];
      for (int ib=0; ib<nBoxes; ++ib) {
        local_box_info[JSTRIDE][ib] = level->my_boxes[ib].jStride;
        local_box_info[KSTRIDE][ib] = level->my_boxes[ib].kStride;
      }
      array_view<const int, 2> box_info_av(N_BOX_FIELDS, nBoxes, &local_box_info[0][0]);


      #if defined(GPU_TILE_DIM) && (GPU_TILE_DIM==-1)
        #ifdef USElxn
          #define Lxn(inck,incj,inci) (x_n_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
          #define lxn(inck,incj,inci) (local_x[l_b*(KBS+GHOSTS)+l_k+inck+GHOSTS][l_j+incj+GHOSTS][l_i+inci+GHOSTS])
        #else // USElxn
          #define lxn(inck,incj,inci) (x_n_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
        #endif // USElxn
        #define lxnp1(inck,incj,inci) (x_np1_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))

        extent<3> e(nBlocks,kDim,jDim);
        PFE_TILE3(e,BBS,KBS,JBS)
          tile_static double this_slice[BBS][KBS+2*GHOSTS][JBS+2*GHOSTS];
          const int block =  tidx.global[0];
          const int kIndex = tidx.global[1];
          const int jIndex = tidx.global[2];
          const int ilo = block_info_av(ILO,block);
          const int jlo = block_info_av(JLO,block);
          const int klo = block_info_av(KLO,block);
          const int k = kIndex + klo;
          const int j = jIndex + jlo;


          const int idim = block_info_av(IDIM,block);
          const int jdim = block_info_av(JDIM,block);
          const int kdim = block_info_av(KDIM,block);

          const int ihi = idim + ilo;
          const int jhi = jdim + jlo;
          const int khi = kdim + klo;

          const int box = block_info_av(BOX,block);

          const int jStride = box_info_av(JSTRIDE,box);
          const int kStride = box_info_av(KSTRIDE,box);


          const int l_b = tidx.local[0];
          const int l_k = tidx.local[1];
          const int l_j = tidx.local[2];

// for a different approach to initialization
//        int iadj = (l_i == 0) ? -1 : 1;
//        int jadj = (l_j == 0) ? -1 : 1;
//        int kadj = (l_k == 0) ? -1 : 1;

          for (int i = ilo; i < ihi; ++i) {
            int ijk = to_ijk(k,j,i);

            #define axn(box,k,j,i) (x_n_av(box,to_ijk(k+GHOSTS,j+GHOSTS,i+GHOSTS)))
            #define sxn(l_b,l_k,l_j) (this_slice[l_b][l_k+GHOSTS][l_j+GHOSTS])

#if defined(STATIC)
          // Broken

                             sxn(l_b,l_k,l_j)    = axn(box,k,j,i);
          if (l_k == 0)      sxn(l_b,-1,l_j)     = axn(box,klo-1,j,i);
          if (l_k == KBS-1)  sxn(l_b,KBS,l_j)    = axn(box,khi,j,i);
          if (l_j==0)        sxn(l_b,l_k,-1)     = axn(box,k,jlo-1,i);
          if (l_j==JBS-1)    sxn(l_b,l_k,JBS)    = axn(box,k,jhi,i);

/*
          if (l_k == 0)      sxn(l_b,l_k-1,l_j) = lxn(-1,0,0);
          if (l_k == KBS-1)  sxn(l_b,l_k+1,l_j)   = lxn(1,0,0);
          if (l_j==0)        sxn(l_b,l_k,l_j-1)    = lxn(0,-1,0);
          if (l_j==JBS-1)    sxn(l_b,l_k,l_j+1)   = lxn(0,1,0);
*/

          tidx.barrier.wait_with_tile_static_memory_fence();
//          tidx.barrier.wait_with_all_memory_fence(); // doesn't help
          double Ax_n =-b*h2inv*(
              + lbi(0,0,1)*( lxn( 0, 0, 1)      - sxn(l_b,l_k,l_j) )
              + lbi(0,0,0)*( lxn( 0, 0,-1)      - sxn(l_b,l_k,l_j) )
              + lbj(0,1,0)*( sxn(l_b,l_k,l_j+1) - sxn(l_b,l_k,l_j) )
              + lbj(0,0,0)*( sxn(l_b,l_k,l_j-1) - sxn(l_b,l_k,l_j) )
              + lbk(1,0,0)*( sxn(l_b,l_k+1,l_j) - sxn(l_b,l_k,l_j) )
              + lbk(0,0,0)*( sxn(l_b,l_k-1,l_j) - sxn(l_b,l_k,l_j) )
            );
          lxnp1(0,0,0) = sxn(l_b,l_k,l_j) + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
#elif defined(SEMISTATIC)
        
          // This one works for fv2
          sxn(l_b,-1, l_j) = axn(box,klo-1,j,i);
          sxn(l_b,KBS,l_j) = axn(box,khi,j,i);
          sxn(l_b,l_k,-1) = axn(box,k,jlo-1,i);
          sxn(l_b,l_k,JBS) = axn(box,k,jhi,i);
          sxn(l_b,l_k,l_j) = axn(box,k,j,i);

          tidx.barrier.wait_with_tile_static_memory_fence();
          double Ax_n =-b*h2inv*(
              + lbi(0,0,1)*( lxn( 0, 0, 1) - sxn(l_b,l_k,l_j) )
              + lbi(0,0,0)*( lxn( 0, 0,-1) - sxn(l_b,l_k,l_j) )
              + lbj(0,1,0)*( axn(box,k,j+1,i) - sxn(l_b,l_k,l_j) )
              + lbj(0,0,0)*( axn(box,k,j-1,i) - sxn(l_b,l_k,l_j) )
              + lbk(1,0,0)*( axn(box,k+1,j,i) - sxn(l_b,l_k,l_j) )
              + lbk(0,0,0)*( axn(box,k-1,j,i) - sxn(l_b,l_k,l_j) )
            );
          lxnp1(0,0,0) = sxn(l_b,l_k,l_j) + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
#elif defined(AV)
          // this one works for fv2
          double temp  = lxn(0,0,0);
          double Ax_n =-b*h2inv*(
              + lbi(0,0,1)*( lxn( 0, 0, 1) - temp )
              + lbi(0,0,0)*( lxn( 0, 0,-1) - temp )
              + lbj(0,1,0)*( axn(box,k,j+1,i) - temp )
              + lbj(0,0,0)*( axn(box,k,j-1,i) - temp )
              + lbk(1,0,0)*( axn(box,k+1,j,i) - temp )
              + lbk(0,0,0)*( axn(box,k-1,j,i) - temp )
            );
          lxnp1(0,0,0) = temp + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
#else
          // as does this one
          double Ax_n = apply_op_ijk_gpu;
          lxnp1(0,0,0) = lxn(0,0,0) + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
#endif
        }
      PFE_END;
      HC_WAIT;
      #elif defined(GPU_TILE_DIM)  && (GPU_TILE_DIM==3)
        extent<3> e(nBlocks*kDim,jDim,iDim);
        PFE_TILE3(e,BBS*KBS,JBS,IBS)
          #ifdef USElxn
//        tile_static double local_x[BBS*(KBS+2*GHOSTS)][JBS+2*GHOSTS][IBS+2*GHOSTS];
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

        #ifdef USElxn
          #define Lxn(inck,incj,inci) (x_n_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
          #define lxn(inck,incj,inci) (local_x[l_b*(KBS+GHOSTS)+l_k+inck+GHOSTS][l_j+incj+GHOSTS][l_i+inci+GHOSTS])
        #else // USElxn
          #define lxn(inck,incj,inci) (x_n_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
        #endif // USElxn
        #define lxnp1(inck,incj,inci) (x_np1_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))

          #ifdef USElxn
          lxn(0,0,iadj) = Lxn(0,0,iadj);
          lxn(0,jadj,0) = Lxn(0,jadj,0);
          lxn(kadj,0,0) = Lxn(kadj,0,0);
          lxn(0,0,0) = Lxn(0,0,0);
          tidx.barrier.wait_with_tile_static_memory_fence();
          #endif // USElxn

          const double Ax_n   = apply_op_ijk_gpu;
          lxnp1(0,0,0) = lxn(0,0,0) + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
        PFE_END;
        HC_WAIT;
        #elif defined(GPU_TILE_DIM) && (GPU_TILE_DIM==1)
          #define lxn(inck,incj,inci) (x_n_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
          #define lxnp1(inck,incj,inci) (x_np1_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))

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

            const double Ax_n   = apply_op_ijk_gpu;
            lxnp1(0,0,0) = lxn(0,0,0) + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
          PFE_END;
          HC_WAIT;
        #elif !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
          #define lxn(inck,incj,inci) (x_n_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
          #define lxnp1(inck,incj,inci) (x_np1_av(box,to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
          extent<1> e(nBlocks*kDim*jDim*iDim);
          parallel_for_each(e, [=] (index<1> idx) restrict(amp)
          {
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

            const double Ax_n   = apply_op_ijk_gpu;
            lxnp1(0,0,0) = lxn(0,0,0) + weight*llambda(ijk)*(lrhs(ijk)-Ax_n);
          });
          HC_WAIT;
      #else // GPU_TILE_DIM
        #error Unrecognized combination of GPU_DIM and GPU_TILE_DIM
      #endif // GPU_TILE_DIM
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
      for(block=0;block<nBlocks;block++) {
        INITIALIZE_CONSTS(__restrict__)
        SERIAL_LOOP
      }
    }
    level->timers.smooth += (double)(getTime()-_timeStart);
  #elif defined(GPU_DIM) && (GPU_DIM==3)
    for(block=0;block<level->num_my_blocks;block++){
      INITIALIZE_CONSTS()
    
      if ((level->box_volume >= GPU_THRESHOLD) && ((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {
  
        extent<3> e(kdim,jdim,idim);

        array_view<const double, 1> x_n_av   (to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS),    &x_n[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        array_view<      double, 1> x_np1_av (to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS),  &x_np1[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        array_view<const double, 1> rhs_av   (to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS),    &rhs[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        array_view<const double, 1> lambda_av(to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS), &lambda[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);

        #define lxn(inck, incj, inci)     (x_n_av   (to_ijk(k+inck+GHOSTS,j+incj+GHOSTS,i+inci+GHOSTS)))
        #define lxnp1(inck, incj, inci)   (x_np1_av (to_ijk(k+inck+GHOSTS,j+incj+GHOSTS,i+inci+GHOSTS)))
        #define lrhs(inck, incj, inci)    (rhs_av   (to_ijk(k+inck+GHOSTS,j+incj+GHOSTS,i+inci+GHOSTS)))
        #define llambda(inck, incj, inci) (lambda_av(to_ijk(k+inck+GHOSTS,j+incj+GHOSTS,i+inci+GHOSTS)))

        #if defined USE_HELMHOLTZ
        array_view<const double, 1> alpha_av(to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS), &alpha[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        #define lalpha (alpha_av(to_ijk(k+GHOSTS,j+GHOSTS,i+GHOSTS)))
        #endif // USE_HELMHOLTZ
        #if defined(STENCIL_VARIABLE_COEFFICIENT)
        array_view<const double, 1> beta_k_av(to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS), &beta_k[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        array_view<const double, 1> beta_j_av(to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS), &beta_j[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        array_view<const double, 1> beta_i_av(to_ijk(kdim+2*GHOSTS,jdim+2*GHOSTS,idim+2*GHOSTS), &beta_i[to_ijk(klo-GHOSTS,jlo-GHOSTS,ilo-GHOSTS)]);
        #define lbk(inck,incj,inci) (beta_k_av(to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
        #define lbj(inck,incj,inci) (beta_j_av(to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
        #define lbi(inck,incj,inci) (beta_i_av(to_ijk(k+GHOSTS+inck,j+GHOSTS+incj,i+GHOSTS+inci)))
        #endif // STENCIL_VARIABLE_COEFFICIENT

        PFE_TILE3(e,KBS,JBS,IBS)
            const int k = tidx.global[0];  // should vary from 0..KBS-1,KBS to 2*KBS-1, etc.
            const int j = tidx.global[1];  // should vary from 0..JBS-1,IBS to 2*JBS-1, etc.
            const int i = tidx.global[2];  // should vary from 0..IBS-1,IBS to 2*IBS-1, etc.
  
            double Ax_n = apply_op_ijk_gpu;
            lxnp1(0,0,0) = lxn(0,0,0) + weight*llambda(0,0,0)*(lrhs(0,0,0)-Ax_n);
        PFE_END;
      } else {
          SERIAL_LOOP
      }
  
    } // block-loop
    level->timers.smooth += (double)(getTime()-_timeStart);
    #else // GPU_DIM
      #error Unsupported GPU_DIM
    #endif // GPU__DIM
  } // s-loop
}

#else // GPU_ARRAY_VIEW
void gpu_smooth(level_type * level, int x_id, int rhs_id, double a, double b){
  if(NUM_SMOOTHS&1){
    fprintf(stderr,"error - NUM_SMOOTHS must be even...\n");
    exit(0);
  }

  #ifdef USE_L1JACOBI
  double weight = 1.0;
  #else
  double weight = 2.0/3.0;
  #endif

  int block,s;

  #ifdef PRINT_DETAILS
  if (level->my_rank == 0) print_smooth_info();
  #endif // PRINT_DETAILS

  #if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
  #endif // *_HC_

  for(s=0;s<NUM_SMOOTHS;s++){
    // exchange ghost zone data... Jacobi ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}


    // apply the smoother... Jacobi ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

    const int nBlocks = level->num_my_blocks;
    const int nBoxes = level->num_my_boxes;

    #if defined(GPU_DIM) && (GPU_DIM==4)
    // parallelize the three inner loops plus the block loop

    const int kDim = level->my_blocks[0].dim.k;
    const int jDim = level->my_blocks[0].dim.j;
    const int iDim = level->my_blocks[0].dim.i;

    // Note that we are inside the s loop, and for a fixed s these bases don't change.
    const int xn_base_for_this_smooth    = ((s&1) == 0) ? x_id : VECTOR_TEMP;
    const int xnp1_base_for_this_smooth  = ((s&1) == 0) ? VECTOR_TEMP : x_id;
    const double h2inv = 1.0/(level->h*level->h);

    const int ghosts = level->box_ghosts;

    #if !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
    // Running 4 loops on GPU, no tiling
    if (iDim*jDim*kDim*nBlocks >= GPU_THRESHOLD) {
      extent<1> e(iDim*jDim*kDim*nBlocks);
      parallel_for_each(e, [=] (index<1> idx) restrict(amp)
      {
        const int block =  idx[0]/(iDim*jDim*kDim);
        const int kIndex = (idx[0] / (iDim*jDim)) % kDim;
        const int jIndex = (idx[0] / iDim) % jDim;
        const int iIndex = idx[0] % iDim;
        INITIALIZE_CONSTS()

        const int k = kIndex + klo;
        const int j = jIndex + jlo;
        const int i = iIndex + ilo;

        LOOP_BODY
      });
      HC_WAIT;
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
      for(block=0;block<nBlocks;block++){
        INITIALIZE_CONSTS(__restrict__)
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
        const int kIndex = tidx.global[0] % kDim;
        const int jIndex = tidx.global[1];
        const int iIndex = tidx.global[2];
        INITIALIZE_CONSTS()

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
      PFE_END;
      HC_WAIT;
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
        for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS(__restrict__)
        SERIAL_LOOP
      }
    } // end if
    #elif GPU_TILE_DIM==1
    // Running 4 loops on GPU, 1D tiling
    if ((nBlocks*kDim*jDim*iDim % (IBS) == 0)) {
      extent<1> e(nBlocks*kDim*jDim*iDim);
      PFE_TILE1(e,IBS)
        const int block =  tidx.global[0] / (iDim*jDim*kDim);
        const int kIndex = (tidx.global[0] / (iDim*jDim)) % kDim;
        const int jIndex = (tidx.global[0] / iDim) % jDim;
        const int iIndex = tidx.global[0] % iDim;
        INITIALIZE_CONSTS()

        const int k = kIndex + klo;
        const int j = jIndex + jlo;
        const int i = iIndex + ilo;

        #ifdef USE_LDS
        #error LDS is currently broken for this case
        const int l_b = tidx.local[0] / KBS;
        const int l_k = tidx.local[0] % KBS;
        const int l_j = tidx.local[1];
        const int l_i = tidx.local[2];
        LDS_LOOP_BODY
        #else // USE_LDS
        LOOP_BODY
        #endif // USE_LDS
      PFE_END;
      HC_WAIT;
    } else {
      PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
        for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS(__restrict__)
        SERIAL_LOOP
      }
    } // end if
    #else // GPU_TILE_DIM
    #error Unrecognized combination of GPU_DIM and GPU_TILE_DIM
    #endif // GPU_TILE_DIM

    #elif defined(GPU_DIM) && (GPU_DIM==3)
    // Don't parallelize the block loop
    // loop over all block/tiles this process owns...
    for(block=0;block<level->num_my_blocks;block++){
      INITIALIZE_CONSTS()
      #if !defined(GPU_TILE_DIM) || (GPU_TILE_DIM==0)
      // Collapsing 3 loops on GPU, no tiling (hence automatic tiling)
      if (kdim*jdim*idim >= GPU_THRESHOLD) {
        extent<1> e(kdim*jdim*idim);
        parallel_for_each(e, [=] (index<1> idx) restrict(amp) {
          const int kIndex = idx[0] / (idim*jdim);
          const int jIndex = (idx[0] / idim) % jdim;
          const int iIndex = idx[0] % idim;
          const int k = kIndex + klo;
          const int j = jIndex + jlo;
          const int i = iIndex + ilo;

          LOOP_BODY
        });
        HC_WAIT;
      } else {
        #pragma omp parallel for collapse(2) private(k,j,i) if (khi > 1)
        SERIAL_LOOP
      }
      #elif (GPU_TILE_DIM==3)
      // Running 3 loops on GPU, 3D tiling
      if (((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {

        extent<3> e(kdim,jdim,idim);

        PFE_TILE3(e,KBS,JBS,IBS)
          const int k = tidx.global[0] + klo;
          const int j = tidx.global[1] + jlo;
          const int i = tidx.global[2] + ilo;

          #ifdef USE_LDS
          const int l_k = tidx.local[0];
          const int l_j = tidx.local[1];
          const int l_i = tidx.local[2];

          LDS_LOOP_BODY
          #else
          LOOP_BODY
          #endif
        PFE_END;
        HC_WAIT;
      } else {
        #pragma omp parallel for collapse(2) if (khi > 1)
        SERIAL_LOOP
      }

      #else // GPU_TILE_DIM
      #error Unknown GPU_DIM/GPU_TILE_DIM combination
      #endif // GPU_TILE_DIM
    } // block-loop
    #else // GPU_DIM
    #error Unsupported GPU_DIM
    #endif // GPU_DIM

    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
}

#endif // GPU_ARRAY_VIEW

#endif // USE_GPU_FOR_SMOOTH

void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
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
   };
#else
   gpu_smooth(level, x_id, rhs_id, a, b);
#endif
#else // USE_GPU_FOR_SMOOTH


  if(NUM_SMOOTHS&1){
    fprintf(stderr,"error - NUM_SMOOTHS must be even...\n");
    exit(0);
  }

  #ifdef USE_L1JACOBI
  double weight = 1.0;
  #else
  double weight = 2.0/3.0;
  #endif

  #ifdef PRINT_DETAILS
  if (level->my_rank == 0) print_smooth_info();
  #endif // PRINT_DETAILS
 
  int block,s;
  for(s=0;s<NUM_SMOOTHS;s++){
    // exchange ghost zone data... Jacobi ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}

    // apply the smoother... Jacobi ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

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
      const int ghosts = level->box_ghosts;
      const int jStride = level->my_boxes[box].jStride;
      const int kStride = level->my_boxes[box].kStride;
      const double h2inv = 1.0/(level->h*level->h);
      const double * __restrict__ rhs    = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
      const double * __restrict__ alpha  = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
      const double * __restrict__ beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
      #ifdef USE_L1JACOBI
      const double * __restrict__ lambda = level->my_boxes[box].vectors[VECTOR_L1INV ] + ghosts*(1+jStride+kStride);
      #else
      const double * __restrict__ lambda = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
      #endif
        const double * __restrict__ x_n;
              double * __restrict__ x_np1;
                      if((s&1)==0){x_n   = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                   x_np1 = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                              else{x_n   = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                   x_np1 = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}

      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        double Ax_n = apply_op_ijk(x_n);
        x_np1[ijk] = x_n[ijk] + weight*lambda[ijk]*(rhs[ijk]-Ax_n);
      }}}
    } // block-loop
    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
#endif // USE_GPU_FOR_SMOOTH
}

//------------------------------------------------------------------------------------------------------------------------------
