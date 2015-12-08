/*** AMD LEGAL HEADER ***/
//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
//------------------------------------------------------------------------------------------------------------------------------


#ifdef USE_L1JACOBI
  #define LAMBDA_BASE (VECTOR_L1INV)
#else
  #define LAMBDA_BASE (VECTOR_DINV)
#endif

#define INITIALIZE_CONSTS \
      const int box = level->my_blocks[block].read.box;    \
      const int ilo = level->my_blocks[block].read.i;      \
      const int jlo = level->my_blocks[block].read.j;      \
      const int klo = level->my_blocks[block].read.k;      \
      const int ihi = level->my_blocks[block].dim.i + ilo; \
      const int jhi = level->my_blocks[block].dim.j + jlo; \
      const int khi = level->my_blocks[block].dim.k + klo; \
      const int ghosts = level->box_ghosts;                \
      const int jStride = level->my_boxes[box].jStride;    \
      const int kStride = level->my_boxes[box].kStride;    \
      const double h2inv = 1.0/(level->h*level->h);        \
      const double * AMP_RESTRICT rhs    = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT alpha  = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT valid  = level->my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT lambda = level->my_boxes[box].vectors[LAMBDA_BASE  ] + ghosts*(1+jStride+kStride); \
        const double * AMP_RESTRICT x_n;   \
              double * AMP_RESTRICT x_np1; \
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
                      double Ax_n = apply_op_ijk_amp(x_n); \
                      x_np1[ijk] = lxn(x_n,0,0,0) + weight*lambda[ijk]*(rhs[ijk]-Ax_n);
#define LOOP_BODY const int ijk = i + j*jStride + k*kStride; \
                  double Ax_n = apply_op_ijk(x_n); \
                  x_np1[ijk] = x_n[ijk] + weight*lambda[ijk]*(rhs[ijk]-Ax_n);

#define SERIAL_LOOP for(int k=klo;k<khi;k++){ \
                    for(int j=jlo;j<jhi;j++){ \
                    for(int i=ilo;i<ihi;i++){ \
                        LOOP_BODY \
                    }}}

#if defined(USE_AMP)
#pragma message "Using GPU for smooth"

using namespace Concurrency;


void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
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

  print_amp_info();

  for(s=0;s<NUM_SMOOTHS;s++){
    // exchange ghost zone data... Jacobi ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}


    // apply the smoother... Jacobi ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

    const int nBlocks = level->num_my_blocks;

    #if defined(AMP_DIM) && (AMP_DIM==4)
    // parallelize the three inner loops plus the block loop

    // In order to parallelize the block loop, we need to know (assume) these
    // don't change from block to block
    const int kDim = level->my_blocks[0].dim.k;
    const int jDim = level->my_blocks[0].dim.j;
    const int iDim = level->my_blocks[0].dim.i;
    #ifdef AMP_ASSERT
    // Make sure they don't change
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

      const int k = kIndex + klo;
      const int j = jIndex + jlo;
      const int i = iIndex + ilo;

      LOOP_BODY

      });
    #elif AMP_TILE_DIM==3
    #pragma message "Running 4 loops on GPU, 3D tiling"
    // Note that we can only tile 3 dimensions.  To do 4, we combine block and k
    if ((nBlocks*kDim % (BBS*KBS) == 0) &&(jDim % JBS == 0) && (iDim % IBS == 0)) {
      extent<3> e(nBlocks*kDim,jDim,iDim);
      parallel_for_each(e.tile<BBS*KBS,JBS,IBS>(),
      [=] (tiled_index<BBS*KBS,JBS,IBS> tidx) restrict(amp) {
        const int block =  tidx.global[0] / kDim;
        const int kIndex = tidx.global[0] % kDim;
        const int jIndex = tidx.global[1];
        const int iIndex = tidx.global[2];
        INITIALIZE_CONSTS

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

      });
    #else // AMP_TILE_DIM
    #error Unrecognized combination of AMP_DIM and AMP_TILE_DIM
    #endif // AMP_TILE_DIM
    } else {
      for(block=0;block<level->num_my_blocks;block++){

        INITIALIZE_CONSTS
        SERIAL_LOOP

      } // block-loop
    } // end if

    #elif defined(AMP_DIM) && (AMP_DIM==3)
    for(block=0;block<level->num_my_blocks;block++){
      INITIALIZE_CONSTS
      #if !defined(AMP_TILE_DIM) || (AMP_TILE_DIM==0)
      #pragma message "Running 3 loops on GPU, no tiling"
      const int kdim = level->my_blocks[block].dim.k;
      const int jdim = level->my_blocks[block].dim.j;
      const int idim = level->my_blocks[block].dim.i;
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
      #pragma messsage "Running 3 loops on GPU, 3D tiling"
      int kdim = level->my_blocks[block].dim.k;
      int jdim = level->my_blocks[block].dim.j;
      int idim = level->my_blocks[block].dim.i;
      if (((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {
        extent<3> e(kdim,jdim,idim);
        parallel_for_each( e.tile<KBS,JBS,IBS>(),
        [=] (tiled_index<KBS,JBS,IBS> tidx) restrict(amp) {
          // The values of k,j,i that we need asssume k in [klo,khi), j in [jlo,jhi), and i in [ilo,ihi).
          // This is important when computing ijk.
          const int k = tidx.global[0] + klo;
          const int j = tidx.global[1] + jlo;
          const int i = tidx.global[2] + ilo;
          // But the values for the static tiles need to be zero-based.

          #ifdef USE_LDS
          const int l_k = tidx.local[0];
          const int l_j = tidx.local[1];
          const int l_i = tidx.local[2];

          LDS_LOOP_BODY
          #else
          LOOP_BODY
          #endif

        });
      } else {
        SERIAL_LOOP
      }

      #else // AMP_TILE_DIM
      #error Unknown AMP_DIM/AMP_TILE_DIM combination
      #endif // AMP_TILE_DIM
    } // block-loop
    #else // AMP_DIM
    #error Unsupported AMP_DIM
    #endif // AMP_DIM

    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
}

#else // USE_AMP
#pragma message "Not using GPU for smooth"
void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
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
  for(s=0;s<NUM_SMOOTHS;s++){
    // exchange ghost zone data... Jacobi ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}

    // apply the smoother... Jacobi ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

    PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
    for(block=0;block<level->num_my_blocks;block++){

      INITIALIZE_CONSTS
      SERIAL_LOOP

    } // box-loop
    level->timers.smooth += (double)(getTime()-_timeStart);
  } // s-loop
}
#endif // USE_AMP

//------------------------------------------------------------------------------------------------------------------------------
