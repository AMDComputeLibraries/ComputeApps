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
// This routines calculates the residual (res=rhs-Ax) using the linear operator specified in the apply_op_ijk macro
// This requires exchanging a ghost zone and/or enforcing a boundary condition.
// NOTE, x_id must be distinct from rhs_id and res_id

#ifdef USE_GPU_FOR_RESIDUAL
#if defined(__KALMAR_AMP__)
  using namespace Concurrency;
#elif defined(__KALMAR_HC__)
  using namespace hc;
#else
#error Either __KALMAR_AMP__ or __KALMAR_HC__ must be defined with USE_GPU_FOR_RESIDUAL
#endif
#endif // USE_GPU_FOR_RESIDUAL
void residual(level_type * level, int res_id, int x_id, int rhs_id, double a, double b){
  // exchange the boundary for x in prep for Ax...
  exchange_boundary(level,x_id,stencil_get_shape());
          apply_BCs(level,x_id,stencil_get_shape());

#if defined(USE_GPU_FOR_RESIDUAL) && defined(__KALMAR_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
#endif


  // now do residual/restriction proper...
  double _timeStart = getTime();
  int block;
  const int nblocks = level->num_my_blocks;

  #ifdef USE_GPU_FOR_RESIDUAL
  const int MY_THRESHOLD = 32;  // wild guess
  if (level->my_blocks[0].dim.k >= MY_THRESHOLD) {
    for(block=0;block<nblocks;block++){
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
      const double h2inv = 1.0/(level->h*level->h);
      const double * GPU_RESTRICT x      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
      const double * GPU_RESTRICT rhs    = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
      const double * GPU_RESTRICT alpha  = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
      const double * GPU_RESTRICT beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
      const double * GPU_RESTRICT beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
      const double * GPU_RESTRICT beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
            double * GPU_RESTRICT res    = level->my_boxes[box].vectors[       res_id] + ghosts*(1+jStride+kStride);

      int kdim = level->my_blocks[block].dim.k;
      int jdim = level->my_blocks[block].dim.j;
      int idim = level->my_blocks[block].dim.i;
      if (((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {
        extent<3> e(kdim,jdim,idim);
        #if defined(__KALMAR_AMP__)
        parallel_for_each(
        e.tile<KBS,JBS,IBS>(),
        [=] (tiled_index<KBS,JBS,IBS> tidx) restrict(amp)
        #elif defined(__KALMAR_HC__)
        const tiled_extent<3> te = e.tile(KBS,JBS,IBS);
        parallel_for_each(
        te,
        [=] (tiled_index<3> tidx) restrict(amp)
        #endif // __KALMAR_*
        {
	  // We aren't going to try to use locals here
          // so just use regular apply_op_ijk
          const int k = tidx.global[0] + klo;
          const int j = tidx.global[1] + jlo;
          const int i = tidx.global[2] + ilo;
          int ijk = i + j*jStride + k*kStride;
          double Ax = apply_op_ijk(x);
          res[ijk] = rhs[ijk]-Ax;
        });
        #if defined(__KALMAR_HC__)
        av.wait();  // ouch.  Is there a better way to sync here?  TODO
        #endif // __KALMAR_HC__
      } else {
        for(k=klo;k<khi;k++){
        for(j=jlo;j<jhi;j++){
        for(i=ilo;i<ihi;i++){
          int ijk = i + j*jStride + k*kStride;
          double Ax = apply_op_ijk(x);
          res[ijk] = rhs[ijk]-Ax;
        }}}
      }
    }
  } else {
  #endif // USE_GPU_FOR_RESIDUAL
  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,nblocks)
  for(block=0;block<nblocks;block++){
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
    const double h2inv = 1.0/(level->h*level->h);
    const double * __restrict__ x      = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    const double * __restrict__ rhs    = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
    const double * __restrict__ alpha  = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
    const double * __restrict__ beta_i = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
    const double * __restrict__ beta_j = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
    const double * __restrict__ beta_k = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
          double * __restrict__ res    = level->my_boxes[box].vectors[       res_id] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      double Ax = apply_op_ijk(x);
      res[ijk] = rhs[ijk]-Ax;
    }}}
  }

  #if defined(USE_GPU_FOR_RESIDUAL)
  }
  #endif // USE_GPU_FOR_RESIDUAL
  level->timers.residual += (double)(getTime()-_timeStart);
}

