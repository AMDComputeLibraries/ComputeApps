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
#if defined(USE_GPU_FOR_BLAS)
#if defined(__KALMAR_AMP__) || defined(__HCC_AMP__)
#define USE_GPU_FOR_ZERO
#define USE_GPU_FOR_SCALE
#define USE_GPU_FOR_NORM
using namespace Concurrency;
#elif defined(__KALMAR_HC__) || defined(__HCC_HC__)
#define USE_GPU_FOR_ZERO
#define USE_GPU_FOR_SCALE
#define USE_GPU_FOR_NORM
using namespace hc;
#else
#error Either __KALMAR_AMP__ / __HCC_AMP__  or __KALMAR_HC__ / __HCC_HC__ must be defined with USE_GPU_FOR_BLAS
#endif // KALMAR or HCC
#else // USE_GPU_FOR_BLAS
#undef USE_GPU_FOR_ZERO
#undef USE_GPU_FOR_SCALE
#undef USE_GPU_FOR_NORM
#endif // USE_GPU_FOR_BLAS

#define MY_THRESH 65536

#if defined(USE_GPU_FOR_ZERO)
void zero_vector(level_type * level, int id_a){
  // zero's the entire grid INCLUDING ghost zones...
  double _timeStart = getTime();
  int block;

#if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
#endif

  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
          int ilo = level->my_blocks[block].read.i;
          int jlo = level->my_blocks[block].read.j;
          int klo = level->my_blocks[block].read.k;
          int ihi = level->my_blocks[block].dim.i + ilo;
          int jhi = level->my_blocks[block].dim.j + jlo;
          int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    const int     dim = level->my_boxes[box].dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts; 
    if(jlo<=  0)jlo-=ghosts; 
    if(klo<=  0)klo-=ghosts; 
    if(ihi>=dim)ihi+=ghosts; 
    if(jhi>=dim)jhi+=ghosts; 
    if(khi>=dim)khi+=ghosts; 

    double * GPU_RESTRICT grid = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    const int kdim = khi-klo;
    const int jdim = jhi-jlo;
    const int idim = ihi-ilo;
    if (kdim*jdim*idim > MY_THRESH) {
      extent<3>e(kdim,jdim,idim);
      parallel_for_each(e, [=] (index<3> idx) restrict(amp) {
        const int k = idx[0] + klo;
        const int j = idx[1] + jlo;
        const int i = idx[2] + ilo;
        const int ijk = i + j*jStride + k*kStride;
        grid[ijk] = 0.0;
      });
      #if defined(__KALMAR_HC__) || defined(__HCC_HC__)
      av.wait();
      #endif
    } else {
      #pragma omp parallel for collapse(2) private(k,j,i) if ((kdim*jdim) > 1)
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid[ijk] = 0.0;
      }}}
    }
  }
  double temp = (double)(getTime()-_timeStart);
  level->timers.blas1 += temp;
  #ifdef BLAS1_DETAIL
  level->timers.blas1_zero_vector += temp;
  #endif
}
#else // USE_GPU_FOR_ZERO
void zero_vector(level_type * level, int id_a){
  // zero's the entire grid INCLUDING ghost zones...
  double _timeStart = getTime();
  int block;

  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
          int ilo = level->my_blocks[block].read.i;
          int jlo = level->my_blocks[block].read.j;
          int klo = level->my_blocks[block].read.k;
          int ihi = level->my_blocks[block].dim.i + ilo;
          int jhi = level->my_blocks[block].dim.j + jlo;
          int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    const int     dim = level->my_boxes[box].dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts; 
    if(jlo<=  0)jlo-=ghosts; 
    if(klo<=  0)klo-=ghosts; 
    if(ihi>=dim)ihi+=ghosts; 
    if(jhi>=dim)jhi+=ghosts; 
    if(khi>=dim)khi+=ghosts; 

    double * GPU_RESTRICT grid = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      grid[ijk] = 0.0;
    }}}
  }
  double temp = (double)(getTime()-_timeStart);
  level->timers.blas1 += temp;
  #ifdef BLAS1_DETAIL
  level->timers.blas1_zero_vector += temp;
  #endif
}
#endif // USE_GPU_FOR_ZERO


//------------------------------------------------------------------------------------------------------------------------------
void init_vector(level_type * level, int id_a, double scalar){
  // initializes the grid to a scalar while zero'ing the ghost zones...
  double _timeStart = getTime();
  int block;

  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
          int ilo = level->my_blocks[block].read.i;
          int jlo = level->my_blocks[block].read.j;
          int klo = level->my_blocks[block].read.k;
          int ihi = level->my_blocks[block].dim.i + ilo;
          int jhi = level->my_blocks[block].dim.j + jlo;
          int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    const int     dim = level->my_boxes[box].dim;

    // expand the size of the block to include the ghost zones...
    if(ilo<=  0)ilo-=ghosts; 
    if(jlo<=  0)jlo-=ghosts; 
    if(klo<=  0)klo-=ghosts; 
    if(ihi>=dim)ihi+=ghosts; 
    if(jhi>=dim)jhi+=ghosts; 
    if(khi>=dim)khi+=ghosts; 

    double * __restrict__ grid = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        int ghostZone = (i<0) || (j<0) || (k<0) || (i>=dim) || (j>=dim) || (k>=dim);
        grid[ijk] = ghostZone ? 0.0 : scalar;
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_init_vector += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}


//------------------------------------------------------------------------------------------------------------------------------
// add vectors id_a (scaled by scale_a) and id_b (scaled by scale_b) and store the result in vector id_c
// i.e. c[] = scale_a*a[] + scale_b*b[]
// note, only non ghost zone values are included in this calculation
void add_vectors(level_type * level, int id_c, double scale_a, int id_a, double scale_b, int id_b){
  double _timeStart = getTime();

  int block;

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
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid_c = level->my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_b = level->my_boxes[box].vectors[id_b] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a*grid_a[ijk] + scale_b*grid_b[ijk];
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_add_vectors += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}


//------------------------------------------------------------------------------------------------------------------------------
// multiply each element of vector id_a by vector id_b and scale, and place the result in vector id_c
// i.e. c[]=scale*a[]*b[]
// note, only non ghost zone values are included in this calculation
void mul_vectors(level_type * level, int id_c, double scale, int id_a, int id_b){
  double _timeStart = getTime();

  int block;

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
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid_c = level->my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_b = level->my_boxes[box].vectors[id_b] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale*grid_a[ijk]*grid_b[ijk];
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_mul_vectors += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}


//------------------------------------------------------------------------------------------------------------------------------
// invert each element of vector id_a, scale by scale_a, and place the result in vector id_c
// i.e. c[]=scale_a/a[]
// note, only non ghost zone values are included in this calculation
void invert_vector(level_type * level, int id_c, double scale_a, int id_a){
  double _timeStart = getTime();

  int block;

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
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid_c = level->my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    double * __restrict__ grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a/grid_a[ijk];
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_invert_vector += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}


//------------------------------------------------------------------------------------------------------------------------------
// scale vector id_a by scale_a and place the result in vector id_c
// i.e. c[]=scale_a*a[]
// note, only non ghost zone values are included in this calculation
#ifdef USE_GPU_FOR_SCALE
void scale_vector(level_type * level, int id_c, double scale_a, int id_a){
  double _timeStart = getTime();

  int block;

#if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
#endif

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
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * GPU_RESTRICT grid_c = level->my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    double * GPU_RESTRICT grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    if (kdim*jdim*idim > MY_THRESH) {
      extent<3>e(kdim,jdim,idim);
      parallel_for_each(e, [=] (index<3> idx) restrict(amp) {
        const int k = idx[0] + klo;
        const int j = idx[1] + jlo;
        const int i = idx[2] + ilo;
        const int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a*grid_a[ijk];
      });
      #if defined(__KALMAR_HC__)
      av.wait();
      #endif
    } else {
      #pragma omp parallel for collapse(2) private(k,j,i) if ((kdim*jdim) > 1)
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a*grid_a[ijk];
      }}}
    }
  }
  double temp = (double)(getTime()-_timeStart);
  level->timers.blas1 += temp;
  #ifdef BLAS1_DETAIL
  level->timers.blas1_scale_vector += temp;
  #endif
}
#else // USE_GPU_FOR_SCALE
void scale_vector(level_type * level, int id_c, double scale_a, int id_a){
  double _timeStart = getTime();

  int block;

  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
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
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * GPU_RESTRICT grid_c = level->my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride);
    double * GPU_RESTRICT grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride);

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
        int ijk = i + j*jStride + k*kStride;
        grid_c[ijk] = scale_a*grid_a[ijk];
    }}}
  }
  double temp = (double)(getTime()-_timeStart);
  level->timers.blas1 += temp;
  #ifdef BLAS1_DETAIL
  level->timers.blas1_scale_vector += temp;
  #endif
}
#endif // USE_GPU_FOR_SCALE


//------------------------------------------------------------------------------------------------------------------------------
// return the dot product of vectors id_a and id_b
// note, only non ghost zone values are included in this calculation
double dot(level_type * level, int id_a, int id_b){
  double _timeStart = getTime();


  int block;
  double a_dot_b_level =  0.0;

  PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,block,level->num_my_blocks,a_dot_b_level)
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
    double * __restrict__ grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    double * __restrict__ grid_b = level->my_boxes[box].vectors[id_b] + ghosts*(1+jStride+kStride);
    double a_dot_b_block = 0.0;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      a_dot_b_block += grid_a[ijk]*grid_b[ijk];
    }}}
    a_dot_b_level+=a_dot_b_block;
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_dot += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif

  #ifdef USE_MPI
  double _timeStartAllReduce = getTime();
  double send = a_dot_b_level;
  MPI_Allreduce(&send,&a_dot_b_level,1,MPI_DOUBLE,MPI_SUM,level->MPI_COMM_ALLREDUCE);
  double _timeEndAllReduce = getTime();
  level->timers.collectives   += (double)(_timeEndAllReduce-_timeStartAllReduce);
  #endif

  return(a_dot_b_level);
}

//------------------------------------------------------------------------------------------------------------------------------
// return the max (infinity) norm of the vector id_a.
// note, only non ghost zone values are included in this calculation

#if defined(USE_GPU_FOR_NORM)
double norm(level_type * level, int id_a){ // implements the max norm
  double _timeStart = getTime();

  int block;
  double max_norm =  0.0;

#if defined(__KALMAR_HC__) || defined(__HCC_HC__)
  accelerator default_acc;
  accelerator_view av = default_acc.get_default_view();
#endif

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
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * GPU_RESTRICT grid   = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    double block_norm = 0.0;

    if ( (idim >= IBS) && ((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {
    double *dummy = new double[(idim/IBS)*(jdim/JBS)*(kdim/KBS)];
    extent<3>e(kdim,jdim,idim);
    #if defined(__KALMAR_AMP__)
    parallel_for_each( e.tile<KBS,JBS,IBS>(),
    [=] (tiled_index<KBS,JBS,IBS> tidx) restrict(amp)
    #elif defined(__KALMAR_HC__)
    parallel_for_each( e.tile(KBS,JBS,IBS),
    [=] (tiled_index<3> tidx) restrict(amp)
    #else
    #error Either __KALMAR_AMP__ or __KALMAR_HC__ must befined when USE_GPU_FOR_NORM is
    #endif
    {
      tile_static double temp[KBS][JBS][IBS];
      const int k = tidx.global[0]+klo;
      const int j = tidx.global[1]+jlo;
      const int i = tidx.global[2]+ilo;
      const int lk = tidx.local[0];
      const int lj = tidx.local[1];
      const int li = tidx.local[2];
      int ijk = i + j*jStride + k*kStride;
      temp[lk][lj][li] = precise_math::fabs(grid[ijk]);

      tidx.barrier.wait_with_tile_static_memory_fence();
      for (int iinc = IBS/2; iinc >= 1; iinc/=2) {
        temp[lk][lj][li] = precise_math::fmax(temp[lk][lj][li],temp[lk][lj][li+iinc]);
      }
      tidx.barrier.wait_with_tile_static_memory_fence();
      for (int jinc = JBS/2; jinc >= 1; jinc/=2) {
        temp[lk][lj][0] = precise_math::fmax(temp[lk][lj][0], temp[lk][lj+jinc][0]);
      }
      tidx.barrier.wait_with_tile_static_memory_fence();
      for (int kinc = KBS/2; kinc >= 1; kinc/=2) {
        temp[lk][0][0] = precise_math::fmax(temp[lk][0][0], temp[lk+kinc][0][0]);
      }
      tidx.barrier.wait_with_tile_static_memory_fence();
      dummy[tidx.tile[0]*(jdim/JBS)*(idim/IBS)+tidx.tile[1]*(idim/IBS)+tidx.tile[2]] = temp[0][0][0];
    });
    #if defined(__KALMAR_HC__)
    av.wait();
    #endif

    for(k=0;k<kdim/KBS;k++){
    for(j=0;j<jdim/JBS;j++){
    for(i=0;i<idim/IBS;i++){ 
      block_norm = fmax(block_norm,dummy[k*(jdim/JBS)*(idim/IBS)+j*(idim/IBS)+i]);
    }}}
    delete[] dummy;
    } else {
      #pragma omp parallel for collapse(2) private(k,j,i) if ((kdim*jdim) > 1)
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){ 
        int ijk = i + j*jStride + k*kStride;
        double fabs_grid_ijk = fabs(grid[ijk]);
        if(fabs_grid_ijk>block_norm){block_norm=fabs_grid_ijk;} // max norm
      }}}
    }
    if(block_norm>max_norm){max_norm = block_norm;}
  } // block list
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_norm += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif

  #ifdef USE_MPI
  double _timeStartAllReduce = getTime();
  double send = max_norm;
  MPI_Allreduce(&send,&max_norm,1,MPI_DOUBLE,MPI_MAX,level->MPI_COMM_ALLREDUCE);
  double _timeEndAllReduce = getTime();
  level->timers.collectives   += (double)(_timeEndAllReduce-_timeStartAllReduce);
  #endif
  return(max_norm);
}
#else // USE_GPU_FOR_NORM
double norm(level_type * level, int id_a){ // implements the max norm
  double _timeStart = getTime();

  int block;
  double max_norm =  0.0;

  PRAGMA_THREAD_ACROSS_BLOCKS_MAX(level,block,level->num_my_blocks,max_norm)
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
    int i,j,k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * GPU_RESTRICT grid   = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    double block_norm = 0.0;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){ 
      int ijk = i + j*jStride + k*kStride;
      double fabs_grid_ijk = fabs(grid[ijk]);
      if(fabs_grid_ijk>block_norm){block_norm=fabs_grid_ijk;} // max norm
    }}}
    if(block_norm>max_norm){max_norm = block_norm;}
  } // block list
  double temp = (double)(getTime()-_timeStart);
  level->timers.blas1 += temp;
  #ifdef BLAS1_DETAIL
  level->timers.blas1_norm += temp;
  #endif

  #ifdef USE_MPI
  double _timeStartAllReduce = getTime();
  double send = max_norm;
  MPI_Allreduce(&send,&max_norm,1,MPI_DOUBLE,MPI_MAX,level->MPI_COMM_ALLREDUCE);
  double _timeEndAllReduce = getTime();
  level->timers.collectives   += (double)(_timeEndAllReduce-_timeStartAllReduce);
  #endif
  return(max_norm);
}
#endif // USE_GPU_FOR_NORM


//------------------------------------------------------------------------------------------------------------------------------
// return the mean (arithmetic average value) of vector id_a
// essentially, this is a l1 norm by a scaling by the inverse of the total (global) number of cells
// note, only non ghost zone values are included in this calculation
double mean(level_type * level, int id_a){
  double _timeStart = getTime();


  int block;
  double sum_level =  0.0;

  PRAGMA_THREAD_ACROSS_BLOCKS_SUM(level,block,level->num_my_blocks,sum_level)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    int i,j,k;
    int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    double sum_block = 0.0;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      sum_block += grid_a[ijk];
    }}}
    sum_level+=sum_block;
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_mean += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
  double ncells_level = (double)level->dim.i*(double)level->dim.j*(double)level->dim.k;

  #ifdef USE_MPI
  double _timeStartAllReduce = getTime();
  double send = sum_level;
  MPI_Allreduce(&send,&sum_level,1,MPI_DOUBLE,MPI_SUM,level->MPI_COMM_ALLREDUCE);
  double _timeEndAllReduce = getTime();
  level->timers.collectives   += (double)(_timeEndAllReduce-_timeStartAllReduce);
  #endif

  double mean_level = sum_level / ncells_level;
  return(mean_level);
}


//------------------------------------------------------------------------------------------------------------------------------
// add the scalar value shift_a to each element of vector id_a and store the result in vector id_c
// note, only non ghost zone values are included in this calculation
void shift_vector(level_type * level, int id_c, int id_a, double shift_a){
  double _timeStart = getTime();
  int block;

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
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid_c = level->my_boxes[box].vectors[id_c] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    double * __restrict__ grid_a = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point


    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      grid_c[ijk] = grid_a[ijk] + shift_a;
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_shift_vector += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}

//------------------------------------------------------------------------------------------------------------------------------
// calculate the error between two vectors (id_a and id_b) using either the max (infinity) norm or the L2 norm
// note, only non ghost zone values are included in this calculation
double error(level_type * level, int id_a, int id_b){
  double h3 = level->h * level->h * level->h;
               add_vectors(level,VECTOR_TEMP,1.0,id_a,-1.0,id_b);            // VECTOR_TEMP = id_a - id_b
  double   max =      norm(level,VECTOR_TEMP);                return(max);   // max norm of error function
  double    L2 = sqrt( dot(level,VECTOR_TEMP,VECTOR_TEMP)*h3);return( L2);   // normalized L2 error ?
}


//------------------------------------------------------------------------------------------------------------------------------
// Color the vector id_a with 1's and 0's
// The pattern is dictated by the number of colors in each dimension and the 'active' color (i,j,kcolor)
// note, only non ghost zone values are included in this calculation
//   e.g. colors_in_each_dim=3, icolor=1, jcolor=2...
//   -+---+---+---+-
//    | 0 | 1 | 0 |
//   -+---+---+---+-
//    | 0 | 0 | 0 |
//   -+---+---+---+-
//    | 0 | 0 | 0 |
//   -+---+---+---+-
//
void color_vector(level_type * level, int id_a, int colors_in_each_dim, int icolor, int jcolor, int kcolor){
  double _timeStart = getTime();
  int block;

  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    const int boxlowi = level->my_boxes[box].low.i;
    const int boxlowj = level->my_boxes[box].low.j;
    const int boxlowk = level->my_boxes[box].low.k;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    int i,j,k;

    for(k=klo;k<khi;k++){double sk=0.0;if( ((k+boxlowk+kcolor)%colors_in_each_dim) == 0 )sk=1.0; // if colors_in_each_dim==1 (don't color), all cells are set to 1.0
    for(j=jlo;j<jhi;j++){double sj=0.0;if( ((j+boxlowj+jcolor)%colors_in_each_dim) == 0 )sj=1.0;
    for(i=ilo;i<ihi;i++){double si=0.0;if( ((i+boxlowi+icolor)%colors_in_each_dim) == 0 )si=1.0;
      int ijk = i + j*jStride + k*kStride;
      grid[ijk] = si*sj*sk;
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_color_vector += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}


//------------------------------------------------------------------------------------------------------------------------------
// Initialize each element of vector id_a with a "random" value.  
// For simplicity, random is defined as -1.0 or +1.0 and is based on whether the coordinates of the element are even or odd
// note, only non ghost zone values are included in this calculation
void random_vector(level_type * level, int id_a){
  double _timeStart = getTime();
  int block;

  PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
  for(block=0;block<level->num_my_blocks;block++){
    const int box = level->my_blocks[block].read.box;
    const int ilo = level->my_blocks[block].read.i;
    const int jlo = level->my_blocks[block].read.j;
    const int klo = level->my_blocks[block].read.k;
    const int ihi = level->my_blocks[block].dim.i + ilo;
    const int jhi = level->my_blocks[block].dim.j + jlo;
    const int khi = level->my_blocks[block].dim.k + klo;
    const int jStride = level->my_boxes[box].jStride;
    const int kStride = level->my_boxes[box].kStride;
    const int  ghosts = level->my_boxes[box].ghosts;
    double * __restrict__ grid = level->my_boxes[box].vectors[id_a] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
    int i,j,k;

    for(k=klo;k<khi;k++){
    for(j=jlo;j<jhi;j++){
    for(i=ilo;i<ihi;i++){
      int ijk = i + j*jStride + k*kStride;
      grid[ijk] = -1.000 + 2.0*(i^j^k^0x1);
    }}}
  }
  #ifdef BLAS1_DETAIL
  {
    double temp = (double)(getTime()-_timeStart);
    level->timers.blas1 += temp;
    level->timers.blas1_random_vector += temp;
  }
  #else
  level->timers.blas1 += (double)(getTime()-_timeStart);
  #endif
}


//------------------------------------------------------------------------------------------------------------------------------
