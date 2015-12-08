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
// Based on Yousef Saad's Iterative Methods for Sparse Linear Algebra, Algorithm 12.1, page 399
//------------------------------------------------------------------------------------------------------------------------------

#define INITIALIZE_CONSTS                               \
      const int box = level->my_blocks[block].read.box; \
      const int ilo = level->my_blocks[block].read.i;   \
      const int jlo = level->my_blocks[block].read.j;   \
      const int klo = level->my_blocks[block].read.k;   \
      const int idim = level->my_blocks[block].dim.i;   \
      const int jdim = level->my_blocks[block].dim.j;   \
      const int kdim = level->my_blocks[block].dim.k;   \
      const int ihi = idim + ilo;                       \
      const int jhi = jdim + jlo;                       \
      const int khi = kdim + klo;                       \
      const int ghosts = level->box_ghosts;             \
      const int jStride = level->my_boxes[box].jStride; \
      const int kStride = level->my_boxes[box].kStride; \
      const double h2inv = 1.0/(level->h*level->h);     \
      const double * AMP_RESTRICT rhs      = level->my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT alpha    = level->my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT beta_i   = level->my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT beta_j   = level->my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT beta_k   = level->my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT Dinv     = level->my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride); \
      const double * AMP_RESTRICT valid    = level->my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); \
            double * AMP_RESTRICT x_np1;                                                                               \
      const double * AMP_RESTRICT x_n;                                                                                 \
      const double * AMP_RESTRICT x_nm1;                                                                               \
                       if((s&1)==0){x_n    = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); \
                                    x_nm1  = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride); \
                                    x_np1  = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}\
                               else{x_n    = level->my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride); \
                                    x_nm1  = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); \
                                    x_np1  = level->my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}\
      const double c1 = chebyshev_c1[s%CHEBYSHEV_DEGREE];                                                              \
      const double c2 = chebyshev_c2[s%CHEBYSHEV_DEGREE];

#define LDS_LOOP_BODY                                             \
            const int ijk = i + j*jStride + k*kStride;            \
            DECLARE_LXN; INITIALIZE_LXN(x_n);                     \
            DECLARE_LBK; INITIALIZE_LBK();                        \
            DECLARE_LBJ; INITIALIZE_LBJ();                        \
            DECLARE_LBI; INITIALIZE_LBI();                        \
            DECLARE_LVAL; INITIALIZE_LVAL();                      \
            tidx.barrier.wait_with_tile_static_memory_fence();    \
            const double lambda = Dinv_ijk_amp();                 \
            const double Ax_n = apply_op_ijk_amp(x_n);            \
            x_np1[ijk] = lxn(x_n,0,0,0) + c1*(lxn(x_n,0,0,0)-x_nm1[ijk]) + c2*lambda*(rhs[ijk]-Ax_n);

#define LOOP_BODY const int ijk = i + j*jStride + k*kStride; \
                  const double Ax_n   = apply_op_ijk(x_n); \
                  const double lambda =     Dinv_ijk(); \
                  x_np1[ijk] = x_n[ijk] + c1*(x_n[ijk]-x_nm1[ijk]) + c2*lambda*(rhs[ijk]-Ax_n);

#define SERIAL_LOOP  for(int k=klo;k<khi;k++){ \
                     for(int j=jlo;j<jhi;j++){ \
                     for(int i=ilo;i<ihi;i++){ \
                       LOOP_BODY; \
                     }}}


# if defined(USE_AMP)
#pragma message "Using GPU for smooth"
using namespace Concurrency;
#define BBS AMP_TILE_BLOCKS
#define KBS AMP_TILE_K
#define JBS AMP_TILE_J
#define IBS AMP_TILE_I
#define MAX_AMP_TILES 1024

#define to_ijk(k,j,i) ((i)+(j)*jStride+(k)*kStride)

#if defined(AMP_DEBUG)
// Until debugging on the GPU is better supported, define some buffers
// Even better, only define the ones you want to use
double buff_xn[100000];
double buff_xnpm1[100000];
double buff_Axn[100000];
double buff_beta_i[20000];
double buff_beta_j[20000];
double buff_beta_k[20000];
double buff_rhs[20000];
double buff_Dinv[20000];
#endif // AMP_DEBUG


void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
  if((CHEBYSHEV_DEGREE*NUM_SMOOTHS)&1){
    fprintf(stderr,"error... CHEBYSHEV_DEGREE*NUM_SMOOTHS must be even for the chebyshev smoother...\n");
    exit(0);
  }
  if( (level->dominant_eigenvalue_of_DinvA<=0.0) && (level->my_rank==0) )
    fprintf(stderr,"dominant_eigenvalue_of_DinvA <= 0.0 !\n");


  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  int s;
  int block;

  print_amp_info();


  // compute the Chebyshev coefficients...
  double beta     = 1.000*level->dominant_eigenvalue_of_DinvA;
//double alpha    = 0.300000*beta;
//double alpha    = 0.250000*beta;
//double alpha    = 0.166666*beta;
  double alpha    = 0.125000*beta;
  double theta    = 0.5*(beta+alpha);		// center of the spectral ellipse
  double delta    = 0.5*(beta-alpha);		// major axis?
  double sigma = theta/delta;
  double rho_n = 1/sigma;			// rho_0
  double chebyshev_c1[CHEBYSHEV_DEGREE];	// + c1*(x_n-x_nm1) == rho_n*rho_nm1
  double chebyshev_c2[CHEBYSHEV_DEGREE];	// + c2*(b-Ax_n)
  chebyshev_c1[0] = 0.0;
  chebyshev_c2[0] = 1/theta;
  for(s=1;s<CHEBYSHEV_DEGREE;s++){
    double rho_nm1 = rho_n;
    rho_n = 1.0/(2.0*sigma - rho_nm1);
    chebyshev_c1[s] = rho_n*rho_nm1;
    chebyshev_c2[s] = rho_n*2.0/delta;
  }

  for(s=0;s<CHEBYSHEV_DEGREE*NUM_SMOOTHS;s++){
    // get ghost zone data... Chebyshev ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}


    const int nBlocks = level->num_my_blocks;
    const int kDim = level->my_blocks[0].dim.k;
    const int jDim = level->my_blocks[0].dim.j;
    const int iDim = level->my_blocks[0].dim.i;


    // apply the smoother... Chebyshev ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

    #ifdef AMP_ASSERT
    for (block=0;block<nBlocks;block++) {
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
    #warning This breaks the 3.3 compiler.  If it breaks yours, try AMP_DIM=3

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

      }); // end pfe
    } else { // need full serial implementation here
      for(block=0;block<level->num_my_blocks;block++){
        INITIALIZE_CONSTS
        SERIAL_LOOP
      } // blocks
    } // end serial

    #elif (AMP_TILE_DIM==3)

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
        SERIAL_LOOP
      } // blocks
    } // end serial
    #else // AMP_TILE_DIM
    #error Unrecognized combination of AMP_DIM and AMP_TILE_DIM
    #endif // AMP_TILE_DIM

    #elif defined(AMP_DIM) && (AMP_DIM==3)

    // Don't parallelize the block loop
    // loop over all block/tiles this process owns...
    for(block=0;block<level->num_my_blocks;block++){
      INITIALIZE_CONSTS

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
      //
      // This is the main version, which tiles the three inner loops on the GPU
      // and optionally copies some arrays into tile_static versions on the GPU
      //
      if ( ((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {

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
    #endif // AMP_TILE_DIM
    } // block-loop

    #elif defined(AMP_ARRAY_VIEW) && (AMP_ARRAY_VIEW==3)
    // This is a very rudimentary array_view implementation, preserved here in
    // case a starting point is desired later

    #pragma message "AMP on 3 loops, array_view version"
    #ifndef OPERATOR_fv2
      #error Only FV2 is currently supported by array_view version
    #endif
    #if defined(STENCIL_FUSE_BC)
      #error Fusing BV is not supported by FV2.
    #endif
    #ifdef USE_HELMHOLTZ
      #error HELMHOLTZ is not yet supported by array_vew version
    #endif
  
    static bool print_array_view_info = false;

    if ( ((idim % IBS) == 0) && ((jdim % JBS) == 0) && ((kdim % KBS) == 0)) {

      extent<3> e(kdim,jdim,idim);
      //
      // This will be a lot simpler if we make all array_view statements the same size and offset.
      // Since all arrays have basically the same declaration, this should be OK.
      //
      // x_n = vectors[base] + ghosts*(1+jStride+kStride) = vectors[base] + ghosts*to_ijk(1,1,1)
      // We essentially read  x_n[to_ijk(klo-1,jlo-1,ilo-1)..to_ijk(khi,jhi,ihi)]
      // This is not quite exact, but should work.
      // So base is x_n[to_ijk(klo-1,jlo-1,ilo-1)]
      // and size is to_ijk(khi,jhi,ihi) - to_ijk(klo-1,jlo-1,ilo-1)+1
      // = to_ijk(khi,jhi,ihi) - to_ijk(klo,jlo,ilo) + to_ijk(1,1,1) + 1
      // = to_ijk(kdim+1,jdim+1,idim+1) + 1
      //
      array_view<const double, 1> x_n_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &x_n[to_ijk(klo-1,jlo-1,ilo-1)]);
      if (print_array_view_info) {
        fprintf(stderr, "klo = %d, khi = %d, jlo = %d, jhi = %d, ilo = %d, ihi = %d\n", klo, khi, jlo, jhi, ilo, ihi);
        fprintf(stderr, "x_n_av: size = %d, base = &x_n[%d]\n", to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
      }
      //
      // Now we'll make the others match, even though they may not reference all the locations.
      //
      array_view<const double, 1> rhs_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &rhs[to_ijk(klo-1,jlo-1,ilo-1)]);
      if (print_array_view_info) {
          fprintf(stderr, "rhs_av: size = %d, base = &rhs[%d]\n", to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
      }
      //
      // Using fv2 implies that Dinv_ijk() is Dinv[ijk]
      //
      array_view<const double, 1> Dinv_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &Dinv[to_ijk(klo-1,jlo-1,ilo-1)]);
      if (print_array_view_info) {
          fprintf(stderr, "Dinv_av: size = %d, base = &Dinv[%d]\n", to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
      }
      #ifdef STENCIL_VARIABLE_COEFFICIENT
        array_view<const double, 1> beta_i_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &beta_i[to_ijk(klo-1,jlo-1,ilo-1)]);
        array_view<const double, 1> beta_j_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &beta_j[to_ijk(klo-1,jlo-1,ilo-1)]);
        array_view<const double, 1> beta_k_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &beta_k[to_ijk(klo-1,jlo-1,ilo-1)]);
        if (print_array_view_info) {
            fprintf(stderr, "beta_i_av: size = %d, base = &&beta_i[%d]\n",
                    to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
            fprintf(stderr, "beta_j_av: size = %d, base = &&beta_j[%d]\n",
                    to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
            fprintf(stderr, "beta_k_av: size = %d, base = &&beta_k[%d]\n",
                    to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
        }
      #endif
      //
      // note that x_nm1 and x_np1 are the same
      //
      array_view<double, 1> x_npm1_av(to_ijk(kdim+1,jdim+1,idim+1)+1, &x_np1[to_ijk(klo-1,jlo-1,ilo-1)]);
      if (print_array_view_info) {
          fprintf(stderr, "x_npm1_av: size = %d, base = &&x_np1[%d]\n", to_ijk(kdim+1,jdim+1,idim+1)+1, to_ijk(klo-1,jlo-1,ilo-1));
          print_array_view_info = false;
      }

      #if defined(AMP_DEBUG)

        //
        // Set up a debugging array that matches the other array shapes.
        //

        // Just initializing these so we can tell they have changed
        for (int iw = 0; iw < 20000; ++iw) {
          buff_xn[iw] = -iw;
          buff_xnpm1[iw] = -iw;
          buff_Axn[iw] = -iw;
          buff_beta_i[iw] = -iw;
          buff_beta_j[iw] = -iw;
          buff_beta_k[iw] = -iw;
          buff_rhs[iw] = -iw;
          buff_Dinv[iw] = -iw;
        }

        double *deb_xn = buff_xn + to_ijk(1,1,1);
        double *deb_xnpm1 = buff_xnpm1 + to_ijk(1,1,1);
        double *deb_Axn = buff_Axn + to_ijk(1,1,1);
        double *deb_beta_i = buff_beta_i + to_ijk(1,1,1);
        double *deb_beta_j = buff_beta_j + to_ijk(1,1,1);
        double *deb_beta_k = buff_beta_k + to_ijk(1,1,1);
        double *deb_rhs = buff_rhs + to_ijk(1,1,1);
        double *deb_Dinv = buff_Dinv + to_ijk(1,1,1);

        // You probably don't really want to use ALL of these simultaneously. Pick the ones you want.
        array_view<double, 1> deb_av_xn(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_xn[to_ijk(klo-1,jlo-1,ilo-1)]);
        array_view<double, 1> deb_av_xnpm1(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_xnpm1[to_ijk(klo-1,jlo-1,ilo-1)]);
        // array_view<double, 1> deb_av_Axn(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_Axn[to_ijk(klo-1,jlo-1,ilo-1)]);
        // array_view<double, 1> deb_av_beta_i(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_beta_i[to_ijk(klo-1,jlo-1,ilo-1)]);
        // array_view<double, 1> deb_av_beta_j(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_beta_j[to_ijk(klo-1,jlo-1,ilo-1)]);
        // array_view<double, 1> deb_av_beta_k(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_beta_k[to_ijk(klo-1,jlo-1,ilo-1)]);
        // array_view<double, 1> deb_av_rhs(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_rhs[to_ijk(klo-1,jlo-1,ilo-1)]);
        // array_view<double, 1> deb_av_Dinv(to_ijk(kdim+1,jdim+1,idim+1)+1, &deb_Dinv[to_ijk(klo-1,jlo-1,ilo-1)]);

        //
        // so deb_av(0) should be deb_av[to_ijk(klo-1,jlo-1,ilo-1)] should be buff_av[0]
        //

        fprintf(stderr, "extent is %d,%d,%d\n", kdim,jdim, idim);
        fprintf(stderr, "KBS=%d,JBS=%d,IBS=%d\n", KBS,JBS,IBS);
        fprintf(stderr, "klo = %d, khi = %d, jlo = %d, jhi = %d, ilo = %d, ihi = %d\n", klo, khi, jlo, jhi, ilo, ihi);
      #endif // AMP_DEBUG

      parallel_for_each(
        e.tile<KBS,JBS,IBS>(),
        [=] (tiled_index<KBS,JBS,IBS> tidx) restrict(amp) {
          const int k = tidx.global[0];
          const int j = tidx.global[1];
          const int i = tidx.global[2];
          const int ijk = to_ijk(k,j,i);

          //  In the original, this is
          //  const double Ax_n   = apply_op_ijk(x_n);
          //  The array_view case is so ugly I didn't want to put it in operators_fv2.c,
          //  so code from there is copied and modified here
          double Ax_n;

          # if defined(AMP_DEBUG)
            deb_av_xn(to_ijk(k+1,j+1,i+1)) = x_n_av(to_ijk(k+1,j+1,i+1)); // this works for interior of x_n
            // deb_av_beta_i(to_ijk(k+1,j+1,i+1)) = beta_i_av(to_ijk(k+1,j+1,i+1));
            // deb_av_beta_j(to_ijk(k+1,j+1,i+1)) = beta_j_av(to_ijk(k+1,j+1,i+1));
            // deb_av_beta_k(to_ijk(k+1,j+1,i+1)) = beta_k_av(to_ijk(k+1,j+1,i+1));
          # endif defined(AMP_DEBUG)

          #ifdef STENCIL_VARIABLE_COEFFICIENT
            Ax_n = -b*h2inv*(
                + beta_i_av(to_ijk(k+1,j+1,i+2))*(x_n_av(to_ijk(k+1,j+1,i+2)) - x_n_av(to_ijk(k+1,j+1,i+1)))
                + beta_i_av(to_ijk(k+1,j+1,i+1  ))*(x_n_av(to_ijk(k+1,j+1,i  )) - x_n_av(to_ijk(k+1,j+1,i+1)))
                + beta_j_av(to_ijk(k+1,j+2,i+1  ))*(x_n_av(to_ijk(k+1,j+2,i+1)) - x_n_av(to_ijk(k+1,j+1,i+1)))
                + beta_j_av(to_ijk(k+1,j+1,i+1  ))*(x_n_av(to_ijk(k+1,j  ,i+1)) - x_n_av(to_ijk(k+1,j+1,i+1)))
                + beta_k_av(to_ijk(k+2,j+1,i+1  ))*(x_n_av(to_ijk(k+2,j+1,i+1)) - x_n_av(to_ijk(k+1,j+1,i+1)))
                + beta_k_av(to_ijk(k+1,j+1,i+1  ))*(x_n_av(to_ijk(k  ,j+1,i+1)) - x_n_av(to_ijk(k+1,j+1,i+1))));
          #ifdef USE_HELMHOLTZ
            Ax_n += alpha_av(to_ijk(k,j,i)*x_n_av(to_ijk(k+1,j+1,i+1)));
          #endif // USE_HELMHOLTZ
          #else // STENCIL_VARIABLE_COEFFICIENT
            Ax_n = a*x_n_av(k+1,j+1,i+1) - b2*h2inv*( x_n_av(k+1,j+1,i+2) + x_n_av(k+1,j+1,i  )
                                                     +x_n_av(k+1,j+2,i+1) + x_n_av(k+1,j  ,i+1)
                                                     +x_n_av(k+2,j+1,i+1) + x_n_av(k  ,j+1,i+1)
                                                     -x_n_av(k+1,j+1,i+1)*6.0);
          #endif // STENCIL_VARIABLE_COEFFICIENT

          # if defined(AMP_DEBUG)
            // deb_av_Axn(to_ijk(k+1,j+1,i+1)) = Ax_n;
          # endif defined(AMP_DEBUG)


          const double lambda =     Dinv_av(to_ijk(k+1,j+1,i+1));
          # if defined(AMP_DEBUG)
            // deb_av_Dinv(to_ijk(k+1,j+1,i+1)) = lambda;
          # endif defined(AMP_DEBUG)
          x_npm1_av(to_ijk(k+1,j+1,i+1)) = x_n_av(to_ijk(k+1,j+1,i+1)) + c1*(x_n_av(to_ijk(k+1,j+1,i+1))
                                         - x_npm1_av(to_ijk(k+1,j+1,i+1)))
                                         + c2*lambda*(rhs_av(to_ijk(k+1,j+1,i+1)) - Ax_n);
          # if defined(AMP_DEBUG)
            deb_av_xnpm1(to_ijk(k+1,j+1,i+1)) = x_npm1_av(to_ijk(k+1,j+1,i+1));
          # endif defined(AMP_DEBUG)
        }
      );
      x_npm1_av.synchronize();

      # if defined(AMP_DEBUG)
      deb_av_xn.synchronize();
      deb_av_xnpm1.synchronize();
      // deb_av_Axn.synchronize();
      // deb_av_beta_i.synchronize();
      // deb_av_beta_j.synchronize();
      // deb_av_beta_k.synchronize();
      // deb_av_Dinv.synchronize();

      printf("c1 = %lg, c2 = %lg\n", c1, c2);
      printf("klo = %d, khi = %d, jlo = %d, jhi = %d, ilo = %d, ihi = %d\n", klo, khi, jlo, jhi, ilo, ihi);
      for(k=klo;k<khi;k++){
      for(j=jlo;j<jhi;j++){
      for(i=ilo;i<ihi;i++){
        fprintf(stderr, "(%d,%d,%d): index = %d,   deb_xn[index] = %lg, deb_xnpm1[index] = %lg, x_np1[index] = %lg\n",
                k, j, i, to_ijk(k,j,i),   deb_xn[to_ijk(k,j,i)], deb_xnpm1[to_ijk(k,j,i)], x_np1[to_ijk(k,j,i)]);
        printf(stderr, "(%d,%d,%d): index = %d,   deb_xn[index] = %lg, deb_xnpm1[index] = %lg, deb_Axn[index] = %lg\n",
                k, j, i, to_ijk(k,j,i),   deb_xn[to_ijk(k,j,i)], deb_xnpm1[to_ijk(k,j,i)], deb_Axn[to_ijk(k,j,i)]);
        fflush(stderr);

      }}}
      // You may want to see only the first AMP iteration to avoid mountains of output, in which case
      // exit(1)
      # endif defined(AMP_DEBUG)

    } else {
        SERIAL_LOOP
    }
    #else // AMP_DIM, AMP_ARRAY_VIEW
    #error Unrecognized VERSION
    #endif // AMP_DIM, AMP_ARRAY_VIEW

    /*
    printf("s = %d, %d blocks, dim [%d,%d,%d], time %lg sec\n",  // DEBUG
            s, level->num_my_blocks, level->my_blocks[0].dim.k, // DEBUG
            level->my_blocks[0].dim.j, level->my_blocks[0].dim.i, // DEBUG
            mytime); // DEBUG
            */
    double mytime = (double)(getTime()-_timeStart);
    level->timers.smooth += mytime;
  } // s-loop
}
#else // USE_AMP
void smooth(level_type * level, int x_id, int rhs_id, double a, double b){
  if((CHEBYSHEV_DEGREE*NUM_SMOOTHS)&1){
    fprintf(stderr,"error... CHEBYSHEV_DEGREE*NUM_SMOOTHS must be even for the chebyshev smoother...\n");
    exit(0);
  }
  if( (level->dominant_eigenvalue_of_DinvA<=0.0) && (level->my_rank==0) )
    fprintf(stderr,"dominant_eigenvalue_of_DinvA <= 0.0 !\n");


  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  int s;
  int block;


  // compute the Chebyshev coefficients...
  double beta     = 1.000*level->dominant_eigenvalue_of_DinvA;
//double alpha    = 0.300000*beta;
//double alpha    = 0.250000*beta;
//double alpha    = 0.166666*beta;
  double alpha    = 0.125000*beta;
  double theta    = 0.5*(beta+alpha);		// center of the spectral ellipse
  double delta    = 0.5*(beta-alpha);		// major axis?
  double sigma = theta/delta;
  double rho_n = 1/sigma;			// rho_0
  double chebyshev_c1[CHEBYSHEV_DEGREE];	// + c1*(x_n-x_nm1) == rho_n*rho_nm1
  double chebyshev_c2[CHEBYSHEV_DEGREE];	// + c2*(b-Ax_n)
  chebyshev_c1[0] = 0.0;
  chebyshev_c2[0] = 1/theta;
  for(s=1;s<CHEBYSHEV_DEGREE;s++){
    double rho_nm1 = rho_n;
    rho_n = 1.0/(2.0*sigma - rho_nm1);
    chebyshev_c1[s] = rho_n*rho_nm1;
    chebyshev_c2[s] = rho_n*2.0/delta;
  }

  for(s=0;s<CHEBYSHEV_DEGREE*NUM_SMOOTHS;s++){
    // get ghost zone data... Chebyshev ping pongs between x_id and VECTOR_TEMP
    if((s&1)==0){exchange_boundary(level,       x_id,stencil_get_shape());apply_BCs(level,       x_id,stencil_get_shape());}
            else{exchange_boundary(level,VECTOR_TEMP,stencil_get_shape());apply_BCs(level,VECTOR_TEMP,stencil_get_shape());}
   
    // apply the smoother... Chebyshev ping pongs between x_id and VECTOR_TEMP
    double _timeStart = getTime();

    PRAGMA_THREAD_ACROSS_BLOCKS(level,block,level->num_my_blocks)
    for(block=0;block<level->num_my_blocks;block++){
      INITIALIZE_CONSTS
      SERIAL_LOOP
    } // block-loop
    double mytime = (double)(getTime()-_timeStart);
    level->timers.smooth += mytime;
  } // s-loop
}
#endif // USE_AMP

