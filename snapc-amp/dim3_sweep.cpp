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
/***********************************************************************
 * Module: dim3_sweep.c
 *
 * This module contains the 2D and 3D mesh sweep logic.
 ***********************************************************************/
#include "snap.h"

// Local variable array macro
#define PSI_1D(ANG)   psi[ANG]
#define PC_1D(ANG)    pc[ANG]
#define DEN_1D(ANG)   den[ANG]

#ifdef ROWORDER
#define HV_2D(ANG, X) hv[ ANG*4                 \
                          + X ]
#else
#define HV_2D(ANG, X) hv[ X*NANG                \
                          + ANG ]
#endif

#ifdef ROWORDER
#define FXHV_2D(ANG, X) fxhv[ ANG*4             \
                              + X ]
#else
#define FXHV_2D(ANG, X) fxhv[ X*NANG            \
                              + ANG ]
#endif

// Simplify array indexing when certain values constant throughout module
#define PSII_3D(ANG, Y, Z)       PSII_4D(ANG, Y, Z, (g-1))
#define PSIJ_3D(ANG, CHUNK, Z)   PSIJ_4D(ANG, CHUNK, Z, (g-1))
#define PSIK_3D(ANG, CHUNK, Y)   PSIK_4D(ANG, CHUNK, Y, (g-1))
#define QTOT_4D(MOM1, X, Y, Z)   QTOT_5D(MOM1, X, Y, Z, (g-1))
#define EC_2D(ANG, MOM1)         EC_3D(ANG, MOM1, (oct-1))
#define VDELT_CONST              VDELT_1D(g-1)
#define PTR_IN_4D(ANG, X, Y, Z)  PTR_IN_6D(ANG, X, Y, Z, (i1-1), (i2-1))
#define PTR_OUT_4D(ANG, X, Y, Z) PTR_OUT_6D(ANG, X, Y, Z, (i1-1), (i2-1))
#define DINV_4D(ANG, X, Y, Z)    DINV_5D(ANG, X, Y, Z, (g-1))
#define FLUX_3D(X, Y, Z)         FLUX_4D(X, Y, Z, (g-1))
#define FLUXM_4D(MOM1, X, Y, Z)  FLUXM_5D(MOM1, X, Y, Z, (g-1))
#define JB_IN_3D(ANG, CHUNK, Z)  JB_IN_4D(ANG, CHUNK, Z, (g-1))
#define JB_OUT_3D(ANG, CHUNK, Z) JB_OUT_4D(ANG, CHUNK, Z, (g-1))
#define KB_IN_3D(ANG, CHUNK, Y)  KB_IN_4D(ANG, CHUNK, Y, (g-1))
#define KB_OUT_3D(ANG, CHUNK, Y) KB_OUT_4D(ANG, CHUNK, Y, (g-1))
#define FLKX_3D(X, Y, Z)         FLKX_4D(X, Y, Z, (g-1))
#define FLKY_3D(X, Y, Z)         FLKY_4D(X, Y, Z, (g-1))
#define FLKZ_3D(X, Y, Z)         FLKZ_4D(X, Y, Z, (g-1))
#define T_XS_3D(X, Y, Z)         T_XS_4D(X, Y, Z, (g-1))

#include <amp.h>
#include <hc.hpp>
#include <hsa_atomic.h>

extern "C" {
using namespace concurrency;

#define WG_BARRIER(SP, LID, GID, TILES, TID)      \
{                                                 \
if(LID==0)                                        \
{                                                 \
 while(atomic_fetch_max(SP,0)&(1<<TID));          \
 atomic_fetch_add(SP, 1<<TID);                    \
 while(atomic_fetch_max(SP,0) < (1<<TILES)-1);    \
}                                                 \
amp_barrier(CLK_GLOBAL_MEM_FENCE);                \
if(GID==0) __hsail_atomic_exchange_unsigned(SP, 0);  \
}
// set the work group size to 64

#define WGS 64

// The default is to allow the CPU to perform the flux reductions.
// The reductions are working correctly for 4 or fewer tiles.  To
// enable GPU reductions, uncomment the following line

//#define USE_AMP_REDUCTION 1

void dim3_sweep_data_init ( dim_sweep_data *dim_sweep_vars,
                            input_data *input_vars)
{
    FMIN = 0;
    FMAX = 0;
    int tiles = NANG/WGS;
    assert((tiles >= 1)&&(tiles*WGS==NANG));


    int size = NX*NY*NZ ;
// global storage for multi tile reductions, but now used to "return"
// psi data to the CPU
    dim_sweep_vars->partialp  =
         (double *) malloc (size*NANG*sizeof (double)) ;
#if defined(USE_AMP_REDUCTION)
    dim_sweep_vars->semaphorep  =
         (unsigned int *) malloc (size*sizeof (unsigned int)) ;
    for(int i = 0;i<size;i++) dim_sweep_vars->semaphorep[i]= 0;
#endif
}

void dim3_sweep_data_free ( dim_sweep_data *dim_sweep_vars)
{
    free(dim_sweep_vars->partialp);
#if defined(USE_AMP_REDUCTION)
    free(dim_sweep_vars->semaphorep);
#endif
}

/***********************************************************************
 *  3-D slab mesh sweeper.
 ***********************************************************************/
void dim3_sweep ( input_data *input_vars, para_data *para_vars,
                  geom_data *geom_vars, sn_data *sn_vars,
                  data_data *data_vars, control_data *control_vars,
                  solvar_data *solvar_vars, dim_sweep_data *dim_sweep_vars,
                  int ich, int id, int d1, int d2, int d3, int d4, int jd,
                  int kd, int jlo, int klo, int jhi, int khi, int jst, int kst,
                  int i1, int i2, int oct, int g, int *ierr )
{
/***********************************************************************
 * Local variables
 ***********************************************************************/
    int ist, d, n, ic, i, j, k, l, ibl, ibr, ibb, ibt, ibf, ibk;

    int z_ind, y_ind, ic_ind, ang, indx1 = 4;

    double sum_hv = 0, sum_hv_tmp = 0,
        sum_wmupsii = 0, sum_wetapsij = 0, sum_wxipsik = 0;

//    double psi[NANG], pc[NANG], den[NANG];
    double hv[NANG*4], fxhv[NANG*4];

/***********************************************************************
 * Set up the sweep order in the i-direction.
 ***********************************************************************/
    ist = -1;
    if ( id == 2 ) ist = 1;

/***********************************************************************
 * Zero out the outgoing boundary arrays and fixup array
 ***********************************************************************/
    for ( z_ind = 0; z_ind < NZ; z_ind++ )
    {
        for ( ic_ind = 0; ic_ind < ICHUNK; ic_ind++ )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                JB_OUT_3D(ang,ic_ind,z_ind) = 0;
            }
        }
    }

    for ( y_ind = 0; y_ind < NY; y_ind++ )
    {
        for ( ic_ind = 0; ic_ind < ICHUNK; ic_ind++ )
        {
            for ( ang = 0; ang < NANG; ang++ )
            {
                KB_OUT_3D(ang,ic_ind,y_ind) = 0;
            }
        }
    }

    for ( i = 0; i < 4; i++)
    {
        for ( ang = 0; ang < NANG; ang++ )
        {
            FXHV_2D(ang, i) = 0;
        }
    }

/***********************************************************************
 * Loop over cells along the diagonals. When only 1 diagonal, it's
 * normal sweep order. Otherwise, nested threading performs mini-KBA.
 ***********************************************************************/
/***********************************************************************
 * Commented out all nested OMP statements because not all compilers support
 * these put them back in if you want.
 ***********************************************************************/
    // diagonal loop
    for ( d = 1; d <= NDIAG; d++ )
    {

        //  line_loop
        for ( n = 1; n <= (DIAG_1D(d-1).lenc); n++ )
        {
            ic = DIAG_1D(d-1).cell_id_vars[n-1].ic;

            i = (ich-1)*ICHUNK + ic;
            if ( ist < 0 ) i = ich*ICHUNK - ic + 1;

// the original code had a if( i<=NX).  This was eliminated for the rest
// of the loops. For the AMP code, ICHUNK should always be set to NX.
// so the case should not be needed. 

            if ( i > NX )
{
printf("i > NX!\n");
exit(0);
}
            {
                j = DIAG_1D(d-1).cell_id_vars[n-1].jc;
                if ( jst < 0 ) j = NY - j + 1;

                k = DIAG_1D(d-1).cell_id_vars[n-1].kc;
                if ( kst < 0 ) k = NZ - k + 1;

/***********************************************************************
 * Left/right boundary conditions, always vacuum.
 ***********************************************************************/
                ibl = 0;
                ibr = 0;

                if ( (i == NX) && (ist == -1) )
                {
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSII_3D(ang,(j-1),(k-1)) = 0;
                    }
                }
                else if ( i == 1 && ist == 1 )
                {
                    switch ( ibl )
                    {
                    case 0:
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSII_3D(ang,(j-1),(k-1)) = 0;
                        }
                    case 1:
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSII_3D(ang,(j-1),(k-1)) = 0;
                        }
                    }
                }

/***********************************************************************
 * Top/bottom boundary condtions. Vacuum at global boundaries, but
 * set to some incoming flux from neighboring proc.
 ***********************************************************************/
                ibb = 0;
                ibt = 0;
                if ( j == jlo )
                {
                    if ( jd == 1 && LASTY )
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIJ_3D(ang,(ic-1),(k-1)) = 0;
                        }
                    }
                    else if ( jd == 2 && FIRSTY )
                    {
                        switch ( ibb )
                        {
                        case 0:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIJ_3D(ang,(ic-1),(k-1)) = 0;
                            }
                        case 1:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIJ_3D(ang,(ic-1),(k-1)) = 0;
                            }
                        }
                    }

                    else
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIJ_3D(ang,(ic-1),(k-1))
                                = JB_IN_3D(ang,(ic-1),(k-1));
                        }
                    }
                }

/***********************************************************************
 * Front/back boundary condtions. Vacuum at global boundaries, but
 * set to some incoming flux from neighboring proc.
 ***********************************************************************/
                ibf = 0;
                ibk = 0;
                if ( k == klo )
                {
                    if ( (kd == 1 && LASTZ) || NDIMEN < 3 )
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIK_3D(ang,(ic-1),(j-1)) = 0;
                        }
                    }
                    else if ( kd == 2 && FIRSTZ )
                    {
                        switch ( ibf )
                        {
                        case 0:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIK_3D(ang,(ic-1),(j-1)) = 0;
                            }
                        case 1:
                            for ( ang = 0; ang < NANG; ang++ )
                            {
                                PSIK_3D(ang,(ic-1),(j-1)) = 0;
                            }
                        }
                    }

                    else
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PSIK_3D(ang,(ic-1),(j-1))
                                = KB_IN_3D(ang,(ic-1),(j-1));
                        }
                    }
                }
/***********************************************************************
 * Clear the flux arrays
 ***********************************************************************/
                if ( oct == 1 )
                {
                    FLUX_4D((i-1),(j-1),(k-1),(g-1)) = 0;

                    for ( indx1 = 0; indx1 < (CMOM-1); indx1++ )
                    {
                        FLUXM_5D(indx1,(i-1),(j-1),(k-1),(g-1)) = 0;
                    }
                }
            }
        }
/***********************************************************************
 * Compute the angular source
 ***********************************************************************/
// implement the mini-KBA sweep with C++AMP
// we use a 2D tiled extent
// the first dimension is the number of cells, X*Y*Z.  We use only a single
// tile in this dimension, so local id is always 1, and global id is n from
// the original loop (0-based)
// The second dimension is number of angles.  Workgroups for GCN are up to 64
// work items, and we tile by that factor.  So NANG must be a multiple of 64.

// all of the above code is run on the CPU, since the if clauses are 
// only run by some of the cells, not a good candidate for GPU work.

// it was first proved with CPU code that we can break the "line loop"
// here and restart it, and breaking also after computing the fluxes.

extent<2> e(DIAG_1D(d-1).lenc,NANG);
parallel_for_each(e.tile<1,WGS>(),
[=]
(tiled_index<1,WGS> idx) restrict (amp)
{
        {

int tiles = NANG/WGS;
int n = idx.global[0];
int ang = idx.global[1];
double psi_amp,pc_amp;
// use tile static for the reductions
double tile_static rtemp[WGS];
            int ic = DIAG_1D(d-1).cell_id_vars[n].ic;
            int i = (ich-1)*ICHUNK + ic;
            if ( ist < 0 ) i = ich*ICHUNK - ic + 1;
            int j = DIAG_1D(d-1).cell_id_vars[n].jc;
            if ( jst < 0 ) j = NY - j + 1;
            int k = DIAG_1D(d-1).cell_id_vars[n].kc;
            if ( kst < 0 ) k = NZ - k + 1;
                    psi_amp = QTOT_4D(0,(i-1),(j-1),(k-1));

                    if ( SRC_OPT == 3 )
                    {
                        psi_amp +=
                            QIM_6D(ang,(i-1),(j-1),(k-1),(oct-1),(g-1));
                    }

                for ( int ll = 2; ll <=CMOM; ll++ )
                {
                        psi_amp +=
                            EC_2D(ang,(ll-1))
                            *QTOT_4D((ll-1),(i-1),(j-1),(k-1));
                }

/***********************************************************************
 * Compute the numerator for the update formula
 ***********************************************************************/
                    pc_amp = psi_amp
                        + PSII_3D(ang,(j-1),(k-1)) *MU_1D(ang)*HI
                        + PSIJ_3D(ang,(ic-1),(k-1))*HJ_1D(ang)
                        + PSIK_3D(ang,(ic-1),(j-1))*HK_1D(ang);

                    if ( VDELT_CONST != 0 )
                    {
                        pc_amp += VDELT_CONST
                            *PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                    }
/***********************************************************************
 * Compute the solution of the center. Use DD for edges. Use fixup
 * if requested.
 ***********************************************************************/
                if ( FIXUP == 0 )
                {
                        psi_amp 
                            = pc_amp*DINV_4D(ang,(i-1),(j-1),(k-1));
                        PSII_3D(ang,(j-1),(k-1))
                            = 2*psi_amp - PSII_3D(ang,(j-1),(k-1));

                        PSIJ_3D(ang,(ic-1),(k-1))
                            = 2*psi_amp - PSIJ_3D(ang,(ic-1),(k-1));

                        if ( NDIMEN == 3 )
                        {
                            PSIK_3D(ang,(ic-1),(j-1))
                                = 2*psi_amp - PSIK_3D(ang,(ic-1),(j-1));
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            PTR_OUT_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                               = 2*psi_amp -
                                PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                        }
                }
// the fixup code has not been implemented in C++AMP.
#if 0

                else
                {

/***********************************************************************
 * Multi-pass set to zero + rebalance fixup. Determine angles
 * that will need fixup first.
 ***********************************************************************/
                    sum_hv = 0;
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        for ( indx1 = 0; indx1 < 4; indx1++ )
                        {
                            HV_2D(ang, indx1) = 1;
                            sum_hv += HV_2D(ang,indx1);
                        }

                        PC_1D(ang) = PC_1D(ang)
                            * DINV_4D(ang,(i-1),(j-1),(k-1));
                    }
                    // fixup_loop
                    while (true)
                    {
                        sum_hv_tmp = 0;
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            FXHV_2D(ang,0) =  2*PC_1D(ang)
                                - PSII_3D(ang,(j-1),(k-1));

                            FXHV_2D(ang,1) =  2*PC_1D(ang)
                                - PSIJ_3D(ang,(ic-1),(k-1));

                            if ( NDIMEN == 3 )
                            {
                                FXHV_2D(ang,2) = 2*PC_1D(ang)
                                    - PSIK_3D(ang,(ic-1),(j-1));
                            }

                            if ( VDELT_CONST != 0 )
                            {
                                FXHV_2D(ang,3) = 2*PC_1D(ang)
                                    - PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1));
                            }

                            for ( indx1 = 0; indx1 < 4; indx1++ )
                            {
                                if ( FXHV_2D(ang,indx1) < 0 )
                                {
                                    HV_2D(ang,indx1) = 0;
                                }
                                sum_hv_tmp += HV_2D(ang,indx1);
                            }
                        }

/***********************************************************************
 * Exit loop when all angles are fixed up
 ***********************************************************************/
                        if ( sum_hv == sum_hv_tmp ) break;

                        sum_hv = sum_hv_tmp;

/***********************************************************************
 * Recompute balance equation numerator and denominator and get
 * new cell average flux
 ***********************************************************************/
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            PC_1D(ang) = PSII_3D(ang,(j-1),(k-1))
                                * MU_1D(ang) * HI * (1+HV_2D(ang,0))
                                + PSIJ_3D(ang,(ic-1),(k-1))
                                * HJ_1D(ang) * (1+HV_2D(ang,1))
                                + PSIK_3D(ang,(ic-1),(j-1))
                                * HK_1D(ang) * (1+HV_2D(ang,2));

                            if ( VDELT_CONST != 0 )
                            {
                                PC_1D(ang) += VDELT_CONST
                                    * PTR_IN_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                    * (1+HV_2D(ang,3));
                            }

                            PC_1D(ang) = PSI_1D(ang) + 0.5*PC_1D(ang);

                            DEN_1D(ang) = T_XS_3D((i-1),(j-1),(k-1))
                                + MU_1D(ang)  * HI * HV_2D(ang,0)
                                + HJ_1D(ang)  * HV_2D(ang,1)
                                + HK_1D(ang)  * HV_2D(ang,2)
                                + VDELT_CONST * HV_2D(ang,3);

                            if ( DEN_1D(ang) > TOLR )
                            {
                                PC_1D(ang) /= DEN_1D(ang);
                            }
                            else
                            {
                                PC_1D(ang) = 0;
                            }
                        }

                    } // end fixup_loop

/***********************************************************************
 * Fixup done, compute edges
 ***********************************************************************/
                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        PSI_1D(ang) = PC_1D(ang);

                        PSII_3D(ang,(j-1),(k-1))
                            = FXHV_2D(ang,0) * HV_2D(ang,0);

                        PSIJ_3D(ang,(ic-1),(k-1))
                            = FXHV_2D(ang,1) * HV_2D(ang,1);

                        if ( NDIMEN == 3 )
                        {
                            PSIK_3D(ang,(ic-1),(j-1))
                                = FXHV_2D(ang,2) * HV_2D(ang,2);
                        }

                        if ( VDELT_CONST != 0 )
                        {
                            PTR_OUT_6D(ang,(i-1),(j-1),(k-1),(i1-1),(i2-1))
                                = FXHV_2D(ang,3) * HV_2D(ang,3);
                        }
                    }
                }
#endif


/***********************************************************************
 * Compute the flux moments
 ***********************************************************************/

int indx= (i-1) + (j-1)*NX + (k-1)*NX*NY;
double *partial = &dim_sweep_vars->partialp[indx*NANG];
unsigned int *semaphore = &dim_sweep_vars->semaphorep[indx];
int gid = idx.global[1];

// Implement the flux and flux moment reductions in the AMP kernel.
// This has been shown to work correctly. 
// But there is a risk of a GPU lockup as shown with
// problem sizes of NANG = 512.  Also it is too slow.
// The enabled alternative is to save the PSI_ID(ang) array, and then compute
// the reductions on the CPU.

#if defined(USE_AMP_REDUCTION)
int lid = idx.local[1];
int tid = idx.tile[1];

   rtemp[lid] = W_1D(ang)*psi_amp;
   int limit = WGS>>1;
   amp_barrier(CLK_LOCAL_MEM_FENCE);
// reduce for each work group
   while(limit>=1)
   {
      if(lid<limit) rtemp[lid] += rtemp[lid+limit];
      limit >>=1;
   }
   amp_barrier(CLK_LOCAL_MEM_FENCE);
// move the result for each work group back to global
   if(tiles >1)
      if(lid==0) partial[tid] = rtemp[0];
// final reduction 
   if(gid == 0)
   {
      double sum_wpsi = rtemp[0];
      if(tiles >1)
         for (int ii=1; ii<tiles;ii++) sum_wpsi+=partial[ii];

      FLUX_4D((i-1),(j-1),(k-1),(g-1)) += sum_wpsi;
   }
   amp_barrier(CLK_GLOBAL_MEM_FENCE);

   for ( int ll = 1; ll <= (CMOM-1); ll++ )
   {

      rtemp[lid] =  EC_2D(ang,(ll))*W_1D(ang)*psi_amp;
      amp_barrier(CLK_LOCAL_MEM_FENCE);

#if 1
      int limit = WGS>>1;

// reduce for each work group
      while(limit>=1)
      {
         if(lid<limit) rtemp[lid] += rtemp[lid+limit];
         limit >>=1;
         amp_barrier(CLK_LOCAL_MEM_FENCE);
      }
// move the result for each work group back to global
      if(tiles >1)
         if(lid==0) partial[tid] = rtemp[0];
      WG_BARRIER(semaphore, lid, gid, tiles, tid);
      amp_barrier(CLK_GLOBAL_MEM_FENCE);
// final reduction
      if(gid == 0)
      {
         double sum_ecwpsi = rtemp[0];
         if(tiles >1)
            for (int ii=1; ii<tiles;ii++) sum_ecwpsi+=partial[ii];

         FLUXM_5D((ll-1),(i-1),(j-1),(k-1),(g-1)) += sum_ecwpsi;
      }
   amp_barrier(CLK_GLOBAL_MEM_FENCE);
#endif
   }
#else
// Save the local psi to global memory and do the reductions 
// sequentially on the CPU.
// We just use the array meant for multi tile reductions to return the data

// This is much faster than doing the reduction in AMP.

   partial[gid] = psi_amp;

#endif
            }
        }
);

/***********************************************************************
 * Restart the diagonal  loop
 ***********************************************************************/
        for ( n = 1; n <= (DIAG_1D(d-1).lenc); n++ )
        {
            ic = DIAG_1D(d-1).cell_id_vars[n-1].ic;
            i = (ich-1)*ICHUNK + ic;
            if ( ist < 0 ) i = ich*ICHUNK - ic + 1;
            {
                j = DIAG_1D(d-1).cell_id_vars[n-1].jc;
                if ( jst < 0 ) j = NY - j + 1;
                k = DIAG_1D(d-1).cell_id_vars[n-1].kc;
                if ( kst < 0 ) k = NZ - k + 1;


#if !defined(USE_AMP_REDUCTION)
//use the saved psi data to compute the reductions on the CPU
                int indx= (i-1) + (j-1)*NX + (k-1)*NX*NY;
                double *partial = &dim_sweep_vars->partialp[indx*(NANG)];
                double sum_wpsi = 0;
                for ( ang = 0; ang < NANG; ang++ )
                {
                    sum_wpsi += W_1D(ang)*partial[ang];
                }
                FLUX_4D((i-1),(j-1),(k-1),(g-1)) += sum_wpsi;

                double sum_ecwpsi;
                for ( int ll = 1; ll <= (CMOM-1); ll++ )
                {
                   sum_ecwpsi = 0;
                   for ( ang = 0; ang < NANG; ang++ )
                   {
                      sum_ecwpsi +=  EC_2D(ang,(ll))*W_1D(ang)*partial[ang];
                   }
                   FLUXM_5D((ll-1),(i-1),(j-1),(k-1),(g-1)) += sum_ecwpsi;
                }
#endif
// we keep the rest of the code on the CPU in it's original form since it
// is largely boundary cases and reductions.

/***********************************************************************
 * Calculate min and max scalar fluxes (not used elsewhere
 * currently)
 ***********************************************************************/
                if ( oct == NOCT )
                {
                    FMIN = MIN( FMIN, FLUX_3D((i-1),(j-1),(k-1)) );
                    FMAX = MAX( FMAX, FLUX_3D((i-1),(j-1),(k-1)) );
                }

/***********************************************************************
 * Save edge fluxes (dummy if checks for unused non-vacuum BCs)
 ***********************************************************************/
                if ( j == jhi )
                {
                    if ( jd==2 && LASTY )
                    {
                        // CONTINUE
                    }
                    else if ( jd == 1 && FIRSTY )
                    {
                        if ( ibb == 1 )
                        {
                            // CONTINUE
                        }
                    }
                    else
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            JB_OUT_3D(ang,(ic-1),(k-1))
                                = PSIJ_3D(ang,(ic-1),(k-1));
                        }
                    }
                }

                if ( k == khi )
                {
                    if ( kd == 2 && LASTZ )
                    {
                        // CONTINUE
                    }
                    else if ( kd==1 && FIRSTZ )
                    {
                        if ( ibf == 1 )
                        {
                            // CONTINUE
                        }
                    }
                    else
                    {
                        for ( ang = 0; ang < NANG; ang++ )
                        {
                            KB_OUT_3D(ang,(ic-1),(j-1))
                                = PSIK_3D(ang,(ic-1),(j-1));
                        }
                    }
                }

/***********************************************************************
 * Compute leakages (not used elsewhere currently)
 ***********************************************************************/
                if ( ((i+id-1) == 1) || ((i+id-1) == (NX+1)) )
                {
                    sum_wmupsii = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_wmupsii
                            += WMU_1D(ang) * PSII_3D(ang,(j-1),(k-1));
                    }

                    FLKX_3D((i+id-1-1),(j-1),(k-1))
                        += ist*sum_wmupsii;
                }

                if ( (jd==1 && FIRSTY) || (jd==2 && LASTY) )
                {
                    sum_wetapsij = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_wetapsij
                            += WETA_1D(ang) * PSIJ_3D(ang,(ic-1),(k-1));
                    }

                    FLKY_3D((i-1),(j+jd-1-1),(k-1))
                        += jst*sum_wetapsij;
                }

                if ( ((kd == 1 && FIRSTZ) || (kd == 2 && LASTZ)) && NDIMEN == 3 )
                {
                    sum_wxipsik = 0;

                    for ( ang = 0; ang < NANG; ang++ )
                    {
                        sum_wxipsik
                            += WXI_1D(ang) * PSIK_3D(ang,(ic-1),(j-1));
                    }

                    FLKZ_3D((i-1),(j-1),(k+kd-1-1))
                        += kst*sum_wxipsik;
                }
            }

/***********************************************************************
 * Finish the loops
 ***********************************************************************/
         } // end line_loop
   } // end diagonal_loop
}
}  //extern "C"
