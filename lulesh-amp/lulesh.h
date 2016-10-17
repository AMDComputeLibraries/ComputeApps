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

#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <hc.hpp>
using namespace hc;

#ifndef __LULESH_H__
#define __LULESH_H__

//#include "CLsetup.hpp"

#ifdef ARRAY_VIEW
#define HCC_ARRAY_STRUC(type, name, size, ptr) array_view<type> name(size, ptr)
#define HCC_ARRAY_OBJECT(type, name) array_view<type> &name
#define HCC_ID(name)
#define HCC_SYNC(name, ptr) name.synchronize()
#else
#define HCC_ARRAY_STRUC(type, name, size, ptr) array<type> name(size); copy(ptr, name)
#define HCC_ARRAY_OBJECT(type, name) array<type> &name
#define HCC_ID(name) ,&name
#define HCC_SYNC(name, ptr) copy(name, ptr)
#endif

//**************************************************
// Allow flexibility for arithmetic representations 
//**************************************************

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))


// Precision specification
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  // 10 bytes on x86

typedef int    Index_t ; // array subscript and loop index
#ifdef SINGLE
typedef real4 Real_t ;  // floating point representation
#else
typedef real8  Real_t ;  // floating point representation
#endif
typedef int    Int_t ;   // integer representation

enum { VolumeError = -1, QStopError = -2 } ;

inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline real4  FABS(real4  arg) { return fabsf(arg) ; }
inline real8  FABS(real8  arg) { return fabs(arg) ; }
inline real10 FABS(real10 arg) { return fabsl(arg) ; }


// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

/*
 * The implementation of the data abstraction used for lulesh
 * resides entirely in the Domain class below.  You can change
 * grouping and interleaving of fields here to maximize data layout
 * efficiency for your underlying architecture or compiler.
 *
 * For example, fields can be implemented as STL objects or
 * raw array pointers.  As another example, individual fields
 * m_x, m_y, m_z could be budled into
 *
 *    struct { Real_t x, y, z ; } *m_coord ;
 *
 * allowing accessor functions such as
 *
 *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
 *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
 *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
 */

class Domain {

   public:

   // Constructor
   Domain(Int_t numRanks, Index_t colLoc,
          Index_t rowLoc, Index_t planeLoc,
          Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

   //
   // ALLOCATION
   //

   void AllocateNodePersistent(Int_t numNode) // Node-centered
   {
      m_x.resize(numNode);  // coordinates
      m_y.resize(numNode);
      m_z.resize(numNode);

      m_xd.resize(numNode); // velocities
      m_yd.resize(numNode);
      m_zd.resize(numNode);

      m_xdd.resize(numNode); // accelerations
      m_ydd.resize(numNode);
      m_zdd.resize(numNode);

      m_fx.resize(numNode);  // forces
      m_fy.resize(numNode);
      m_fz.resize(numNode);

      m_nodalMass.resize(numNode);  // mass
   }

   void AllocateElemPersistent(Int_t numElem) // Elem-centered
   {
      m_matElemlist.resize(numElem) ;
      m_nodelist.resize(8*numElem);

      // elem connectivities through face
      m_lxim.resize(numElem);
      m_lxip.resize(numElem);
      m_letam.resize(numElem);
      m_letap.resize(numElem);
      m_lzetam.resize(numElem);
      m_lzetap.resize(numElem);

      m_elemBC.resize(numElem);

      m_e.resize(numElem);
      m_p.resize(numElem);

      m_q.resize(numElem);
      m_ql.resize(numElem);
      m_qq.resize(numElem);

      m_v.resize(numElem);

      m_volo.resize(numElem);
      m_delv.resize(numElem);
      m_vdov.resize(numElem);

      m_arealg.resize(numElem);

      m_ss.resize(numElem);

      m_elemMass.resize(numElem);
   }

   void AllocateGradients(Int_t numElem, Int_t allElem)
   {
      // Position gradients
      m_delx_xi.resize(numElem) ;
      m_delx_eta.resize(numElem) ;
      m_delx_zeta.resize(numElem) ;

      // Velocity gradients
      m_delv_xi.resize(allElem) ;
      m_delv_eta.resize(allElem);
      m_delv_zeta.resize(allElem) ;
   }

   /* Temporaries should not be initialized in bulk but */
   /* this is a runnable placeholder for now */
   void AllocateElemTemporary(size_t size)
   {
      m_dxx.resize(size) ;
      m_dyy.resize(size) ;
      m_dzz.resize(size) ;

      m_delv_xi.resize(size) ;
      m_delv_eta.resize(size) ;
      m_delv_zeta.resize(size) ;

      m_delx_xi.resize(size) ;
      m_delx_eta.resize(size) ;
      m_delx_zeta.resize(size) ;

      m_vnew.resize(size) ;
   }

   /* Temporaries moved from subroutines to avoid allocation overhead */
   void AllocateRoutinePersistent(Int_t numElem, Int_t numNode) 
   {
      dev_mindthydro.resize(numElem/BLOCKSIZE) ;
      dev_mindtcourant.resize(numElem/BLOCKSIZE) ;
      p_vnewc.resize(numElem) ;
      p_e_old.resize(numElem) ;
      p_delvc.resize(numElem) ;
      p_p_old.resize(numElem) ;
      p_q_old.resize(numElem) ;
      p_compression.resize(numElem) ;
      p_compHalfStep.resize(numElem) ;
      p_qq_old.resize(numElem) ;
      p_ql_old.resize(numElem) ;
      p_work.resize(numElem) ;
      p_p_new.resize(numElem) ;
      p_e_new.resize(numElem) ;
      p_q_new.resize(numElem) ;
      p_bvc.resize(numElem) ;
      p_pbvc.resize(numElem) ;
      p_dvdx.resize(numElem*8) ;
      p_dvdy.resize(numElem*8) ;
      p_dvdz.resize(numElem*8) ;
      p_x8n.resize(numElem*8) ;
      p_y8n.resize(numElem*8) ;
      p_z8n.resize(numElem*8) ;
      p_sigxx.resize(numElem) ;
      p_sigyy.resize(numElem) ;
      p_sigzz.resize(numElem) ;
      p_determ.resize(numElem) ;
      p_fx_elem.resize(numElem*8) ;
      p_fy_elem.resize(numElem*8) ;
      p_fz_elem.resize(numElem*8) ;
      ppHalfStep.resize(numElem) ;
   }

   void DeallocateGradients()
   {
      m_delx_zeta.clear() ;
      m_delx_eta.clear() ;
      m_delx_xi.clear() ;

      m_delv_zeta.clear() ;
      m_delv_eta.clear() ;
      m_delv_xi.clear() ;
   }

   void AllocateStrains(Int_t numElem)
   {
      m_dxx.resize(numElem) ;
      m_dyy.resize(numElem) ;
      m_dzz.resize(numElem) ;
   }

   void DeallocateStrains()
   {
      m_dzz.clear() ;
      m_dyy.clear() ;
      m_dxx.clear() ;
   }

   void AllocateNodeElemIndexes()
   {
        Index_t i,j,nidx;
        /* set up node-centered indexing of elements */
        m_nodeElemCount.resize(m_numNode);
        for (i=0;i<m_numNode;i++) m_nodeElemCount[i]=0;

        m_nodeElemCornerList.resize(m_numNode*8);
        for (i=0;i<m_numElem;i++) {
            for (j=0;j<8;j++) {
                nidx=nodelist(i,j);
                m_nodeElemCornerList[nidx+m_numNode*m_nodeElemCount[nidx]++] = i+m_numElem*j;
                if (m_nodeElemCount[nidx]>8) {
                    std::cerr << "Node degree is higher than 8!\n"; 
                    exit(1);
                }
            }
        }
   }
   
   //
   // ACCESSORS
   //

   // Routine temporaries
   Real_t& mindthydro(Index_t idx)    { return dev_mindthydro[idx] ; }
   Real_t& mindtcourant(Index_t idx)  { return dev_mindtcourant[idx] ; }
   Real_t& vnewc(Index_t idx)         { return p_vnewc[idx] ; }
   Real_t& e_old(Index_t idx)         { return p_e_old[idx] ; }
   Real_t& delvc(Index_t idx)         { return p_delvc[idx] ; }
   Real_t& p_old(Index_t idx)         { return p_p_old[idx] ; }
   Real_t& q_old(Index_t idx)         { return p_q_old[idx] ; }
   Real_t& compression(Index_t idx)   { return p_compression[idx] ; }
   Real_t& compHalfStep(Index_t idx)  { return p_compHalfStep[idx] ; }
   Real_t& qq_old(Index_t idx)        { return p_qq_old[idx] ; }
   Real_t& ql_old(Index_t idx)        { return p_ql_old[idx] ; }
   Real_t& work(Index_t idx)          { return p_work[idx] ; }
   Real_t& p_new(Index_t idx)         { return p_p_new[idx] ; }
   Real_t& q_new(Index_t idx)         { return p_q_new[idx] ; }
   Real_t& e_new(Index_t idx)         { return p_e_new[idx] ; }
   Real_t& bvc(Index_t idx)           { return p_bvc[idx] ; }
   Real_t& pbvc(Index_t idx)          { return p_pbvc[idx] ; }
   Real_t& pHalfStep(Index_t idx)     { return ppHalfStep[idx] ; }
   Real_t& dvdx(Index_t idx)           { return p_dvdx[idx] ; }
   Real_t& dvdy(Index_t idx)           { return p_dvdy[idx] ; }
   Real_t& dvdz(Index_t idx)           { return p_dvdz[idx] ; }
   Real_t& x8n(Index_t idx)           { return p_x8n[idx] ; }
   Real_t& y8n(Index_t idx)           { return p_y8n[idx] ; }
   Real_t& z8n(Index_t idx)           { return p_z8n[idx] ; }
   Real_t& sigxx(Index_t idx)         { return p_sigxx[idx] ; }
   Real_t& sigyy(Index_t idx)         { return p_sigyy[idx] ; }
   Real_t& sigzz(Index_t idx)         { return p_sigzz[idx] ; }
   Real_t& determ(Index_t idx)        { return p_determ[idx] ; }
   Real_t& fx_elem(Index_t idx)        { return p_fx_elem[idx] ; }
   Real_t& fy_elem(Index_t idx)        { return p_fy_elem[idx] ; }
   Real_t& fz_elem(Index_t idx)        { return p_fz_elem[idx] ; }

   // Node-centered

   // Nodal coordinates
   Real_t& x(Index_t idx)    { return m_x[idx] ; }
   Real_t& y(Index_t idx)    { return m_y[idx] ; }
   Real_t& z(Index_t idx)    { return m_z[idx] ; }

   // Nodal velocities
   Real_t& xd(Index_t idx)   { return m_xd[idx] ; }
   Real_t& yd(Index_t idx)   { return m_yd[idx] ; }
   Real_t& zd(Index_t idx)   { return m_zd[idx] ; }

   // Nodal accelerations
   Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }
   Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }
   Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }

   // Nodal forces
   Real_t& fx(Index_t idx)   { return m_fx[idx] ; }
   Real_t& fy(Index_t idx)   { return m_fy[idx] ; }
   Real_t& fz(Index_t idx)   { return m_fz[idx] ; }

   // Nodal mass
   Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }

   // Nodes on symmertry planes
   Index_t symmX(Index_t idx) { return m_symmX[idx] ; }
   Index_t symmY(Index_t idx) { return m_symmY[idx] ; }
   Index_t symmZ(Index_t idx) { return m_symmZ[idx] ; }
   bool symmXempty()          { return m_symmX.empty(); }
   bool symmYempty()          { return m_symmY.empty(); }
   bool symmZempty()          { return m_symmZ.empty(); }

   //
   // Element-centered
   //
   Index_t&  regStartPosition(Index_t idx) { return m_regStartPosition[idx] ; }

   Index_t&  regElemSize(Index_t idx) { return m_regElemSize[idx] ; }
   Index_t&  regNumList(Index_t idx) { return m_regNumList[idx] ; }
   Index_t*  regNumList()            { return &m_regNumList[0] ; }
   Index_t*  regElemlist(Int_t r)    { return m_regElemlist[r] ; }
   Index_t&  regElemlist(Int_t r, Index_t idx) { return m_regElemlist[r][idx] ; }

   Index_t&  matElemlist(Index_t idx) { return m_matElemlist[idx] ; }
   Index_t*  nodelist(Index_t idx)    { return &m_nodelist[Index_t(8)*idx] ; }
   Index_t&  nodelist(Index_t idx,Index_t nidx)    { return m_nodelist[idx+nidx*m_numElem] ; }

   // elem connectivities through face
   Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }
   Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }
   Index_t&  letam(Index_t idx) { return m_letam[idx] ; }
   Index_t&  letap(Index_t idx) { return m_letap[idx] ; }
   Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }
   Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }

   // elem face symm/free-surface flag
   Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }

   // Principal strains - temporary
   Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }
   Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }
   Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }

   // Velocity gradient - temporary
   Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }
   Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }
   Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }

   // Position gradient - temporary
   Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }
   Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }
   Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }

   // Energy
   Real_t& e(Index_t idx)          { return m_e[idx] ; }

   // Pressure
   Real_t& p(Index_t idx)          { return m_p[idx] ; }

   // Artificial viscosity
   Real_t& q(Index_t idx)          { return m_q[idx] ; }

   // Linear term for q
   Real_t& ql(Index_t idx)         { return m_ql[idx] ; }
   // Quadratic term for q
   Real_t& qq(Index_t idx)         { return m_qq[idx] ; }

   // Relative volume
   Real_t& v(Index_t idx)          { return m_v[idx] ; }
   Real_t& delv(Index_t idx)       { return m_delv[idx] ; }

   // Reference volume
   Real_t& volo(Index_t idx)       { return m_volo[idx] ; }

   // volume derivative over volume
   Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }

   // Element characteristic length
   Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }

   // Sound speed
   Real_t& ss(Index_t idx)         { return m_ss[idx] ; }

   // Element mass
   Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }

   Index_t nodeElemCount(Index_t idx)
   { return m_nodeElemStart[idx+1] - m_nodeElemStart[idx] ; }

   Index_t *nodeElemCornerList(Index_t idx)
   { return &m_nodeElemCornerList[m_nodeElemStart[idx]] ; }

   // Parameters 

   // Cutoffs
   Real_t u_cut() const               { return m_u_cut ; }
   Real_t e_cut() const               { return m_e_cut ; }
   Real_t p_cut() const               { return m_p_cut ; }
   Real_t q_cut() const               { return m_q_cut ; }
   Real_t v_cut() const               { return m_v_cut ; }

   // Other constants (usually are settable via input file in real codes)
   Real_t hgcoef() const              { return m_hgcoef ; }
   Real_t qstop() const               { return m_qstop ; }
   Real_t monoq_max_slope() const     { return m_monoq_max_slope ; }
   Real_t monoq_limiter_mult() const  { return m_monoq_limiter_mult ; }
   Real_t ss4o3() const               { return m_ss4o3 ; }
   Real_t qlc_monoq() const           { return m_qlc_monoq ; }
   Real_t qqc_monoq() const           { return m_qqc_monoq ; }
   Real_t qqc() const                 { return m_qqc ; }

   Real_t eosvmax() const             { return m_eosvmax ; }
   Real_t eosvmin() const             { return m_eosvmin ; }
   Real_t pmin() const                { return m_pmin ; }
   Real_t emin() const                { return m_emin ; }
   Real_t dvovmax() const             { return m_dvovmax ; }
   Real_t refdens() const             { return m_refdens ; }

   // Timestep controls, etc...
   Real_t& time()                 { return m_time ; }
   Real_t& deltatime()            { return m_deltatime ; }
   Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
   Real_t& deltatimemultub()      { return m_deltatimemultub ; }
   Real_t& stoptime()             { return m_stoptime ; }
   Real_t& dtcourant()            { return m_dtcourant ; }
   Real_t& dthydro()              { return m_dthydro ; }
   Real_t& dtmax()                { return m_dtmax ; }
   Real_t& dtfixed()              { return m_dtfixed ; }

   Int_t&  cycle()                { return m_cycle ; }
   Index_t&  numRanks()           { return m_numRanks ; }

   Index_t&  colLoc()             { return m_colLoc ; }
   Index_t&  rowLoc()             { return m_rowLoc ; }
   Index_t&  planeLoc()           { return m_planeLoc ; }
   Index_t&  tp()                 { return m_tp ; }

   Index_t&  sizeX()              { return m_sizeX ; }
   Index_t&  sizeY()              { return m_sizeY ; }
   Index_t&  sizeZ()              { return m_sizeZ ; }
   Index_t&  numReg()             { return m_numReg ; }
   Int_t&  cost()             { return m_cost ; }
   Index_t&  numElem()            { return m_numElem ; }
   Index_t&  numNode()            { return m_numNode ; }
   
   Index_t&  maxPlaneSize()       { return m_maxPlaneSize ; }
   Index_t&  maxEdgeSize()        { return m_maxEdgeSize ; }
   
   //
   // MPI-Related additional data
   //

#if USE_MPI   
   // Communication Work space 
   Real_t *commDataSend ;
   Real_t *commDataRecv ;
   
   // Maximum number of block neighbors 
   MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
   MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

  //private:

   void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
   void SetupThreadSupportStructures();
   void CreateRegionIndexSets(Int_t nreg, Int_t balance);
   void SetupCommBuffers(Int_t edgeNodes);
   void SetupSymmetryPlanes(Int_t edgeNodes);
   void SetupElementConnectivities(Int_t edgeElems);
   void SetupBoundaryConditions(Int_t edgeElems);

   //
   // IMPLEMENTATION
   //

   /* Node-centered */
   std::vector<Real_t> m_x ;  /* coordinates */
   std::vector<Real_t> m_y ;
   std::vector<Real_t> m_z ;

   std::vector<Real_t> m_xd ; /* velocities */
   std::vector<Real_t> m_yd ;
   std::vector<Real_t> m_zd ;

   std::vector<Real_t> m_xdd ; /* accelerations */
   std::vector<Real_t> m_ydd ;
   std::vector<Real_t> m_zdd ;

   std::vector<Real_t> m_fx ;  /* forces */
   std::vector<Real_t> m_fy ;
   std::vector<Real_t> m_fz ;

   std::vector<Real_t> m_nodalMass ;  /* mass */

   std::vector<Index_t> m_symmX ;  /* symmetry plane nodesets */
   std::vector<Index_t> m_symmY ;
   std::vector<Index_t> m_symmZ ;

   /* for GPU */
   std::vector<Int_t> m_nodeElemCount ;
   std::vector<Index_t> m_nodeElemCornerList ;
    
   // Element-centered

   // Region information
   Int_t    m_numReg ;
   Int_t    m_cost; //imbalance cost
   Index_t *m_regStartPosition; //Regions start positions
   Index_t *m_regElemSize ;   // Size of region sets
   Index_t *m_regNumList ;    // Region number per domain element
   Index_t **m_regElemlist ;  // region indexset 

   std::vector<Index_t>  m_nodelist ;     /* elemToNode connectivity */
   std::vector<Index_t>  m_matElemlist ;     /* elemToNode connectivity */

   std::vector<Index_t>  m_lxim ;  /* element connectivity across each face */
   std::vector<Index_t>  m_lxip ;
   std::vector<Index_t>  m_letam ;
   std::vector<Index_t>  m_letap ;
   std::vector<Index_t>  m_lzetam ;
   std::vector<Index_t>  m_lzetap ;

   std::vector<Int_t>    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

   std::vector<Real_t> m_dxx ;  /* principal strains -- temporary */
   std::vector<Real_t> m_dyy ;
   std::vector<Real_t> m_dzz ;

   std::vector<Real_t> m_delv_xi ;    /* velocity gradient -- temporary */
   std::vector<Real_t> m_delv_eta ;
   std::vector<Real_t> m_delv_zeta ;

   std::vector<Real_t> m_delx_xi ;    /* coordinate gradient -- temporary */
   std::vector<Real_t> m_delx_eta ;
   std::vector<Real_t> m_delx_zeta ;
   
   std::vector<Real_t> m_e ;   /* energy */

   std::vector<Real_t> m_p ;   /* pressure */
   std::vector<Real_t> m_q ;   /* q */
   std::vector<Real_t> m_ql ;  /* linear term for q */
   std::vector<Real_t> m_qq ;  /* quadratic term for q */

   std::vector<Real_t> m_v ;     /* relative volume */
   std::vector<Real_t> m_volo ;  /* reference volume */
   std::vector<Real_t> m_vnew ;  /* new relative volume -- temporary */
   std::vector<Real_t> m_delv ;  /* m_vnew - m_v */
   std::vector<Real_t> m_vdov ;  /* volume derivative over volume */

   std::vector<Real_t> m_arealg ;  /* characteristic length of an element */
   
   std::vector<Real_t> m_ss ;      /* "sound speed" */

   std::vector<Real_t> m_elemMass ;  /* mass */

   // work arrays from all routines moved here to eliminate per routine
   // std::vector and array|array_view allocations
   std::vector<Real_t> dev_mindthydro;
   std::vector<Real_t> dev_mindtcourant;
   std::vector<Real_t> p_vnewc;
   std::vector<Real_t> p_e_old;
   std::vector<Real_t> p_delvc;
   std::vector<Real_t> p_q_old;
   std::vector<Real_t> p_p_old;
   std::vector<Real_t> p_compression;
   std::vector<Real_t> p_compHalfStep;
   std::vector<Real_t> p_qq_old;
   std::vector<Real_t> p_ql_old;
   std::vector<Real_t> p_work;
   std::vector<Real_t> p_p_new;
   std::vector<Real_t> p_e_new;
   std::vector<Real_t> p_q_new;
   std::vector<Real_t> p_bvc;
   std::vector<Real_t> p_pbvc;
   std::vector<Real_t> ppHalfStep;
   std::vector<Real_t> p_dvdx;
   std::vector<Real_t> p_dvdy;
   std::vector<Real_t> p_dvdz;
   std::vector<Real_t> p_x8n;
   std::vector<Real_t> p_y8n;
   std::vector<Real_t> p_z8n;
   std::vector<Real_t> p_sigxx;
   std::vector<Real_t> p_sigyy;
   std::vector<Real_t> p_sigzz;
   std::vector<Real_t> p_determ;
   std::vector<Real_t> p_fx_elem;
   std::vector<Real_t> p_fy_elem;
   std::vector<Real_t> p_fz_elem;

   // Cutoffs (treat as constants)
   const Real_t  m_e_cut ;             // energy tolerance 
   const Real_t  m_p_cut ;             // pressure tolerance 
   const Real_t  m_q_cut ;             // q tolerance 
   const Real_t  m_v_cut ;             // relative volume tolerance 
   const Real_t  m_u_cut ;             // velocity tolerance 

   // Other constants (usually setable, but hardcoded in this proxy app)

   const Real_t  m_hgcoef ;            // hourglass control 
   const Real_t  m_ss4o3 ;
   const Real_t  m_qstop ;             // excessive q indicator 
   const Real_t  m_monoq_max_slope ;
   const Real_t  m_monoq_limiter_mult ;
   const Real_t  m_qlc_monoq ;         // linear term coef for q 
   const Real_t  m_qqc_monoq ;         // quadratic term coef for q 
   const Real_t  m_qqc ;
   const Real_t  m_eosvmax ;
   const Real_t  m_eosvmin ;
   const Real_t  m_pmin ;              // pressure floor 
   const Real_t  m_emin ;              // energy floor 
   const Real_t  m_dvovmax ;           // maximum allowable volume change 
   const Real_t  m_refdens ;           // reference density 

   // Variables to keep track of timestep, simulation time, and cycle
   Real_t  m_dtcourant ;         // courant constraint 
   Real_t  m_dthydro ;           // volume change constraint 
   Int_t   m_cycle ;             // iteration count for simulation 
   Real_t  m_dtfixed ;           // fixed time increment 
   Real_t  m_time ;              // current time 
   Real_t  m_deltatime ;         // variable time increment 
   Real_t  m_deltatimemultlb ;
   Real_t  m_deltatimemultub ;
   Real_t  m_dtmax ;             // maximum allowable time increment 
   Real_t  m_stoptime ;          // end time for simulation 


   Int_t   m_numRanks ;

   Index_t m_colLoc ;
   Index_t m_rowLoc ;
   Index_t m_planeLoc ;
   Index_t m_tp ;

   Index_t m_sizeX ;
   Index_t m_sizeY ;
   Index_t m_sizeZ ;
   Index_t m_numElem ;
   Index_t m_numNode ;

   Index_t m_maxPlaneSize ;
   Index_t m_maxEdgeSize ;

   // OMP hack 
   Index_t *m_nodeElemStart ;

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

} ;

/* GPU mesh */

/* Given a number of bytes, nbytes, and a byte alignment, align, (e.g., 2,
 * 4, 8, or 16), return the smallest integer that is larger than nbytes and
 * a multiple of align.
 */
#define PAD_DIV(nbytes, align)  (((nbytes) + (align) - 1) / (align))
#define PAD(nbytes, align)  (PAD_DIV((nbytes),(align)) * (align))

#define MINEQ(a,b) (a)=(((a)<(b))?(a):(b))


// Modified to pass ARRAY_OBJECT references
// this simplifies the call list for all of the subroutines
struct MeshGPU {

    MeshGPU(
HCC_ARRAY_OBJECT(Index_t, matElemlist),
HCC_ARRAY_OBJECT(Real_t, ss),
HCC_ARRAY_OBJECT(Real_t, arealg),
HCC_ARRAY_OBJECT(Real_t, vdov),
HCC_ARRAY_OBJECT(Index_t, nodelist),
HCC_ARRAY_OBJECT(Real_t, x),
HCC_ARRAY_OBJECT(Real_t, y),
HCC_ARRAY_OBJECT(Real_t, z),
HCC_ARRAY_OBJECT(Real_t, xd),
HCC_ARRAY_OBJECT(Real_t, yd),
HCC_ARRAY_OBJECT(Real_t, zd),
HCC_ARRAY_OBJECT(Real_t, fx),
HCC_ARRAY_OBJECT(Real_t, fy),
HCC_ARRAY_OBJECT(Real_t, fz),
HCC_ARRAY_OBJECT(Real_t, elemMass),
HCC_ARRAY_OBJECT(Int_t, nodeElemCount),
HCC_ARRAY_OBJECT(Index_t, nodeElemCornerList),
HCC_ARRAY_OBJECT(Real_t, v),
HCC_ARRAY_OBJECT(Real_t, volo),
HCC_ARRAY_OBJECT(Real_t, vnew),
HCC_ARRAY_OBJECT(Real_t, vnewc),
HCC_ARRAY_OBJECT(Real_t, xdd),
HCC_ARRAY_OBJECT(Real_t, ydd),
HCC_ARRAY_OBJECT(Real_t, zdd),
HCC_ARRAY_OBJECT(Real_t, nodalMass),
HCC_ARRAY_OBJECT(Index_t, symmX),
HCC_ARRAY_OBJECT(Index_t, symmY),
HCC_ARRAY_OBJECT(Index_t, symmZ),
HCC_ARRAY_OBJECT(Real_t, delv),
HCC_ARRAY_OBJECT(Real_t, dxx),
HCC_ARRAY_OBJECT(Real_t, dyy),
HCC_ARRAY_OBJECT(Real_t, dzz),

HCC_ARRAY_OBJECT(Real_t, delx_zeta),
HCC_ARRAY_OBJECT(Real_t, delv_zeta),
HCC_ARRAY_OBJECT(Real_t, delx_xi),
HCC_ARRAY_OBJECT(Real_t, delv_xi),
HCC_ARRAY_OBJECT(Real_t, delx_eta),
HCC_ARRAY_OBJECT(Real_t, delv_eta),
HCC_ARRAY_OBJECT(Index_t, elemBC),
HCC_ARRAY_OBJECT(Index_t, lxim),
HCC_ARRAY_OBJECT(Index_t, lxip),
HCC_ARRAY_OBJECT(Index_t, letam),
HCC_ARRAY_OBJECT(Index_t, letap),
HCC_ARRAY_OBJECT(Index_t, lzetam),
HCC_ARRAY_OBJECT(Index_t, lzetap),

HCC_ARRAY_OBJECT(Real_t, qq),
HCC_ARRAY_OBJECT(Real_t, ql),
HCC_ARRAY_OBJECT(Real_t, e),
HCC_ARRAY_OBJECT(Real_t, p),
HCC_ARRAY_OBJECT(Real_t, q),
HCC_ARRAY_OBJECT(Real_t, e_old),
HCC_ARRAY_OBJECT(Real_t, p_old),
HCC_ARRAY_OBJECT(Real_t, q_old),
HCC_ARRAY_OBJECT(Real_t, delvc),
HCC_ARRAY_OBJECT(Real_t, compression),
HCC_ARRAY_OBJECT(Real_t, compHalfStep),
HCC_ARRAY_OBJECT(Real_t, qq_old),
HCC_ARRAY_OBJECT(Real_t, ql_old),
HCC_ARRAY_OBJECT(Real_t, work),
HCC_ARRAY_OBJECT(Real_t, p_new),
HCC_ARRAY_OBJECT(Real_t, e_new),
HCC_ARRAY_OBJECT(Real_t, q_new),
HCC_ARRAY_OBJECT(Real_t, bvc),
HCC_ARRAY_OBJECT(Real_t, pbvc),
HCC_ARRAY_OBJECT(Real_t, pHalfStep),
HCC_ARRAY_OBJECT(Real_t, sigxx),
HCC_ARRAY_OBJECT(Real_t, sigyy),
HCC_ARRAY_OBJECT(Real_t, sigzz),
HCC_ARRAY_OBJECT(Real_t, determ),
HCC_ARRAY_OBJECT(Real_t, dvdx),
HCC_ARRAY_OBJECT(Real_t, dvdy),
HCC_ARRAY_OBJECT(Real_t, dvdz),
HCC_ARRAY_OBJECT(Real_t, x8n),
HCC_ARRAY_OBJECT(Real_t, y8n),
HCC_ARRAY_OBJECT(Real_t, z8n),
HCC_ARRAY_OBJECT(Real_t, fx_elem),
HCC_ARRAY_OBJECT(Real_t, fy_elem),
HCC_ARRAY_OBJECT(Real_t, fz_elem),
HCC_ARRAY_OBJECT(Real_t, mindthydro),
HCC_ARRAY_OBJECT(Real_t, mindtcourant)
    ) :  matElemlist(matElemlist),
    ss(ss), arealg(arealg), vdov(vdov),
    nodelist(nodelist),
    x(x), y(y), z(z),
    xd(xd), yd(yd), zd(zd),
    fx(fx), fy(fy), fz(fz),
    elemMass(elemMass),
    nodeElemCount(nodeElemCount), nodeElemCornerList(nodeElemCornerList),
    v(v), volo(volo), vnew(vnew), vnewc(vnewc),
    xdd(xdd), ydd(ydd), zdd(zdd),
    nodalMass(nodalMass),
    symmX(symmX), symmY(symmY), symmZ(symmZ),
    delv(delv),
    dxx(dxx), dyy(dyy), dzz(dzz), 
    delx_zeta(delx_zeta), delv_zeta(delv_zeta),
    delx_xi(delx_xi), delv_xi(delv_xi),
    delx_eta(delx_eta), delv_eta(delv_eta),

    elemBC(elemBC),
    lxim(lxim),
    lxip(lxip),
    letam(letam),
    letap(letap),
    lzetam(lzetam),
    lzetap(lzetap),
    qq(qq),
    ql(ql),
    e(e),
    p(p),
    q(q),
    e_old(e_old),
    p_old(p_old),
    q_old(q_old),
    delvc(delvc),
    compression(compression),
    compHalfStep(compHalfStep),
    qq_old(qq_old),
    ql_old(ql_old),
    work(work),
    p_new(p_new),
    e_new(e_new),
    q_new(q_new),
    bvc(bvc),
    pbvc(pbvc),
    pHalfStep(pHalfStep),
    sigxx(sigxx),
    sigyy(sigyy),
    sigzz(sigzz),

    determ(determ),
    dvdx(dvdx),
    dvdy(dvdy),
    dvdz(dvdz),
    x8n(x8n),
    y8n(y8n),
    z8n(z8n),
    fx_elem(fx_elem),
    fy_elem(fy_elem),
    fz_elem(fz_elem),
    mindthydro(mindthydro),
    mindtcourant(mindtcourant)
  {
    std::cout << "New Initialization of GPU complete" << std::endl;
  }
    
   /******************/
   /* Implementation */
   /******************/

HCC_ARRAY_OBJECT(Index_t, matElemlist);
HCC_ARRAY_OBJECT(Real_t, ss);
HCC_ARRAY_OBJECT(Real_t, arealg);
HCC_ARRAY_OBJECT(Real_t, vdov);
HCC_ARRAY_OBJECT(Index_t, nodelist);
HCC_ARRAY_OBJECT(Real_t, x);
HCC_ARRAY_OBJECT(Real_t, y);
HCC_ARRAY_OBJECT(Real_t, z);
HCC_ARRAY_OBJECT(Real_t, xd);
HCC_ARRAY_OBJECT(Real_t, yd);
HCC_ARRAY_OBJECT(Real_t, zd);
HCC_ARRAY_OBJECT(Real_t, fx);
HCC_ARRAY_OBJECT(Real_t, fy);
HCC_ARRAY_OBJECT(Real_t, fz);

HCC_ARRAY_OBJECT(Real_t, elemMass);
HCC_ARRAY_OBJECT(Int_t, nodeElemCount);
HCC_ARRAY_OBJECT(Index_t, nodeElemCornerList);
HCC_ARRAY_OBJECT(Real_t, v);
HCC_ARRAY_OBJECT(Real_t, volo);
HCC_ARRAY_OBJECT(Real_t, vnew);
HCC_ARRAY_OBJECT(Real_t, vnewc);
HCC_ARRAY_OBJECT(Real_t, xdd);
HCC_ARRAY_OBJECT(Real_t, ydd);
HCC_ARRAY_OBJECT(Real_t, zdd);
HCC_ARRAY_OBJECT(Real_t, nodalMass);
HCC_ARRAY_OBJECT(Index_t, symmX);
HCC_ARRAY_OBJECT(Index_t, symmY);
HCC_ARRAY_OBJECT(Index_t, symmZ);
HCC_ARRAY_OBJECT(Real_t, delv);
HCC_ARRAY_OBJECT(Real_t, dxx);
HCC_ARRAY_OBJECT(Real_t, dyy);
HCC_ARRAY_OBJECT(Real_t, dzz);

HCC_ARRAY_OBJECT(Real_t, delx_zeta);
HCC_ARRAY_OBJECT(Real_t, delv_zeta);
HCC_ARRAY_OBJECT(Real_t, delx_xi);
HCC_ARRAY_OBJECT(Real_t, delv_xi);
HCC_ARRAY_OBJECT(Real_t, delx_eta);
HCC_ARRAY_OBJECT(Real_t, delv_eta);
HCC_ARRAY_OBJECT(Index_t, elemBC);
HCC_ARRAY_OBJECT(Index_t, lxim);
HCC_ARRAY_OBJECT(Index_t, lxip);
HCC_ARRAY_OBJECT(Index_t, letam);
HCC_ARRAY_OBJECT(Index_t, letap);
HCC_ARRAY_OBJECT(Index_t, lzetam);
HCC_ARRAY_OBJECT(Index_t, lzetap);
HCC_ARRAY_OBJECT(Real_t, qq);
HCC_ARRAY_OBJECT(Real_t, ql);
HCC_ARRAY_OBJECT(Real_t, e);
HCC_ARRAY_OBJECT(Real_t, p);
HCC_ARRAY_OBJECT(Real_t, q);
HCC_ARRAY_OBJECT(Real_t, e_old);
HCC_ARRAY_OBJECT(Real_t, p_old);
HCC_ARRAY_OBJECT(Real_t, q_old);
HCC_ARRAY_OBJECT(Real_t, delvc);
HCC_ARRAY_OBJECT(Real_t, compression);
HCC_ARRAY_OBJECT(Real_t, compHalfStep);
HCC_ARRAY_OBJECT(Real_t, qq_old);
HCC_ARRAY_OBJECT(Real_t, ql_old);
HCC_ARRAY_OBJECT(Real_t, work);
HCC_ARRAY_OBJECT(Real_t, p_new);
HCC_ARRAY_OBJECT(Real_t, e_new);
HCC_ARRAY_OBJECT(Real_t, q_new);
HCC_ARRAY_OBJECT(Real_t, bvc);
HCC_ARRAY_OBJECT(Real_t, pbvc);
HCC_ARRAY_OBJECT(Real_t, pHalfStep);
HCC_ARRAY_OBJECT(Real_t, sigxx);
HCC_ARRAY_OBJECT(Real_t, sigyy);
HCC_ARRAY_OBJECT(Real_t, sigzz);
HCC_ARRAY_OBJECT(Real_t, determ);
HCC_ARRAY_OBJECT(Real_t, dvdx);
HCC_ARRAY_OBJECT(Real_t, dvdy);
HCC_ARRAY_OBJECT(Real_t, dvdz);
HCC_ARRAY_OBJECT(Real_t, x8n);
HCC_ARRAY_OBJECT(Real_t, y8n);
HCC_ARRAY_OBJECT(Real_t, z8n);
HCC_ARRAY_OBJECT(Real_t, fx_elem);
HCC_ARRAY_OBJECT(Real_t, fy_elem);
HCC_ARRAY_OBJECT(Real_t, fz_elem);
HCC_ARRAY_OBJECT(Real_t, mindthydro);
HCC_ARRAY_OBJECT(Real_t, mindtcourant);
};

typedef Real_t &(Domain::* Domain_member )(Index_t) ;

struct cmdLineOpts {
   Int_t its; // -i 
   Int_t nx;  // -s 
   Int_t numReg; // -r 
   Int_t numFiles; // -f
   Int_t showProg; // -p
   Int_t quiet; // -q
   Int_t viz; // -v 
   Int_t cost; // -c
   Int_t balance; // -b
};



// Function Prototypes

// lulesh-par
static inline
Real_t CalcElemVolume( const Real_t x[8],
                       const Real_t y[8],
                       const Real_t z[8]) restrict(amp);

Real_t CalcElemVolume_nonamp( const Real_t x[8], const Real_t y[8], const Real_t z[8] );


// lulesh-util
void ParseCommandLineOptions(int argc, char *argv[],
                             Int_t myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
                               Int_t nx,
                               Int_t numRanks);

// lulesh-viz
void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// lulesh-init
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side);
                    
#endif                    
