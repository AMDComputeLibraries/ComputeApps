/*
	lulesh_kernels.h
	
	function prototypes for the C++ AMP kernels defined in lulesh_kerenls.cc
	
	These are called in lulesh.cc
*/

#include <amp.h>
#include "lulesh.h"

#ifndef __LULESH_KERNELS__
#define __LULESH_KERNELS__

void InitStressTermsForElems_kernel(int numElem, 
						      Real_t *sigxx,  Real_t *sigyy, Real_t *sigzz, 
						      Real_t *p,  Real_t *q, 
						      concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
						      //concurrency::index<1> &idx) restrict(amp);

void IntegrateStressForElems_kernel( Index_t numElem,  Index_t *nodelist,
                                      Real_t *x,  Real_t *y,  Real_t *z,
                                      Real_t *fx_elem,  Real_t *fy_elem,  Real_t *fz_elem,
                                      Real_t *sigxx,  Real_t *sigyy,  Real_t *sigzz,
                                      Real_t *determ, 
                                      concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                      //concurrency::index<1> &idx) restrict(amp);
                                      
void AddNodeForcesFromElems_kernel( Index_t numNode,
                                    Int_t *nodeElemCount,  Index_t *nodeElemCornerList,
                                    Real_t *fx_elem,  Real_t *fy_elem,  Real_t *fz_elem,
                                    Real_t *fx_node,  Real_t *fy_node,  Real_t *fz_node, 
                                    concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                    //concurrency::index<1> &idx) restrict(amp);
                                    
void AddNodeForcesFromElems2_kernel(Index_t numNode,
                                    Int_t *nodeElemCount, Index_t *nodeElemCornerList,
                                    Real_t *fx_elem,  Real_t *fy_elem,  Real_t *fz_elem,
                                    Real_t *fx_node,  Real_t *fy_node,  Real_t *fz_node, 
                                    concurrency::tiled_index<64> &idx) restrict(amp);
                                    //concurrency::index<1> &idx) restrict(amp);

void CalcHourglassControlForElems_kernel(Int_t numElem, Index_t *nodelist,
                                        Real_t *x, Real_t *y, Real_t *z,
                                        Real_t *determ, Real_t *volo, Real_t *v,
                                        Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                        Real_t *x8n, Real_t *y8n, Real_t *z8n, 
                                        concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                        //concurrency::index<1> &idx) restrict(amp);
                                        
void CalcFBHourglassForceForElems_kernel(const  Real_t *determ,
    						           const  Real_t *x8n, const Real_t *y8n, const Real_t *z8n,
    							      const  Real_t *dvdx, const  Real_t *dvdy, const Real_t *dvdz,
    								 Real_t hourg,
    								 Index_t numElem, const Index_t *nodelist,
    								 const Real_t *ss, const Real_t *elemMass,
    								 const Real_t *xd, const Real_t *yd, const Real_t *zd,
     							 Real_t *fx_elem, Real_t *fy_elem, Real_t *fz_elem, 
     							 concurrency::tiled_index<64> &idx) restrict(amp);
     							 //concurrency::index<1> &idx) restrict(amp);

void CalcAccelerationForNodes_kernel(int numNode,
                                    Real_t *xdd, Real_t *ydd, Real_t *zdd,
                                    Real_t *fx, Real_t *fy, Real_t *fz,
                                    Real_t *nodalMass, 
                                    concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                    //concurrency::index<1> &idx) restrict(amp);
                                    
void ApplyAccelerationBoundaryConditionsForNodes_kernel(int numNodeBC, 
										      Real_t *xdd, Real_t *ydd, Real_t *zdd,
     										 Index_t *symmX, Index_t *symmY, Index_t *symmZ, 
     										 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     										 //concurrency::index<1> &idx) restrict(amp);                                    
                                    
void CalcVelocityForNodes_kernel(int numNode, const Real_t dt, const Real_t u_cut,
                                 Real_t *xd, Real_t *yd, Real_t *zd,
                                 Real_t *xdd, Real_t *ydd, Real_t *zdd, 
                                 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                 //concurrency::index<1> &idx) restrict(amp);
                                 
void CalcPositionForNodes_kernel(int numNode, Real_t dt,
                                 Real_t *x, Real_t *y, Real_t *z,
                                 Real_t *xd, Real_t *yd, Real_t *zd, 
                                 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                 //concurrency::index<1> &idx) restrict(amp);
                                 
 void CalcKinematicsForElems_kernel(Index_t numElem, Real_t dt,
     					      Index_t *nodelist, Real_t *volo, Real_t *v,
     						 Real_t *x, Real_t *y, Real_t *z, 
     						 Real_t *xd, Real_t *yd, Real_t *zd,
     						 Real_t *vnew, Real_t *delv, Real_t *arealg,
     						 Real_t *dxx, Real_t *dyy, Real_t *dzz, 
     						 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     						 //concurrency::index<1> &idx) restrict(amp);
     								
void CalcLagrangeElementsPart2_kernel(Index_t numElem,
     						   Real_t *dxx, Real_t *dyy,  Real_t *dzz,
     						   Real_t *vdov, 
     						   concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     						   //concurrency::index<1> &idx) restrict(amp);

void CalcMonotonicQGradientsForElems_kernel(Index_t numElem, Index_t *nodelist,
     							    Real_t *x, Real_t *y, Real_t *z, 
     							    Real_t *xd, Real_t *yd, Real_t *zd,
     							    Real_t *volo, Real_t *vnew,
     							    Real_t *delx_zeta, Real_t *delv_zeta,
     							    Real_t *delx_xi, Real_t *delv_xi,
     							    Real_t *delx_eta, Real_t *delv_eta, 
     							    concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     							    //concurrency::index<1> &idx) restrict(amp);

void CalcMonotonicQRegionForElems_kernel(Index_t regionStart,
    								 Real_t qlc_monoq,
    								 Real_t qqc_monoq,
    								 Real_t monoq_limiter_mult,
    								 Real_t monoq_max_slope,
    								 Real_t ptiny,
    								 Index_t elength,
     							 Index_t *matElemlist, Index_t *elemBC,
     							 Index_t *lxim, Index_t *lxip,
     							 Index_t *letam, Index_t *letap,
     							 Index_t *lzetam, Index_t *lzetap,
     							 Real_t *delv_xi, Real_t *delv_eta, Real_t *delv_zeta,
     							 Real_t *delx_xi, Real_t *delx_eta, Real_t *delx_zeta,
     							 Real_t *vdov, Real_t *elemMass, Real_t *volo, Real_t *vnew,
     							 Real_t *qq, Real_t *ql,
     							 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     							 // concurrency::index<1> &idx) restrict(amp);
     									
void CalcPressureForElems_kernel(Index_t regionStart,
                                 Index_t *matElemlist,
                                 Real_t* p_new,  Real_t* bvc,
                                 Real_t* pbvc,  Real_t* e_old,
                                 Real_t* compression,  Real_t *vnewc,
                                 Real_t pmin,
                                 Real_t p_cut, Real_t eosvmax,
                                 Index_t length, Real_t c1s, 
                                 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                 //concurrency::index<1> &idx) restrict(amp);
                                                                                                                                                          
void CalcEnergyForElemsPart1_kernel(Index_t length,Real_t emin,
     					      Real_t *e_old, Real_t *delvc, 
     						 Real_t *p_old, Real_t *q_old, 
     						 Real_t *work, Real_t *e_new, 
     						 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     						 //concurrency::index<1> &idx) restrict(amp);
     							    
void CalcEnergyForElemsPart2_kernel(Index_t length,Real_t rho0,Real_t e_cut,Real_t emin,
     						 Real_t *compHalfStep, Real_t *delvc, Real_t *pbvc, Real_t *bvc, 
     						 Real_t *pHalfStep, Real_t *ql, Real_t *qq, 
     						 Real_t *p_old, Real_t *q_old, Real_t *work,
     						 Real_t *e_new, Real_t *q_new, 
     						 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     						 //concurrency::index<1> &idx) restrict(amp);
     								
void CalcEnergyForElemsPart3_kernel(Index_t regionStart, Index_t *matElemlist, 
							 Index_t length, Real_t rho0, Real_t sixth,
							 Real_t e_cut, Real_t emin,
     						 Real_t *pbvc, Real_t *vnewc, Real_t *bvc, 
     						 Real_t *p_new, Real_t *ql, Real_t *qq,
     						 Real_t *p_old, Real_t *q_old, Real_t *pHalfStep, 
     						 Real_t *q_new, Real_t *delvc, Real_t *e_new, 
     						 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     						 //concurrency::index<1> &idx) restrict(amp);
     								
void CalcEnergyForElemsPart4_kernel(Index_t regionStart, Index_t *matElemlist, 
							 Index_t length, Real_t rho0, Real_t q_cut,
     						 Real_t *delvc, Real_t *pbvc, Real_t *e_new, 
     						 Real_t *vnewc, Real_t *bvc, Real_t *p_new, 
     						 Real_t *ql, Real_t *qq, Real_t *q_new, 
     						 concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     						 //concurrency::index<1> &idx) restrict(amp);
 
void CalcSoundSpeedForElems_kernel(Index_t regionStart, Real_t *vnewc, Real_t rho0,  
						     Real_t *enewc, Real_t *pnewc, Real_t *pbvc,
                                   Real_t *bvc, Real_t ss4o3, Index_t nz, 
                                   Index_t *matElemlist, Real_t *ss, 
                                   concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                   //concurrency::index<1> &idx) restrict(amp);
                                  
void EvalEOSForElemsPart1_kernel(Index_t regionStart, Index_t length,
						   Real_t eosvmin,Real_t eosvmax,
     					   Index_t *matElemlist,
     					   Real_t *e, Real_t *delv, Real_t *p, 
     					   Real_t *q, Real_t *qq, Real_t *ql,
     					   Real_t *vnewc,
     					   Real_t *e_old, Real_t *delvc, 
     					   Real_t *p_old, Real_t *q_old,
     					   Real_t *compression, Real_t *compHalfStep,
     					   Real_t *qq_old, Real_t *ql_old, Real_t *work, 
     					   concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     					   //concurrency::index<1> &idx) restrict(amp);
     							
void EvalEOSForElemsPart2_kernel(Index_t regionStart, Index_t length,
     					   Index_t *matElemlist, Real_t *p_new, Real_t *e_new, 
     					   Real_t *q_new, Real_t *p, Real_t *e, Real_t *q, 
     					   concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     					   //concurrency::index<1> &idx) restrict(amp);
     							 
void ApplyMaterialPropertiesForElemsPart1_kernel(Index_t length,Real_t eosvmin,Real_t eosvmax,
     								    Index_t *matElemlist, Real_t *vnew, Real_t *vnewc, 
     								    concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     								    //concurrency::index<1> &idx) restrict(amp);
     											  
void UpdateVolumesForElems_kernel(Index_t numElem,Real_t v_cut,
                                  Real_t *vnew, Real_t *v, 
                                  concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
                                  //concurrency::index<1> &idx) restrict(amp);
                                  
void CalcCourantConstraintForElems_kernel(Index_t regionStart, Index_t length,Real_t qqc2,
     							  Index_t *matElemlist, Real_t *ss, Real_t *vdov, 
     							  Real_t *arealg, Real_t *mindtcourant, 
     							  concurrency::tiled_index<BLOCKSIZE> &idx) restrict(amp);
     									  
void CalcHydroConstraintForElems_kernel(Index_t regionStart, Index_t length, Real_t dvovmax,
     							Index_t *matElemlist, Real_t *vdov, Real_t *mindthydro, 
     							concurrency::tiled_index<BLOCKSIZE> &t_idx) restrict(amp);

#endif
