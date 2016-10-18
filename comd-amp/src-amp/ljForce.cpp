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

/// Computes forces for the 12-6 Lennard Jones (LJ) potential.
///
/// The Lennard-Jones model is not a good representation for the
/// bonding in copper, its use has been limited to constant volume
/// simulations where the embedding energy contribution to the cohesive
/// energy is not included in the two-body potential
///
/// The parameters here are taken from Wolf and Phillpot and fit to the
/// room temperature lattice constant and the bulk melt temperature
/// Ref: D. Wolf and S.Yip eds. Materials Interfaces (Chapman & Hall
///      1992) Page 230.
///
/// Notes on LJ:
///
/// http://en.wikipedia.org/wiki/Lennard_Jones_potential
///
/// The total inter-atomic potential energy in the LJ model is:
///
/// \f[
///   E_{tot} = \sum_{ij} U_{LJ}(r_{ij})
/// \f]
/// \f[
///   U_{LJ}(r_{ij}) = 4 \epsilon
///           \left\{ \left(\frac{\sigma}{r_{ij}}\right)^{12}
///           - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\}
/// \f]
///
/// where \f$\epsilon\f$ and \f$\sigma\f$ are the material parameters in the potential.
///    - \f$\epsilon\f$ = well depth
///    - \f$\sigma\f$   = hard sphere diameter
///
///  To limit the interation range, the LJ potential is typically
///  truncated to zero at some cutoff distance. A common choice for the
///  cutoff distance is 2.5 * \f$\sigma\f$.
///  This implementation can optionally shift the potential slightly
///  upward so the value of the potential is zero at the cuotff
///  distance.  This shift has no effect on the particle dynamics.
///
///
/// The force on atom i is given by
///
/// \f[
///   F_i = -\nabla_i \sum_{jk} U_{LJ}(r_{jk})
/// \f]
///
/// where the subsrcipt i on the gradient operator indicates that the
/// derivatives are taken with respect to the coordinates of atom i.
/// Liberal use of the chain rule leads to the expression
///
/// \f{eqnarray*}{
///   F_i &=& - \sum_j U'_{LJ}(r_{ij})\hat{r}_{ij}\\
///       &=& \sum_j 24 \frac{\epsilon}{r_{ij}} \left\{ 2 \left(\frac{\sigma}{r_{ij}}\right)^{12}
///               - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\} \hat{r}_{ij}
/// \f}
///
/// where \f$\hat{r}_{ij}\f$ is a unit vector in the direction from atom
/// i to atom j.
/// 
///

#include "ljForce.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "constants.h"
#include "parallel.h"
#include "linkCells.h"
#include "memUtils.h"

#include <hc.hpp>
using namespace hc;

#define POT_SHIFT 1.0

static int ljForce(SimFlat* s);
static void ljPrint(FILE* file, BasePotential* pot);

void ljDestroy(BasePotential** inppot)
{
   if ( ! inppot ) return;
   LjPotential* pot = (LjPotential*)(*inppot);
   if ( ! pot ) return;
   comdFree(pot);
   *inppot = NULL;

   return;
}

/// Initialize an Lennard Jones potential for Copper.
BasePotential* initLjPot(void)
{
   LjPotential *pot = (LjPotential*)comdMalloc(sizeof(LjPotential));
   pot->force = ljForce;
   pot->print = ljPrint;
   pot->destroy = ljDestroy;
   pot->sigma = 2.315;	                  // Angstrom
   pot->epsilon = 0.167;                  // eV
   pot->mass = 63.55 * amuToInternalMass; // Atomic Mass Units (amu)

   pot->lat = 3.615;                      // Equilibrium lattice const in Angs
   strcpy(pot->latticeType, "FCC");       // lattice type, i.e. FCC, BCC, etc.
   pot->cutoff = 2.5*pot->sigma;          // Potential cutoff in Angs

   strcpy(pot->name, "Cu");
   pot->atomicNo = 29;

   return (BasePotential*) pot;
}

void ljPrint(FILE* file, BasePotential* pot)
{
   LjPotential* ljPot = (LjPotential*) pot;
   fprintf(file, "  Potential type   : Lennard-Jones\n");
   fprintf(file, "  Species name     : %s\n", ljPot->name);
   fprintf(file, "  Atomic number    : %d\n", ljPot->atomicNo);
   fprintf(file, "  Mass             : %lg amu\n", ljPot->mass / amuToInternalMass); // print in amu
   fprintf(file, "  Lattice Type     : %s\n", ljPot->latticeType);
   fprintf(file, "  Lattice spacing  : %lg Angstroms\n", ljPot->lat);
   fprintf(file, "  Cutoff           : %lg Angstroms\n", ljPot->cutoff);
   fprintf(file, "  Epsilon          : %lg eV\n", ljPot->epsilon);
   fprintf(file, "  Sigma            : %lg Angstroms\n", ljPot->sigma);
}

int ljForce(SimFlat* s)
{
   LjPotential* pot = (LjPotential *) s->pot;
   real_t sigma = pot->sigma;
   real_t epsilon = pot->epsilon;
   real_t rCut = pot->cutoff;
   real_t rCut2 = rCut*rCut;

   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;
   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);
   int nNbrBoxes = 27;
   
   // zero forces and energy
   real_t ePot = 0.0;
   s->ePotential = 0.0;
   int fSize = s->boxes->nTotalBoxes*MAXATOMS;
   int nBoxes = s->boxes->nTotalBoxes;
   extent<1> boxesExt(s->boxes->nLocalBoxes * MAXATOMS);
   tiled_extent<1> tBoxesExt(boxesExt, MAXATOMS);
   completion_future fut;   
   
   for (int ii=0; ii<fSize; ++ii)
   {
     s->atoms->f[ii*3 + 0] = 0.;
     s->atoms->f[ii*3 + 1] = 0.;
     s->atoms->f[ii*3 + 2] = 0.;     
     s->atoms->U[ii] = 0.;
   }

   HCC_ARRAY_STRUC(real_t, U, fSize, s->atoms->U);
   HCC_ARRAY_STRUC(real_t, f, fSize*3, s->atoms->f);
   HCC_ARRAY_STRUC(real_t, r, fSize*3, s->atoms->r);
   HCC_ARRAY_STRUC(int, nAtoms, nBoxes, s->boxes->nAtoms);
   HCC_ARRAY_STRUC(int, nbrBoxes, nBoxes * nNbrBoxes, s->boxes->nbrBoxes);

	/* With tiling, each i-th box works in one tile each with several threads. 
	 * Each thread in the tile works on each i-th atom.
	 * Number of threads in a tile is equal to 64, i.e., the wavefront size.
	 * This means that some threads can be left ideal if an i-th box does not
	 * contain #MAXATOMS atoms.
	 * Important to mention that no data-copy is required with CPP AMP and 
	 * host pointers are directly passed to the GPU without the need to create 
	 * different data containers.
	 * An optimized approach to allocate extent is to have each tile work on
	 * two boxes because for LJ max. atoms in a box = 32. Therefore, 64/32 = 2
	 * boxes can be computed within a wavefront.
	 */

   // loop over local boxes
   fut = parallel_for_each(tBoxesExt, [=
				       HCC_ID(U)
				       HCC_ID(f)
				       HCC_ID(r)
				       HCC_ID(nAtoms)
				       HCC_ID(nbrBoxes)](tiled_index<1> t_idx) restrict(amp){
       int nIBox = nAtoms[t_idx.tile[0]];
       // loop over neighbors of iBox
       for (int jTmp=0; jTmp<nNbrBoxes; jTmp++){
	 int jBox = nbrBoxes[t_idx.tile[0]*nNbrBoxes + jTmp];
	 if(jBox >= 0){
	   int nJBox = nAtoms[jBox];
	   int iOff = t_idx.tile[0] * MAXATOMS;
	   int ii = t_idx.local[0];
	   // loop over atoms in iBox
	   if(ii < nIBox){
	     // loop over atoms in jBox
	     for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++){
	       real_t dr[3];
	       real_t r2 = 0.0;
	       for (int m=0; m<3; m++){
		 dr[m] = r[(iOff + ii)*3 + m] - r[jOff*3 + m];
		 r2+=dr[m]*dr[m];
	       }

	       if ( r2 <= rCut2 && r2 > 0.0){ 
		 // Important note:
		 // from this point on r actually refers to 1.0/r
		 r2 = 1.0/r2;
		 real_t r6 = s6 * (r2*r2*r2);
		 real_t eLocal = r6 * (r6 - 1.0) - eShift;
		 U[iOff + ii] += 0.5*eLocal;

		 // different formulation to avoid sqrt computation
		 real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
		 for (int m=0; m<3; m++){
		   f[(iOff + ii)*3 + m] -= dr[m]*fr;
		 }
	       }
	     } // loop over atoms in jBox
	   } // close if ii < nIBox
	 } // close if jBox >= 0
       } // loop over neighbor boxes
     } // loop over local boxes in system
     );
   fut.wait();

   HCC_SYNC(U,s->atoms->U);
   HCC_SYNC(f,s->atoms->f);   
   

   /* A loop over all the atoms is requrired to reduce the ePot value.
    * Otherwise, update to ePot is required to be atomic which will 
    * lead to slow performance. 
    */
   for (int iBox=0; iBox<s->boxes->nLocalBoxes; iBox++){
     for(int iAtom = 0; iAtom < s->boxes->nAtoms[iBox]; iAtom++){
       int iOff = iBox * MAXATOMS + iAtom;
       ePot += s->atoms->U[iOff];
     }
   }
   ePot = ePot*4.0*epsilon;
   s->ePotential = ePot;

   return 0;
}
