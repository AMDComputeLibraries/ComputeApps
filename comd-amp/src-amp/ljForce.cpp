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
/// \file
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

#include <amp.h>
using namespace concurrency;

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
   //fprintf(file, "  Mass             : "FMT1" amu\n", ljPot->mass / amuToInternalMass); // print in amu
   fprintf(file, "  Mass             : %lg amu\n", ljPot->mass / amuToInternalMass); // print in amu
   fprintf(file, "  Lattice Type     : %s\n", ljPot->latticeType);
   //fprintf(file, "  Lattice spacing  : "FMT1" Angstroms\n", ljPot->lat);
   //fprintf(file, "  Cutoff           : "FMT1" Angstroms\n", ljPot->cutoff);
   //fprintf(file, "  Epsilon          : "FMT1" eV\n", ljPot->epsilon);
   //fprintf(file, "  Sigma            : "FMT1" Angstroms\n", ljPot->sigma);
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

   // zero forces and energy
   real_t ePot = 0.0;
   s->ePotential = 0.0;
   int fSize = s->boxes->nTotalBoxes*MAXATOMS;
		
   parallel_for_each(
		extent<1>(fSize), [=](index<1> idx) restrict(amp)
		{
			s->atoms->f[idx[0]][0] = 0.;
			s->atoms->f[idx[0]][1] = 0.;
			s->atoms->f[idx[0]][2] = 0.;
			s->atoms->U[idx[0]] = 0.;
		}
	);

#if 0
   for (int ii=0; ii<fSize; ++ii)
   {
      zeroReal3(s->atoms->f[ii]);
      s->atoms->U[ii] = 0.;
   }
#endif

   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);

   int nNbrBoxes = 27;

/* CPP AMP implementation which does not use any tiling 
 * Not the most optimal version and is present here only for
 * reference and comparative purposes
 */

#if 0
   // No tiling
   extent<1> boxesExt(s->boxes->nLocalBoxes);
	parallel_for_each(
			boxesExt, [=](index<1> idx) restrict(amp)
   {
      int nIBox = s->boxes->nAtoms[idx[0]];
      // loop over neighbors of iBox
      for (int jTmp=0; jTmp<nNbrBoxes; jTmp++)
      {
         int jBox = s->boxes->nbrBoxes[idx[0]][jTmp];
        
		 if(jBox < 0)
			return;
         //assert(jBox>=0);
         
         int nJBox = s->boxes->nAtoms[jBox];
		
         // loop over atoms in iBox

         for (int iOff=idx[0]*MAXATOMS,ii=0; ii<nIBox; ii++,iOff++)
         {
            // loop over atoms in jBox
            for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++)
            {
               real_t dr[3];
               real_t r2 = 0.0;
               for (int m=0; m<3; m++)
               {
                  dr[m] = s->atoms->r[iOff][m]-s->atoms->r[jOff][m];
                  r2+=dr[m]*dr[m];
               }

               if ( r2 > rCut2 || r2 <= 0.0) continue;

               // Important note:
               // from this point on r actually refers to 1.0/r
               r2 = 1.0/r2;
               real_t r6 = s6 * (r2*r2*r2);
               real_t eLocal = r6 * (r6 - 1.0) - eShift;
               s->atoms->U[iOff] += 0.5*eLocal;

               // different formulation to avoid sqrt computation
               real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
               for (int m=0; m<3; m++)
               {
				  s->atoms->f[iOff][m] -= dr[m]*fr;
               }
            } // loop over atoms in jBox
         } // loop over atoms in iBox
      } // loop over neighbor boxes
   } // loop over local boxes in system
	);
#endif

	// With tiling
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
#define NO_OPT 1

#if NO_OPT
   extent<1> boxesExt(s->boxes->nLocalBoxes * MAXATOMS);
#else
   int max_atoms_in_box = 32; // Max atoms in a box for LJ, as computed during initialization
   int tile_size = 64; // Wavefront size on AMD GPUs
   int extent_size = ((s->boxes->nLocalBoxes * max_atoms_in_box / tile_size) + 1) * tile_size;

   extent<1> boxesExt(extent_size);
#endif

   // loop over local boxes
	parallel_for_each(
			boxesExt.tile<MAXATOMS>(), [=](tiled_index<MAXATOMS> t_idx) restrict(amp)
   {
#if NO_OPT
      int nIBox = s->boxes->nAtoms[t_idx.tile[0]];
#else
	  int ii_local = t_idx.local[0];
	  int box_id = t_idx.tile[0] * (tile_size / max_atoms_in_box) + ii_local / max_atoms_in_box;
	  int nIBox = s->boxes->nAtoms[box_id];
#endif
      // loop over neighbors of iBox
      for (int jTmp=0; jTmp<nNbrBoxes; jTmp++)
      {
#if NO_OPT
         int jBox = s->boxes->nbrBoxes[t_idx.tile[0]][jTmp];
#else
         int jBox = s->boxes->nbrBoxes[box_id][jTmp];
#endif
        
		 if(jBox < 0)
			return;
         //assert(jBox>=0);
         
         int nJBox = s->boxes->nAtoms[jBox];
		
#if NO_OPT
		 int iOff = t_idx.tile[0] * MAXATOMS;
		 int ii = t_idx.local[0];
#else
		 int iOff = box_id * MAXATOMS;
		 int ii = ii_local % max_atoms_in_box;
#endif
         // loop over atoms in iBox

		 if(ii < nIBox)
         {
            // loop over atoms in jBox
            for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++)
            {
               real_t dr[3];
               real_t r2 = 0.0;
               for (int m=0; m<3; m++)
               {
                  dr[m] = s->atoms->r[iOff + ii][m]-s->atoms->r[jOff][m];
                  r2+=dr[m]*dr[m];
               }

               if ( r2 > rCut2 || r2 <= 0.0) continue;

               // Important note:
               // from this point on r actually refers to 1.0/r
               r2 = 1.0/r2;
               real_t r6 = s6 * (r2*r2*r2);
               real_t eLocal = r6 * (r6 - 1.0) - eShift;
               s->atoms->U[iOff + ii] += 0.5*eLocal;

               // different formulation to avoid sqrt computation
               real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
               for (int m=0; m<3; m++)
               {
                  s->atoms->f[iOff + ii][m] -= dr[m]*fr;
               }
            } // loop over atoms in jBox
         } // loop over atoms in iBox
      } // loop over neighbor boxes
   } // loop over local boxes in system
	);

	/* A loop over all the atoms is requrired to reduce the ePot value.
	 * Otherwise, update to ePot is required to be atomic which will 
	 * lead to slow performance. 
	 */
	for (int iBox=0; iBox<s->boxes->nLocalBoxes; iBox++)
	{
		for(int iAtom = 0; iAtom < s->boxes->nAtoms[iBox]; iAtom++)
		{
			int iOff = iBox * MAXATOMS + iAtom;
			ePot += s->atoms->U[iOff];
		}
	}
   ePot = ePot*4.0*epsilon;
   s->ePotential = ePot;

   return 0;
}
