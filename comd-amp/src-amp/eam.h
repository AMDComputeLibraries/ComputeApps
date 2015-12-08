
/// \file
/// Compute forces for the Embedded Atom Model (EAM).

#ifndef __EAM_H
#define __EAM_H

#include "mytype.h"
#include "CoMDTypes.h"

struct BasePotentialSt;
struct LinkCellSt;

/// Pointers to the data that is needed in the load and unload functions
/// for the force halo exchange.
/// \see loadForceBuffer
/// \see unloadForceBuffer
typedef struct ForceExchangeDataSt
{
   real_t* dfEmbed; //<! derivative of embedding energy
   struct LinkCellSt* boxes;
}ForceExchangeData;

/// Handles interpolation of tabular data.
///
/// \see initInterpolationObject
/// \see interpolate
typedef struct InterpolationObjectSt 
{
   int n;          //!< the number of values in the table
   real_t x0;      //!< the starting ordinate range
   real_t invDx;   //!< the inverse of the table spacing
   real_t* values; //!< the abscissa values
} InterpolationObject;

/// Derived struct for an EAM potential.
/// Uses table lookups for function evaluation.
/// Polymorphic with BasePotential.
/// \see BasePotential
typedef struct EamPotentialSt 
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t mass;            //!< mass of atoms in intenal units
   real_t lat;             //!< lattice spacing (angs) of unit cell
   char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
   char  name[3];	   //!< element name
   int	 atomicNo;	   //!< atomic number  
   int  (*force)(SimFlat* s); //!< function pointer to force routine
   void (*print)(FILE* file, BasePotential* pot);
   void (*destroy)(BasePotential** pot); //!< destruction of the potential
   InterpolationObject* phi;  //!< Pair energy
   InterpolationObject* rho;  //!< Electron Density
   InterpolationObject* f;    //!< Embedding Energy

   real_t* rhobar;        //!< per atom storage for rhobar
   real_t* dfEmbed;       //!< per atom storage for derivative of Embedding
   HaloExchange* forceExchange;
   ForceExchangeData* forceExchangeData;
} EamPotential;

struct BasePotentialSt* initEamPot(const char* dir, const char* file, const char* type);
#endif
