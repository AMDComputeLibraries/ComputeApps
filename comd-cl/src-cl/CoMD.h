#ifndef COMD_H
#define COMD_H
#include "CoMDTypes.h"
#include "mycommand.h"

static SimFlat* initSimulation(Command cmd);
static void destroySimulation(SimFlat** ps);

static void initSubsystems(void);
static void finalizeSubsystems(void);

static BasePotential* initPotential(
   int doeam, const char* potDir, const char* potName, const char* potType);
static SpeciesData* initSpecies(BasePotential* pot);
static Validate* initValidate(SimFlat* s);
static void validateResult(const Validate* val, SimFlat *sim);

static void sumAtoms(SimFlat* s);
static void printThings(SimFlat* s, int iStep, double elapsedTime);
static void printSimulationDataYaml(FILE* file, SimFlat* s);
static void sanityChecks(Command cmd, double cutoff, double latticeConst, char latticeType[8]);

#endif
