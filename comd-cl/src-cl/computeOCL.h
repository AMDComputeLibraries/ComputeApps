#ifndef __PMD_H_
#define __PMD_H_

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "helpers.h"
#include "mytype.h"
#include "mycommand.h"
#include "CoMDTypes.h"
#include "ljForce.h"
#include "decomposition.h"
#include "constants.h"

#define DEBUGLEVEL 0
#define PMDDEBUGPRINTF(xxx,...) {if(xxx>DEBUGLEVEL) printf(__VA_ARGS__);}
#define fPMDDEBUGPRINTF(xxx,...) {if(xxx>DEBUGLEVEL) fprintf(__VA_ARGS__);}


HostSimAos hostSimAos;
DevSimAos devSimAos;

double ts, te;

OclSimSoa* initOclSimSoa(SimFlat* sim, Command cmd);

void computeIterationSoa2(SimFlat* sim, OclSimSoa* oclSim);

void computeIterationSoa(SimFlat* sim, HostSimSoa* hostSim, DevSimSoa* devSim);

void computeIterationAos(SimFlat* sim);

void finishOclSoa(SimFlat* sim, HostSimSoa* hostSim, DevSimSoa* devSim);

void finishOclAos();

void computeInitSoa(SimFlat* sim, HostSimSoa* hostSim, DevSimSoa* devSim);

void computeInitAos(SimFlat* sim);

#endif
