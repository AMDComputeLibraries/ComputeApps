
#pragma once

/** \file
* \brief Calculate particle population
*/

#include "global.h"
#include "problem.h"
#include "allocate.h"
#include "comms.h"

/** \brief Calculate particle population: sum of scalar flux weighted by spatial volume */
void calculate_population(struct problem * problem, struct rankinfo * rankinfo, struct memory * memory, double *population);
