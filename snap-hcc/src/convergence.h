
#pragma once

/** \file
* \brief Convergence checking routines (computed on the host)
*/

#include <stdbool.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "comms.h"
#include "problem.h"
#include "allocate.h"

/** \brief Check inner convergence - requires MPI_Allreduce*/
int inner_convergence(const struct problem * problem, const struct rankinfo * rankinfo, const struct memory * memory);

/** \brief Check outer convergence, saving the max difference - requires MPI_Allreduce */
bool outer_convergence(const struct problem * problem, const struct rankinfo * rankinfo, const struct memory * memory, double * max_diff);
