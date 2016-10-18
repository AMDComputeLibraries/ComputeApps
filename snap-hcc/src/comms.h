
#pragma once

/** \file
* \brief Communication routines (setup and sweep data transfer)
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "global.h"
#include "allocate.h"
#include "buffers.h"

#include "profiler.h"


/** \brief Check MPI error codes */
void check_mpi(const int err, const char *msg);

/** \brief Setup cartesian communicator and find your MPI rank */
void setup_comms(struct problem * problem, struct rankinfo * rankinfo);

/** \brief Just call MPI_Finalize */
void finish_comms(void);

/** \brief Discover the ranks of your neighbours */
void calculate_neighbours(MPI_Comm comms,  struct problem * problem, struct rankinfo * rankinfo);

/** \brief Receive chunk number of XY planes starting at position z_pos */
void recv_boundaries(int z_pos, const int octant, const int istep, const int jstep, const int kstep, struct problem * problem, struct rankinfo * rankinfo, struct memory * memory, struct buffers * buffers);
/** \brief Send chunk number of XY planes starting at position z_pos */
void send_boundaries(int z_pos, const int octant, const int istep, const int jstep, const int kstep, struct problem * problem, struct rankinfo * rankinfo, struct memory * memory, struct buffers * buffers);
