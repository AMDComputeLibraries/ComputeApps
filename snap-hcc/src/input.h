
#pragma once

/** \file
* \brief Handles reading in problem data from file
*/

/** \brief Define a macro so getline() is declared in stdlib.h */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <mpi.h>

#include "problem.h"

/** \brief Read the input data from a file and populare the problem structure
*
* Note: should only be called by master rank.
* - nx: Number of cells in x direction
* - ny: Number of cells in y direction
* - nz: Number of cells in z direction
* - lx: Physical size in x direction
* - ly: Physical size in y direction
* - lz: Physical size in z direction
* - ng: Number of energy groups
* - nang: Number of angles per octant
* - nmom: Number of moments
* - iitm: Maximum number of inner iterations per outer
* - oitm: Maximum number of outer iterations per timestep
* - nsteps: Number of timesteps
* - tf: Physical time to simulate
* - epsi: Convergence criteria
* - npex: MPI decomposition: number of processors in x direction
* - npey: MPI decomposition: number of processors in y direction
* - npez: MPI decomposition: number of processors in z direction
* - chunk: Number of x-y planes to calculate before communication
* - multigpu: Specifies if there is more than one GPU per physical node (optional)
*/
void read_input(char *file, struct problem *problem);

/** \brief Send problem data from master to all MPI ranks */
void broadcast_problem(struct problem *problem, int rank);

/** \brief Check MPI decomposition is valid */
void check_decomposition(struct problem * input);
