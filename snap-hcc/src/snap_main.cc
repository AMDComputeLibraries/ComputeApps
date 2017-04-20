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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#include "global.h"
#include "comms.h"
#include "input.h"
#include "problem.h"
#include "allocate.h"
#include "source.h"
#include "sweep.h"
#include "scalar_flux.h"
#include "convergence.h"
#include "population.h"
#include "profiler.h"

#include "buffers.h"

double sweep_mpi_time = 0.0;
double sweep_mpi_recv_time = 0.0;

/** \mainpage
* SNAP-MPI is a cut down version of the SNAP mini-app which allows us to
* investigate MPI decomposition schemes with GPU offload for node-level 
* computation.
*
* The MPI scheme used is KBA, expanding into hybrid-KBA.
*/

/** \brief Cartesian communicator */
extern MPI_Comm snap_comms;

/** \brief Print out starting information */
void print_banner(void);

/** \brief Print out the input paramters */
void print_input(struct problem * problem);

/** \brief Print out the timing report */
void print_timing_report(struct timers * timers, struct problem * problem, unsigned int total_iterations);

#define MAX_INFO_STRING 256
#define STARS "********************************************************"

/** \brief Main function, contains iteration loops */
int main(int argc, char **argv)
{
    int mpi_err = MPI_Init(&argc, &argv);
    check_mpi(mpi_err, "MPI_Init");

    struct timers timers;
    zero_timers(&timers);
    timers.setup_time = wtime();

    int rank,size;
    size_t nsize;
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    check_mpi(mpi_err, "Getting MPI rank");

    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    check_mpi(mpi_err, "Getting MPI size");

    struct problem problem;

    if (rank == 0)
    {
        print_banner();

        // Check for two files on CLI
        if (argc != 2)
        {
            fprintf(stderr, "Usage: ./snap snap.in\n");
            exit(EXIT_FAILURE);
        }
        read_input(argv[1], &problem);
        if ((problem.npex * problem.npey * problem.npez) != size)
        {
            fprintf(stderr, "Input error: wanted %d ranks but executing with %d\n", problem.npex*problem.npey*problem.npez, size);
            exit(EXIT_FAILURE);
        }
        check_decomposition(&problem);

    }

    // Set dx, dy, dz, dt values
    problem.dx = problem.lx / (double)problem.nx;
    problem.dy = problem.ly / (double)problem.ny;
    problem.dz = problem.lz / (double)problem.nz;
    problem.dt = problem.tf / (double)problem.nsteps;

    // Broadcast the global variables
    broadcast_problem(&problem, rank);

    // Echo input file to screen
    if (rank == 0)
        print_input(&problem);

    // Set up communication neighbours
    struct rankinfo rankinfo;
    setup_comms(&problem, &rankinfo);

    check_device_memory_requirements(&problem, &rankinfo);

    // Allocate the problem arrays
    struct memory memory;
    allocate_memory(&problem, &rankinfo, &memory);
    // define device arrays
    struct buffers buffers;
#include "hcc_arrays.h"
    // Set up problem
    init_quadrature_weights(&problem, &buffers);
    calculate_cosine_coefficients(&problem, &buffers, memory.mu, memory.eta, memory.xi);
    calculate_scattering_coefficients(&problem, &buffers, memory.mu, memory.eta, memory.xi);
    init_material_data(&problem, &buffers, memory.mat_cross_section);
    init_fixed_source(&problem, &rankinfo, &buffers);
    init_scattering_matrix(&problem, &buffers, memory.mat_cross_section);
    init_velocities(&problem, &buffers);

    struct plane* planes;
    unsigned int num_planes;
    init_planes(&planes, &num_planes, &problem, &rankinfo);
#include "hcc_planes.h"

    // Zero out the angular flux buffers
    nsize = problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
    for (int i = 0;i<8; i++)
    {
        zero_buffer(buffers.angular_flux_in[i], 0, nsize);
        zero_buffer(buffers.angular_flux_out[i], 0, nsize);
    }

    // Zero out the outer source, because later moments are +=
    zero_buffer(buffers.outer_source, 0, problem.cmom*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);

//hcc sync and copy for array view case
    hc::completion_future outer_source_future = buffers.outer_source->get_accelerator_view().create_marker();
    outer_source_future.wait();

    if (rankinfo.rank == 0)
        timers.setup_time = wtime() - timers.setup_time;

    bool innerdone, outerdone;

    // Timers
    if (rankinfo.rank == 0)
        timers.simulation_time = wtime();

    if (rankinfo.rank == 0)
    {
        printf("%s\n", STARS);
        printf("  Iteration Monitor\n");
        printf("%s\n", STARS);
    }

    unsigned int total_iterations = 0;


    //----------------------------------------------
    // Timestep loop
    //----------------------------------------------
    for (unsigned int t = 0; t < problem.nsteps; t++)
    {
        unsigned int outer_iterations = 0;
        unsigned int inner_iterations = 0;
        if (rankinfo.rank == 0)
        {
            printf(" Timestep %d\n", t);
            printf("   %-10s %-15s %-10s\n", "Outer", "Difference", "Inners");
        }

        // Zero out the scalar flux and flux moments
        zero_buffer(buffers.scalar_flux, 0, problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        if (problem.cmom-1 > 0)
            zero_buffer(buffers.scalar_flux_moments, 0, (problem.cmom-1)*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);

//hcc sync and copy for array view case
#if ARRAY_VIEW
    buffers.scalar_flux_moments.synchronize();
#else
    hc::completion_future sfm_future = buffers.scalar_flux_moments->get_accelerator_view().create_marker();
    sfm_future.wait();
#endif

        // Swap angluar flux pointers (not for the first timestep)
        if (t > 0)
            swap_angular_flux_buffers(&buffers);

        //----------------------------------------------
        // Outers
        //----------------------------------------------
        for (unsigned int o = 0; o < problem.oitm; o++)
        {
            init_velocity_delta(&problem, &buffers);
            calculate_dd_coefficients(&problem, &buffers);
            calculate_denominator(&problem, &rankinfo, &buffers);

            compute_outer_source(&problem, &rankinfo, &buffers);

            // Get the scalar flux back
            copy_back_scalar_flux(&problem, &rankinfo, &buffers, memory.old_outer_scalar_flux, false);

            //----------------------------------------------
            // Inners
            //----------------------------------------------
            inner_iterations += problem.ng;
            unsigned int i;
            for (i = 0; i < problem.iitm; i++)
            {
                compute_inner_source(&problem, &rankinfo, &buffers);

                // Get the scalar flux back
                copy_back_scalar_flux(&problem, &rankinfo, &buffers, memory.old_inner_scalar_flux, false);
                hc::completion_future inner_copy_event = 
                   buffers.scalar_flux->get_accelerator_view().create_marker();

                double sweep_tick;
                if (profiling && rankinfo.rank == 0)
                {
                    // We must wait for the transfer to finish before we enqueue the next transfer,
                    // or MPI_Recv to get accurate timings
                    sweep_tick = wtime();
                }

                // Sweep each octant in turn
                int octant = 0;
                for (int istep = -1; istep < 2; istep += 2)
                    for (int jstep = -1; jstep < 2; jstep += 2)
                        for (int kstep = -1; kstep < 2; kstep += 2)
                        {
                            // Zero the z buffer every octant - we just do KBA
                            zero_buffer(buffers.flux_k, 0, problem.nang*problem.ng*rankinfo.nx*rankinfo.ny);


                            for (unsigned int z_pos = 0; z_pos < rankinfo.nz; z_pos += problem.chunk)
                            {
                                double tick = wtime();
                                recv_boundaries(z_pos, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &buffers);
                                sweep_mpi_recv_time += wtime() - tick;
                                for (unsigned int p = 0; p < num_planes; p++)
                                {
                                    sweep_plane(z_pos, octant, istep, jstep, kstep, p, planes, &problem, &rankinfo, &buffers);
                                }
                                send_boundaries(z_pos, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &buffers);
                            }

                            if (profiling && rankinfo.rank == 0)
                                chunk_profiler(&timers);

                            octant += 1;
                        }

                if (profiling && rankinfo.rank == 0)
                {
                    // The last send boundaries is either a blocking read of blocking MPI_Send,
                    // so we know everything in the queue is done
                    timers.sweep_time += wtime() - sweep_tick;
                }

                // Put barrier on compute queue so we don't update the scalar flux until we know it's back on the host
                inner_copy_event.wait();

                // Compute the Scalar Flux
                compute_scalar_flux(&problem, &rankinfo,  &buffers);
                if (problem.cmom-1 > 0)
                    compute_scalar_flux_moments(&problem, &rankinfo, &buffers);

                // Put a marker on the compute queue
                hc::completion_future scalar_compute_event= 
                   buffers.scalar_flux->get_accelerator_view().create_marker();
                scalar_compute_event.wait();

                // Get the new scalar flux back and check inner convergence
                copy_back_scalar_flux(&problem, &rankinfo, &buffers, memory.scalar_flux, true);


                double conv_tick = wtime();

                int inners_left = inner_convergence(&problem, &rankinfo, &memory);
                innerdone = inners_left?false:true;
                inner_iterations += inners_left;
                if (profiling && rankinfo.rank == 0)
                    timers.convergence_time += wtime() - conv_tick;

                // Do any profiler updates for timings
                if (rankinfo.rank == 0)
                    inner_profiler(&timers, &problem);

                if (innerdone)
                {
                    i += 1;
                    break;
                }
            }
            //----------------------------------------------
            // End of Inners
            //----------------------------------------------

            // Check outer convergence
            // We don't need to copy back the new scalar flux again as it won't have changed from the last inner
            double max_outer_diff;
            double conv_tick = wtime();
            outerdone = outer_convergence(&problem, &rankinfo, &memory, &max_outer_diff) && innerdone;

            if (profiling && rankinfo.rank == 0)
                timers.convergence_time += wtime() - conv_tick;

            total_iterations += i;
            outer_iterations++;

            if (rankinfo.rank == 0)
                printf("     %-9u %-15.4e %-10u\n", o, max_outer_diff, i);

            // Do any profiler updates for timings
            if (rankinfo.rank == 0)
                outer_profiler(&timers);

            if (outerdone)
                break;

        }
        //----------------------------------------------
        // End of Outers
        //----------------------------------------------

        // Exit the time loop early if outer not converged
        if (!outerdone)
        {
            if (rankinfo.rank == 0)
                printf(" * Stopping because not converged *\n");
            break;
        }

        // Print loop statistics for comparison purposes
        if (rankinfo.rank == 0)
        {
            printf("\n");
            printf("  Timestep= %4d   No. Outers= %4d    No. Inners= %4d\n"
                   ,t,outer_iterations,inner_iterations);
        }

        // Calculate particle population and print out the value
        double population;
        calculate_population(&problem, &rankinfo, &memory, &population);
        if (rankinfo.rank == 0)
        {
            // Get exponent of outer convergence criteria
            int places;
            frexp(100.0 * problem.epsi, &places);
            places = ceil(fabs(places / log2(10)));
            char format[100];
            sprintf(format, "   Population: %%.%dlf\n", places);
            printf("\n");
            printf(format, population);
            printf("\n");
        }

    }
    //----------------------------------------------
    // End of Timestep
    //----------------------------------------------

    hc::completion_future scalar_compute_event= 
        buffers.scalar_flux->get_accelerator_view().create_marker();
    scalar_compute_event.wait();

    if (rankinfo.rank == 0)
    {
        timers.simulation_time = wtime() - timers.simulation_time;

        print_timing_report(&timers, &problem, total_iterations);
    }

    free_memory(&memory);

    finish_comms();

    return EXIT_SUCCESS;
}

void print_banner(void)
{
    printf("\n");
    printf(" SNAP: SN (Discrete Ordinates) Application Proxy\n");
    printf(" MPI+HCC port\n");
    time_t rawtime;
    struct tm * timeinfo;
    char timestring[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestring, 80, "%c", timeinfo);
    printf(" Run on %s\n", timestring);
    printf("\n");
}

void print_input(struct problem * problem)
{
    printf("\n%s\n", STARS);
    printf(  "  Input Parameters\n");
    printf(  "%s\n", STARS);

    printf(" Geometry\n");
    printf("   %-30s %.3lf x %.3lf x %.3lf\n", "Problem size:", problem->lx, problem->ly, problem->lz);
    printf("   %-30s %5u x %5u x %5u\n", "Cells:", problem->nx, problem->ny, problem->nz);
    printf("   %-30s %.3lf x %.3lf x %.3lf\n", "Cell size:", problem->dx, problem->dy, problem->dz);
    printf("\n");

    printf(" Discrete Ordinates\n");
    printf("   %-30s %u\n", "Angles per octant:", problem->nang);
    printf("   %-30s %u\n", "Moments:", problem->nmom);
    printf("   %-30s %u\n", "\"Computational\" moments:", problem->cmom);
    printf("\n");

    printf(" Energy groups\n");
    printf("   %-30s %u\n", "Number of groups:", problem->ng);
    printf("\n");

    printf(" Timesteps\n");
    printf("   %-30s %u\n", "Timesteps:", problem->nsteps);
    printf("   %-30s %.3lf\n", "Simulation time:", problem->tf);
    printf("   %-30s %.3lf\n", "Time delta:", problem->dt);
    printf("\n");

    printf(" Iterations\n");
    printf("   %-30s %u\n", "Max outers per timestep:", problem->oitm);
    printf("   %-30s %u\n", "Max inners per outer:", problem->iitm);

    printf("   Stopping criteria\n");
    printf("     %-28s %.2E\n", "Inner convergence:", problem->epsi);
    printf("     %-28s %.2E\n", "Outer convergence:", 100.0*problem->epsi);
    printf("\n");

    printf(" MPI decomposition\n");
    printf("   %-30s %u x %u x %u\n", "Rank layout:", problem->npex, problem->npey, problem->npez);
    printf("   %-30s %u\n", "Chunk size:", problem->chunk);
    printf("\n");

}


void print_timing_report(struct timers * timers, struct problem * problem, unsigned int total_iterations)
{
    printf("\n%s\n", STARS);
    printf(  "  Timing Report\n");
    printf(  "%s\n", STARS);

    printf(" %-30s %6.3lfs\n", "Setup", timers->setup_time);
    if (profiling)
    {
        printf(" %-30s %6.3lfs\n", "Outer source", timers->outer_source_time);
        printf(" %-30s %6.3lfs\n", "Outer parameters", timers->outer_params_time);
        printf(" %-30s %6.3lfs\n", "Inner source", timers->inner_source_time);
        printf(" %-30s %6.3lfs\n", "Sweeps", timers->sweep_time);
        printf("   %-28s %6.3lfs\n", "MPI Send time", sweep_mpi_time);
        printf("   %-28s %6.3lfs\n", "MPI Recv time", sweep_mpi_recv_time);
        printf("   %-28s %6.3lfs\n", "PCIe transfer time", timers->sweep_transfer_time);
        printf("   %-28s %6.3lfs\n", "Compute time", timers->sweep_time-sweep_mpi_time-sweep_mpi_recv_time-timers->sweep_transfer_time);
        printf(" %-30s %6.3lfs\n", "Scalar flux reductions", timers->reduction_time);
        printf(" %-30s %6.3lfs\n", "Convergence checking", timers->convergence_time);
        printf(" %-30s %6.3lfs\n", "Other", timers->simulation_time - timers->outer_source_time - timers->outer_params_time - timers->inner_source_time - timers->sweep_time - timers->reduction_time - timers->convergence_time);
    }
        printf(" %-30s %6.3lfs\n", "Total simulation", timers->simulation_time);

        printf("\n");
        printf(" %-30s %6.3lfns\n", "Grind time",
            1.0E9 * timers->simulation_time /
            (double)(problem->nx*problem->ny*problem->nz*problem->nang*8*problem->ng*total_iterations)
            );

        printf( "%s\n", STARS);

}

