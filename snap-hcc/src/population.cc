
#include "population.h"

void calculate_population(
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct memory * memory,
    double *population
    )
{
    const double volume = problem->dx * problem->dy * problem->dz;
    double total = 0.0;
    for (unsigned int k = 0; k < rankinfo->nz; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
                for (unsigned int g = 0; g < problem->ng; g++)
                {
                    total += volume *
                        memory->scalar_flux[SCALAR_FLUX_INDEX(g,i,j,k,problem->ng,rankinfo->nx,rankinfo->ny)];
                }
    int err = MPI_Reduce(&total, population, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    check_mpi(err, "Population reduction");
}
