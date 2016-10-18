
#include "input.h"

void read_input(char *file, struct problem *problem)
{
    FILE *fp;
    fp = fopen(file, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s\n", file);
        exit(EXIT_FAILURE);
    }
    char *line = NULL;
    size_t read;
    size_t len = 0;

    // Set multigpu to false in case it's not in the file
    problem->multigpu = false;

    // Set default npex to be compatible with fortren input decks
    problem->npex = 1;

    // Read the lines in the file
    while ((read = getline(&line, &len, fp)) != -1)
    {
        // Cycle over whitespace
        int i = 0;
        while (isspace(line[i]))
            i++;

        if (strncmp(line+i, "nx", strlen("nx")) == 0)
        {
            i += strlen("nx");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->nx = atoi(line+i);
        }
        else if (strncmp(line+i, "ny", strlen("ny")) == 0)
        {
            i += strlen("ny");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->ny = atoi(line+i);
        }
        else if (strncmp(line+i, "nz", strlen("nz")) == 0)
        {
            i += strlen("nz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->nz = atoi(line+i);
        }
        else if (strncmp(line+i, "lx", strlen("lx")) == 0)
        {
            i += strlen("lx");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->lx = atof(line+i);
        }
        else if (strncmp(line+i, "ly", strlen("ly")) == 0)
        {
            i += strlen("ly");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->ly = atof(line+i);
        }
        else if (strncmp(line+i, "lz", strlen("lz")) == 0)
        {
            i += strlen("lz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->lz = atof(line+i);
        }
        else if (strncmp(line+i, "ng", strlen("ng")) == 0)
        {
            i += strlen("ng");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->ng = atoi(line+i);
        }
        else if (strncmp(line+i, "nang", strlen("nang")) == 0)
        {
            i += strlen("nang");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->nang = atoi(line+i);
        }
        else if (strncmp(line+i, "nmom", strlen("nmom")) == 0)
        {
            i += strlen("nmom");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->nmom = atoi(line+i);
        }
        else if (strncmp(line+i, "iitm", strlen("iitm")) == 0)
        {
            i += strlen("iitm");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->iitm = atoi(line+i);
        }
        else if (strncmp(line+i, "oitm", strlen("oitm")) == 0)
        {
            i += strlen("oitm");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->oitm = atoi(line+i);
        }
        else if (strncmp(line+i, "nsteps", strlen("nsteps")) == 0)
        {
            i += strlen("nsteps");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->nsteps = atoi(line+i);
        }
        else if (strncmp(line+i, "tf", strlen("tf")) == 0)
        {
            i += strlen("tf");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->tf = atof(line+i);
        }
        else if (strncmp(line+i, "epsi", strlen("epsi")) == 0)
        {
            i += strlen("epsi");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->epsi = atof(line+i);
        }
        else if (strncmp(line+i, "npex", strlen("npex")) == 0)
        {
            i += strlen("npex");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->npex = atoi(line+i);
        }
        else if (strncmp(line+i, "npey", strlen("npey")) == 0)
        {
            i += strlen("npey");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->npey = atoi(line+i);
        }
        else if (strncmp(line+i, "npez", strlen("npez")) == 0)
        {
            i += strlen("npez");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->npez = atoi(line+i);
        }
        else if (strncmp(line+i, "ichunk", strlen("ichunk")) == 0)
        {
            i += strlen("ichunk");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            problem->chunk = atoi(line+i);
        }
        else if (strncmp(line+i, "multigpu", strlen("multigpu")) == 0)
        {
            i += strlen("multigpu");
            problem->multigpu = true;
        }
    }
    free(line);
}

void broadcast_problem(struct problem *problem, int rank)
{
    unsigned int ints[] = {
        problem->nx,
        problem->ny,
        problem->nz,
        problem->ng,
        problem->nang,
        problem->nmom,
        problem->iitm,
        problem->oitm,
        problem->nsteps,
        problem->npex,
        problem->npey,
        problem->npez,
        problem->chunk,
        problem->multigpu
    };
    double doubles[] = {
        problem->lx,
        problem->ly,
        problem->lz,
        problem->dx,
        problem->dy,
        problem->dz,
        problem->dt,
        problem->tf,
        problem->epsi
    };
    MPI_Bcast(ints, 14, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(doubles, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        problem->nx = ints[0];
        problem->ny = ints[1];
        problem->nz = ints[2];
        problem->ng = ints[3];
        problem->nang = ints[4];
        problem->nmom = ints[5];
        problem->iitm = ints[6];
        problem->oitm = ints[7];
        problem->nsteps = ints[8];
        problem->npex = ints[9];
        problem->npey = ints[10];
        problem->npez = ints[11];
        problem->chunk = ints[12];
        problem->multigpu = ints[13];

        problem->lx = doubles[0];
        problem->ly = doubles[1];
        problem->lz = doubles[2];
        problem->dx = doubles[3];
        problem->dy = doubles[4];
        problem->dz = doubles[5];
        problem->dt = doubles[6];
        problem->tf = doubles[7];
        problem->epsi = doubles[8];
    }
    problem->cmom = problem->nmom * problem->nmom;
}

void check_decomposition(struct problem * input)
{
    bool err = false;

    // Check we have at least a 1x1x1 processor array
    if (input->npex < 1)
    {
        fprintf(stderr, "Input error: npex must be >= 1\n");
        err = true;
    }
    if (input->npey < 1)
    {
        fprintf(stderr, "Input error: npey must be >= 1\n");
        err = true;
    }

    // Check npez = 1 (regular KBA for now)
    if (input->npez != 1)
    {
        fprintf(stderr, "Input error: npez must equal 1 (for KBA)\n");
        err = true;
    }

    // Check grid divides across processor array
    if (input->nx % input->npex != 0)
    {
        fprintf(stderr, "Input error: npex should divide nx\n");
        err = true;
    }
    if (input->ny % input->npey != 0)
    {
        fprintf(stderr, "Input error: npey should divide ny\n");
        err = true;
    }
    if (input->nz % input->npez != 0)
    {
        fprintf(stderr, "Input error: npez should divide nz\n");
        err = true;
    }

    // Check chunk size divides nz
    if (input->nz % input->chunk != 0)
    {
        fprintf(stderr, "Input error: chunk should divide nz\n");
        err = true;
    }

    if (err)
        exit(EXIT_FAILURE);
}

