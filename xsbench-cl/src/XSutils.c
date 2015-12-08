/*******************************************************************************
Copyright (c) 2015 Advanced Micro Devices, Inc. 

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
#include "XSbench_header.h"

// Allocates nuclide matrix
NuclideGridPoint ** gpmatrix(size_t m, size_t n)
{
	int i,j;
	NuclideGridPoint * full = (NuclideGridPoint *) malloc( m * n *
	                          sizeof( NuclideGridPoint ) );
	NuclideGridPoint ** M = (NuclideGridPoint **) malloc( m *
	                          sizeof(NuclideGridPoint *) );

	for( i = 0, j=0; i < m*n; i++ )
		if( i % n == 0 )
			M[j++] = &full[i];

	return M;
}

// Frees nuclide matrix
void gpmatrix_free( NuclideGridPoint ** M )
{
	if (M && *M)
	{
	   free( *M );
	}
	if ( M )
	{
	   free( M );
	}
}

// Compare function for two grid points. Used for sorting during init
int NGP_compare( const void * a, const void * b )
{
	NuclideGridPoint *i, *j;

	i = (NuclideGridPoint *) a;
	j = (NuclideGridPoint *) b;

	if( i->energy > j->energy )
		return 1;
	else if ( i->energy < j->energy)
		return -1;
	else
		return 0;
}

// Prints program logo
void logo(int version)
{
	border_print();
	printf(
	"                   __   __ ___________                 _                        \n"
	"                   \\ \\ / //  ___| ___ \\               | |                       \n"
	"                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
	"                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
	"                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
	"                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n"
    );
	border_print();
	center_print("Developed at Argonne National Laboratory", 79);
	char v[100];
	sprintf(v, "Version: %d", version);
	center_print(v, 79);
	border_print();
}

// Prints Section titles in center of 80 char terminal
void center_print(const char *s, int width)
{
	int length = strlen(s);
	int i;
	for (i=0; i<=(width-length)/2; i++) {
		fputs(" ", stdout);
	}
	fputs(s, stdout);
	fputs("\n", stdout);
}

void border_print(void)
{
	printf(
	"==================================================================="
	"=============\n");
}

// Prints comma separated integers - for ease of reading
void fancy_int( unsigned long a )
{
    if( a < 1000 )
        printf("%d\n",a);

    else if( a >= 1000 && a < 1000000 )
        printf("%d,%03d\n", a / 1000, a % 1000);

    else if( a >= 1000000 && a < 1000000000 )
        printf("%d,%03d,%03d\n", a / 1000000, (a % 1000000) / 1000, a % 1000 );

    else if( a >= 1000000000 )
        printf("%u,%03u,%03u,%03u\n",
               a / 1000000000,
               (a % 1000000000) / 1000000,
               (a % 1000000) / 1000,
               a % 1000 );
    else
        printf("%u\n",a);
}

// Binary Search function for nuclide grid
// Returns ptr to energy less than the quarry that is closest to the quarry
int binary_search( NuclideGridPoint * A, double quarry, int n )
{
	int min = 0;
	int max = n-1;
	int mid;
	
	// checks to ensure we're not reading off the end of the grid
	if( A[0].energy > quarry )
		return 0;
	else if( A[n-1].energy < quarry )
		return n-2;
	
	// Begins binary search	
	while( max >= min )
	{
		mid = min + floor( (max-min) / 2.0);
		if( A[mid].energy < quarry )
			min = mid+1;
		else if( A[mid].energy > quarry )
			max = mid-1;
		else
			return mid;
	}
	return max;
}

// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn(unsigned long * seed)
{
	double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (*seed) ) % m;
	*seed = n1;
	ret = (double) n1 / m;
	return ret;
}

Inputs read_CLI( int argc, char * argv[] )
{
	Inputs input;

	memset(&input, 0, sizeof(Inputs));
	
	// defaults to max threads on the system	
	input.nthreads = omp_get_num_procs();
	
	// defaults to 355 (corresponding to H-M Large benchmark)
	input.n_isotopes = 355;
	
	// defaults to 11303 (corresponding to H-M Large benchmark)
	input.n_gridpoints = 11303;
	
	// defaults to 15,000,000
	input.lookups = 15000000;
	
	// defaults to H-M Large benchmark
	input.HM = (char *) malloc( 6 * sizeof(char) );
	strcpy(input.HM, "small");
	/*
	input.HM[0] = 'l' ; 
	input.HM[1] = 'a' ; 
	input.HM[2] = 'r' ; 
	input.HM[3] = 'g' ; 
	input.HM[4] = 'e' ; 
	input.HM[5] = '\0';
	*/
	
#ifdef __USE_AMD_OCL__
	input.tloops = 1;
	input.run_cpu = false;
#endif
        input.savegrids = false;
        input.restoregrids = false;
        strcpy(input.file_name, "grids");

	// Check if user sets these
	int user_g = 0;
	
	// Collect Raw Input
	for( int i = 1; i < argc; i++ )
	{
		char * arg = argv[i];

		// nthreads (-t)
		if( strcmp(arg, "-t") == 0 )
		{
			if( ++i < argc )
				input.nthreads = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// n_gridpoints (-g)
		else if( strcmp(arg, "-g") == 0 )
		{	
			if( ++i < argc )
			{
				user_g = 1;
				input.n_gridpoints = atoi(argv[i]);
			}
			else
				print_CLI_error();
		}
		// lookups (-l)
		else if( strcmp(arg, "-l") == 0 )
		{
			if( ++i < argc )
				input.lookups = atoi(argv[i]);
			else
				print_CLI_error();
		}
		// HM (-s)
		else if( strcmp(arg, "-s") == 0 )
		{	
			if( ++i < argc )
				input.HM = argv[i];
			else
				print_CLI_error();
		}
#ifdef __USE_AMD_OCL__
		else if( strcmp(arg, "-k") == 0 )
		{	
			if( ++i < argc )
				input.tloops = atoi(argv[i]);
			else
				print_CLI_error();
		}
		else if( strcmp(arg, "-c") == 0 )
		{
			input.run_cpu = true;
		}

#endif
                else if( strcmp(arg, "-v") == 0 )
                {
                        input.savegrids = true;
                }
                else if( strcmp(arg, "-r") == 0 )
                {
                        input.restoregrids = true;
                }
                else if( strcmp(arg, "-f") == 0 )
                {
			if( ++i < argc )
				strcpy(input.file_name,argv[i]);
			else
				print_CLI_error();
                }
                else
			print_CLI_error();
	}
       if(input.savegrids)
               input.restoregrids = false;

	// Validate Input

	// Validate nthreads
	if( input.nthreads < 1 )
		print_CLI_error();
	
	// Validate n_isotopes
	if( input.n_isotopes < 1 )
		print_CLI_error();
	
	// Validate n_gridpoints
	if( input.n_gridpoints < 1 )
		print_CLI_error();

	// Validate lookups
	if( input.lookups < 1 )
		print_CLI_error();
	
	// Validate HM size
	if( strcasecmp(input.HM, "small") != 0 &&
		strcasecmp(input.HM, "large") != 0 &&
		strcasecmp(input.HM, "XL") != 0 &&
		strcasecmp(input.HM, "XXL") != 0 )
		print_CLI_error();
	
	// Set HM size specific parameters
	// (defaults to large)
	if( strcasecmp(input.HM, "small") == 0 )
		input.n_isotopes = 68;
	else if( strcasecmp(input.HM, "XL") == 0 && user_g == 0 )
		input.n_gridpoints = 238847; // sized to make 120 GB XS data
	else if( strcasecmp(input.HM, "XXL") == 0 && user_g == 0 )
		input.n_gridpoints = 238847 * 2.1; // 252 GB XS data

	// Return input struct
	return input;
}

void print_CLI_error(void)
{
	printf("Usage: ./XSBench <options>\n");
	printf("Options include:\n");
	printf("  -t <threads>     Number of OpenMP threads to run\n");
	printf("  -s <size>        Size of H-M Benchmark to run (small, large, XL, XXL)\n");
	printf("  -g <gridpoints>  Number of gridpoints per nuclide (overrides -s defaults)\n");
	printf("  -l <lookups>     Number of Cross-section (XS) lookups\n");
	printf("Default is equivalent to: -s large -l 15000000\n");
	printf("See readme for full description of default run values\n");
        printf("  -v               Save grids to disk for restore\n");
        printf("  -r               Restore grids from disk\n");
 
	exit(4);
}

// RNG Used for Verification Option.
// This one has a static seed (must be set manually in source).
// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn_v(void)
{
	static unsigned long seed = 1337;
	double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (seed) ) % m;
	seed = n1;
	ret = (double) n1 / m;
	return ret;
}

unsigned int hash(unsigned char *str, int nbins)
{
	unsigned int hash = 5381;
	int c;

	while (c = *str++)
		hash = ((hash << 5) + hash) + c;

	return hash % nbins;
}

unsigned int hash_l(unsigned char *str, int len, int nbins)
{
	unsigned int hash = 5381;
	int c;
	for(int l = 0; l < len;l++)
	{
	    c = *str++;
		hash = ((hash << 5) + hash) + c;
	}
	return hash % nbins;
}
