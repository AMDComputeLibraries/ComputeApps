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

void counter_init( int *eventset, int *num_papi_events )
{
	char error_str[PAPI_MAX_STR_LEN];
	//  int events[] = {PAPI_TOT_INS,PAPI_BR_INS,PAPI_SR_INS};
	int events[] = {PAPI_TOT_CYC,PAPI_L3_TCM};
	int stat;

	int thread = omp_get_thread_num();
	if( thread == 0 )
		printf("Initializing PAPI counters...\n");

	*num_papi_events = sizeof(events) / sizeof(int);

	if ((stat = PAPI_thread_init((long unsigned int (*)(void)) omp_get_thread_num)) != PAPI_OK){
		PAPI_perror("PAPI_thread_init");
		exit(1);
	}

	if ( (stat= PAPI_create_eventset(eventset)) != PAPI_OK){
		PAPI_perror("PAPI_create_eventset");
		exit(1);
	}

	for( int i = 0; i < *num_papi_events; i++ ){
		if ((stat=PAPI_add_event(*eventset,events[i])) != PAPI_OK){
			PAPI_perror("PAPI_add_event");
			exit(1);
		}
	}

	if ((stat=PAPI_start(*eventset)) != PAPI_OK){
		PAPI_perror("PAPI_start");
		exit(1);
	}
}

// Stops the papi counters and prints results
void counter_stop( int * eventset, int num_papi_events )
{
	int * events = malloc(num_papi_events * sizeof(int));
	int n = num_papi_events;
	PAPI_list_events( *eventset, events, &n );
	PAPI_event_info_t info;

	long_long * values = malloc( num_papi_events * sizeof(long_long));
	PAPI_stop(*eventset, values);
	int thread = omp_get_thread_num();
	#ifndef HAVE_AMP
	#pragma omp critical (papi)
	#endif
	{
		printf("Thread %d\n", thread);
		for( int i = 0; i < num_papi_events; i++ )
		{
			PAPI_get_event_info(events[i], &info);
			printf("%-15lld\t%s\t%s\n", values[i],info.symbol,info.long_descr);
		}
		free(events);
		free(values);	
	}
}
