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


int scan16(local int *scratch, int lid)
{
	int offset = 1;

	int sum = 0;
	for(int d = 4096>>4; d > 0; d >>= 4)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if(lid < d)
		{
			int ai = offset * (16*lid + 1) - 1;
			int bi = offset * (16*lid + 2) - 1;
			int ci = offset * (16*lid + 3) - 1;
			int di = offset * (16*lid + 4) - 1;
			int ei = offset * (16*lid + 5) - 1;
			int fi = offset * (16*lid + 6) - 1;
			int gi = offset * (16*lid + 7) - 1;
			int hi = offset * (16*lid + 8) - 1;
			int ii = offset * (16*lid + 9) - 1;
			int ji = offset * (16*lid + 10) - 1;
			int ki = offset * (16*lid + 11) - 1;
			int li = offset * (16*lid + 12) - 1;
			int mi = offset * (16*lid + 13) - 1;
			int ni = offset * (16*lid + 14) - 1;
			int oi = offset * (16*lid + 15) - 1;
			int pi = offset * (16*lid + 16) - 1;

			scratch[pi] += scratch[ai] + scratch[bi] + scratch[ci] + scratch[di]
							+ scratch[ei] + scratch[fi] + scratch[gi] +
							scratch[hi] + scratch[ii] + scratch[ji] + scratch[ki] + 
							+ scratch[li] + scratch[mi] + scratch[ni] +  scratch[oi] ;
		}
		offset <<= 4;
	}
	

	if(lid == 0) 
	{
		sum = scratch[4096-1];
		scratch[4096-1] = 0;
	}

	for(int d = 1; d < 4096; d <<= 4)
	{
		offset >>= 4;
		barrier(CLK_LOCAL_MEM_FENCE);

		if(lid < d)
		{
			int ai = offset * (16*lid + 1) - 1;
			int bi = offset * (16*lid + 2) - 1;
			int ci = offset * (16*lid + 3) - 1;
			int di = offset * (16*lid + 4) - 1;
			int ei = offset * (16*lid + 5) - 1;
			int fi = offset * (16*lid + 6) - 1;
			int gi = offset * (16*lid + 7) - 1;
			int hi = offset * (16*lid + 8) - 1;
			int ii = offset * (16*lid + 9) - 1;
			int ji = offset * (16*lid + 10) - 1;
			int ki = offset * (16*lid + 11) - 1;
			int li = offset * (16*lid + 12) - 1;
			int mi = offset * (16*lid + 13) - 1;
			int ni = offset * (16*lid + 14) - 1;
			int oi = offset * (16*lid + 15) - 1;
			int pi = offset * (16*lid + 16) - 1;

			int t1 = scratch[ai] + scratch[pi];
			int t2 = scratch[bi];
			int t3 = scratch[ci];
			int t4 = scratch[di];
			int t5 = scratch[ei];
			int t6 = scratch[fi];
			int t7 = scratch[gi];
			int t8 = scratch[hi];
			int t9 = scratch[ii];
			int t10 = scratch[ji];
			int t11 = scratch[ki];
			int t12 = scratch[li];
			int t13 = scratch[mi];
			int t14 = scratch[ni];
			int t15 = scratch[oi];

			scratch[ai] = scratch[pi];
			scratch[bi] = t1;
			scratch[ci] = t1 + t2;
			scratch[di] = t1 + t2 + t3;
			scratch[ei] = t1 + t2 + t3 + t4;
			scratch[fi] = t1 + t2 + t3 + t4 + t5;
			scratch[gi] = t1 + t2 + t3 + t4 + t5 + t6;
			scratch[hi] = t1 + t2 + t3 + t4 + t5 + t6 + t7;
			scratch[ii] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;
			scratch[ji] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9;
			scratch[ki] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10;
			scratch[li] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11;
			scratch[mi] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12;
			scratch[ni] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13;
			scratch[oi] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14;
			scratch[pi] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
		}
	}

	return sum;
}

kernel void pfxSumBoxes ( global int *boxes, 
								int length,
								local int *scratch
							)
{

	int tid = get_global_id(0);
	int lid = get_local_id(0);

	// 256 work items can do 4096 elems
	int size = (length < 4096) ? 256 : (length/4096 + 1) * 256;
	//int size = (length < 1024) ? 256 : (length/1024 + 1) * 256;

	int sum = 0, sum_p = 0; 
	local int sum_0[1];
	sum_0[0] = 0;

	tid = get_global_id(0);

	while (tid < size)
	{
		
		for(int i = 0; i < 16; i++)
		{
			scratch[16 * lid + i] =  0;
			if(16 * tid + i < length)
				scratch[16 * lid + i] = boxes[16 * tid + i];
		}
		
#if 0
		for(int i = 0; i < 4; i++)
		{
			scratch[4 * lid + i] =  0;
			if(4 * tid + i < length)
				scratch[4 * lid + i] = hist[4 * tid + i];
		}
#endif
		sum = scan16(scratch, lid);
		//sum = scan4(scratch, lid);
		barrier(CLK_LOCAL_MEM_FENCE);

		if(lid == 0)
			sum_0[0] += sum_p;
		
		sum_p = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		
		for(int i = 0; i < 16; i++)
		{
			if(16 * tid + i < length)
				boxes[16 * tid + i] = sum_0[0] + scratch[16 * lid + i];
		}
		
#if 0

		for(int i = 0; i < 4; i++)
		{
			if(4 * tid + i < length)
				hist[4 * tid + i] = sum_0[0] + scratch[4 * lid + i];
		}
#endif

		tid += 256;
	}
}

