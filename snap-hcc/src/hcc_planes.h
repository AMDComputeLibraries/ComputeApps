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

/** \file
* \brief Instantiate hcc plane array objects
*/

/** \brief 
This file included only once in main.
It instantiates hc::array plane objects for the life of the program.
*/
    buffers.planes = (GPU_VEC(struct cell_id) **)  malloc(sizeof(GPU_VEC(struct cell_id)*)*num_planes);

    for (unsigned int p = 0; p < num_planes; p++)
    {
        std::vector<struct cell_id> vcell_ids (planes[p].num_cells);

        for (int j=0;j<planes[p].num_cells;j++)
        {
           vcell_ids[j].i = planes[p].cell_ids[j].i;
           vcell_ids[j].j = planes[p].cell_ids[j].j;
           vcell_ids[j].k = planes[p].cell_ids[j].k;
        }
        buffers.planes[p] =  new GPU_VEC(struct cell_id) (planes[p].num_cells);
        copy(vcell_ids.data(),*buffers.planes[p]);
    }
