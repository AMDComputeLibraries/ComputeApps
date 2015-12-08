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
v3.0
OpenCL with bitonic sort.  A sorted energy grid save/restore feature is added
to support faster initialization.
This version is also a code clean of v2.0, and potentially confusing HSA 
ifdefs have been removed.

v2.0
This version is an OpenCL implementation using a bitonic sort method to
rearrage the input data for better performance.  The code has HSA defines and
some files, but these are not tested in this version.
The testgrid program is removed in this version.

v1.0
This version of XSBench uses OpenCL and is based off of the AMD struct of
arrays version.

The original XSBench code uses arrays of structurs to hold the nuclide grid data
and the unionized grid data. This version changes to a structure of arrays approach.
Because of this, sorting was moved from qsort to a simple sort algorithm which is currently
slower than qsort. 

Included with this code is a simple test code that tests the grid creation and sorting.
It could be used to perform one or two particle iterations. Read the comments in the code. 
The test code has variables to specify the number of isotopes and the number of gridpoints. 
The Materials.c code assumes two models that hard-code array sizes. Setting the number of 
isotopes to something different in the test code will likely result in a core dump due to 
mismatch of array sizes.


