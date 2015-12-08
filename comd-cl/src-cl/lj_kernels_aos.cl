//Initial implementation of the MD code

/** 
  Since OpenCL doesn't pick up #include properly, we need to manually switch real_t from 
  float to double in each kernel file individually.
**/

#define N_MAX_NEIGHBORS 27
#define PERIODIC 1

#define KERN_DIAG 0

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* CL_REAL_T is set to single or double depending on compile time flags */
typedef CL_REAL_T real_t; 
typedef CL_REAL4_T cl_real4; 

// Simple version without local blocking to check for correctness
__kernel void ljForce(
        __global real_t* xPos,
        __global real_t* yPos,
        __global real_t* zPos,
        __global real_t* fx,
        __global real_t* fy,
        __global real_t* fz,
        __global real_t* energy,
        __global real_t* dcx,
        __global real_t* dcy,
        __global real_t* dcz,
        __global real_t* bounds,
        __global int* neighborList,
        __global int* nNeighbors,
        __global int* nAtoms,
        const real_t sigma,
        const real_t epsilon,
        const real_t cutoff) 
{

    int iAtom = get_global_id(0);
    int iBox = get_global_id(1);
    int maxAtoms = get_global_size(0);

    real_t dx, dy, dz;
    real_t r, r2, r6;
    real_t fr, e;

    real_t dxbox, dybox, dzbox;

    // accumulate local force value
    real_t fxItem, fyItem, fzItem;

    real_t rCut = cutoff;
    real_t rCut2 = rCut*rCut;
    real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

    // zero out forces on particles
    fxItem = 0.0;
    fyItem = 0.0;
    fzItem = 0.0;

    e = 0.0;

    int iOffset = iBox*maxAtoms; //N_MAX_ATOMS;
    int iParticle = iOffset + iAtom;

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box

#if(KERN_DIAG) 
        //if (iBox < 2) printf("i = %d, %f, %f, %f\n", iParticle, xPos[iParticle], yPos[iParticle], zPos[iParticle]);

        //printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
#endif

        for (int j = 0; j<nNeighbors[iBox]; j++)
	{// loop over neighbor cells
            int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
            int jOffset = jBox*maxAtoms; //N_MAX_ATOMS;

            // compute box center offsets
            dxbox = dcx[iBox] - dcx[jBox];
            dybox = dcy[iBox] - dcy[jBox];
            dzbox = dcz[iBox] - dcz[jBox];

            // correct for periodic 
            if(PERIODIC)
	    {
                if (dxbox<-0.5*bounds[0]) dxbox += bounds[0];
                else if (dxbox > 0.5*bounds[0] ) dxbox -= bounds[0];
                if (dybox<-0.5*bounds[1]) dybox += bounds[1];
                else if (dybox > 0.5*bounds[1] ) dybox -= bounds[1];
                if (dzbox<-0.5*bounds[2]) dzbox += bounds[2];
                else if (dzbox > 0.5*bounds[2] ) dzbox -= bounds[2];
            }

            // printf("dxbox, dybox, dzbox = %f, %f, %f\n", dxbox, dybox, dzbox);

            for (int jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
	    {// loop over all groups in neighbor cell 

                int jParticle = jOffset + jAtom; // global offset of particle

                dx = xPos[iParticle] - xPos[jParticle] + dxbox;;
                dy = yPos[iParticle] - yPos[jParticle] + dybox;;
                dz = zPos[iParticle] - zPos[jParticle] + dzbox;;

#if(KERN_DIAG) 
                //printf("dx, dy, dz = %f, %f, %f\n", dx, dy, dz);
                //printf("i = %d, j = %d, %f, %f, %f\n", iParticle, jParticle, xPos[jParticle], yPos[jParticle], zPos[jParticle]);
#endif

                r2 = dx*dx + dy*dy + dz*dz;

                if ( r2 <= rCut2 && r2 > 0.0)
		{// no divide by zero

#if(KERN_DIAG) 
                    printf("%d, %d, %f\n", iParticle, jParticle, r2);
                    //printf("r2, rCut = %f, %f\n", r2, rCut);
#endif

                    // reciprocal of r2 now
                    r2 = (real_t)1.0/r2;

                    r6 = r2*r2*r2;

                    e += r6*(s6*r6 - 1.0);

#if(KERN_DIAG) 
                    //printf("%d, %d, %f\n", iParticle, jParticle, r2);
                    //printf("iParticle = %d, jParticle = %d, i_b = %d, r6 = %f\n", iParticle, jParticle, i_b, r6);
#endif

                    fr = -4.0*epsilon*s6*r2*r6*(12.0*r6*s6 - 6.0);

                    fxItem += dx*fr;
                    fyItem += dy*fr;
                    fzItem += dz*fr;

                } else {
                }


            } // loop over all atoms
        } // loop over neighbor cells

        fx[iParticle] = fxItem;
        fy[iParticle] = fyItem;
        fz[iParticle] = fzItem;

        // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
        energy[iParticle] = e*2.0*epsilon*s6;

    }
}


__kernel void ljForceAos(
        __global cl_real4* pos,
        __global cl_real4* f,
        __global real_t* energy,
        __global cl_real4* dc,
        __global cl_real4* bounds,
        __global int* neighborList,
        __global int* nNeighbors,
        __global int* nAtoms,
        const real_t sigma,
        const real_t epsilon,
        const real_t cutoff) 
{

    int iAtom = get_global_id(0);
    int iBox = get_global_id(1);
    int maxAtoms = get_global_size(0);

    cl_real4 dr;
    real_t r, r2, r6;
    real_t fr, e;

    cl_real4 drBox;

    // accumulate local force value
    cl_real4 fItem;

    real_t rCut = cutoff;
    real_t rCut2 = rCut*rCut;
    real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

    int j, j_local;
    int jBox, jAtom;

    int iOffset, jOffset;
    int iParticle, jParticle;

    // zero out forces on particles
    fItem.x = 0.0;
    fItem.y = 0.0;
    fItem.z = 0.0;

    e = 0.0;

    iOffset = iBox*maxAtoms; //N_MAX_ATOMS;
    iParticle = iOffset + iAtom;

    if (iAtom < nAtoms[iBox])
    {// each thread executes on a single atom in the box

#if(KERN_DIAG) 
        //if (iBox < 2) printf("i = %d, %e, %e, %e\n", iParticle, pos[iParticle].x, pos[iParticle].y, pos[iParticle].z);

        //printf("iBox = %d, nNeighbors = %d\n", iBox, nNeighbors[iBox]);
        //printf("iBox = %d, nAtoms = %d\n", iBox, nAtoms[iBox]);
#endif

        for (j = 0; j<nNeighbors[iBox]; j++)
	{// loop over neighbor cells
            int jBox = neighborList[iBox*N_MAX_NEIGHBORS + j];
            jOffset = jBox*maxAtoms; //N_MAX_ATOMS;

            // compute box center offsets
            drBox = dc[iBox] - dc[jBox];

            // correct for periodic 
            if(PERIODIC)
	    {
                if (drBox.x<-0.5*bounds[0].x) drBox.x += bounds[0].x;
                else if (drBox.x > 0.5*bounds[0].x ) drBox.x -= bounds[0].x;
                if (drBox.y<-0.5*bounds[0].y) drBox.y += bounds[0].y;
                else if (drBox.y > 0.5*bounds[0].y ) drBox.y -= bounds[0].y;
                if (drBox.z<-0.5*bounds[0].z) drBox.z += bounds[0].z;
                else if (drBox.z > 0.5*bounds[0].z ) drBox.z -= bounds[0].z;
            }

            // printf("dxbox, dybox, dzbox = %f, %f, %f\n", drBox.x, drBox.y, drBox.z);

            for (jAtom = 0; jAtom<nAtoms[jBox]; jAtom++)
	    {// loop over all groups in neighbor cell 

                jParticle = jOffset + jAtom; // global offset of particle

                dr = pos[iParticle] - pos[jParticle] + drBox;

#if(KERN_DIAG) 
                //printf("dx, dy, dz = %f, %f, %f\n", dr.x, dr.y, dr.z);
                //printf("i = %d, j = %d, %f, %f, %f\n", iParticle, jParticle, pos.x[jParticle], pos.y[jParticle], pos.z[jParticle]);
#endif

                r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

                if ( r2 <= rCut2 && r2 > 0.0)
		{// no divide by zero

#if(KERN_DIAG) 
                    //printf("%d, %d, %f\n", iParticle, jParticle, r2);
                    //printf("r2, rCut = %f, %f\n", r2, rCut);
#endif

                    // reciprocal of r2 now
                    r2 = (real_t)1.0/r2;

                    r6 = r2*r2*r2;

                    e += r6*(s6*r6 - 1.0);

#if(KERN_DIAG) 
                    //printf("%d, %d, %f\n", iParticle, jParticle, r2);
                    //printf("iParticle = %d, jParticle = %d, i_b = %d, r6 = %f\n", iParticle, jParticle, i_b, r6);
#endif

                    fr = -4.0*epsilon*s6*r2*r6*(12.0*r6*s6 - 6.0);

                    fItem.x += dr.x*fr;
                    fItem.y += dr.y*fr;
                    fItem.z += dr.z*fr;

                } else {
                }


            } // loop over all atoms
        } // loop over neighbor cells

        f[iParticle] = fItem;

        // since we loop over all particles, each particle contributes 1/2 the pair energy to the total
        energy[iParticle] = e*2.0*epsilon*s6;

    }
}

/*
   __kernel void ljForce(
   __global real_t* xPos,
   __global real_t* yPos,
   __global real_t* zPos,
   __global real_t* fx,
   __global real_t* fy,
   __global real_t* fz,
   __global real_t* energy,
   __global int* neighborList,
   __global int* nNeighbors,
   __local real_t* x_ii,
   __local real_t* y_ii,
   __local real_t* z_ii,
   __local real_t* x_ij,
   __local real_t* y_ij,
   __local real_t* z_ij,
   const real_t sigma,
   const real_t epsilon,
   const int nCells) 
   {

   int iAtom = get_global_id(0);
   int iBox = get_global_id(1);
   int iLocal = get_local_id(0);
   int n_groups = get_num_groups(0);
   int n_items = get_local_size(0);



#if(KERN_DIAG) 
printf("Number of work groups: %d\n", n_groups);
printf("Number of work items: %d\n", n_items);
#endif

real_t dx, dy, dz;
real_t r, r2, r6;
real_t fr, e;
real_t fx_ii, fy_ii, fz_ii;

real_t rCut = 5.0*sigma;
real_t rCut2 = rCut*rCut;
real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

int j, j_local;
int i_b, i_p;
int j_b;

int cell_offset, jOffset;
int group_offset;
int iParticle, jParticle;

// zero out forces on particles
fx_ii = 0.0;
fy_ii = 0.0;
fz_ii = 0.0;

e = 0.0;

i_b = get_group_id(0);

cell_offset = iBox*N_MAX_ATOMS;
group_offset = n_items*i_b;
iParticle = group_offset + iLocal;

#if(KERN_DIAG) 
//printf("iParticle = %d\n", iParticle);
//printf("i_global = %d, iLocal = %d, i_b = %d, n_items = %d\n", i_global, iLocal, i_b, n_items);
#endif

// load particle data into local arrays
x_ii[iLocal] = xPos[iParticle + cell_offset];
y_ii[iLocal] = yPos[iParticle + cell_offset];
z_ii[iLocal] = zPos[iParticle + cell_offset];

barrier(CLK_LOCAL_MEM_FENCE);

#if(KERN_DIAG) 
//printf("x_ii, y_ii, z_ii = %f, %f, %f\n", x_ii[iLocal], y_ii[iLocal], z_ii[iLocal]);
printf("%d, %f, %f, %f\n", iParticle, x_ii[iLocal], y_ii[iLocal], z_ii[iLocal]);
#endif

for (j = 0; j<nNeighbors[iBox]; j++)
{// loop over neighbor cells
    jOffset = neighborList[iBox*N_MAX_NEIGHBORS + j]*N_MAX_ATOMS;
    for (j_b = 0; j_b<n_groups; j_b++)
    {// loop over all groups in neighbor cell 

        // use iLocal to load data in blocks of size n_items
        x_ij[iLocal] = xPos[iLocal + j_b*n_items + jOffset];
        y_ij[iLocal] = yPos[iLocal + j_b*n_items + jOffset];
        z_ij[iLocal] = zPos[iLocal + j_b*n_items + jOffset];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (j_local=0;j_local < n_items; j_local ++)
	{// loop over all atoms in group

            jParticle = j_local+ j_b*n_items; // global offset of particle

            dx = x_ii[iLocal] - x_ij[j_local];
            dy = y_ii[iLocal] - y_ij[j_local];
            dz = z_ii[iLocal] - z_ij[j_local];

#if(KERN_DIAG) 
            //printf("dx, dy, dz = %f, %f, %f\n", dx, dy, dz);
            printf("%d, %f, %f, %f\n", jParticle, x_ij[j_local], y_ij[j_local], z_ij[j_local]);
            printf("%d, %d, %f, %f, %f\n", iParticle, jParticle, dx, dy, dz);
#endif

            r2 = dx*dx + dy*dy + dz*dz;

#if(KERN_DIAG) 
            printf("%d, %d, %f\n", iParticle, jParticle, r2);
            //printf("r2, rCut = %f, %f\n", r2, rCut);
#endif

            if ( r2 <= rCut2 && r2 > 0.0)
	    {// no divide by zero

                // reciprocal of r2 now
                r2 = (real_t)1.0/r2;

                r6 = r2*r2*r2;

                e += r6*(s6*r6 - 1.0);

#if(KERN_DIAG) 
                //printf("%d, %d, %f\n", iParticle, jParticle, r2);
                //printf("iParticle = %d, jParticle = %d, i_b = %d, r6 = %f\n", iParticle, jParticle, i_b, r6);
#endif

                fr = 4.0*epsilon*s6*r2*r6*(12.0*r6*s6 - 6.0);

                fx_ii += dx*fr;
                fy_ii += dy*fr;
                fz_ii += dz*fr;

            } else {
            }

        } // loop over all atoms in group

    } // loop over all groups in neighbor cell
} // loop over neighbor cells

fx[iParticle + cell_offset] = fx_ii;
fy[iParticle + cell_offset] = fy_ii;
fz[iParticle + cell_offset] = fz_ii;

barrier(CLK_LOCAL_MEM_FENCE);

// since we loop over all particles, each particle contributes 1/2 the pair energy to the total
energy[iParticle + cell_offset] = e*2.0*epsilon*s6;

}
*/
