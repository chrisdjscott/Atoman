
/*******************************************************************************
 ** Calculate Steinhardt order parameters
 ** Copyright Chris Scott 2014
 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_legendre.h>
#include "boxeslib.h"
#include "neb_list.h"
#include "utilities.h"
#include "bond_order.h"

//double plgndr(int, int, double);
void Ylm(int, int, double, double, double*, double*);
void convertToSphericalCoordinates(double, double, double, double, double*, double*);
void complex_qlm(int, int*, struct NeighbourList*, double*, double*, int*, struct AtomStructureResults*);
void calculate_Q(int, struct AtomStructureResults*);


/*******************************************************************************
 ** Compute Y_lm (spherical harmonics)
 *******************************************************************************/
void Ylm(int l, int m, double theta, double phi, double *realYlm, double *imgYlm)
{
    double factor, arg;
    
    
    /* call GSL spherical P_lm function */
    factor = gsl_sf_legendre_sphPlm(l, m, cos(theta));
    
    arg = (double) m * phi;
    *realYlm = factor * cos(arg);
    *imgYlm = factor * sin(arg);
}

/*******************************************************************************
 ** Convert to spherical coordinates
 *******************************************************************************/
void convertToSphericalCoordinates(double xdiff, double ydiff, double zdiff, double sep, double *phi, double *theta)
{
    *theta = acos(zdiff / sep);
    *phi = atan2(ydiff, xdiff);
}

/*******************************************************************************
 ** Compute complex q_lm (sum over eq. 3 from Stutowski paper), for each atom
 *******************************************************************************/
void complex_qlm(int NVisibleIn, int *visibleAtoms, struct NeighbourList *nebList, double *pos, double *cellDims, int *PBC, struct AtomStructureResults *results)
{
    int i, index, index2, visIndex2, m, visIndex;
    double realYlm, complexYlm;
    double xpos1, ypos1, zpos1, xpos2, ypos2, zpos2;
    double sepVec[3], theta, phi, real_part, img_part;
    
    
    /* loop over atoms */
    for (visIndex = 0; visIndex < NVisibleIn; visIndex++)
    {
        /* pos 1 */
        index = visibleAtoms[visIndex];
        xpos1 = pos[3*index];
        ypos1 = pos[3*index+1];
        zpos1 = pos[3*index+2];
        
        /* loop over m, l = 6 */
        for (m = -6; m < 7; m++)
        {
            real_part = 0.0;
            img_part = 0.0;
            for (i = 0; i < nebList[visIndex].neighbourCount; i++)
            {
                /* pos 2 */
                visIndex2 = nebList[visIndex].neighbour[i];
                index2 = visibleAtoms[visIndex2];
                xpos2 = pos[3*index2];
                ypos2 = pos[3*index2+1];
                zpos2 = pos[3*index2+2];
                
                /* separation vector */
                atomSeparationVector(sepVec, xpos1, ypos1, zpos1, xpos2, ypos2, zpos2, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                
                /* convert to spherical coordinates */
                convertToSphericalCoordinates(sepVec[0], sepVec[1], sepVec[2], nebList[visIndex].neighbourSep[i], &phi, &theta);
                
                /* calc Ylm */
                if (m < 0)
                {
                    Ylm(6, abs(m), theta, phi, &realYlm, &complexYlm);
                    realYlm = pow(-1.0, m) * realYlm;
                    complexYlm = pow(-1.0, m) * complexYlm;
                }
                else
                {
                    Ylm(6, m, theta, phi, &realYlm, &complexYlm);
                }
                
                /* sum */
                real_part += realYlm;
                img_part += complexYlm;
            }
            
            /* divide by num nebs */
            results[visIndex].realQ6[m+6] = real_part / ((double) nebList[visIndex].neighbourCount);
            results[visIndex].imgQ6[m+6] = img_part / ((double) nebList[visIndex].neighbourCount);
        }
        
        /* loop over m, l = 4 */
        for (m = -4; m < 5; m++)
        {
            real_part = 0.0;
            img_part = 0.0;
            for (i = 0; i < nebList[visIndex].neighbourCount; i++)
            {
                /* pos 2 */
                visIndex2 = nebList[visIndex].neighbour[i];
                index2 = visibleAtoms[visIndex2];
                xpos2 = pos[3*index2];
                ypos2 = pos[3*index2+1];
                zpos2 = pos[3*index2+2];
                
                /* separation vector */
                atomSeparationVector(sepVec, xpos1, ypos1, zpos1, xpos2, ypos2, zpos2, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                
                /* convert to spherical coordinates */
                convertToSphericalCoordinates(sepVec[0], sepVec[1], sepVec[2], nebList[visIndex].neighbourSep[i], &phi, &theta);
                
                /* calc Ylm */
                if (m < 0)
                {
                    Ylm(4, abs(m), theta, phi, &realYlm, &complexYlm);
                    realYlm = pow(-1.0, m) * realYlm;
                    complexYlm = pow(-1.0, m) * complexYlm;
                }
                else
                {
                    Ylm(4, m, theta, phi, &realYlm, &complexYlm);
                }
                
                /* sum */
                real_part += realYlm;
                img_part += complexYlm;
            }
            
            /* divide by num nebs */
            results[visIndex].realQ4[m+4] = real_part / ((double) nebList[visIndex].neighbourCount);
            results[visIndex].imgQ4[m+4] = img_part / ((double) nebList[visIndex].neighbourCount);
        }
    }
}

/*******************************************************************************
 ** calculate Q4/6 from complex q_lm's
 *******************************************************************************/
void calculate_Q(int NVisibleIn, struct AtomStructureResults *results)
{
    int i, m;
    double sumQ6, sumQ4;
    
    
    for (i = 0; i < NVisibleIn; i++)
    {
        sumQ6 = 0.0;
        for (m = 0; m < 13; m++)
        {
            sumQ6 += results[i].realQ6[m] * results[i].realQ6[m] + results[i].imgQ6[m] * results[i].imgQ6[m];
        }
        results[i].Q6 = pow(((4.0 * M_PI / 13.0) * sumQ6), 0.5);
        
        sumQ4 = 0.0;
        for (m = 0; m < 9; m++)
        {
            sumQ4 += results[i].realQ4[m] * results[i].realQ4[m] + results[i].imgQ4[m] * results[i].imgQ4[m];
        }
        results[i].Q4 = pow(((4.0 * M_PI / 9.0) * sumQ4), 0.5);
    }
}






/*******************************************************************************
 ** bond order filter
 *******************************************************************************/
int bondOrderFilter(int NVisibleIn, int* visibleAtoms, int posDim, double *pos, double maxBondDistance, 
                    int scalarsDim, double *scalarsQ4, double *scalarsQ6, double *minPos, double *maxPos, double *cellDims, int *PBC, 
                    int NScalars, double *fullScalars, int filterQ4Enabled, double minQ4, double maxQ4, int filterQ6Enabled, 
                    double minQ6, double maxQ6)
{
    int i, j, index, NVisible;
    int maxSep2;
    double approxBoxWidth, q4, q6;
    double *visiblePos;
    struct Boxes *boxes;
    struct NeighbourList *nebList;
    struct AtomStructureResults *results;
    
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisibleIn * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    /* box visible atoms */
    approxBoxWidth = maxBondDistance;
    maxSep2 = maxBondDistance * maxBondDistance;
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    
    /* build neighbour list (this should be separate function) */
    nebList = constructNeighbourList(NVisibleIn, visiblePos, boxes, cellDims, PBC, maxSep2);
    
    /* only required for building neb list */
    free(visiblePos);
    freeBoxes(boxes);
    
    /* allocate results structure */
    results = malloc(NVisibleIn * sizeof(struct AtomStructureResults));
    if (results == NULL)
    {
        printf("ERROR: could not allocate results\n");
        exit(50);
    }
    
    /* first calc q_lm for each atom over all m values */
    complex_qlm(NVisibleIn, visibleAtoms, nebList, pos, cellDims, PBC, results);
    
    /* calculate Q4 and Q6 */
    calculate_Q(NVisibleIn, results);
    
    /* do filtering here, storing results along the way */
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        q4 = results[i].Q4;
        q6 = results[i].Q6;
        
        if (filterQ4Enabled && (q4 < minQ4 || q4 > maxQ4))
            continue;
        
        if (filterQ6Enabled && (q6 < minQ6 || q6 > maxQ6))
            continue;
        
        /* add */
        visibleAtoms[NVisible] = visibleAtoms[i];
        
        /* store scalars */
        scalarsQ4[NVisible] = q4;
        scalarsQ6[NVisible] = q6;
        
        /* handle full scalars array */
        for (j = 0; j < NScalars; j++)
        {
            fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
        }
        
        NVisible++;
    }
    
    /* free */
    freeNeighbourList(nebList, NVisibleIn);
    free(results);
    
    return NVisible;
}
