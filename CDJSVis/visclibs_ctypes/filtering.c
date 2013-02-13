
/*******************************************************************************
 ** Copyright Chris Scott 2011
 ** Filtering routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../visclibs/utilities.h"
#include "../visclibs/boxeslib.h"
#include "filtering.h"


/*******************************************************************************
 ** Specie filter
 *******************************************************************************/
int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie)
{
    int i, j, index, match, NVisible;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        match = 0;
        for (j=0; j<visSpecDim; j++)
        {
            if (specie[index] == visSpec[j])
            {
                match = 1;
                break;
            }
        }
        
        if (match)
        {
            visibleAtoms[NVisible] = index;
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Slice filter
 *******************************************************************************/
int sliceFilter(int NVisibleIn, int *visibleAtoms, int posDim, double *pos, double x0,
                double y0, double z0, double xn, double yn, double zn, int invert)
{
    int i, NVisible, index;
    double mag, xd, yd, zd, dotProd, distanceToPlane;
    
    /* normalise (xn, yn, zn) */
    mag = sqrt(x0 * x0 + y0 * y0 + z0 * z0);
    xn = xn / mag;
    yn = yn / mag;
    zn = zn / mag;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        xd = pos[3*index] - x0;
        yd = pos[3*index+1] - y0;
        zd = pos[3*index+2] - z0;
        
        dotProd = xd * xn + yd * yn + zd * zn;
        distanceToPlane = dotProd / mag;
        
        if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
        {
            visibleAtoms[NVisible] = index;
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Crop sphere filter
 *******************************************************************************/
int cropSphereFilter(int NVisibleIn, int *visibleAtoms, int posDim, double *pos, double xCentre, 
                     double yCentre, double zCentre, double radius, double *cellDims, 
                     int *PBC, int invertSelection)
{
    int i, NVisible, index;
    double radius2, sep2;
    
    
    radius2 = radius * radius;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                 xCentre, yCentre, zCentre, 
                                 cellDims[0], cellDims[1], cellDims[2], 
                                 PBC[0], PBC[1], PBC[2]);
        
        if (sep2 < radius2)
        {
            if (invertSelection)
            {
                visibleAtoms[NVisible] = index;
                NVisible++;
            }
        }
        else
        {
            if (!invertSelection)
            {
                visibleAtoms[NVisible] = index;
                NVisible++;
            }
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Crop filter
 *******************************************************************************/
int cropFilter(int NVisibleIn, int* visibleAtoms, int posDim, double* pos, double xmin, double xmax,
               double ymin, double ymax, double zmin, double zmax, int xEnabled, int yEnabled, int zEnabled)
{
    int i, index, NVisible;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (xEnabled == 1)
        {
            if (pos[3*index] < xmin || pos[3*index] > xmax)
            {
                continue;
            }
        }
        
        if (yEnabled == 1)
        {
            if (pos[3*index+1] < ymin || pos[3*index+1] > ymax)
            {
                continue;
            }
        }
        
        if (zEnabled == 1)
        {
            if (pos[3*index+2] < zmin || pos[3*index+2] > zmax)
            {
                continue;
            }
        }
        
        visibleAtoms[NVisible] = index;
        NVisible++;
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Displacement filter
 *******************************************************************************/
int displacementFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int posDim, double *pos, int refPosDim, double *refPos, 
                       double *cellDims, int *PBC, double minDisp, double maxDisp)
{
    int i, NVisible, index;
    double sep2, maxDisp2, minDisp2;
    
    
    minDisp2 = minDisp * minDisp;
    maxDisp2 = maxDisp * maxDisp;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                 refPos[3*index], refPos[3*index+1], refPos[3*index+2], 
                                 cellDims[0], cellDims[1], cellDims[2], 
                                 PBC[0], PBC[1], PBC[2]);
        
        if (sep2 <= maxDisp2 && sep2 >= minDisp2)
        {
            visibleAtoms[NVisible] = index;
            scalars[NVisible] = sqrt(sep2);
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Kinetic energy filter
 *******************************************************************************/
int KEFilter(int NVisibleIn, int* visibleAtoms, int KEDim, double *KE, double minKE, double maxKE)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (KE[index] < minKE || KE[index] > maxKE)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Potential energy filter
 *******************************************************************************/
int PEFilter(int NVisibleIn, int* visibleAtoms, int PEDim, double *PE, double minPE, double maxPE)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (PE[index] < minPE || PE[index] > maxPE)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Charge energy filter
 *******************************************************************************/
int chargeFilter(int NVisibleIn, int* visibleAtoms, int chargeDim, double *charge, double minCharge, double maxCharge)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (charge[index] < minCharge || charge[index] > maxCharge)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 * Calculate coordination number
 *******************************************************************************/
int coordNumFilter(int NVisible, int *visibleAtoms, double *pos, int *specie, int NSpecies, double *bondMinArray, double *bondMaxArray, 
                   double approxBoxWidth, double *cellDims, int *PBC, double *minPos, double *maxPos,
                   double *coordArray, int minCoordNum, int maxCoordNum)
{
    int i, j, k, index, index2, visIndex;
    int speca, specb, count, NVisibleNew;
    int boxIndex, boxNebList[27];
    double *visiblePos, sep2, sep;
    double sepVec[3];
    struct Boxes *boxes;
    
    
//    printf("BONDS CLIB\n");
//    printf("N VIS: %d\n", NVisible);
//    
//    for (i=0; i<NSpecies; i++)
//    {
//        for (j=i; j<NSpecies; j++)
//        {
//            printf("%d - %d: %lf -> %lf\n", i, j, bondMinArray[i*NSpecies+j], bondMaxArray[i*NSpecies+j]);
//        }
//    }
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisible * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    /* box visible atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisible, visiblePos, boxes);
    
    /* free visible pos */
    free(visiblePos);
    
    /* zero coord array */
    for (i=0; i<NVisible; i++)
    {
        coordArray[i] = 0;
    }
    
    /* loop over visible atoms */
    count = 0;
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        
        speca = specie[index];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j=0; j<27; j++)
        {
            boxIndex = boxNebList[j];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                visIndex = boxes->boxAtoms[boxIndex][k];
                index2 = visibleAtoms[visIndex];
                
                if (index >= index2)
                {
                    continue;
                }
                
                specb = specie[index2];
                
                if (bondMinArray[speca*NSpecies+specb] == 0.0 && bondMaxArray[speca*NSpecies+specb] == 0.0)
                {
                    continue;
                }
                
                /* atomic separation */
                sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                sep = sqrt(sep2);
                
                /* check if these atoms are bonded */
                if (sep >= bondMinArray[speca*NSpecies+specb] && sep <= bondMaxArray[speca*NSpecies+specb])
                {
                    coordArray[i]++;
                    coordArray[visIndex]++;
                    
                    count++;
                }
            }
        }
    }
    
    /* filter */
    NVisibleNew = 0;
    for (i=0; i<NVisible; i++)
    {
        if (coordArray[i] >= minCoordNum && coordArray[i] <= maxCoordNum)
        {
            visibleAtoms[NVisibleNew] = visibleAtoms[i];
            coordArray[NVisibleNew] = coordArray[i];
            
            NVisibleNew++;
        }
    }
    
    /* free */
    freeBoxes(boxes);
    
    return NVisibleNew;
}

