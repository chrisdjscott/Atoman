
/*******************************************************************************
 ** Adaptive Common Neighbour Analysis (Stutowski...)
 ** Copyright Chris Scott 2014
 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constants.h"
#include "boxeslib.h"
#include "neb_list.h"
#include "utilities.h"
#include "acna.h"

/* function prototypes */
int compare_two_nebs(const void *, const void *);

/* some parameters/constants */
int MAX_REQUIRED_NEBS = 16;
int MIN_REQUIRED_NEBS = 12;





/*******************************************************************************
 ** Function that compares two elements in a neighbour list
 *******************************************************************************/
int compare_two_nebs(const void * a, const void * b)
{
    const struct Neighbour *n1 = a;
    const struct Neighbour *n2 = b;
    
    if (n1->sep2 < n2->sep2)
    {
        return -1;
    }
    else if (n1->sep2 > n2->sep2)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*******************************************************************************
 ** perform adaptive common neighbour analysis
 *******************************************************************************/
int adaptiveCommonNeighbourAnalysis(int NVisibleIn, int* visibleAtoms, int posDim, double *pos, int scalarsDim, double *scalars, 
                                    double *minPos, double *maxPos, double *cellDims, int *PBC, int NScalars, double *fullScalars,
                                    double maxBondDistance)
{
    int i, NVisible, index, j;
    double *visiblePos, approxBoxWidth, maxSep2;
    struct Boxes *boxes;
    struct NeighbourList2 *nebList;
    
    
/* first we construct neighbour list, containing indexes and separation */
    
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
    
    /* create neighbour list */
    nebList = constructNeighbourList2(NVisibleIn, visiblePos, boxes, cellDims, PBC, maxSep2);
    
    /* only required for building neb list */
    free(visiblePos);
    freeBoxes(boxes);
    
    printf("atom 0 has %d nebs\n", nebList[0].neighbourCount);
    
/* now we order the neighbour lists */
    
    /* if less than min neighbours, mark as disorderd!!! */
    
    /* sort neighbours by distance */
    for (i = 0; i < NVisibleIn; i++)
    {
        if (i==0)
        {
            printf("NEBS %d:\n", i);
            for (j=0; j<nebList[i].neighbourCount; j++)
            {
                printf("  VIS %8d; SEP %lf\n", nebList[i].neighbour[j].index, sqrt(nebList[i].neighbour[j].sep2));
            }
        }
        
        qsort(nebList[i].neighbour, nebList[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
        
        /* check sorted (debugging...) */
        if (i==0)
        {
            printf("NEBS %d:\n", i);
            for (j=0; j<nebList[i].neighbourCount; j++)
            {
                printf("  VIS %8d; SEP %lf\n", nebList[i].neighbour[j].index, sqrt(nebList[i].neighbour[j].sep2));
            }
        }
        
    }
    
    
    
    
    
    
/* there should be option to only show atoms of specific structure type */
    
    
    
    
    /* free */
    freeNeighbourList2(nebList, NVisibleIn);
    
    return 0;
}



